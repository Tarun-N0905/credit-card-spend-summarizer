"""
src/agents/nodes.py

LangGraph node implementations for the Credit Card Spend Summarizer.

Graph flow:
    history_loader → router → kb_search   → response → END
                           └→ sql_search  ↗
"""

import datetime
import json
import re
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI

from src.core.settings import get_settings
from src.agents.prompts import (
    ROUTER_PROMPT_TEMPLATE,
    NL2SQL_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
    SPEND_SUMMARY_PROMPT_TEMPLATE,
)
from src.agents.schemas import (
    AgentResponse,
    SpendSummaryResponse,
    CategoryBreakdown,
    TopMerchant,
    InternationalSpend,
    RewardPointsSummary,
)
from src.services.rag_service import retrieve, format_context
from src.services.sql_service import execute_sql
from src.core.db import (
    get_or_create_conversation,
    save_message,
    get_conversation_messages,
)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    session_id: str
    conversation_history: Optional[list]
    route: Optional[str]
    chunks: Optional[list]
    sql_executed: Optional[str]
    sql_results: Optional[list]
    spend_context: Optional[dict]
    sql_queries_run: Optional[list]
    response: Optional[object]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key)


def _format_history(history: list) -> str:
    if not history:
        return ""
    return "\n".join(
        f"{m.get('role','user').capitalize()}: {m.get('content','')}"
        for m in history
    )


def _enrich_query(query: str, history: list) -> str:
    """Prepend recent history if the query lacks explicit card/month."""
    has_card  = bool(re.search(r'\bCC-\d+\b', query, re.IGNORECASE))
    has_month = bool(
        re.search(r'\b(20\d{2})[-/](0[1-9]|1[0-2])\b', query) or
        re.search(
            r'\b(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+20\d{2}\b',
            query, re.IGNORECASE,
        )
    )
    if has_card and has_month:
        return query
    history_text = _format_history(history[-6:])
    if not history_text:
        return query
    return f"[Conversation so far]\n{history_text}\n\n[New question]\n{query}"


def _extract_card_and_month(text: str):
    card_match = re.search(r'\bCC-\d+\b', text, re.IGNORECASE)
    card_id = card_match.group(0).upper() if card_match else None

    month_match = re.search(r'\b(20\d{2})[-/](0[1-9]|1[0-2])\b', text)
    if month_match:
        billing_month = f"{month_match.group(1)}-{month_match.group(2)}"
    else:
        names = {
            'january':'01','february':'02','march':'03','april':'04',
            'may':'05','june':'06','july':'07','august':'08',
            'september':'09','october':'10','november':'11','december':'12',
        }
        nm = re.search(r'\b(' + '|'.join(names) + r')\s+(20\d{2})\b', text, re.IGNORECASE)
        billing_month = f"{nm.group(2)}-{names[nm.group(1).lower()]}" if nm else None

    return card_id, billing_month


def _is_spend_summary(query: str) -> bool:
    keywords = [
        'summar','breakdown','spend','spent','spending','category',
        'merchant','international','reward point','mom','month-over-month',
        'compare','fee waiver','on track',
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def _persist_turn(state: AgentState, assistant_reply: str) -> None:
    session_id = state.get("session_id", "")
    if not session_id:
        return
    try:
        conv_id = get_or_create_conversation(session_id)
        save_message(conv_id, role="user",      content=state["query"])
        save_message(conv_id, role="assistant", content=assistant_reply)
    except Exception as e:
        print(f"[_persist_turn] failed: {e}")


# ─────────────────────────────────────────────
# Spend-summary SQL (7 queries)
# ─────────────────────────────────────────────

def _run_spend_summary_queries(card_id: str, billing_month: str):
    year_int  = int(billing_month.split('-')[0])
    month_int = int(billing_month.split('-')[1])
    prev_month = (
        f"{year_int-1}-12" if month_int == 1
        else f"{year_int}-{month_int-1:02d}"
    )

    ctx: dict = {"card_id": card_id, "billing_month": billing_month}
    sql_list  = []

    queries = {
        "customer_info": (
            f"SELECT c.full_name, cc.card_variant, cc.reward_points, "
            f"cc.credit_limit, cc.available_limit, cc.outstanding_amt "
            f"FROM credit_cards cc "
            f"JOIN customers c ON cc.customer_id = c.customer_id "
            f"WHERE cc.card_id = '{card_id}' LIMIT 1"
        ),
        "billing_statement": (
            f"SELECT billing_month, start_date::text, end_date::text, due_date::text, "
            f"opening_balance, total_purchases, total_payments, total_fees, "
            f"total_refunds, closing_balance, min_amount_due, reward_pts_earned "
            f"FROM billing_statements "
            f"WHERE card_id = '{card_id}' AND billing_month = '{billing_month}' LIMIT 1"
        ),
        "category_breakdown": (
            f"SELECT ct.category_name, COUNT(*) AS txn_count, "
            f"SUM(ct.amount) AS total_spend, SUM(ct.reward_pts_earned) AS points_earned "
            f"FROM card_transactions ct "
            f"JOIN billing_statements bs ON bs.card_id = ct.card_id "
            f"  AND bs.billing_month = '{billing_month}' "
            f"WHERE ct.card_id = '{card_id}' "
            f"  AND ct.txn_date BETWEEN bs.start_date AND bs.end_date "
            f"  AND ct.txn_type = 'purchase' "
            f"GROUP BY ct.category_name ORDER BY total_spend DESC"
        ),
        "top_merchants": (
            f"SELECT ct.merchant_name, SUM(ct.amount) AS total, COUNT(*) AS txns "
            f"FROM card_transactions ct "
            f"JOIN billing_statements bs ON bs.card_id = ct.card_id "
            f"  AND bs.billing_month = '{billing_month}' "
            f"WHERE ct.card_id = '{card_id}' "
            f"  AND ct.txn_date BETWEEN bs.start_date AND bs.end_date "
            f"  AND ct.txn_type = 'purchase' "
            f"GROUP BY ct.merchant_name ORDER BY total DESC LIMIT 5"
        ),
        "international_transactions": (
            f"SELECT ct.txn_date::text, ct.merchant_name, ct.amount, "
            f"ct.original_currency, ct.original_amount, ct.category_name "
            f"FROM card_transactions ct "
            f"JOIN billing_statements bs ON bs.card_id = ct.card_id "
            f"  AND bs.billing_month = '{billing_month}' "
            f"WHERE ct.card_id = '{card_id}' "
            f"  AND ct.is_international = TRUE "
            f"  AND ct.txn_date BETWEEN bs.start_date AND bs.end_date "
            f"  AND ct.txn_type = 'purchase' "
            f"ORDER BY ct.txn_date DESC"
        ),
        "reward_points": (
            f"SELECT cc.reward_points AS current_balance, "
            f"cc.reward_points * 0.25 AS redemption_value_inr, "
            f"COALESCE(SUM(rt.points_earned), 0) AS earned_this_cycle, "
            f"COALESCE(SUM(rt.points_redeemed), 0) AS redeemed_this_cycle "
            f"FROM credit_cards cc "
            f"LEFT JOIN reward_transactions rt ON rt.card_id = cc.card_id "
            f"  AND rt.txn_date BETWEEN ("
            f"    SELECT start_date FROM billing_statements "
            f"    WHERE card_id = '{card_id}' AND billing_month = '{billing_month}') "
            f"  AND ("
            f"    SELECT end_date FROM billing_statements "
            f"    WHERE card_id = '{card_id}' AND billing_month = '{billing_month}') "
            f"WHERE cc.card_id = '{card_id}' GROUP BY cc.reward_points"
        ),
        "mom_comparison": (
            f"SELECT TO_CHAR(txn_date, 'YYYY-MM') AS month, "
            f"SUM(amount) FILTER (WHERE txn_type = 'purchase') AS total_purchases, "
            f"COUNT(*) FILTER (WHERE txn_type = 'purchase') AS txn_count "
            f"FROM card_transactions WHERE card_id = '{card_id}' "
            f"  AND TO_CHAR(txn_date, 'YYYY-MM') IN ('{billing_month}', '{prev_month}') "
            f"GROUP BY TO_CHAR(txn_date, 'YYYY-MM') ORDER BY month"
        ),
        "fee_waiver": (
            f"SELECT cc.card_id, cc.card_variant, SUM(ct.amount) AS ytd_spend, "
            f"CASE cc.card_variant "
            f"  WHEN 'NorthStar Classic'   THEN 50000 "
            f"  WHEN 'NorthStar Gold'      THEN 100000 "
            f"  WHEN 'NorthStar Platinum'  THEN 300000 "
            f"  WHEN 'NorthStar Signature' THEN 700000 "
            f"  ELSE 100000 END AS fee_waiver_target, "
            f"GREATEST(0, CASE cc.card_variant "
            f"  WHEN 'NorthStar Classic'   THEN 50000 "
            f"  WHEN 'NorthStar Gold'      THEN 100000 "
            f"  WHEN 'NorthStar Platinum'  THEN 300000 "
            f"  WHEN 'NorthStar Signature' THEN 700000 "
            f"  ELSE 100000 END - SUM(ct.amount)) AS remaining_to_waiver "
            f"FROM credit_cards cc "
            f"JOIN card_transactions ct ON cc.card_id = ct.card_id "
            f"WHERE cc.card_id = '{card_id}' AND ct.txn_type = 'purchase' "
            f"  AND EXTRACT(YEAR FROM ct.txn_date) = {year_int} "
            f"GROUP BY cc.card_id, cc.card_variant"
        ),
    }

    for key, sql in queries.items():
        sql_list.append(sql)
        try:
            ctx[key] = execute_sql(sql)
        except Exception as e:
            print(f"[spend_summary] {key} query failed: {e}")
            ctx[key] = []

    return ctx, sql_list


# ─────────────────────────────────────────────
# Node 0: History Loader
# ─────────────────────────────────────────────

def history_loader_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    if not session_id:
        return {**state, "conversation_history": []}
    try:
        conv_id = get_or_create_conversation(session_id)
        messages = get_conversation_messages(conv_id)
        history = messages[-6:] if messages else []
        return {**state, "conversation_history": history}
    except Exception as e:
        print(f"[history_loader_node] failed: {e}")
        return {**state, "conversation_history": []}


# ─────────────────────────────────────────────
# Node 1: Router
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    try:
        history  = state.get("conversation_history") or []
        enriched = _enrich_query(state["query"], history)
        result   = (ROUTER_PROMPT_TEMPLATE | _get_llm()).invoke({"query": enriched})
        route    = result.content.strip().lower()

        if "sql_query" in route:
            route = "sql_query"
        elif "knowledge_base" in route:
            route = "knowledge_base"
        else:
            print(f"[router_node] unexpected route '{route}', defaulting to knowledge_base")
            route = "knowledge_base"

        print(f"[router_node] route='{route}'")
        return {**state, "route": route}
    except Exception as e:
        print(f"[router_node] failed: {e}")
        return {**state, "route": "knowledge_base"}


# ─────────────────────────────────────────────
# Node 2: KB Search
# ─────────────────────────────────────────────

def kb_search_node(state: AgentState) -> AgentState:
    try:
        chunks = retrieve(state["query"])
        print(f"[kb_search_node] {len(chunks)} chunks retrieved")
        return {**state, "chunks": chunks}
    except Exception as e:
        print(f"[kb_search_node] failed: {e}")
        return {**state, "chunks": []}


# ─────────────────────────────────────────────
# Node 3: SQL Search
# ─────────────────────────────────────────────

def sql_search_node(state: AgentState) -> AgentState:
    history  = state.get("conversation_history") or []
    enriched = _enrich_query(state["query"], history)

    if _is_spend_summary(state["query"]):
        card_id, billing_month = _extract_card_and_month(enriched)
        if card_id and billing_month:
            print(f"[sql_search_node] spend-summary path card={card_id} month={billing_month}")
            try:
                ctx, sql_list = _run_spend_summary_queries(card_id, billing_month)
                return {**state, "spend_context": ctx, "sql_queries_run": sql_list,
                        "sql_executed": None, "sql_results": None}
            except Exception as e:
                print(f"[sql_search_node] spend-summary failed: {e}")
                return {**state, "spend_context": {"error": str(e)}, "sql_queries_run": [],
                        "sql_executed": None, "sql_results": None}
        else:
            print(f"[sql_search_node] could not parse card/month from query, falling back to NL2SQL")

    # Generic NL2SQL path
    try:
        result = (NL2SQL_PROMPT_TEMPLATE | _get_llm()).invoke({"query": enriched})
        sql = result.content.strip()
        sql = re.sub(r'^```(?:sql)?\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql).strip()
        print(f"[sql_search_node] generated SQL: {sql[:120]}")
        rows = execute_sql(sql)
        return {**state, "sql_executed": sql, "sql_results": rows,
                "spend_context": None, "sql_queries_run": [sql]}
    except Exception as e:
        print(f"[sql_search_node] NL2SQL failed: {e}")
        return {**state, "sql_executed": "", "sql_results": [],
                "spend_context": None, "sql_queries_run": []}


# ─────────────────────────────────────────────
# Node 4: Response
# ─────────────────────────────────────────────

def response_node(state: AgentState) -> AgentState:
    try:
        llm          = _get_llm()
        route        = state.get("route", "knowledge_base")
        history_text = _format_history(state.get("conversation_history") or [])

        # ── KB path ───────────────────────────────────────────────────────
        if route == "knowledge_base":
            chunks  = state.get("chunks") or []
            context = format_context(chunks)
            result  = (KB_GENERATION_PROMPT_TEMPLATE | llm).invoke({
                "query": state["query"], "context": context, "history": history_text,
            })
            answer = result.content.strip()

            data_sources, page_numbers, doc_names = [], [], []
            for chunk in chunks:
                doc  = getattr(chunk, "document_name", None) or (chunk.get("document_name") if isinstance(chunk, dict) else None)
                page = getattr(chunk, "page_number",   None) or (chunk.get("page_number")   if isinstance(chunk, dict) else None)
                sec  = getattr(chunk, "section_name",  None) or (chunk.get("section_name")  if isinstance(chunk, dict) else None)
                scr  = getattr(chunk, "score",         None) or (chunk.get("score")         if isinstance(chunk, dict) else None)
                if doc and doc not in doc_names:
                    doc_names.append(doc)
                if page and str(page) not in page_numbers:
                    page_numbers.append(str(page))
                data_sources.append({"document": doc, "page": page, "section": sec, "score": scr})

            response = AgentResponse(
                query=state["query"], answer=answer,
                data_sources=data_sources,
                page_no=", ".join(page_numbers) if page_numbers else "N/A",
                document_name=", ".join(doc_names) if doc_names else "N/A",
                sql_query_executed=None, route_taken="knowledge_base",
            )
            _persist_turn(state, answer)
            print("[response_node] KB answer generated")
            return {**state, "response": response}

        # ── Spend-summary path ────────────────────────────────────────────
        spend_context = state.get("spend_context")
        if spend_context and "error" not in spend_context:
            context_json = json.dumps(spend_context, indent=2, default=str)
            result = (SPEND_SUMMARY_PROMPT_TEMPLATE | llm).invoke({
                "context_json": context_json, "history": history_text,
            })
            raw = result.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw).strip()

            try:
                llm_out      = json.loads(raw)
                summary_text = llm_out.get("summary_text", "Summary unavailable.") if isinstance(llm_out, dict) else str(llm_out)
                tip          = llm_out.get("tip", "") if isinstance(llm_out, dict) else ""
            except Exception:
                print("[response_node] LLM JSON parse failed, using raw text")
                summary_text = raw
                tip          = ""

            card_id           = spend_context.get("card_id", "")
            billing_month_key = spend_context.get("billing_month", "")
            try:
                billing_month_label = datetime.datetime.strptime(billing_month_key, "%Y-%m").strftime("%B %Y")
            except Exception:
                billing_month_label = billing_month_key

            customer_info = spend_context.get("customer_info") or []
            customer_name = "Customer"
            if customer_info:
                item = customer_info[0]
                customer_name = item.get("full_name", "Customer") if isinstance(item, dict) else str(item)
            customer_name = customer_name.split()[0]

            billing     = spend_context.get("billing_statement") or []
            total_spend = 0.0
            if billing:
                item = billing[0]
                total_spend = float(item.get("total_purchases") or 0) if isinstance(item, dict) else 0.0

            category_rows      = spend_context.get("category_breakdown") or []
            total_transactions = sum(int(r.get("txn_count", 0)) for r in category_rows)
            category_breakdown = [
                CategoryBreakdown(
                    category=r.get("category_name", "Unknown"),
                    amount=float(r.get("total_spend") or 0),
                    count=int(r.get("txn_count", 0)),
                    pct_of_total=round(float(r.get("total_spend") or 0) / total_spend * 100, 1) if total_spend else 0.0,
                )
                for r in category_rows
            ]

            merchant_rows = spend_context.get("top_merchants") or []
            top_merchants = [
                TopMerchant(merchant_name=r.get("merchant_name", "Unknown"), amount=float(r.get("total") or 0))
                for r in merchant_rows
            ]

            intl_rows  = spend_context.get("international_transactions") or []
            intl_total = sum(float(r.get("amount") or 0) for r in intl_rows)

            reward_rows   = spend_context.get("reward_points") or []
            points_earned = int(reward_rows[0].get("earned_this_cycle") or 0) if reward_rows else 0

            mom_rows       = spend_context.get("mom_comparison") or []
            mom_change_pct = None
            if len(mom_rows) == 2:
                prev = float(mom_rows[0].get("total_purchases") or 0)
                curr = float(mom_rows[1].get("total_purchases") or 0)
                if prev > 0:
                    mom_change_pct = round((curr - prev) / prev * 100, 1)

            response = SpendSummaryResponse(
                card_id=card_id, customer_name=customer_name,
                billing_month=billing_month_label, total_spend=total_spend,
                total_transactions=total_transactions, category_breakdown=category_breakdown,
                top_merchants=top_merchants,
                international_spend=InternationalSpend(total_amount=intl_total, transaction_count=len(intl_rows)),
                reward_points_earned=RewardPointsSummary(points_earned=points_earned, inr_value=round(points_earned * 0.25, 2)),
                mom_change_pct=mom_change_pct, summary_text=summary_text, tip=tip,
                route_taken="sql_query", sql_queries_executed=state.get("sql_queries_run"),
            )
            _persist_turn(state, summary_text)
            print("[response_node] spend-summary assembled")
            return {**state, "response": response}

        # ── Generic SQL path ──────────────────────────────────────────────
        sql_executed = state.get("sql_executed", "")
        sql_results  = state.get("sql_results") or []
        result = (SQL_ANSWER_PROMPT_TEMPLATE | llm).invoke({
            "query": state["query"],
            "sql_executed": sql_executed,
            "sql_results": json.dumps(sql_results, indent=2, default=str),
            "history": history_text,
        })
        answer = result.content.strip()
        response = AgentResponse(
            query=state["query"], answer=answer, data_sources=[],
            page_no="N/A", document_name="credit_card_account_data",
            sql_query_executed=sql_executed, route_taken="sql_query",
        )
        _persist_turn(state, answer)
        print("[response_node] generic SQL answer generated")
        return {**state, "response": response}

    except Exception as e:
        print(f"[response_node] failed: {e}")
        return {**state, "response": AgentResponse(
            query=state["query"],
            answer="Sorry, I encountered an error generating a response. Please try again.",
            data_sources=[], page_no="N/A", document_name="N/A",
            sql_query_executed=None, route_taken=state.get("route", "unknown"),
        )}
