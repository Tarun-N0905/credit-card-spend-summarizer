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

    # ── Generic NL2SQL path ─────────────────────────────
    try:
        # LLM generates SQL dynamically for any query
        result = (NL2SQL_PROMPT_TEMPLATE | _get_llm()).invoke({"query": enriched})
        sql = result.content.strip()
        # Remove any markdown fences
        sql = re.sub(r'^```(?:sql)?\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql).strip()
        print(f"[sql_search_node] generated SQL: {sql[:120]}")

        # Execute SQL
        rows = execute_sql(sql)

        return {**state,
                "sql_executed": sql,
                "sql_results": rows,
                "spend_context": None,
                "sql_queries_run": [sql]}

    except Exception as e:
        print(f"[sql_search_node] NL2SQL failed: {e}")
        return {**state,
                "sql_executed": "",
                "sql_results": [],
                "spend_context": None,
                "sql_queries_run": []}

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
