"""
src/agents/nodes.py

LangGraph node implementations for the Credit Card Spend Summarizer.

Graph flow:
    history_loader → router → kb_agent → kb_tool_node → response → END
                           ├→ sql_search               ↗
                           ├→ both                    ↗
                           └→ general                      → END

Key design decisions
────────────────────
1.  KB retrieval is now fully LLM-driven.
    A bound-tool LLM (kb_agent_llm) decides per-query whether to call
    `hybrid_search_tool` (hybrid: vector + FTS + rerank) or
    `vector_search_tool` (pure cosine similarity).  No keyword matching.

2.  SQL is 100 % NL2SQL — no hardcoded queries, no static spend-summary
    pipeline.  Every SQL path generates SQL via NL2SQL_PROMPT_TEMPLATE,
    executes it, and hands the raw results to the response LLM.

3.  Tool execution uses LangGraph's ToolNode so tool-call ↔ tool-result
    message pairs are handled correctly by the framework.
"""

import json
import re
from typing import TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.core.settings import get_settings
from src.agents.prompts import (
    COMBINED_ANSWER_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
    NL2SQL_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
)
from src.agents.schemas import AgentResponse
from src.services.rag_service import retrieve as hybrid_retrieve, format_context
from src.services.sql_service import execute_sql
from src.retrieval.vector_search import search_semantic
from src.core.db import (
    get_or_create_conversation,
    save_message,
    get_conversation_messages,
)

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval tools — LLM picks one per KB query
# ─────────────────────────────────────────────────────────────────────────────


@tool
def hybrid_search_tool(query: str) -> str:
    """
    Search the credit card knowledge base using HYBRID search
    (vector similarity + full-text search + Cohere rerank).

    Use this when the query involves nuanced policy language, benefit
    comparisons, fee structures, eligibility criteria, or multi-concept
    questions where keyword matching adds recall on top of semantic search.

    Examples:
    - "What are the lounge access benefits?"
    - "How does the fee-waiver threshold work for Platinum?"
    - "Compare reward rates across card variants"
    """
    chunks = hybrid_retrieve(query, top_k=5)
    return format_context(chunks)


@tool
def vector_search_tool(query: str) -> str:
    """
    Search the credit card knowledge base using VECTOR-ONLY search
    (pure cosine similarity, no keyword boosting).

    Use this when the query is conversational or semantically clear but
    unlikely to benefit from exact keyword matching — e.g. follow-up
    questions, rephrased queries, or short factual lookups.

    Examples:
    - "What is the annual fee?"
    - "Tell me about cashback on dining"
    - "Is there an EMI conversion feature?"
    """
    chunks = search_semantic(query, top_k=5, rerank=True)
    return format_context(chunks)


# Registry of KB tools — passed to ToolNode (in graph.py) and bound to the KB agent LLM
KB_TOOLS = [hybrid_search_tool, vector_search_tool]


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    query: str
    session_id: str
    conversation_history: Optional[list]
    route: Optional[str]
    messages: Optional[list]  # LangGraph ToolNode reads/writes this
    chunks: Optional[list]  # raw RetrievedChunk objects for image extraction
    kb_context: Optional[str]  # formatted string passed to response LLM
    sql_executed: Optional[str]
    sql_results: Optional[list]
    sql_queries_run: Optional[list]
    sql_facts: Optional[str]
    response: Optional[object]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key
    )


def _get_kb_agent_llm() -> ChatOpenAI:
    """LLM bound with KB retrieval tools so it can choose vector vs hybrid."""
    s = get_settings()
    llm = ChatOpenAI(model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key)
    return llm.bind_tools(KB_TOOLS)


def _format_history(history: list) -> str:
    if not history:
        return ""
    return "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in history
    )


def _enrich_query(query: str, history: list) -> str:
    """Prepend recent history when the query lacks an explicit card-ID or month."""
    has_card = bool(re.search(r"\bCC-\d+\b", query, re.IGNORECASE))
    has_month = bool(
        re.search(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b", query)
        or re.search(
            r"\b(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+20\d{2}\b",
            query,
            re.IGNORECASE,
        )
    )
    if has_card and has_month:
        return query
    history_text = _format_history(history[-6:])
    if not history_text:
        return query
    return f"[Conversation so far]\n{history_text}\n\n[New question]\n{query}"


def _extract_image_paths(chunks: list) -> list[str]:
    """Collect image_path values from image-type chunks for Streamlit rendering."""
    seen: set[str] = set()
    paths: list[str] = []
    for chunk in chunks or []:
        content_type = getattr(chunk, "content_type", None) or (
            chunk.get("content_type") if isinstance(chunk, dict) else None
        )
        if content_type != "image":
            continue
        metadata = (
            getattr(chunk, "metadata", None)
            or (chunk.get("metadata") if isinstance(chunk, dict) else None)
            or {}
        )
        path = (
            metadata.get("image_path")
            if isinstance(metadata, dict)
            else getattr(metadata, "image_path", None)
        )
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _persist_turn(state: AgentState, assistant_reply: str) -> None:
    session_id = state.get("session_id", "")
    if not session_id:
        return
    try:
        conv_id = get_or_create_conversation(session_id)
        save_message(conv_id, role="user", content=state["query"])
        save_message(conv_id, role="assistant", content=assistant_reply)
    except Exception as e:
        print(f"[_persist_turn] failed: {e}")


def _run_nl2sql(enriched_query: str) -> tuple[str, list]:
    """
    Generate SQL from a natural-language query and execute it.
    Returns (sql_string, rows_list).  Never raises — returns ("", []) on error.
    """
    try:
        result = (NL2SQL_PROMPT_TEMPLATE | _get_llm()).invoke({"query": enriched_query})
        sql = result.content.strip()
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql).strip()
        print(f"[_run_nl2sql] generated SQL: {sql[:160]}")
        rows = execute_sql(sql)
        print(f"[_run_nl2sql] {len(rows)} row(s) returned")
        return sql, rows
    except Exception as e:
        print(f"[_run_nl2sql] failed: {e}")
        return "", []


def _run_kb_agent(query: str) -> tuple[str, list]:
    """
    Let the LLM choose between hybrid_search_tool (hybrid) and
    vector_search_tool (pure vector), call the retrieval function directly,
    and return (formatted_context_string, raw_chunks_for_image_extraction).
    Used by both_node. The knowledge_base route uses kb_agent_node + graph ToolNode instead.

    Returns ("No relevant documents found.", []) on any failure.
    """
    kb_llm = _get_kb_agent_llm()

    # Ask the LLM which tool to call
    messages = [HumanMessage(content=query)]
    ai_msg: AIMessage = kb_llm.invoke(messages)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if not tool_calls:
        # LLM answered without a tool call — fall back to hybrid
        print("[_run_kb_agent] LLM made no tool call; falling back to hybrid retrieval")
        chunks = hybrid_retrieve(query, top_k=5)
        return format_context(chunks), chunks

    tool_name = tool_calls[0].get("name", "")
    print(f"[_run_kb_agent] LLM selected tool: {tool_name!r}")

    # Call the retrieval function directly — no ToolNode needed here.
    # ToolNode is only used as a proper graph node (kb_tool_node in graph.py)
    # for the knowledge_base route. The both_node path uses this helper
    # directly and needs raw chunks, so we retrieve once and derive context.
    try:
        if tool_name == "vector_search_tool":
            raw_chunks = search_semantic(query, top_k=5, rerank=True)
        else:
            raw_chunks = hybrid_retrieve(query, top_k=5)
    except Exception:
        raw_chunks = []

    context_str = (
        format_context(raw_chunks) if raw_chunks else "No relevant documents found."
    )
    return context_str, raw_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Node 0 — History Loader
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Router
# ─────────────────────────────────────────────────────────────────────────────


def router_node(state: AgentState) -> AgentState:
    """Classify query → knowledge_base | sql_query | both | general."""
    try:
        history = state.get("conversation_history") or []
        enriched = _enrich_query(state["query"], history)
        result = (ROUTER_PROMPT_TEMPLATE | _get_llm()).invoke({"query": enriched})
        route = result.content.strip().lower()

        if "both" in route:
            route = "both"
        elif "sql_query" in route:
            route = "sql_query"
        elif "knowledge_base" in route:
            route = "knowledge_base"
        elif "general" in route:
            route = "general"
        else:
            print(f"[router_node] unexpected route '{route}', defaulting to general")
            route = "general"

        print(f"[router_node] route='{route}'")
        return {**state, "route": route}
    except Exception as e:
        print(f"[router_node] failed: {e}")
        return {**state, "route": "general"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — KB Agent  (LLM picks vector vs hybrid via tool call)
# ─────────────────────────────────────────────────────────────────────────────


def kb_agent_node(state: AgentState) -> AgentState:
    """
    Fire the tool-bound LLM so it emits an AIMessage with tool_calls.
    The actual tool execution happens in the next graph node (kb_tool_node,
    a LangGraph ToolNode) which reads state["messages"] automatically.

    Falls back to hybrid retrieval and short-circuits kb_context if the LLM
    returns no tool call (e.g. it answered directly).
    """
    try:
        kb_llm = _get_kb_agent_llm()
        human_msg = HumanMessage(content=state["query"])
        ai_msg: AIMessage = kb_llm.invoke([human_msg])

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # LLM answered without a tool call — fall back to hybrid and
            # short-circuit: store kb_context so response_node can use it.
            print("[kb_agent_node] no tool call; falling back to hybrid retrieval")
            chunks = hybrid_retrieve(state["query"], top_k=5)
            return {
                **state,
                "messages": [human_msg, ai_msg],
                "chunks": chunks,
                "kb_context": format_context(chunks),
            }

        tool_name = tool_calls[0].get("name", "")
        print(f"[kb_agent_node] LLM selected tool: {tool_name!r}")
        return {**state, "messages": [human_msg, ai_msg]}

    except Exception as e:
        print(f"[kb_agent_node] failed: {e}")
        return {
            **state,
            "messages": [],
            "chunks": [],
            "kb_context": "No relevant documents found.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — SQL Search  (pure NL2SQL — no hardcoded queries)
# ─────────────────────────────────────────────────────────────────────────────


def sql_search_node(state: AgentState) -> AgentState:
    """
    Convert the user query to SQL via the NL2SQL LLM chain, execute it,
    and store the results.  A single NL2SQL call covers all question types
    (transactions, balances, rewards, spend summaries, fee-waiver checks, etc.)
    — there are no hardcoded SQL templates or static query pipelines.
    """
    history = state.get("conversation_history") or []
    enriched = _enrich_query(state["query"], history)

    sql, rows = _run_nl2sql(enriched)

    return {
        **state,
        "sql_executed": sql,
        "sql_results": rows,
        "sql_queries_run": [sql] if sql else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — General (catch-all)
# ─────────────────────────────────────────────────────────────────────────────


def general_node(state: AgentState) -> AgentState:
    """
    Handle greetings, capability questions, and off-topic requests.
    Sets the response directly and exits without going through response_node.
    """
    try:
        history_text = _format_history(state.get("conversation_history") or [])
        result = (GENERAL_PROMPT_TEMPLATE | _get_llm()).invoke(
            {
                "query": state["query"],
                "history": history_text,
            }
        )
        answer = result.content.strip()

        response = AgentResponse(
            query=state["query"],
            answer=answer,
            data_sources=[],
            page_no="N/A",
            document_name="N/A",
            sql_query_executed=None,
            route_taken="general",
            image_paths=None,
        )
        _persist_turn(state, answer)
        print("[general_node] answer generated")
        return {**state, "response": response}

    except Exception as e:
        print(f"[general_node] failed: {e}")
        return {
            **state,
            "response": AgentResponse(
                query=state["query"],
                answer="Sorry, I encountered an error. Please try again.",
                data_sources=[],
                page_no="N/A",
                document_name="N/A",
                sql_query_executed=None,
                route_taken="general",
                image_paths=None,
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Both  (SQL + KB, then merge in response_node)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sql_facts(rows: list) -> str:
    """Pull key account facts from SQL rows to enrich the KB context string."""
    if not rows:
        return ""
    facts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, label in [
            ("card_variant", "Card variant"),
            ("card_id", "Card ID"),
            ("full_name", "Customer name"),
            ("billing_month", "Billing month"),
            ("total_purchases", "Total spend"),
            ("reward_pts_earned", "Reward points earned"),
            ("reward_points", "Current reward balance"),
        ]:
            if row.get(key) is not None:
                facts.append(f"{label}: {row[key]}")
    if not facts:
        return ""
    return "\n[Account facts from live data]\n" + "\n".join(dict.fromkeys(facts)) + "\n"


def both_node(state: AgentState) -> AgentState:
    """
    Run NL2SQL first, then KB retrieval (LLM-selected tool), and store
    both results for response_node to merge into a single answer.
    """
    history = state.get("conversation_history") or []
    enriched = _enrich_query(state["query"], history)

    # 1. SQL — pure NL2SQL, no hardcoded queries
    sql, rows = _run_nl2sql(enriched)
    sql_facts = _extract_sql_facts(rows)

    # 2. KB — LLM chooses vector vs hybrid
    try:
        kb_context, raw_chunks = _run_kb_agent(state["query"])
        print(f"[both_node] kb: {len(raw_chunks)} raw chunks")
    except Exception as e:
        print(f"[both_node] kb failed: {e}")
        kb_context, raw_chunks = "No relevant documents found.", []

    return {
        **state,
        "chunks": raw_chunks,
        "kb_context": kb_context,
        "sql_executed": sql,
        "sql_results": rows,
        "sql_facts": sql_facts,
        "sql_queries_run": [sql] if sql else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Response
# ─────────────────────────────────────────────────────────────────────────────


def response_node(state: AgentState) -> AgentState:
    """
    Generate the final customer-facing answer.

    Route handling:
      knowledge_base → KB_GENERATION_PROMPT  (uses kb_context from state)
      sql_query      → SQL_ANSWER_PROMPT     (uses sql_executed + sql_results)
      both           → COMBINED_ANSWER_PROMPT (kb_context + sql_results together)
    """
    try:
        llm = _get_llm()
        route = state.get("route", "knowledge_base")
        history_text = _format_history(state.get("conversation_history") or [])

        # ── KB-only path ─────────────────────────────────────────────────────
        if route == "knowledge_base":
            # kb_context may already be set (fallback path in kb_agent_node)
            # or it comes from ToolMessage results written by kb_tool_node.
            kb_context = state.get("kb_context")
            if not kb_context:
                tool_messages = [
                    m
                    for m in (state.get("messages") or [])
                    if isinstance(m, ToolMessage)
                ]
                kb_context = (
                    "\n\n".join(tm.content for tm in tool_messages)
                    or "No relevant documents found."
                )
            chunks = state.get("chunks") or []

            result = (KB_GENERATION_PROMPT_TEMPLATE | llm).invoke(
                {
                    "query": state["query"],
                    "context": kb_context,
                    "history": history_text,
                }
            )
            answer = result.content.strip()

            data_sources, page_numbers, doc_names = [], [], []
            for chunk in chunks:
                doc = getattr(chunk, "document_name", None) or (
                    chunk.get("document_name") if isinstance(chunk, dict) else None
                )
                page = getattr(chunk, "page_number", None) or (
                    chunk.get("page_number") if isinstance(chunk, dict) else None
                )
                sec = getattr(chunk, "section_name", None) or (
                    chunk.get("section_name") if isinstance(chunk, dict) else None
                )
                scr = getattr(chunk, "score", None) or (
                    chunk.get("score") if isinstance(chunk, dict) else None
                )
                if doc and doc not in doc_names:
                    doc_names.append(doc)
                if page and str(page) not in page_numbers:
                    page_numbers.append(str(page))
                data_sources.append(
                    {"document": doc, "page": page, "section": sec, "score": scr}
                )

            image_paths = _extract_image_paths(chunks) or None

            response = AgentResponse(
                query=state["query"],
                answer=answer,
                data_sources=data_sources,
                page_no=", ".join(page_numbers) if page_numbers else "N/A",
                document_name=", ".join(doc_names) if doc_names else "N/A",
                sql_query_executed=None,
                route_taken="knowledge_base",
                image_paths=image_paths,
            )
            _persist_turn(state, answer)
            print(
                f"[response_node/kb] answer generated — {len(image_paths or [])} image(s)"
            )
            return {**state, "response": response}

        # ── SQL-only path ─────────────────────────────────────────────────────
        if route == "sql_query":
            sql_executed = state.get("sql_executed") or ""
            sql_results = state.get("sql_results") or []

            result = (SQL_ANSWER_PROMPT_TEMPLATE | llm).invoke(
                {
                    "query": state["query"],
                    "sql_executed": sql_executed,
                    "sql_results": json.dumps(sql_results, indent=2, default=str),
                    "history": history_text,
                }
            )
            answer = result.content.strip()

            response = AgentResponse(
                query=state["query"],
                answer=answer,
                data_sources=[],
                page_no="N/A",
                document_name="credit_card_account_data",
                sql_query_executed=sql_executed or None,
                route_taken="sql_query",
                image_paths=None,
            )
            _persist_turn(state, answer)
            print("[response_node/sql] answer generated")
            return {**state, "response": response}

        # ── Both path ─────────────────────────────────────────────────────────
        if route == "both":
            chunks = state.get("chunks") or []
            kb_context = state.get("kb_context") or "No relevant documents found."
            sql_results = state.get("sql_results") or []
            sql_json = (
                json.dumps(sql_results, indent=2, default=str)
                if sql_results
                else "No account data found."
            )
            sql_executed = state.get("sql_executed", "")

            data_sources, page_numbers, doc_names = [], [], []
            for chunk in chunks:
                doc = getattr(chunk, "document_name", None) or (
                    chunk.get("document_name") if isinstance(chunk, dict) else None
                )
                page = getattr(chunk, "page_number", None) or (
                    chunk.get("page_number") if isinstance(chunk, dict) else None
                )
                sec = getattr(chunk, "section_name", None) or (
                    chunk.get("section_name") if isinstance(chunk, dict) else None
                )
                scr = getattr(chunk, "score", None) or (
                    chunk.get("score") if isinstance(chunk, dict) else None
                )
                if doc and doc not in doc_names:
                    doc_names.append(doc)
                if page and str(page) not in page_numbers:
                    page_numbers.append(str(page))
                data_sources.append(
                    {"document": doc, "page": page, "section": sec, "score": scr}
                )

            image_paths = _extract_image_paths(chunks)

            result = (COMBINED_ANSWER_PROMPT_TEMPLATE | llm).invoke(
                {
                    "query": state["query"],
                    "kb_context": kb_context,
                    "sql_results": sql_json,
                    "history": history_text,
                }
            )
            answer = result.content.strip()

            response = AgentResponse(
                query=state["query"],
                answer=answer,
                data_sources=data_sources,
                page_no=", ".join(page_numbers) if page_numbers else "N/A",
                document_name=", ".join(doc_names) if doc_names else "N/A",
                sql_query_executed=sql_executed or None,
                route_taken="both",
                image_paths=image_paths or None,
            )
            _persist_turn(state, answer)
            print(f"[response_node/both] merged answer — {len(image_paths)} image(s)")
            return {**state, "response": response}

        # ── Unexpected route — safe fallback ──────────────────────────────────
        raise ValueError(f"response_node received unexpected route: {route!r}")

    except Exception as e:
        print(f"[response_node] failed: {e}")
        return {
            **state,
            "response": AgentResponse(
                query=state["query"],
                answer="Sorry, I encountered an error generating a response. Please try again.",
                data_sources=[],
                page_no="N/A",
                document_name="N/A",
                sql_query_executed=None,
                route_taken=state.get("route", "unknown"),
                image_paths=None,
            ),
        }
