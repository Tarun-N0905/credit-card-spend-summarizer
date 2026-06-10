"""
src/agents/nodes.py

LangGraph node implementations for the Credit Card Spend Summarizer.

Graph flow:
    history_loader → router → kb_agent  → kb_tool_node  → response → END
                           ├→ sql_agent → sql_tool_node ↗
                           ├→ both                      ↗
                           └→ general                        → END

Key design decisions
────────────────────
1.  KB retrieval is fully LLM-driven.
    A bound-tool LLM (kb_agent_llm) decides per-query whether to call
    `hybrid_search_tool` (hybrid: vector + FTS + rerank) or
    `vector_search_tool` (pure cosine similarity).  No keyword matching.

2.  SQL is fully LLM-driven via tools.
    A bound-tool LLM (sql_agent_llm) decides whether to call
    `nl2sql_execute` (single query) or `nl2sql_execute_multi` (two queries
    for comparisons / multi-dataset questions).  No keyword matching, no
    hardcoded query pipelines.

3.  Both routes use LangGraph ToolNode so tool-call ↔ tool-result message
    pairs are handled correctly by the framework.
"""

import json
import re
import operator
from typing import Annotated, TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from src.api.v1.core.settings import get_settings
from src.api.v1.agents.prompts import (
    COMBINED_ANSWER_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
    SQL_AGENT_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
)
from src.api.v1.agents.schemas import AgentResponse
from src.api.v1.retrieval.hybrid_search import search_hybrid
from src.api.v1.core.db import (
    get_or_create_conversation,
    save_message,
    get_conversation_messages,
)
from src.api.v1.tools.kb_tools import KB_TOOLS, hybrid_search_tool, vector_search_tool
from src.api.v1.tools.sql_tools import SQL_TOOLS, _run_nl2sql

# State


class AgentState(TypedDict):
    query: str
    session_id: str
    conversation_history: Optional[list]
    route: Optional[str]
    messages: Annotated[
        list, operator.add
    ]  # LangGraph ToolNode reads/writes this; operator.add merges across nodes
    chunks: Optional[list]  # raw RetrievedChunk objects for image extraction
    kb_context: Optional[str]  # formatted string passed to response LLM
    sql_executed: Optional[str]
    sql_results: Optional[list]
    sql_queries_run: Optional[list]
    sql_facts: Optional[str]
    response: Optional[object]


# Helpers


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


def _get_sql_agent_llm() -> ChatOpenAI:
    """LLM bound with SQL tools so it can choose single vs multi query."""
    s = get_settings()
    llm = ChatOpenAI(model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key)
    return llm.bind_tools(SQL_TOOLS)


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


def _parse_sql_tool_messages(messages: list) -> tuple[str, list]:
    """
    Extract SQL and results from ToolMessage(s) written by sql_tool_node.

    Handles both single-tool (nl2sql_execute) and multi-tool
    (nl2sql_execute_multi) outputs.  Returns (sql_summary_str, merged_rows).
    """
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "", []

    all_sqls: list[str] = []
    all_rows: list[dict] = []

    for tm in tool_messages:
        try:
            payload = json.loads(tm.content)
        except Exception:
            continue

        if "sql" in payload:
            # nl2sql_execute output: {"sql": "...", "results": [...]}
            if payload.get("sql"):
                all_sqls.append(payload["sql"])
            all_rows.extend(payload.get("results") or [])

        elif "query_a" in payload and "query_b" in payload:
            # nl2sql_execute_multi output
            for key in ("query_a", "query_b"):
                part = payload[key]
                if part.get("sql"):
                    all_sqls.append(part["sql"])
                all_rows.extend(part.get("results") or [])

    sql_summary = "\n\n".join(all_sqls)
    return sql_summary, all_rows


# Node 0 — History Loader


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


# Node 1 — Router


def router_node(state: AgentState) -> AgentState:
    """Classify query → knowledge_base | sql_query | both | general."""
    try:
        history = state.get("conversation_history") or []
        enriched = _enrich_query(state["query"], history)
        result = (ROUTER_PROMPT_TEMPLATE | _get_llm()).invoke(
            {"query": enriched},
            config={
                "run_name": "router",
                "metadata": {
                    "node": "router_node",
                    "session_id": state.get("session_id", ""),
                    "query": state["query"],
                },
            },
        )
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


# Node 2 — KB Agent  (LLM picks vector vs hybrid via tool call)


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
        ai_msg: AIMessage = kb_llm.invoke(
            [human_msg],
            config={
                "run_name": "kb_agent",
                "metadata": {
                    "node": "kb_agent_node",
                    "session_id": state.get("session_id", ""),
                    "query": state["query"],
                },
            },
        )

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            print("[kb_agent_node] no tool call; falling back to hybrid retrieval")
            from src.api.v1.services.rag_service import format_context

            chunks = search_hybrid(state["query"], top_k=5)
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


# Node 3 — SQL Agent  (LLM picks single vs multi query via tool call)


def sql_agent_node(state: AgentState) -> AgentState:
    """
    Fire the tool-bound SQL agent LLM so it emits an AIMessage with tool_calls.
    The LLM decides whether to call:
      • nl2sql_execute       — single NL→SQL query
      • nl2sql_execute_multi — two independent NL→SQL queries (comparisons)

    No keyword matching, no hardcoded routing — the LLM owns the decision.

    The actual tool execution happens in the next graph node (sql_tool_node,
    a LangGraph ToolNode) which reads state["messages"] automatically.

    Falls back to direct _run_nl2sql and short-circuits sql_executed /
    sql_results if the LLM returns no tool call.
    """
    try:
        sql_llm = _get_sql_agent_llm()
        history = state.get("conversation_history") or []
        history_text = _format_history(history)
        enriched = _enrich_query(state["query"], history)

        human_msg = HumanMessage(
            content=SQL_AGENT_PROMPT_TEMPLATE.format_messages(
                query=enriched,
                history=history_text,
            )[
                -1
            ].content  # extract rendered human turn content
        )
        ai_msg: AIMessage = sql_llm.invoke(
            [human_msg],
            config={
                "run_name": "sql_agent",
                "metadata": {
                    "node": "sql_agent_node",
                    "session_id": state.get("session_id", ""),
                    "query": state["query"],
                    "route": state.get("route", ""),
                },
            },
        )

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # LLM answered without a tool call — fall back to direct NL2SQL
            print("[sql_agent_node] no tool call; falling back to direct NL2SQL")
            sql, rows = _run_nl2sql(enriched)
            return {
                **state,
                "messages": [human_msg, ai_msg],
                "sql_executed": sql,
                "sql_results": rows,
                "sql_queries_run": [sql] if sql else [],
            }

        tool_name = tool_calls[0].get("name", "")
        print(f"[sql_agent_node] LLM selected tool: {tool_name!r}")
        return {**state, "messages": [human_msg, ai_msg]}

    except Exception as e:
        print(f"[sql_agent_node] failed: {e}")
        return {
            **state,
            "messages": [],
            "sql_executed": "",
            "sql_results": [],
            "sql_queries_run": [],
        }


# Node 4 — General (catch-all)


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
            },
            config={
                "run_name": "general",
                "metadata": {
                    "node": "general_node",
                    "session_id": state.get("session_id", ""),
                    "query": state["query"],
                },
            },
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


# Node 5 — Response


def response_node(state: AgentState) -> AgentState:
    """
    Generate the final customer-facing answer.

    Route handling:
      knowledge_base → KB_GENERATION_PROMPT  (uses kb_context from state or ToolMessages)
      sql_query      → SQL_ANSWER_PROMPT     (uses sql_executed + sql_results;
                                              reads from ToolMessages if agent ran)
      both           → COMBINED_ANSWER_PROMPT (kb_context + sql_results together)
    """
    try:
        llm = _get_llm()
        route = state.get("route", "knowledge_base")
        history_text = _format_history(state.get("conversation_history") or [])

        #  KB-only path
        if route == "knowledge_base":
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
                },
                config={
                    "run_name": "response_kb",
                    "metadata": {
                        "node": "response_node",
                        "route": "knowledge_base",
                        "session_id": state.get("session_id", ""),
                        "query": state["query"],
                    },
                },
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
            print(
                f"[response_node/kb] answer generated — {len(image_paths or [])} image(s)"
            )
            return {**state, "response": response}

        #  SQL-only path
        if route == "sql_query":
            # Prefer state values set by fallback in sql_agent_node.
            # If the tool loop ran, read from ToolMessages instead.
            sql_executed = state.get("sql_executed") or ""
            sql_results = state.get("sql_results") or []

            if not sql_executed:
                sql_executed, sql_results = _parse_sql_tool_messages(
                    state.get("messages") or []
                )

            result = (SQL_ANSWER_PROMPT_TEMPLATE | llm).invoke(
                {
                    "query": state["query"],
                    "sql_executed": sql_executed,
                    "sql_results": json.dumps(sql_results, indent=2, default=str),
                    "history": history_text,
                },
                config={
                    "run_name": "response_sql",
                    "metadata": {
                        "node": "response_node",
                        "route": "sql_query",
                        "session_id": state.get("session_id", ""),
                        "query": state["query"],
                        "sql_executed": sql_executed,
                    },
                },
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
            print("[response_node/sql] answer generated")
            return {**state, "response": response}

        #  Both path
        if route == "both":
            messages = state.get("messages") or []

            # Build tool_call_id → tool_name from all AIMessages.
            # Safer than relying on ToolMessage.name which may be None
            # depending on LangChain version.
            kb_tool_names = {"hybrid_search_tool", "vector_search_tool"}
            sql_tool_names = {"nl2sql_execute", "nl2sql_execute_multi"}

            call_id_to_tool: dict[str, str] = {}
            for m in messages:
                if isinstance(m, AIMessage):
                    for tc in getattr(m, "tool_calls", None) or []:
                        call_id_to_tool[tc["id"]] = tc["name"]

            kb_tool_msgs, sql_tool_msgs = [], []
            for m in messages:
                if not isinstance(m, ToolMessage):
                    continue
                tool_name = call_id_to_tool.get(m.tool_call_id, "") or m.name or ""
                if tool_name in kb_tool_names:
                    kb_tool_msgs.append(m)
                elif tool_name in sql_tool_names:
                    sql_tool_msgs.append(m)

            print(
                f"[response_node/both] kb_tool_msgs={len(kb_tool_msgs)} "
                f"sql_tool_msgs={len(sql_tool_msgs)}"
            )

            # KB context from ToolMessages, fallback to state
            kb_context = (
                "\n\n".join(m.content for m in kb_tool_msgs)
                or state.get("kb_context")
                or "No relevant documents found."
            )

            # SQL — parse from ToolMessages, fallback to state
            sql_executed, sql_results = _parse_sql_tool_messages(sql_tool_msgs)
            if not sql_executed:
                sql_executed = state.get("sql_executed") or ""
                sql_results = state.get("sql_results") or []

            chunks = state.get("chunks") or []
            sql_json = (
                json.dumps(sql_results, indent=2, default=str)
                if sql_results
                else "No account data found."
            )

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
                },
                config={
                    "run_name": "response_both",
                    "metadata": {
                        "node": "response_node",
                        "route": "both",
                        "session_id": state.get("session_id", ""),
                        "query": state["query"],
                        "sql_executed": sql_executed,
                    },
                },
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
