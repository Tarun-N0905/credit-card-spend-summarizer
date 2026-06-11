import json
import logging
import operator
from functools import lru_cache
from typing import Annotated, TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from src.api.v1.core.settings import get_settings
from src.api.v1.agents.prompts import (
    RouteDecision,
    COMBINED_ANSWER_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
    SQL_AGENT_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
)
from src.api.v1.agents.schemas import AgentResponse
from src.api.v1.tools.kb_tools import KB_TOOLS, hybrid_search_tool
from src.api.v1.tools.sql_tools import SQL_TOOLS, _run_nl2sql

logger = logging.getLogger(__name__)

KB_CONFIDENCE_THRESHOLD = 0.4


class AgentState(TypedDict):
    query: str
    session_id: str
    route: Optional[str]
    messages: Annotated[list, operator.add]
    chunks: Optional[list]
    kb_context: Optional[str]
    sql_executed: Optional[str]
    sql_results: Optional[list]
    sql_queries_run: Optional[list]
    sql_facts: Optional[str]
    response: Optional[object]


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key
    )


@lru_cache(maxsize=1)
def _get_kb_agent_llm():
    s = get_settings()
    llm = ChatOpenAI(model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key)
    return llm.bind_tools(KB_TOOLS)


@lru_cache(maxsize=1)
def _get_sql_agent_llm():
    s = get_settings()
    llm = ChatOpenAI(model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key)
    return llm.bind_tools(SQL_TOOLS)


def _history_text_from_messages(messages: list) -> str:
    """
    Build a plain-text history string from the replayed checkpoint messages.
    Takes the last 6 Human/AI messages, skips ToolMessages.
    """
    human_ai = [
        m for m in (messages or []) if isinstance(m, (HumanMessage, AIMessage))
    ][-8:]
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in human_ai
    )


def _enrich_query(query: str, messages: list) -> str:
    history_text = _history_text_from_messages(messages)
    if not history_text:
        return query
    return f"[Conversation so far]\n{history_text}\n\n[New question]\n{query}"


def _extract_image_paths(chunks: list) -> list[str]:
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
    """Extract SQL and results from ToolMessage(s) written by sql_tool_node."""
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
            if payload.get("sql"):
                all_sqls.append(payload["sql"])
            all_rows.extend(payload.get("results") or [])
        elif "query_a" in payload and "query_b" in payload:
            for key in ("query_a", "query_b"):
                part = payload[key]
                if part.get("sql"):
                    all_sqls.append(part["sql"])
                all_rows.extend(part.get("results") or [])

    return "\n\n".join(all_sqls), all_rows


def router_node(state: AgentState) -> dict:
    """Classify query → knowledge_base | sql_query | both | general."""
    try:
        enriched = _enrich_query(state["query"], state.get("messages") or [])
        router_chain = ROUTER_PROMPT_TEMPLATE | _get_llm().with_structured_output(
            RouteDecision
        )
        decision: RouteDecision = router_chain.invoke(
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
        route = decision.route
        logger.info("[router_node] route='%s'", route)
        # Only return the field we're changing — NOT {**state, ...}
        return {"route": route}
    except Exception as e:
        logger.error("[router_node] failed: %s", e)
        return {"route": "general"}


def kb_agent_node(state: AgentState) -> dict:
    """
    Fire the tool-bound LLM so it emits an AIMessage with tool_calls.
    Falls back to hybrid_search_tool (via .invoke) so a proper ToolMessage
    is written to state["messages"] — keeps the both-path kb_tool_msgs logic intact.
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
            logger.warning(
                "[kb_agent_node] no tool_calls — falling back to hybrid_search_tool directly"
            )
            raw = hybrid_search_tool.invoke({"query": state["query"]})
            fallback_tool_msg = ToolMessage(
                content=raw if isinstance(raw, str) else json.dumps(raw),
                tool_call_id="fallback",
                name="hybrid_search_tool",
            )
            return {"messages": [ai_msg, fallback_tool_msg]}

        return {"messages": [ai_msg]}

    except Exception as e:
        logger.error("[kb_agent_node] failed: %s", e)
        return {"messages": []}


def sql_agent_node(state: AgentState) -> dict:
    try:
        sql_llm = _get_sql_agent_llm()
        human_msg = HumanMessage(content=state["query"])
        ai_msg: AIMessage = sql_llm.invoke(
            [human_msg],
            config={
                "run_name": "sql_agent",
                "metadata": {
                    "node": "sql_agent_node",
                    "session_id": state.get("session_id", ""),
                    "query": state["query"],
                },
            },
        )
        return {"messages": [ai_msg]}
    except Exception as e:
        logger.error("[sql_agent_node] failed: %s", e)
        return {"messages": []}


def response_node(state: AgentState) -> dict:
    try:
        llm = _get_llm()
        route = state.get("route", "general")
        messages = state.get("messages") or []
        history_text = _history_text_from_messages(messages)

        #  general
        if route == "general":
            result = (GENERAL_PROMPT_TEMPLATE | llm).invoke(
                {"query": state["query"], "history": history_text},
                config={
                    "run_name": "response_general",
                    "metadata": {
                        "node": "response_node",
                        "route": "general",
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
            # Return response AND an AIMessage so the checkpoint stores the answer
            return {
                "response": response,
                "messages": [AIMessage(content=answer)],
            }

        #  knowledge_base
        if route == "knowledge_base":
            chunks = state.get("chunks") or []
            kb_context = state.get("kb_context")
            if not kb_context:
                tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
                kb_context = (
                    "\n\n".join(tm.content for tm in tool_messages)
                    or "No relevant documents found."
                )

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
            logger.info(
                "[response_node/kb] answer generated — %d image(s)",
                len(image_paths or []),
            )
            return {
                "response": response,
                "messages": [AIMessage(content=answer)],
            }

        #  sql_query
        if route == "sql_query":
            sql_executed = state.get("sql_executed") or ""
            sql_results = state.get("sql_results") or []
            if not sql_executed:
                sql_executed, sql_results = _parse_sql_tool_messages(messages)

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
            logger.info("[response_node/sql] answer generated")
            return {
                "response": response,
                "messages": [AIMessage(content=answer)],
            }

        #  both
        if route == "both":
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

            logger.info(
                "[response_node/both] kb_tool_msgs=%d sql_tool_msgs=%d",
                len(kb_tool_msgs),
                len(sql_tool_msgs),
            )

            kb_context = (
                "\n\n".join(m.content for m in kb_tool_msgs)
                or state.get("kb_context")
                or "No relevant documents found."
            )

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
            logger.info(
                "[response_node/both] merged answer — %d image(s)", len(image_paths)
            )
            return {
                "response": response,
                "messages": [AIMessage(content=answer)],
            }

        raise ValueError(f"response_node received unexpected route: {route!r}")

    except Exception as e:
        logger.error("[response_node] failed: %s", e)
        error_answer = (
            "Sorry, I encountered an error generating a response. Please try again."
        )
        return {
            "response": AgentResponse(
                query=state["query"],
                answer=error_answer,
                data_sources=[],
                page_no="N/A",
                document_name="N/A",
                sql_query_executed=None,
                route_taken=state.get("route", "unknown"),
                image_paths=None,
            ),
            "messages": [AIMessage(content=error_answer)],
        }
