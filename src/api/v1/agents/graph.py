import json
import logging
from typing import Generator

import psycopg
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from src.api.v1.agents.nodes import (
    AgentState,
    router_node,
    kb_agent_node,
    sql_agent_node,
    response_node,
    KB_TOOLS,
    SQL_TOOLS,
    _get_llm,
    _parse_sql_tool_messages,
    _history_text_from_messages,  # ← single source of truth (was duplicated)
)
from src.api.v1.agents.prompts import (
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
    COMBINED_ANSWER_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
)
from src.api.v1.core.settings import get_settings

logger = logging.getLogger(__name__)


def _build_checkpointer() -> PostgresSaver:
    settings = get_settings()
    conn = psycopg.connect(settings.pg_connection_string, autocommit=True)
    saver = PostgresSaver(conn)
    saver.setup()
    logger.info("[checkpointer] PostgresSaver ready")
    return saver


def build_agent_graph(checkpointer: PostgresSaver):
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)

    graph.add_node("kb_agent", kb_agent_node)
    graph.add_node("kb_tool_node", ToolNode(KB_TOOLS))

    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_tool_node", ToolNode(SQL_TOOLS))

    graph.add_node("response", response_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "general"),
        {
            "knowledge_base": "kb_agent",
            "sql_query": "sql_agent",
            "both": "kb_agent",
            "general": "response",
        },
    )

    graph.add_edge("kb_agent", "kb_tool_node")

    graph.add_conditional_edges(
        "kb_tool_node",
        lambda state: "sql_agent" if state.get("route") == "both" else "response",
        {
            "sql_agent": "sql_agent",
            "response": "response",
        },
    )

    graph.add_edge("sql_agent", "sql_tool_node")
    graph.add_edge("sql_tool_node", "response")
    graph.add_edge("response", END)

    compiled = graph.compile(checkpointer=checkpointer)
    graph_image = compiled.get_graph().draw_mermaid_png()
    with open("reference/tool_workflow.png", "wb") as f:
        f.write(graph_image)
    logger.info("[build_agent_graph] compiled with PostgresSaver checkpointer")
    return compiled


_checkpointer = _build_checkpointer()
credit_card_agent = build_agent_graph(_checkpointer)


def _thread_config(session_id: str, mode: str = "sync") -> dict:
    return {
        "configurable": {"thread_id": session_id},
        "run_name": "credit_card_agent",
        "metadata": {"session_id": session_id, "mode": mode},
    }


def _initial_state(query: str, session_id: str) -> AgentState:
    """
    Only supply the fields for this turn. The checkpointer merges these
    with the replayed state (messages list) from prior turns automatically.
    """
    return {
        "query": query,
        "session_id": session_id,
        "route": "",
        "messages": [HumanMessage(content=query)],
        "chunks": [],
        "kb_context": None,
        "sql_executed": None,
        "sql_results": None,
        "sql_queries_run": [],
        "sql_facts": None,
        "response": None,
    }


def run_credit_card_agent(query: str, session_id: str = "") -> dict:
    try:
        final_state = credit_card_agent.invoke(
            _initial_state(query, session_id),
            config=_thread_config(session_id, mode="sync"),
        )

        response = final_state.get("response")
        if response is None:
            raise ValueError("Agent produced no response object")

        if hasattr(response, "model_dump"):
            response = response.model_dump()

        return response

    except Exception as e:
        logger.error("[run_credit_card_agent] error: %s", e)
        return {
            "query": query,
            "answer": f"Sorry, I encountered an error: {e}",
            "data_sources": [],
            "page_no": "N/A",
            "document_name": "error",
            "route_taken": "error",
            "image_paths": None,
        }


# _history_text_from_messages removed — imported from nodes.py


def _build_stream_prompt(final_state: AgentState):
    """
    Reconstruct the exact prompt used by response_node and return
    (chain, input_dict, metadata_dict) for streaming.
    History is derived from the replayed messages in state.
    """
    llm = _get_llm()
    route = final_state.get("route", "knowledge_base")
    messages = final_state.get("messages") or []
    history_text = _history_text_from_messages(messages)

    metadata = {
        "route_taken": route,
        "page_no": "N/A",
        "document_name": "N/A",
        "sql_query_executed": None,
        "image_paths": None,
    }

    if route == "knowledge_base":
        kb_context = final_state.get("kb_context")
        if not kb_context:
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            kb_context = (
                "\n\n".join(tm.content for tm in tool_messages)
                or "No relevant documents found."
            )
        return (
            KB_GENERATION_PROMPT_TEMPLATE | llm,
            {
                "query": final_state["query"],
                "context": kb_context,
                "history": history_text,
            },
            metadata,
        )

    if route == "sql_query":
        sql_executed = final_state.get("sql_executed") or ""
        sql_results = final_state.get("sql_results") or []
        if not sql_executed:
            sql_executed, sql_results = _parse_sql_tool_messages(messages)
        metadata["sql_query_executed"] = sql_executed or None
        metadata["document_name"] = "credit_card_account_data"
        return (
            SQL_ANSWER_PROMPT_TEMPLATE | llm,
            {
                "query": final_state["query"],
                "sql_executed": sql_executed,
                "sql_results": json.dumps(sql_results, indent=2, default=str),
                "history": history_text,
            },
            metadata,
        )

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

        kb_context = (
            "\n\n".join(m.content for m in kb_tool_msgs)
            or final_state.get("kb_context")
            or "No relevant documents found."
        )
        sql_executed, sql_results = _parse_sql_tool_messages(sql_tool_msgs)
        if not sql_executed:
            sql_executed = final_state.get("sql_executed") or ""
            sql_results = final_state.get("sql_results") or []

        metadata["sql_query_executed"] = sql_executed or None
        return (
            COMBINED_ANSWER_PROMPT_TEMPLATE | llm,
            {
                "query": final_state["query"],
                "kb_context": kb_context,
                "sql_results": (
                    json.dumps(sql_results, indent=2, default=str)
                    if sql_results
                    else "No account data found."
                ),
                "history": history_text,
            },
            metadata,
        )

    # general
    return (
        GENERAL_PROMPT_TEMPLATE | llm,
        {"query": final_state["query"], "history": history_text},
        metadata,
    )


def run_credit_card_agent_stream(
    query: str, session_id: str = ""
) -> Generator[str, None, None]:
    try:
        final_state = credit_card_agent.invoke(
            _initial_state(query, session_id),
            config=_thread_config(session_id, mode="stream"),
        )
        logger.info("[stream] graph done, route=%s", final_state.get("route"))

        chain, inputs, metadata = _build_stream_prompt(final_state)

        for chunk in chain.stream(inputs):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield f"data: {json.dumps(token)}\n\n"

        yield "data: [DONE]\n\n"

        response_obj = final_state.get("response")
        if response_obj is not None:
            response_dict = (
                response_obj.model_dump()
                if hasattr(response_obj, "model_dump")
                else response_obj
            )
            for key in (
                "route_taken",
                "page_no",
                "document_name",
                "sql_query_executed",
                "image_paths",
            ):
                if key in response_dict:
                    metadata[key] = response_dict[key]

        yield f"data: [META] {json.dumps(metadata)}\n\n"

    except Exception as e:
        logger.error("[stream] error: %s", e)
        yield f"data: [ERROR] {str(e)}\n\n"
