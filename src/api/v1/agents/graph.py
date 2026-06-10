import json
import logging
from typing import Generator

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.api.v1.agents.nodes import (
    AgentState,
    history_loader_node,
    router_node,
    kb_agent_node,
    sql_agent_node,
    response_node,
    KB_TOOLS,
    SQL_TOOLS,
    _get_llm,
    _format_history,
    _parse_sql_tool_messages,
)
# NOTE: general_node removed — general route handled inside response_node
from src.api.v1.agents.prompts import (
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
    COMBINED_ANSWER_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
)
from src.api.v1.core.settings import get_settings

logger = logging.getLogger(__name__)


def build_agent_graph():
    graph = StateGraph(AgentState)

    #  Nodes
    graph.add_node("history_loader", history_loader_node)
    graph.add_node("router", router_node)

    # KB agent + tool executor
    graph.add_node("kb_agent", kb_agent_node)
    graph.add_node("kb_tool_node", ToolNode(KB_TOOLS))

    # SQL agent + tool executor
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_tool_node", ToolNode(SQL_TOOLS))

    graph.add_node("response", response_node)

    #  Entry point
    graph.set_entry_point("history_loader")

    #  history_loader → router
    graph.add_edge("history_loader", "router")

    #  router → agent nodes (general now hits response_node directly)
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "general"),
        {
            "knowledge_base": "kb_agent",
            "sql_query": "sql_agent",
            "both": "kb_agent",  # enters the shared pipeline at kb_agent
            "general": "response",
        },
    )

    #  KB flow: agent → tool node → response
    graph.add_edge("kb_agent", "kb_tool_node")

    #  kb_tool_node branches: KB-only → response, both → sql_agent
    graph.add_conditional_edges(
        "kb_tool_node",
        lambda state: "sql_agent" if state.get("route") == "both" else "response",
        {
            "sql_agent": "sql_agent",
            "response": "response",
        },
    )

    #  SQL flow: agent → tool node → response
    graph.add_edge("sql_agent", "sql_tool_node")
    graph.add_edge("sql_tool_node", "response")

    graph.add_edge("response", END)

    compiled = graph.compile()
    logger.info("[build_agent_graph] compiled successfully")
    return compiled


credit_card_agent = build_agent_graph()


def run_credit_card_agent(query: str, session_id: str = "") -> dict:
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "conversation_history": None,
        "route": "",
        "messages": [],
        "chunks": [],
        "kb_context": None,
        "sql_executed": None,
        "sql_results": None,
        "sql_queries_run": [],
        "sql_facts": None,
        "response": None,
    }

    try:
        final_state = credit_card_agent.invoke(
            initial_state,
            config={
                "run_name": "credit_card_agent",
                "metadata": {
                    "session_id": session_id,
                    "query": query,
                    "mode": "sync",
                },
            },
        )

        response = final_state.get("response")
        if response is None:
            raise ValueError("Agent produced no response object")

        if hasattr(response, "model_dump"):
            response = response.model_dump()

        return response

    except Exception as e:
        return {
            "query": query,
            "answer": f"Sorry, I encountered an error: {e}",
            "data_sources": [],
            "page_no": "N/A",
            "document_name": "error",
            "route_taken": "error",
            "image_paths": None,
        }


# Streaming support
def _build_stream_prompt(final_state: AgentState):
    """
    Reconstruct the exact prompt inputs used by response_node, but return
    the (prompt_template, input_dict) so the caller can call .stream() on it.

    Returns (chain, input_dict, metadata_dict) where metadata carries
    route_taken, page_no, document_name, sql_query_executed, image_paths.
    """
    llm = _get_llm()
    route = final_state.get("route", "knowledge_base")
    history_text = _format_history(final_state.get("conversation_history") or [])
    messages = final_state.get("messages") or []

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
        chain = KB_GENERATION_PROMPT_TEMPLATE | llm
        inputs = {
            "query": final_state["query"],
            "context": kb_context,
            "history": history_text,
        }
        return chain, inputs, metadata

    if route == "sql_query":
        sql_executed = final_state.get("sql_executed") or ""
        sql_results = final_state.get("sql_results") or []
        if not sql_executed:
            sql_executed, sql_results = _parse_sql_tool_messages(messages)
        metadata["sql_query_executed"] = sql_executed or None
        metadata["document_name"] = "credit_card_account_data"
        chain = SQL_ANSWER_PROMPT_TEMPLATE | llm
        inputs = {
            "query": final_state["query"],
            "sql_executed": sql_executed,
            "sql_results": json.dumps(sql_results, indent=2, default=str),
            "history": history_text,
        }
        return chain, inputs, metadata

    if route == "both":
        from langchain_core.messages import AIMessage

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
        chain = COMBINED_ANSWER_PROMPT_TEMPLATE | llm
        inputs = {
            "query": final_state["query"],
            "kb_context": kb_context,
            "sql_results": (
                json.dumps(sql_results, indent=2, default=str)
                if sql_results
                else "No account data found."
            ),
            "history": history_text,
        }
        return chain, inputs, metadata

    # general — falls through to GENERAL_PROMPT_TEMPLATE
    chain = GENERAL_PROMPT_TEMPLATE | llm
    inputs = {"query": final_state["query"], "history": history_text}
    return chain, inputs, metadata


def run_credit_card_agent_stream(
    query: str, session_id: str = ""
) -> Generator[str, None, None]:
    """
    Run the full LangGraph pipeline (routing + retrieval/SQL), then stream
    the final LLM answer token-by-token as plain text chunks.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "conversation_history": None,
        "route": "",
        "messages": [],
        "chunks": [],
        "kb_context": None,
        "sql_executed": None,
        "sql_results": None,
        "sql_queries_run": [],
        "sql_facts": None,
        "response": None,
    }

    try:
        # Step 1 — run the full graph (retrieval/SQL happen here)
        final_state = credit_card_agent.invoke(
            initial_state,
            config={
                "run_name": "credit_card_agent_stream",
                "metadata": {
                    "session_id": session_id,
                    "query": query,
                    "mode": "stream",
                },
            },
        )
        route = final_state.get("route", "unknown")
        logger.info("[stream] graph done, route=%s", route)

        # Step 2 — rebuild the prompt and stream the answer.
        chain, inputs, metadata = _build_stream_prompt(final_state)

        for chunk in chain.stream(inputs):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield f"data: {json.dumps(token)}\n\n"

        yield "data: [DONE]\n\n"

        # Step 3 — emit metadata so the client can show sources etc
        response_obj = final_state.get("response")
        if response_obj is not None:
            if hasattr(response_obj, "model_dump"):
                response_dict = response_obj.model_dump()
            else:
                response_dict = response_obj
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
