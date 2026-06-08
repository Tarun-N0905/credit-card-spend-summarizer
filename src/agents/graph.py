"""
src/agents/graph.py

LangGraph state machine (ToolNode as an actual graph node).

Flow:
    history_loader → router → kb_agent → kb_tool_node → response → END
                           ├→ sql_search                ↗
                           ├→ both                     ↗
                           └→ general                       → END
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agents.nodes import (
    AgentState,
    history_loader_node,
    router_node,
    kb_agent_node,  # NEW
    sql_search_node,
    general_node,
    both_node,
    response_node,
    KB_TOOLS,  # NEW
)


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("history_loader", history_loader_node)
    graph.add_node("router", router_node)

    # NEW AGENT + TOOL NODES
    graph.add_node("kb_agent", kb_agent_node)
    graph.add_node("kb_tool_node", ToolNode(KB_TOOLS))

    graph.add_node("sql_search", sql_search_node)
    graph.add_node("general", general_node)
    graph.add_node("both", both_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("history_loader")

    graph.add_edge("history_loader", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "general"),
        {
            "knowledge_base": "kb_agent",
            "both": "both",
            "sql_query": "sql_search",
            "general": "general",
        },
    )

    # KB TOOL FLOW
    graph.add_edge("kb_agent", "kb_tool_node")
    graph.add_edge("kb_tool_node", "response")

    # OTHER FLOWS
    graph.add_edge("sql_search", "response")
    graph.add_edge("both", "response")

    graph.add_edge("response", END)
    graph.add_edge("general", END)

    compiled = graph.compile()
    print("[build_agent_graph] compiled successfully")
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
        "sql_executed": None,
        "sql_results": None,
        "sql_queries_run": [],
        "sql_facts": None,
        "response": None,
    }

    try:
        final_state = credit_card_agent.invoke(initial_state)

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
