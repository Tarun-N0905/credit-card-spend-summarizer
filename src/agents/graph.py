"""
src/agents/graph.py

LangGraph state machine for the Credit Card Spend Summarizer.

Flow:
    history_loader → router → kb_search   → response → END
                           └→ sql_search  ↗
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from src.agents.nodes import (
    AgentState,
    history_loader_node,
    router_node,
    kb_search_node,
    sql_search_node,
    response_node,
)


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("history_loader", history_loader_node)
    graph.add_node("router",         router_node)
    graph.add_node("kb_search",      kb_search_node)
    graph.add_node("sql_search",     sql_search_node)
    graph.add_node("response",       response_node)

    graph.set_entry_point("history_loader")
    graph.add_edge("history_loader", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "knowledge_base"),
        {"knowledge_base": "kb_search", "sql_query": "sql_search"},
    )

    graph.add_edge("kb_search",  "response")
    graph.add_edge("sql_search", "response")
    graph.add_edge("response",   END)

    compiled = graph.compile()
    print("[build_agent_graph] compiled successfully")
    return compiled


# Singleton — compiled once at module load
credit_card_agent = build_agent_graph()


def run_credit_card_agent(query: str, session_id: str = "") -> dict:
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "conversation_history": None,
        "route": "",
        "chunks": [],
        "sql_executed": None,
        "sql_results": None,
        "spend_context": None,
        "sql_queries_run": [],
        "response": None,
    }

    try:
        final_state = credit_card_agent.invoke(initial_state)
        response    = final_state.get("response")

        if response is None:
            raise ValueError("Agent produced no response object")

        if hasattr(response, "model_dump"):
            response = response.model_dump()

        print(f"[run_credit_card_agent] route={response.get('route_taken','?')}")
        return response

    except Exception as e:
        print(f"[run_credit_card_agent] error: {e}")
        return {
            "query":         query,
            "answer":        f"Sorry, I encountered an error: {e}",
            "data_sources":  [],
            "page_no":       "N/A",
            "document_name": "error",
            "route_taken":   "error",
        }
