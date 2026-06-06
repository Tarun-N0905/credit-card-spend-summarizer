# src/agents/graph.py

"""
LangGraph state machine for the credit card RAG agent.

Workflow:
    router_node
        ├─→ "knowledge_base" ──→ kb_search_node ──→ response_node ──→ END
        └─→ "sql_query" ─────────────────────────→ sql_search_node ──→ response_node ──→ END

The router decides which path based on query classification.
Both paths converge at response_node to generate the final answer.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.nodes import (
    AgentState,
    router_node,
    kb_search_node,
    sql_search_node,
    response_node,
)

logger = logging.getLogger(__name__)


def _route_decision(state: AgentState) -> Literal["knowledge_base", "sql_query"]:
    """Routing function: determine next node based on router decision."""
    route = state.get("route", "knowledge_base")
    return route


def build_agent_graph():
    """Build and compile the LangGraph state machine."""

    # Create the graph
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("kb_search", kb_search_node)
    graph.add_node("sql_search", sql_search_node)
    graph.add_node("response", response_node)

    # Set entry point
    graph.set_entry_point("router")

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "knowledge_base": "kb_search",
            "sql_query": "sql_search",
        }
    )

    # KB path: search → response → END
    graph.add_edge("kb_search", "response")
    graph.add_edge("response", END)

    # SQL path: search → response → END
    graph.add_edge("sql_search", "response")

    # Compile the graph
    compiled_agent = graph.compile()
    logger.info("[build_agent_graph] LangGraph compiled successfully")

    return compiled_agent


# Singleton: compile once at module load, reuse for all requests
credit_card_agent = build_agent_graph()


def run_credit_card_agent(query: str) -> dict:
    """
    Execute the credit card agent.

    Args:
        query: User's natural language question

    Returns:
        Dict with final response (see AgentResponse schema)
    """
    initial_state = {
        "query": query,
        "route": "",
        "router_reason": "",
        "retrieved_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "response": {}
    }

    try:
        logger.info(f"[run_credit_card_agent] Processing query: {query[:80]}...")
        final_state = credit_card_agent.invoke(initial_state)
        response = final_state.get("response", {})
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        logger.info(f"[run_credit_card_agent] Route taken: {response.get('route_taken', 'unknown')}")
        return response

    except Exception as exc:
        logger.error(f"[run_credit_card_agent] Error: {exc}")
        return {
            "query": query,
            "answer": f"Sorry, I encountered an error processing your query: {exc}",
            "data_sources": "error",
            "page_no": "N/A",
            "document_name": "error",
            "route_taken": "error"
        }