"""
src/agents/graph.py

LangGraph state machine — both KB and SQL use ToolNode-backed agent nodes.

Flow:
    history_loader → router → kb_agent  → kb_tool_node  → response → END
                           ├→ sql_agent → sql_tool_node ↗
                           ├→ both → kb_agent → kb_tool_node → sql_agent → sql_tool_node ↗
                           └→ general                                           → END

Key design decisions
────────────────────
• KB path   : kb_agent_node  emits tool_calls → kb_tool_node  (ToolNode) executes them
• SQL path  : sql_agent_node emits tool_calls → sql_tool_node (ToolNode) executes them
• Both path : reuses the same kb_agent → kb_tool_node → sql_agent → sql_tool_node pipeline
              via a dedicated "both" entry node that simply sets the route, keeping
              ToolNode-backed execution consistent across all paths.
• No keyword matching anywhere — the LLM bound to each agent decides which tool to call.
• response_node reads ToolMessage results from state["messages"] for all paths.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agents.nodes import (
    AgentState,
    history_loader_node,
    router_node,
    kb_agent_node,
    sql_agent_node,
    general_node,
    response_node,
    KB_TOOLS,
    SQL_TOOLS,
)


def build_agent_graph():
    graph = StateGraph(AgentState)

    #  Nodes
    graph.add_node("history_loader", history_loader_node)
    graph.add_node("router", router_node)

    # KB agent + tool executor
    graph.add_node("kb_agent", kb_agent_node)
    graph.add_node("kb_tool_node", ToolNode(KB_TOOLS))

    # SQL agent + tool executor  ← NEW
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_tool_node", ToolNode(SQL_TOOLS))

    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    #  Entry point  
    graph.set_entry_point("history_loader")

    #  history_loader → router  
    graph.add_edge("history_loader", "router")

    #  router → agent nodes  
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "general"),
        {
            "knowledge_base": "kb_agent",
            "sql_query": "sql_agent",
            "both": "kb_agent",  # enters the shared pipeline at kb_agent
            "general": "general",
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

    #  general flow  
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
        "kb_context": None,
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
