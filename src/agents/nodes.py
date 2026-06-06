import logging
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.settings import get_settings
from src.retrieval.schema import RetrievedChunk
from src.agents.prompts import (
    ROUTER_PROMPT_TEMPLATE,
    NL2SQL_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
)
from src.agents.schemas import AgentResponse
from src.services.rag_service import retrieve, format_context
from src.services.sql_service import query as sql_query

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    route: Optional[str]
    chunks: Optional[list[RetrievedChunk]]
    sql_executed: Optional[str]
    sql_results: Optional[str]
    response: Optional[AgentResponse]


# ─────────────────────────────────────────────
# LLM singleton
# ─────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


# ─────────────────────────────────────────────
# Node 1: Router
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Classify query as 'knowledge_base' or 'sql_query'."""
    try:
        llm = _get_llm()
        chain = ROUTER_PROMPT_TEMPLATE | llm
        result = chain.invoke({"query": state["query"]})
        route = result.content.strip().lower()

        if route not in ("knowledge_base", "sql_query"):
            logger.warning("router_node: unexpected route '%s', defaulting to knowledge_base", route)
            route = "knowledge_base"

        logger.info("router_node: routed to '%s'", route)
        return {**state, "route": route}
    except Exception as e:
        logger.error("router_node failed: %s", e)
        return {**state, "route": "knowledge_base"}


# ─────────────────────────────────────────────
# Node 2: KB Search
# ─────────────────────────────────────────────

def kb_search_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from knowledge base via rag_service."""
    try:
        chunks = retrieve(state["query"])
        logger.info("kb_search_node: retrieved %d chunks", len(chunks))
        return {**state, "chunks": chunks}
    except Exception as e:
        logger.error("kb_search_node failed: %s", e)
        return {**state, "chunks": []}




# ─────────────────────────────────────────────
# Node 3: SQL Search
# ─────────────────────────────────────────────

def sql_search_node(state: AgentState) -> AgentState:
    """Generate SQL from NL query and execute on cc_db via sql_service."""
    try:
        sql_executed, results = sql_query(state["query"])
        logger.info("sql_search_node: SQL executed: %s", sql_executed)
        return {**state, "sql_executed": sql_executed, "sql_results": results}
    except Exception as e:
        logger.error("sql_search_node failed: %s", e)
        return {**state, "sql_executed": "", "sql_results": f"Error: {e}"}


# ─────────────────────────────────────────────
# Node 4: Response
# ─────────────────────────────────────────────

def response_node(state: AgentState) -> AgentState:
    """Generate final answer based on route taken."""
    try:
        llm = _get_llm()
        route = state.get("route", "knowledge_base")

        if route == "knowledge_base":
            chunks = state.get("chunks", [])
            context = format_context(chunks)

            chain = KB_GENERATION_PROMPT_TEMPLATE | llm
            result = chain.invoke({"query": state["query"], "context": context})
            answer = result.content.strip()

            # Build source metadata
            data_sources = []
            page_numbers = []
            doc_names = []
            for chunk in chunks:
                if chunk.document_name and chunk.document_name not in doc_names:
                    doc_names.append(chunk.document_name)
                if chunk.page_number and chunk.page_number not in page_numbers:
                    page_numbers.append(str(chunk.page_number))
                data_sources.append({
                    "document": chunk.document_name,
                    "page": chunk.page_number,
                    "section": chunk.section_name,
                    "score": chunk.score,
                })

            response = AgentResponse(
                query=state["query"],
                answer=answer,
                data_sources=data_sources,
                page_no=", ".join(page_numbers) if page_numbers else "N/A",
                document_name=", ".join(doc_names) if doc_names else "N/A",
                sql_query_executed=None,
                route_taken="knowledge_base",
            )

        else:  # sql_query
            sql_executed = state.get("sql_executed", "")
            sql_results = state.get("sql_results", "No results.")

            chain = SQL_ANSWER_PROMPT_TEMPLATE | llm
            result = chain.invoke({
                "query": state["query"],
                "sql_results": sql_results,
            })
            answer = result.content.strip()

            response = AgentResponse(
                query=state["query"],
                answer=answer,
                data_sources=[],
                page_no="N/A",
                document_name="N/A",
                sql_query_executed=sql_executed,
                route_taken="sql_query",
            )

        logger.info("response_node: answer generated for route '%s'", route)
        return {**state, "response": response}

    except Exception as e:
        logger.error("response_node failed: %s", e)
        fallback = AgentResponse(
            query=state["query"],
            answer="Sorry, I encountered an error generating a response. Please try again.",
            data_sources=[],
            page_no="N/A",
            document_name="N/A",
            sql_query_executed=None,
            route_taken=state.get("route", "unknown"),
        )
        return {**state, "response": fallback}
