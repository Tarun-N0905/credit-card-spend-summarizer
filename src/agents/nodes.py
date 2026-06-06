"""
src/agents/nodes.py

Agent action nodes for the credit card RAG workflow.
Each node represents a step in the reasoning process.

Nodes:
  1. router_node — classifies query (knowledge_base vs sql_query)
  2. kb_search_node — retrieves relevant KB documents
  3. sql_search_node — generates and executes SQL
  4. rerank_node — reranks KB results using Cohere
  5. response_node — generates final answer
"""

import os
import logging
from typing import Literal
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.core.db import get_sql_database
from src.core.settings import settings
from src.retrieval.hybrid_search import search_hybrid
from src.retrieval.reranker import rerank_results
from src.retrieval.schemas import RetrievedChunk
from src.agents.prompts import (
    ROUTER_PROMPT_TEMPLATE,
    NL2SQL_PROMPT_TEMPLATE,
    KB_GENERATION_PROMPT_TEMPLATE,
    SQL_ANSWER_PROMPT_TEMPLATE,
)
from src.agents.schemas import AgentResponse

logger = logging.getLogger(__name__)


# ── State Type ─────────────────────────────────────────────────────────────
class AgentState(dict):
    """Shared state flowing through all agent nodes."""
    pass


# ── Helper: Get LLM ────────────────────────────────────────────────────────

def _get_llm():
    """Create OpenAI LLM instance."""
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )


# ── Node 0: Router ────────────────────────────────────────────────────────
# Classifies the query: is it about KB docs or credit card account data?

class _RouteDecision(BaseModel):
    """Router output."""
    route: Literal["knowledge_base", "sql_query"]
    reason: str


def router_node(state: AgentState) -> AgentState:
    """Route the query to knowledge_base or sql_query path."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)
    
    chain = ROUTER_PROMPT_TEMPLATE | structured_llm
    decision = chain.invoke({"query": state["query"]})
    
    logger.info(f"[router_node] Route → '{decision.route}' | Reason: {decision.reason}")
    
    return {
        **state,
        "route": decision.route,
        "router_reason": decision.reason
    }


# ── Node 1A: KB Search (Knowledge Base) ───────────────────────────────────

def kb_search_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from ingested KB documents using hybrid search."""
    query = state["query"]
    
    try:
        # Use hybrid search (vector + FTS + RRF + reranking)
        chunks = search_hybrid(query, top_k=5)
        logger.info(f"[kb_search_node] Retrieved {len(chunks)} KB chunks")
        
        # Convert RetrievedChunk to LangChain Document for consistency
        docs = [
            Document(
                page_content=chunk.chunk_text,
                metadata={
                    "id": chunk.id,
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "section": chunk.section_name,
                    "score": chunk.score,
                    **chunk.metadata
                }
            )
            for chunk in chunks
        ]
        
        return {**state, "retrieved_docs": docs}
    
    except Exception as exc:
        logger.error(f"[kb_search_node] Error: {exc}")
        return {**state, "retrieved_docs": []}


# ── Node 1B: SQL Search (Account Data) ────────────────────────────────────

def sql_search_node(state: AgentState) -> AgentState:
    """Generate SQL, execute it, and return results."""
    llm = _get_llm()
    db = get_sql_database()
    query = state["query"]
    
    try:
        # ── Step 1: Get DB schema ──────────────────────────────────────────
        schema_info = db.get_table_info()
        
        # ── Step 2: Generate SQL ───────────────────────────────────────────
        sql_chain = NL2SQL_PROMPT_TEMPLATE | llm
        raw_sql = sql_chain.invoke({
            "schema": schema_info,
            "question": query
        })
        
        # Parse content (handle both string and list formats)
        content = raw_sql.content
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        
        generated_sql = content.strip().strip("```").strip()
        if generated_sql.lower().startswith("sql"):
            generated_sql = generated_sql[3:].strip()
        
        logger.info(f"[sql_search_node] Generated SQL:\n{generated_sql}")
        
        # ── Step 3: Execute SQL ───────────────────────────────────────────
        try:
            sql_result = db.run(generated_sql)
            logger.info(f"[sql_search_node] Result (truncated): {str(sql_result)[:200]}")
        except Exception as exc:
            sql_result = f"SQL execution error: {exc}"
            logger.error(f"[sql_search_node] SQL failed: {exc}")
        
        return {
            **state,
            "generated_sql": generated_sql,
            "sql_result": str(sql_result)
        }
    
    except Exception as exc:
        logger.error(f"[sql_search_node] Error: {exc}")
        return {
            **state,
            "generated_sql": "",
            "sql_result": f"Error generating SQL: {exc}"
        }


# ── Node 2: Rerank (KB Path Only) ─────────────────────────────────────────

def rerank_node(state: AgentState) -> AgentState:
    """Rerank KB documents using Cohere cross-encoder."""
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        logger.info("[rerank_node] No docs to rerank")
        return {**state, "reranked_docs": []}
    
    try:
        # Convert LangChain Documents back to RetrievedChunk for reranking
        chunks = [
            RetrievedChunk(
                id=doc.metadata.get("id", ""),
                chunk_text=doc.page_content,
                score=doc.metadata.get("score", 0.0),
                content_type=doc.metadata.get("content_type"),
                page_number=doc.metadata.get("page_number"),
                section_name=doc.metadata.get("section"),
                metadata=doc.metadata,
                position=None
            )
            for doc in docs
        ]
        
        # Rerank using Cohere
        reranked_chunks = rerank_results(
            query=state["query"],
            chunks=chunks,
            top_k=3
        )
        
        # Convert back to Documents
        reranked_docs = [
            Document(
                page_content=chunk.chunk_text,
                metadata={
                    "id": chunk.id,
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "section": chunk.section_name,
                    "score": chunk.score,
                    **chunk.metadata
                }
            )
            for chunk in reranked_chunks
        ]
        
        logger.info(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking")
        return {**state, "reranked_docs": reranked_docs}
    
    except Exception as exc:
        logger.error(f"[rerank_node] Error: {exc}")
        return {**state, "reranked_docs": docs}


# ── Node 3: Generate Response ──────────────────────────────────────────────

def response_node(state: AgentState) -> AgentState:
    """Generate final answer based on retrieved context."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AgentResponse)
    
    route = state.get("route", "knowledge_base")
    query = state["query"]
    
    try:
        if route == "knowledge_base":
            # KB path: use reranked docs
            reranked_docs = state.get("reranked_docs", [])
            
            if not reranked_docs:
                # Fallback: no docs retrieved
                answer = "I couldn't find relevant information in the knowledge base. Could you rephrase your question?"
                response = AgentResponse(
                    query=query,
                    answer=answer,
                    data_sources="No documents found",
                    page_no="N/A",
                    document_name="KB_documents",
                    sql_query_executed=None,
                    route_taken="knowledge_base"
                )
            else:
                # Format context
                context = "\n\n".join([
                    f"[Source: {doc.metadata.get('section', 'Unknown Section')} | "
                    f"Page: {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
                    for doc in reranked_docs
                ])
                
                # Generate answer
                chain = KB_GENERATION_PROMPT_TEMPLATE | structured_llm
                response = chain.invoke({"context": context, "query": query})
                
                # Add route info
                response.route_taken = "knowledge_base"
                response.document_name = "credit_card_kb_documents"
        
        else:  # sql_query route
            # SQL path: use SQL results
            sql_result = state.get("sql_result", "No results")
            generated_sql = state.get("generated_sql", "")
            
            # Generate answer from SQL results
            chain = SQL_ANSWER_PROMPT_TEMPLATE | structured_llm
            response = chain.invoke({
                "query": query,
                "sql": generated_sql,
                "result": sql_result
            })
            
            # Add route info
            response.route_taken = "sql_query"
            response.sql_query_executed = generated_sql
            response.page_no = "N/A"
            response.document_name = "credit_card_account_data"
        
        logger.info("[response_node] Answer generated")
        return {**state, "response": response.model_dump()}
    
    except Exception as exc:
        logger.error(f"[response_node] Error: {exc}")
        response = AgentResponse(
            query=query,
            answer=f"Error generating response: {exc}",
            data_sources="Error",
            page_no="N/A",
            document_name="error",
            route_taken=route
        )
        return {**state, "response": response.model_dump()}
