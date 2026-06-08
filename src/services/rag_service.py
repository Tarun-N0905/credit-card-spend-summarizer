import logging
from src.retrieval.hybrid_search import search_hybrid
from src.retrieval.reranker import rerank_results
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Run hybrid search (vector + FTS + RRF) then rerank with Cohere.
    Returns top_k RetrievedChunk objects ready for LLM context.
    """
    try:
        chunks = search_hybrid(query, top_k=top_k * 4)  # fetch more, rerank trims
        if not chunks:
            logger.warning("rag_service: no chunks returned for query: %s", query)
            return []
        reranked = rerank_results(query, chunks, top_k=top_k)
        logger.info(
            "rag_service: retrieved %d chunks for query: %s", len(reranked), query
        )
        return reranked
    except Exception as e:
        logger.error("rag_service: retrieval failed: %s", e)
        return []


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format RetrievedChunk list into a single context string for the LLM prompt.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}]"
        if chunk.document_name:
            header += f" {chunk.document_name}"
        if chunk.page_number:
            header += f" (Page {chunk.page_number})"
        if chunk.section_name:
            header += f" — {chunk.section_name}"
        parts.append(f"{header}\n{chunk.chunk_text}")

    return "\n\n".join(parts)
