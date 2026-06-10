from langchain_core.tools import tool
from src.api.v1.services.rag_service import format_context
from src.api.v1.retrieval.vector_search import search_semantic
from src.api.v1.retrieval.hybrid_search import search_hybrid


@tool
def hybrid_search_tool(query: str) -> str:
    """
    Search the credit card knowledge base using HYBRID search
    (vector similarity + full-text search + Cohere rerank).

    Use this when the query involves nuanced policy language, benefit
    comparisons, fee structures, eligibility criteria, or multi-concept
    questions where keyword matching adds recall on top of semantic search.

    Examples:
    - "What are the lounge access benefits?"
    - "How does the fee-waiver threshold work for Platinum?"
    - "Compare reward rates across card variants"
    """
    chunks = search_hybrid(query, top_k=5)
    return format_context(chunks)


@tool
def vector_search_tool(query: str) -> str:
    """
    Search the credit card knowledge base using VECTOR-ONLY search
    (pure cosine similarity, no keyword boosting).

    Use this when the query is conversational or semantically clear but
    unlikely to benefit from exact keyword matching — e.g. follow-up
    questions, rephrased queries, or short factual lookups.

    Examples:
    - "What is the annual fee?"
    - "Tell me about cashback on dining"
    - "Is there an EMI conversion feature?"
    """
    chunks = search_semantic(query, top_k=5, rerank=True)
    return format_context(chunks)


# Registry — imported by nodes.py and graph.py
KB_TOOLS = [hybrid_search_tool, vector_search_tool]
