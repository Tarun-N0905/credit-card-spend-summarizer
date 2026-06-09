import logging
from langchain_core.tools import tool
from src.api.v1.services.rag_service import retrieve, format_context

logger = logging.getLogger(__name__)


@tool
def knowledge_base_tool(query: str) -> str:
    """
    Search the ingested credit card knowledge base (PDFs) for information about
    card features, benefits, fees, reward programs, terms and conditions, and policies.
    Use this for any question about how the card works, not account-specific data.
    """
    try:
        chunks = retrieve(query)
        context = format_context(chunks)
        logger.info("knowledge_base_tool: retrieved %d chunks", len(chunks))
        return context
    except Exception as e:
        logger.error("knowledge_base_tool failed: %s", e)
        return f"Knowledge base lookup failed: {e}"
