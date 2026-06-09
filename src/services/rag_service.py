import logging
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


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
