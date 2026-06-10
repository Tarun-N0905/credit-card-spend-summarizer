import logging

from langchain_openai import OpenAIEmbeddings

from src.api.v1.core.settings import settings

logger = logging.getLogger(__name__)


_BATCH_SIZE = 512

# Singleton embeddings client — instantiated once, reused across calls.
_embeddings_client: OpenAIEmbeddings | None = None


def _get_client() -> OpenAIEmbeddings:
    """Return the singleton OpenAIEmbeddings client, creating it if needed.

    Using a module-level singleton avoids re-initialising the HTTP client
    on every embed call, which matters during bulk ingestion where thousands
    of chunks are embedded in sequence.
    """
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(
            model=settings.openai_embeddings_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings_client


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings in batches.

    Splits the input into batches of _BATCH_SIZE and calls the OpenAI
    embeddings API once per batch. Results are concatenated and returned
    in the same order as the input list.

    """
    if not texts:
        raise ValueError("embed_documents requires at least one text string")

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        logger.debug(
            "Embedding batch %d–%d of %d",
            batch_start,
            batch_start + len(batch),
            len(texts),
        )
        batch_embeddings = client.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts → %d vectors", len(texts), len(all_embeddings))
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string for retrieval.

    Uses embed_query (not embed_documents) as OpenAI applies a different
    instruction prefix for query embeddings vs document embeddings,
    which improves retrieval relevance.

    Args:
        text : The user query string to embed.

    Returns:
        1536-dimensional float vector.
    """
    client = _get_client()
    return client.embed_query(text)
