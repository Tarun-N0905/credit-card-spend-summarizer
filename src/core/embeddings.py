"""
src/core/embeddings.py

Thin wrapper around OpenAI's text-embedding-3-small model.

All embedding calls go through this module so the model name and batching
logic live in one place. Callers pass a list of strings and get back a
list of float vectors — no OpenAI SDK details leak outside this file.

Vector dimension: 1536  (fixed for text-embedding-3-small)
This must match the VECTOR(1536) column in document_chunks.
"""

import logging

from langchain_openai import OpenAIEmbeddings

from src.core.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI enforces a max of 2048 inputs per embed request.
# We use 512 as the batch size — large enough to be efficient, small enough
# to stay well under the limit and keep memory usage predictable.
# ---------------------------------------------------------------------------
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

    Batching is important during ingestion — embedding one chunk at a time
    is slow and wastes API round trips. A single large request would hit
    the 2048-input limit, so we window it ourselves.

    Args:
        texts : List of strings to embed. Must be non-empty.

    Returns:
        List of 1536-dimensional float vectors, one per input string,
        in the same order as the input.

    Raises:
        ValueError : If texts is empty.
        Exception  : Propagates OpenAI API errors to the caller.
    """
    if not texts:
        raise ValueError("embed_documents requires at least one text string")

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start: batch_start + _BATCH_SIZE]
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
