import hashlib
import logging
import re

from src.api.v1.core.db import get_existing_hashes

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """
    Normalise a chunk's text before hashing.
    """
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_hash(text: str) -> str:
    """
    Compute a SHA256 hex digest of the normalised chunk text.
    """
    normalised = _normalise(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """
    Filter out exact-duplicate chunks using SHA256 hashing.
    """
    existing_hashes: set[str] = get_existing_hashes()

    unique_chunks: list[dict] = []
    seen_in_batch: set[str] = set()
    skipped = 0

    for chunk in chunks:
        chunk_hash = compute_hash(chunk["content"])

        if chunk_hash in existing_hashes or chunk_hash in seen_in_batch:
            skipped += 1
            logger.debug("Duplicate skipped (hash=%s…)", chunk_hash[:12])
            continue

        seen_in_batch.add(chunk_hash)
        unique_chunks.append({**chunk, "chunk_hash": chunk_hash})

    logger.info("Deduplication: %d unique, %d skipped", len(unique_chunks), skipped)
    return unique_chunks, skipped
