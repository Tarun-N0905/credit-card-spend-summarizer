"""
src/ingestion/deduplication.py

Exact-duplicate removal using SHA256 hashing.

Strategy:
  - Normalise chunk text (lowercase, strip, collapse whitespace)
  - Compute SHA256 hash of the normalised text
  - Check the hash against existing hashes in `document_chunks`
  - Keep the chunk only if its hash has not been seen before

Rules (from blueprint):
  - Remove exact duplicates only
  - Keep similar content
  - Keep overlapping content
  - No file hashing, no embedding deduplication, no similarity deduplication
"""

import hashlib
import logging
import re

from src.api.v1.core.db import get_existing_hashes

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Normalise a chunk's text before hashing.

    Normalisation steps:
      1. Lowercase — "Fee Waiver" and "fee waiver" are the same
      2. Strip leading/trailing whitespace
      3. Collapse all internal whitespace (spaces, tabs, newlines) to
         a single space — formatting differences don't create false uniques
      4. Remove unicode whitespace variants that wouldn't be caught by \\s

    Args:
        text : Raw chunk content string.

    Returns:
        Normalised string ready for hashing.
    """
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_hash(text: str) -> str:
    """Compute a SHA256 hex digest of the normalised chunk text.

    The hash is stored in `document_chunks.chunk_hash` (VARCHAR 64) and
    used as the uniqueness key for deduplication across all documents.

    Args:
        text : Raw chunk content (normalisation is applied internally).

    Returns:
        64-character lowercase hex string.
    """
    normalised = _normalise(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Filter out exact-duplicate chunks using SHA256 hashing.

    Process:
      1. Compute a hash for every chunk in the incoming list
      2. Fetch all hashes already stored in `document_chunks` from the DB
      3. Also deduplicate within the current batch itself (a single PDF
         can repeat the same paragraph in multiple sections)
      4. Return only chunks whose hash has not been seen before

    Args:
        chunks : List of chunk dicts from the splitting stage. Each dict
                 must have a "content" key.

    Returns:
        Tuple of (unique_chunks, skipped_count) where:
          unique_chunks  — list of chunks with their computed hash injected
                           under the key "chunk_hash"
          skipped_count  — number of exact duplicates that were dropped
    """
    # Fetch hashes that are already stored in the database so we don't
    # re-insert chunks from previous ingestion runs of the same document
    # or from other documents that share identical text.
    existing_hashes: set[str] = get_existing_hashes()

    unique_chunks: list[dict] = []
    seen_in_batch: set[str] = set()
    skipped = 0

    for chunk in chunks:
        chunk_hash = compute_hash(chunk["content"])

        # Drop if already in the DB or already seen in this batch
        if chunk_hash in existing_hashes or chunk_hash in seen_in_batch:
            skipped += 1
            logger.debug("Duplicate skipped (hash=%s…)", chunk_hash[:12])
            continue

        seen_in_batch.add(chunk_hash)
        # Inject the hash into the chunk dict so store_chunks() can write
        # it directly to the chunk_hash column without recomputing it.
        unique_chunks.append({**chunk, "chunk_hash": chunk_hash})

    logger.info("Deduplication: %d unique, %d skipped", len(unique_chunks), skipped)
    return unique_chunks, skipped
