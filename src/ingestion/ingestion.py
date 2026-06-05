

# Run directly for development:
#   uv run python -m src.ingestion.ingestion path/to/file.pdf
# """

import logging
import pathlib
from dotenv import load_dotenv
from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document
from src.ingestion.deduplication import deduplicate_chunks

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking configuration
#
# Long text elements (paragraphs that span half a page or more) are split
# into overlapping windows so that a single dense paragraph does not
# dominate a retrieval result and context from surrounding sentences is
# preserved across chunk boundaries.
#
# _TEXT_CHUNK_SIZE    — maximum characters per text chunk
# _TEXT_CHUNK_OVERLAP — characters shared between adjacent chunks so that
#                       sentences cut at a boundary still appear in both
#
# Tables and images are never split — stored as atomic units.
# ---------------------------------------------------------------------------
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long string into overlapping character windows.

    Walks through the text in steps of (chunk_size - overlap). Each window
    is exactly chunk_size characters (or shorter at the end). The overlap
    ensures sentences cut at a boundary appear in both the preceding and
    following chunk, preserving retrieval context.

    This is a lightweight alternative to langchain_text_splitters.

    Args:
        text       : The source string to split.
        chunk_size : Maximum characters per chunk.
        overlap    : Characters shared between consecutive chunks.

    Returns:
        List of string chunks.
    """
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += step
    return chunks


def run_ingestion(file_path: str) -> dict:
    """Run the full ingestion pipeline for a single PDF file.

    Steps:
      1. Register the document in the `documents` table → stable doc_id
      2. Parse PDF with Docling → typed elements (text / table / image)
      3. Split long text elements into overlapping chunks
      4. Deduplicate chunks via SHA256 hash (exact duplicates only)
      5. Embed all chunks in batches and store in `document_chunks`

    Args:
        file_path : Absolute or relative path to the source PDF.

    Returns:
        Dict with "status", "doc_id", "chunks_ingested", and
        "chunks_skipped" (duplicates dropped by deduplication).
    """
    resolved = pathlib.Path(file_path).resolve()

    # ── Step 1: Register (or update) the document record ─────────────────
    # upsert_document() inserts into the `documents` table and returns a
    # UUID. Re-ingesting the same filename reuses the same doc_id so old
    # chunk rows can be cleaned up by doc_id if needed.
    doc_id = upsert_document(resolved.name, str(resolved))
    logger.info("doc_id=%s  file=%s", doc_id, file_path)

    # ── Step 2: Parse the PDF ─────────────────────────────────────────────
    # parse_document() runs the full Docling pipeline and returns a flat
    # list. Each element: {content, content_type, metadata{page_number,
    # section, source_file, element_type, position}}
    logger.info("Parsing: %s", file_path)
    parsed_elements = parse_document(file_path)
    logger.info("Docling produced %d elements", len(parsed_elements))

    # ── Step 3: Split long text elements into overlapping chunks ──────────
    # Tables and images are stored as atomic units — never split.
    # Long text elements are windowed with overlap so sentences at
    # boundaries appear in both the preceding and following chunk.
    chunks: list[dict] = []
    for elem in parsed_elements:
        if (
            elem["content_type"] == "text"
            and len(elem["content"]) > _TEXT_CHUNK_SIZE
        ):
            for sub in _split_text(elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP):
                # Each sub-chunk inherits the parent element's full metadata
                chunks.append({
                    "content": sub,
                    "content_type": elem["content_type"],
                    "metadata": elem["metadata"],
                })
        else:
            chunks.append(elem)

    logger.info("%d chunks after splitting", len(chunks))

    # ── Step 4: Deduplicate via SHA256 hash ───────────────────────────────
    # deduplicate_chunks() normalises each chunk's content, computes a
    # SHA256 hash, checks existing hashes in document_chunks, and returns
    # only the chunks whose hash has not been seen before.
    # Exact duplicates are removed; similar/overlapping content is kept.
    unique_chunks, skipped = deduplicate_chunks(chunks)
    logger.info(
        "%d unique chunks ready for embedding (%d duplicates skipped)",
        len(unique_chunks),
        skipped,
    )

    # ── Step 5: Embed chunks and store in document_chunks ─────────────────
    # store_chunks() calls the embeddings model in batches, then INSERTs
    # each row into `document_chunks` with its embedding vector,
    # page/section metadata, position JSONB, and search_vector tsvector.
    count = store_chunks(unique_chunks, doc_id)
    logger.info("Stored %d chunks → document_chunks", count)

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_ingested": count,
        "chunks_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Run ingestion directly:
#   uv run python -m src.ingestion.ingestion
#   uv run python -m src.ingestion.ingestion path/to/file.pdf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data\documents\KB_Credit_Card_Spend_Summarizer.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")

    result = run_ingestion(str(pdf_path))
    print(f"\nIngestion complete: {result}")
