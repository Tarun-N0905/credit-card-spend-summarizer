"""
src/ingestion/ingestion.py

Section-aware chunking for multimodal RAG.

Design goals
------------
- Preserve semantic structure.
- Keep tables atomic.
- Keep images atomic.
- Chunk text by paragraphs, not raw characters.
- Maintain section context by repeating headings.
- Use 500-char chunks with 70-char overlap.
"""

import logging
import pathlib
import sys

from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document
from src.ingestion.deduplication import deduplicate_chunks

load_dotenv()

logger = logging.getLogger(__name__)

TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 70


def _split_text_with_overlap(
    text: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Sliding window splitter used only inside a section after
    paragraph aggregation.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks


def build_section_documents(parsed_elements: list[dict]) -> list[dict]:
    """
    Build semantically meaningful chunks.

    Assumptions:
    - section headers are NOT emitted as standalone chunks by docling_parser.
    - every element already carries metadata["section"].
    - tables and images remain atomic.
    """

    chunks: list[dict] = []

    current_section = None
    paragraph_buffer: list[str] = []
    current_metadata = None

    def flush_text_buffer():
        nonlocal paragraph_buffer
        nonlocal current_metadata
        nonlocal current_section

        if not paragraph_buffer:
            return

        combined_text = "\n\n".join(paragraph_buffer)

        sub_chunks = _split_text_with_overlap(
            combined_text,
            TEXT_CHUNK_SIZE,
            TEXT_CHUNK_OVERLAP,
        )

        for sub_chunk in sub_chunks:

            if current_section:
                content = f"{current_section}\n\n{sub_chunk}"
            else:
                content = sub_chunk

            chunks.append(
                {
                    "content": content,
                    "content_type": "text",
                    "metadata": current_metadata,
                }
            )

        paragraph_buffer = []

    for elem in parsed_elements:

        content = elem["content"].strip()

        if not content:
            continue

        metadata = elem["metadata"]
        content_type = elem["content_type"]

        section_name = metadata.get("section")

        # Section changed
        if section_name != current_section:
            flush_text_buffer()
            current_section = section_name

        # Tables/images remain atomic
        if content_type in ("table", "image"):

            flush_text_buffer()

            chunks.append(
                {
                    "content": content,
                    "content_type": content_type,
                    "metadata": metadata,
                }
            )

            continue

        # Normal text
        current_metadata = metadata
        paragraph_buffer.append(content)

    flush_text_buffer()

    return chunks


def run_ingestion(
    file_path: str,
    original_filename: str,
) -> dict:
    """
    Full ingestion pipeline.

    Steps
    -----
    1. Register document
    2. Parse PDF with Docling
    3. Build section-aware chunks
    4. Deduplicate
    5. Store embeddings + chunks
    """

    resolved = pathlib.Path(file_path).resolve()

    # Register document
    doc_id = upsert_document(
        original_filename,
        str(resolved),
    )

    logger.info(
        "doc_id=%s file=%s",
        doc_id,
        file_path,
    )

    # Parse
    logger.info("Parsing document: %s", file_path)

    parsed_elements = parse_document(file_path)

    logger.info(
        "Docling extracted %d elements",
        len(parsed_elements),
    )

    # Build chunks
    chunks = build_section_documents(parsed_elements)

    logger.info(
        "%d chunks after section-aware chunking",
        len(chunks),
    )

    # Deduplicate
    unique_chunks, skipped = deduplicate_chunks(chunks)

    logger.info(
        "%d unique chunks ready for embedding (%d skipped)",
        len(unique_chunks),
        skipped,
    )

    # Store
    count = store_chunks(
        unique_chunks,
        doc_id,
    )

    logger.info(
        "Stored %d chunks",
        count,
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_ingested": count,
        "chunks_skipped": skipped,
    }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/documents/KB_Credit_Card_Spend_Summarizer.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")

    result = run_ingestion(
        str(pdf_path),
        pdf_path.name,
    )

    print(f"\nIngestion complete: {result}")
