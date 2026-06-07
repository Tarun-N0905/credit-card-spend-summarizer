import logging
import pathlib

from dotenv import load_dotenv
import sys
from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document
from src.ingestion.deduplication import deduplicate_chunks

load_dotenv()

logger = logging.getLogger(__name__)


# Chunking Configuration
_TEXT_CHUNK_SIZE = 500
_TEXT_CHUNK_OVERLAP = 50


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character windows.
    """

    chunks: list[str] = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step

    return chunks


def build_section_documents(parsed_elements: list[dict]) -> list[dict]:
    """
    Merge section headings with their associated content.

    Example:

        Rewards Program
        ├─ Paragraph A
        ├─ Paragraph B
        └─ Paragraph C

    becomes

        Rewards Program

        Paragraph A

        Paragraph B

        Paragraph C

    Tables and images remain atomic elements.
    """

    structured_elements: list[dict] = []

    current_section_title: str | None = None
    current_text_parts: list[str] = []
    current_metadata: dict | None = None

    for elem in parsed_elements:

        content_type = elem["content_type"]

        
        # Tables and images remain atomic
        
        if content_type in ("table", "image"):

            if current_text_parts:
                structured_elements.append(
                    {
                        "content": "\n\n".join(current_text_parts),
                        "content_type": "text",
                        "metadata": current_metadata,
                    }
                )

                current_text_parts = []
                current_section_title = None
                current_metadata = None

            structured_elements.append(elem)
            continue

        metadata = elem["metadata"]

        element_type = (
            metadata.get("element_type", "")
            .lower()
        )

        is_heading = (
            "section_header" in element_type
            or element_type == "title"
        )

        
        # New section starts
        if is_heading:

            if current_text_parts:
                structured_elements.append(
                    {
                        "content": "\n\n".join(current_text_parts),
                        "content_type": "text",
                        "metadata": current_metadata,
                    }
                )

            current_section_title = elem["content"]
            current_metadata = metadata

            current_text_parts = [current_section_title]

        
        # Regular text content
        else:

            if current_section_title is None:

                current_section_title = "Document Content"
                current_metadata = metadata

                current_text_parts = [current_section_title]

            current_text_parts.append(elem["content"])

    
    # Flush final section
    
    if current_text_parts:
        structured_elements.append(
            {
                "content": "\n\n".join(current_text_parts),
                "content_type": "text",
                "metadata": current_metadata,
            }
        )

    return structured_elements


def run_ingestion(file_path: str) -> dict:
    """
    Full ingestion pipeline.

    Steps:
        1. Register document
        2. Parse PDF with Docling
        3. Merge headings + section content
        4. Split long text sections (500 chars, 50 overlap)
        5. Deduplicate
        6. Store embeddings + chunks
    """

    resolved = pathlib.Path(file_path).resolve()

    
    # Step 1: Register document
    
    doc_id = upsert_document(
        resolved.name,
        str(resolved),
    )

    logger.info(
        "doc_id=%s file=%s",
        doc_id,
        file_path,
    )

    
    # Step 2: Parse document
    
    logger.info("Parsing document: %s", file_path)

    parsed_elements = parse_document(file_path)

    logger.info(
        "Docling extracted %d elements",
        len(parsed_elements),
    )

    
    # Step 3: Merge headings with content
    
    structured_elements = build_section_documents(parsed_elements)

    logger.info(
        "%d structured elements after section merge",
        len(structured_elements),
    )

    
    # Step 4: Chunk text
    
    chunks: list[dict] = []

    for elem in structured_elements:

        if (
            elem["content_type"] == "text"
            and len(elem["content"]) > _TEXT_CHUNK_SIZE
        ):

            sub_chunks = _split_text(
                elem["content"],
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP,
            )

            for sub in sub_chunks:

                chunks.append(
                    {
                        "content": sub,
                        "content_type": "text",
                        "metadata": elem["metadata"],
                    }
                )

        else:
            chunks.append(elem)

    logger.info(
        "%d chunks after text splitting",
        len(chunks),
    )

    
    # Step 5: Deduplication
    unique_chunks, skipped = deduplicate_chunks(chunks)

    logger.info(
        "%d unique chunks ready for embedding (%d skipped)",
        len(unique_chunks),
        skipped,
    )

    
    # Step 6: Store chunks 
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



# Direct execution
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path(
            "data/documents/KB_Credit_Card_Spend_Summarizer.pdf"
        )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path.resolve()}"
        )

    result = run_ingestion(str(pdf_path))

    print(f"\nIngestion complete: {result}")