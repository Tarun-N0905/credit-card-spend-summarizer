"""
src/ingestion/docling_parser.py

Converts a raw PDF into a flat list of typed content chunks using Docling's
layout analysis pipeline.

Each chunk carries:
  content      — text representation of the element
  content_type — "text" | "table" | "image"
  metadata     — page_number, section, source_file, element_type, position

Images are NOT stored as base64 in the database. Instead, raw bytes are sent
to GPT-4o vision and only the resulting description text is stored as chunk
content. A failed vision call never aborts the pipeline — a safe fallback is
used instead.
"""

import base64
import io
import logging
import os
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vision model used during ingestion to describe embedded images.
# The chat model does not support vision, so a separate env var is used.
_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
_VISION_MAX_TOKENS = 512

_VISION_PROMPT = (
    "You are analyzing an image extracted from a financial document "
    "(credit card terms, reward guide, fee schedule, etc). "
    "Describe what this image shows in detail — include any data, "
    "labels, figures, or visual structure visible. "
    "Be specific and factual. "
    "Output only the description, no preamble."
)

# Text chunking — tuned for 3–4 page PDFs.
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100

# Docling label taxonomy we care about:
#   section_header  — numbered or unnumbered section headings
#   title           — document-level title
#   text / paragraph— body paragraphs
#   list_item       — bullet / numbered list items
#   caption         — figure / table captions
#   footnote        — footnotes at the bottom of a page
#   table           — tabular data (Docling reconstructs cell structure)
#   picture         — embedded raster / vector images
#   chart           — chart/graph images (rendered image, no raw data)
#   page_header     — running header printed on every page  ← SKIPPED (noise)
#   page_footer     — running footer printed on every page  ← SKIPPED (noise)

_SKIP_LABELS = {"page_header", "page_footer"}
_HEADING_LABELS = {"section_header", "title"}
_TABLE_LABELS = {"table"}
_IMAGE_LABELS = {"picture", "figure", "chart"}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _describe_image_with_vision_model(img_b64: str, page_no: int | None) -> str:
    """Send a base64-encoded image to GPT-4o and return a text description.

    The description becomes the chunk's searchable text content — far more
    useful than a sparse caption for embedding and retrieval.

    A failed vision call never aborts ingestion. Any exception is caught,
    logged as a warning, and a safe fallback string is returned.

    Args:
        img_b64 : Base64-encoded PNG string of the extracted image.
        page_no : Page number the image was found on (used in fallback text).

    Returns:
        Descriptive string from the vision model, or a fallback placeholder.
    """
    try:
        llm = ChatOpenAI(model=_VISION_MODEL, max_tokens=_VISION_MAX_TOKENS)
        message = HumanMessage(
            content=[
                {"type": "text", "text": _VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        )
        response = llm.invoke([message])
        description = response.content.strip()
        logger.debug("Vision model response for page %s: %s", page_no, description)
        return description

    except Exception as exc:
        logger.warning(
            "Image description failed on page %s — using fallback. Error: %s",
            page_no,
            exc,
        )
        return f"Image on page {page_no} — description unavailable"


def _extract_image_b64(node, doc) -> str | None:
    """Extract a PIL image from a Docling picture node and base64-encode it.

    Tries two extraction paths to handle differences across Docling versions:
      1. node.get_image(doc) — preferred; uses pre-rendered PIL Images
         produced when generate_picture_images=True in pipeline options.
      2. node.image.pil_image — fallback attribute on older Docling builds.

    Returns:
        Base64-encoded PNG string, or None if extraction fails.
    """
    try:
        if hasattr(node, "get_image"):
            pil_img = node.get_image(doc)
            if pil_img:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

        # Fallback for older Docling versions
        if hasattr(node, "image") and node.image:
            pil_img = getattr(node.image, "pil_image", None)
            if pil_img:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

    except Exception as exc:
        logger.warning("Image extraction error: %s", exc)

    return None


def _save_image_locally(
    img_b64: str,
    source_file: str,
    page_no: int,
    image_idx: int,
) -> str:
    """Save an extracted image to disk and return its path.

    Images are saved under data/images/<doc_name>/ with filenames
    like page_2_img_1.png for easy traceability.
    """
    doc_name = Path(source_file).stem
    image_dir = Path("data/images") / doc_name
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"page_{page_no}_img_{image_idx}.png"
    img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    img.save(image_path)

    return str(image_path)


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------


def _table_to_text(node, doc) -> str:
    """Convert a Docling table node into readable plain text.

    Extraction strategy (in order of preference):
      1. DataFrame export  — structured row/column text
      2. HTML export       — tags stripped, whitespace collapsed
      3. node.text         — raw fallback
    """

    # Strategy 1: DataFrame
    if hasattr(node, "export_to_dataframe"):
        try:
            df = node.export_to_dataframe()
            if df is not None and not df.empty:
                headers = [str(c).strip() for c in df.columns]

                # If Docling used numeric column indices, promote first row as header
                if all(h.isdigit() for h in headers) and len(df) > 0:
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
                    headers = [str(c).strip() for c in df.columns]

                rows_text = []
                for _, row in df.iterrows():
                    pairs = [
                        f"{h}: {str(v).strip()}"
                        for h, v in zip(headers, row)
                        if str(v).strip() not in ("", "nan", "None")
                        and not h.lower().startswith("unnamed")
                    ]
                    if pairs:
                        rows_text.append(" | ".join(pairs))

                if rows_text:
                    # Guard: if the average number of populated pairs per row
                    # is below 1.5 the DataFrame structure is likely degenerate
                    # (column values bleeding, repeated prefix artefacts, etc).
                    # Fall through to HTML export rather than persisting with
                    # garbled output.
                    total_pairs = sum(
                        len(r.split(" | ")) for r in rows_text
                    )
                    avg_pairs = total_pairs / len(rows_text)
                    if avg_pairs >= 1.5:
                        return "\n".join(rows_text)
                    logger.debug(
                        "DataFrame output looks garbled (avg_pairs=%.2f) — "
                        "falling back to HTML export",
                        avg_pairs,
                    )

        except Exception as exc:
            logger.debug("DataFrame export failed: %s", exc)

    # Strategy 2: HTML export
    if hasattr(node, "export_to_html"):
        try:
            import re

            html = node.export_to_html(doc)
            text = re.sub(r"<[^>]+>", " ", html or "")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text
        except Exception as exc:
            logger.debug("HTML export failed: %s", exc)

    # Strategy 3: Raw text fallback
    return getattr(node, "text", "") or ""


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------


def _split_text(text: str) -> list[str]:
    """Split a long text block into overlapping chunks.

    For short texts (under _CHUNK_SIZE chars) no splitting is done.
    Separators never include the empty string so the splitter will not
    cut mid-word; a chunk may slightly exceed _CHUNK_SIZE rather than
    produce a partial token.

    Args:
        text : Raw paragraph or body text.

    Returns:
        List of one or more text strings ready for embedding.
    """
    if len(text) <= _CHUNK_SIZE:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of typed content chunks using Docling.

    Each returned chunk is a dict with three keys:
      content      — text or image description
      content_type — "text" | "table" | "image"
      metadata     — dict with content_type, element_type, section,
                     page_number, source_file, position (bounding box)

    Pipeline steps:
      1. Configure Docling with OCR, table structure, and picture rendering
      2. Run the full conversion pipeline on the PDF
      3. Walk the element tree, skipping page headers/footers
      4. Route each node to the appropriate handler:
           text/paragraphs → split into overlapping chunks with section prefix
           tables          → text representation (single chunk)
           pictures/charts → GPT-4o vision description (single chunk)

    Args:
        file_path : Absolute or relative path to the source PDF.

    Returns:
        List of chunk dicts ready for deduplication, embedding, and storage.
    """

    # ── Step 1: Configure Docling pipeline ───────────────────────────────
    # do_ocr=True             — run OCR on scanned/rasterised pages
    # do_table_structure=True — detect table grid and reconstruct rows/cols
    # generate_picture_images — render each picture element to a PIL Image
    #
    # CPU accelerator is pinned to avoid MPS float64 crash on Apple Silicon.
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    # ── Step 2: Convert the PDF ───────────────────────────────────────────
    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    current_section = "Document Content"
    source_file = os.path.basename(file_path)
    image_counter = 0

    # ── Step 3: Walk the document element tree ────────────────────────────
    for item in doc.iterate_items():
        node, _ = item if isinstance(item, tuple) else (item, None)

        label = str(getattr(node, "label", "")).lower()

        # Skip noisy repeating headers/footers
        if label in _SKIP_LABELS:
            continue

        # Extract page number and bounding box
        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        # Snapshot mutable state at this point in the loop
        # (avoids stale closure issues if chunks were ever lazily evaluated)
        snapshot_section = current_section
        snapshot_page = page_no
        snapshot_position = position

        def _make_metadata(content_type: str, element_type: str) -> dict:
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": snapshot_section,
                "page_number": snapshot_page,
                "source_file": source_file,
                "position": snapshot_position,
            }

        # ── Section headings & document title ─────────────────────────
        if label in _HEADING_LABELS:
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
            continue

        # ── Tables ────────────────────────────────────────────────────
        elif label in _TABLE_LABELS:
            table_text = _table_to_text(node, doc).strip()
            if table_text:
                parsed_chunks.append(
                    {
                        "content": table_text,
                        "content_type": "table",
                        "metadata": _make_metadata("table", "table"),
                    }
                )

        # ── Pictures, figures, and charts ─────────────────────────────
        elif any(img_label in label for img_label in _IMAGE_LABELS):
            image_counter += 1
            img_b64 = _extract_image_b64(node, doc)
            metadata = _make_metadata("image", "picture")

            if img_b64 is None:
                content = (
                    getattr(node, "text", "") or f"Image on page {page_no}"
                ).strip()
            else:
                metadata["image_path"] = _save_image_locally(
                    img_b64=img_b64,
                    source_file=source_file,
                    page_no=page_no,
                    image_idx=image_counter,
                )
                content = _describe_image_with_vision_model(img_b64, page_no)

            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": metadata,
                }
            )

        # ── Plain text: paragraphs, list items, captions, footnotes ───
        else:
            text = getattr(node, "text", "")
            if not text or not text.strip():
                continue

            for sub_chunk in _split_text(text.strip()):
                parsed_chunks.append(
                    {
                        "content": sub_chunk,
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

    logger.info(
        "parse_document: %d chunks extracted from %s",
        len(parsed_chunks),
        source_file,
    )
    return parsed_chunks
