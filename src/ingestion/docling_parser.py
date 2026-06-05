

"""
src/ingestion/docling_parser.py

Responsible for converting a raw PDF into a flat list of typed content
chunks using Docling's layout analysis pipeline.

Each chunk carries:
  content      — text representation of the element
  content_type — "text" | "table" | "image"
  metadata     — page_number, section, source_file, element_type, position

Images are NOT stored as base64 in the database. Instead, the raw bytes
are passed to GPT-4o vision (via _describe_image_with_vision_model) and
only the resulting description text is stored as the chunk content.

If the vision API call fails for any reason, ingestion continues with a
fallback description — a failed image never aborts the pipeline.
"""

import base64
import io
import logging
import os

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

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vision model used exclusively during ingestion to describe images.
# The chat model (gpt-5.4) does not support vision, so a separate
# OPENAI_VISION_MODEL env var points to gpt-4o.
# ---------------------------------------------------------------------------
_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

_VISION_PROMPT = (
    "You are analyzing an image extracted from a financial document "
    "(credit card terms, reward guide, fee schedule, etc). "
    "Describe what this image shows in detail — include any data, "
    "labels, figures, or visual structure visible. "
    "Be specific and factual. "
    "Output only the description, no preamble."
)

# ---------------------------------------------------------------------------
# Docling label taxonomy (DocItemLabel enum values we care about):
#
#   section_header  — numbered or unnumbered section headings
#   title           — document-level title
#   text / paragraph— body paragraphs
#   list_item       — bullet / numbered list items
#   caption         — figure / table captions
#   footnote        — footnotes at the bottom of a page
#   table           — tabular data (Docling reconstructs cell structure)
#   picture         — embedded raster / vector images
#   chart           — chart/graph images (rendered image, no raw data)
#   page_header     — running header printed on every page  ← NOISE, skipped
#   page_footer     — running footer printed on every page  ← NOISE, skipped
# ---------------------------------------------------------------------------


def _describe_image_with_vision_model(img_b64: str, page_no: int | None) -> str:
    """Send a base64-encoded image to GPT-4o and return a text description.

    The description becomes the chunk's searchable text content — far more
    useful than a sparse caption for embedding and retrieval.

    A failed vision call must NEVER abort ingestion. Any exception is caught,
    logged as a warning, and a safe fallback string is returned so the pipeline
    continues processing the rest of the document.

    Args:
        img_b64 : Base64-encoded PNG string of the extracted image.
        page_no : Page number the image was found on (used in fallback text).

    Returns:
        A descriptive string from the vision model, or a fallback placeholder.
    """
    try:
        llm = ChatOpenAI(model=_VISION_MODEL, max_tokens=512)
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
        print(response.content.strip())  # Debug: log the raw vision response
        return response.content.strip()
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


def _table_to_text(node, doc) -> str:
    """Convert a Docling table node into clean plain-text rows.

    Serialises each row as "Col1: val1  |  Col2: val2" so that column
    context travels with every value and embeddings are meaningful.

    Fallback chain:
      1. export_to_dataframe() — preferred; structured rows/cols
      2. export_to_html() with tags stripped — when DataFrame is unavailable
      3. node.text — last resort raw text

    Args:
        node : Docling table DocItem node.
        doc  : The parent DoclingDocument (needed for some export methods).

    Returns:
        Plain-text representation of the table, or empty string if extraction
        fails entirely.
    """
    # ── Strategy 1: DataFrame ─────────────────────────────────────────────
    if hasattr(node, "export_to_dataframe"):
        try:
            df = node.export_to_dataframe()
            if df is not None and not df.empty:
                rows_text: list[str] = []
                headers = [str(c).strip() for c in df.columns]
                for _, row in df.iterrows():
                    pairs = [
                        f"{h}: {str(v).strip()}"
                        for h, v in zip(headers, row)
                        if str(v).strip() not in ("", "nan", "None")
                    ]
                    if pairs:
                        rows_text.append("  |  ".join(pairs))
                if rows_text:
                    return "\n".join(rows_text)
        except Exception as exc:
            logger.debug("DataFrame export failed: %s", exc)

    # ── Strategy 2: HTML stripped of tags ─────────────────────────────────
    if hasattr(node, "export_to_html"):
        try:
            import re as _re
            raw_html = node.export_to_html(doc)
            text = _re.sub(r"<[^>]+>", " ", raw_html or "")
            text = _re.sub(r"\s+", " ", text).strip()
            if text:
                return text
        except Exception as exc:
            logger.debug("HTML export failed: %s", exc)

    # ── Strategy 3: Raw text attribute ────────────────────────────────────
    return getattr(node, "text", "") or ""


def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of typed content chunks using Docling.

    Each returned chunk is a dict with three keys:
      content      — text or description of the element
      content_type — "text" | "table" | "image"
      metadata     — dict with content_type, element_type, section,
                     page_number, source_file, position (bounding box)

    Pipeline steps:
      1. Configure Docling with OCR, table structure, and picture rendering
      2. Run the full conversion pipeline on the PDF
      3. Walk the element tree, skipping page headers/footers
      4. Route each node to the appropriate handler:
           - Headings/text  → plain text chunk
           - Tables         → text representation chunk
           - Pictures/charts → GPT-4o vision description chunk

    Args:
        file_path : Absolute or relative path to the source PDF.

    Returns:
        List of chunk dicts ready for deduplication, embedding, and storage.
    """

    # ── Step 1: Configure Docling pipeline ───────────────────────────────
    # do_ocr=True              — run OCR on scanned/rasterised pages
    # do_table_structure=True  — detect table grid and reconstruct rows/cols
    # generate_picture_images  — render each picture element to a PIL Image
    #
    # accelerator_options: CPU is pinned to avoid MPS float64 crash on Apple
    # Silicon (Docling's layout model uses float64 which MPS rejects).
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
    # Tracks the most recently seen section heading so every chunk carries
    # the section name it belongs to — useful for filtered retrieval.
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    # ── Step 3: Walk the document element tree ────────────────────────────
    for item in doc.iterate_items():
        if isinstance(item, tuple):
            node, _ = item   # iterate_items() yields (node, level)
        else:
            node = item      # older Docling versions yield bare nodes

        # label is a DocItemLabel enum — convert to lowercase string
        label = str(getattr(node, "label", "")).lower()

        # ── Skip page headers/footers ─────────────────────────────────
        # These repeat on every page and pollute retrieval results.
        if label in ("page_header", "page_footer"):
            continue

        # ── Extract page number and bounding box ──────────────────────
        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _make_metadata(content_type: str, element_type: str) -> dict:
            """Build a metadata dict stored alongside every chunk.

            content_type  — "text" | "table" | "image"
            element_type  — raw Docling label ("section_header", "table", …)
            position      — bounding box JSONB {l, t, r, b} or None
            """
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
            }

        # ── Section headings & document title ─────────────────────────
        # Update current_section so all subsequent chunks carry the
        # correct section name until the next heading is encountered.
        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append({
                    "content": text,
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                })

        # ── Tables ────────────────────────────────────────────────────
        elif "table" in label:
            table_text = _table_to_text(node, doc)
            if table_text.strip():
                parsed_chunks.append({
                    "content": table_text.strip(),
                    "content_type": "table",
                    "metadata": _make_metadata("table", "table"),
                })

        # ── Pictures, figures, and charts ─────────────────────────────
        # Base64 is used only to call the vision model. The description
        # text is stored — no base64 is written to the database.
        elif "picture" in label or "figure" in label or label == "chart":
            img_b64 = _extract_image_b64(node, doc)

            # Skip decorative images (logos, dividers) below 100px
            if img_b64 is None:
                caption = getattr(node, "text", "") or f"Image on page {page_no}"
                content = caption.strip()
            else:
                content = _describe_image_with_vision_model(img_b64, page_no)

            parsed_chunks.append({
                "content": content,
                "content_type": "image",
                "metadata": _make_metadata("image", "picture"),
            })

        # ── Plain text: paragraphs, list items, captions, footnotes ───
        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append({
                    "content": text.strip(),
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                })

    logger.info("parse_document: %d elements extracted from %s", len(parsed_chunks), source_file)
    return parsed_chunks
