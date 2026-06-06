from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChunk:
    id: str
    chunk_text: str
    score: float

    content_type: str | None
    page_number: int | None
    section_name: str | None

    metadata: dict[str, Any]
    position: dict[str, Any] | None
