"""
src/agents/schemas.py

Response data models for the credit card agent.
Aligned with the Capstone BFSI-CC-003 API specification.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union


class AgentResponse(BaseModel):
    """
    Generic response used for knowledge_base, general, and sql_query routes.
    """

    query: str
    answer: str
    data_sources: Union[list, str]
    page_no: str
    document_name: str
    sql_query_executed: Optional[str] = None
    route_taken: str

    # Populated only on knowledge_base route when retrieved chunks contain images.
    # Each entry is a file-system path that Streamlit can read with st.image().
    image_paths: Optional[List[str]] = Field(
        default=None,
        description="Absolute paths to images referenced in the answer (kb route only)",
    )
