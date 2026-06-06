"""
src/agents/schemas.py

Response data models for the credit card agent.
Defines the structure of agent responses returned to the chat endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional


class AgentResponse(BaseModel):
    """Structured response from the credit card agent.
    
    Fields:
        query: The original user query
        answer: The generated response text
        data_sources: Where the answer came from (KB documents, SQL queries, etc.)
        page_no: Page numbers from retrieved documents (if applicable)
        document_name: Name(s) of documents used (KB or database)
        sql_query_executed: The SQL query used (if SQL route was taken)
        route_taken: "knowledge_base" or "sql_query" — indicates which path the router chose
    """
    query: str = Field(description="The user's original query")
    answer: str = Field(description="The generated response")
    data_sources: str = Field(description="Source of the answer (e.g., KB document, SQL query)")
    page_no: str = Field(description="Page numbers from documents (or 'N/A' for SQL)")
    document_name: str = Field(description="Name of document/database used")
    sql_query_executed: Optional[str] = Field(
        default=None, 
        description="The SQL query executed (if applicable)"
    )
    route_taken: str = Field(description="'knowledge_base' or 'sql_query'")
