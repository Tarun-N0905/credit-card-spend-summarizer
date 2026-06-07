"""
src/agents/schemas.py

Response data models for the credit card agent.
Aligned with the Capstone BFSI-CC-003 API specification.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union


class CategoryBreakdown(BaseModel):
    """One row in the category spend breakdown."""
    category: str
    amount: float
    count: int
    pct_of_total: float


class TopMerchant(BaseModel):
    """One entry in the top merchants list."""
    merchant_name: str
    amount: float


class InternationalSpend(BaseModel):
    """International spend summary."""
    total_amount: float
    transaction_count: int


class RewardPointsSummary(BaseModel):
    """Reward points earned this billing cycle."""
    points_earned: int
    inr_value: float


class SpendSummaryResponse(BaseModel):
    """
    Structured response from the credit card spend summarizer.
    Matches the API spec in Capstone BFSI-CC-003.
    """
    card_id: str = Field(description="Card identifier")
    customer_name: str = Field(description="First name of cardholder")
    billing_month: str = Field(description="Human-readable billing month, e.g. 'March 2026'")
    total_spend: float = Field(description="Total purchase amount for the billing period")
    total_transactions: int = Field(description="Count of purchase transactions")
    category_breakdown: List[CategoryBreakdown] = Field(
        description="Spend broken down by category with percentage of total"
    )
    top_merchants: List[TopMerchant] = Field(
        description="Top 5 merchants by spend amount"
    )
    international_spend: InternationalSpend = Field(
        description="Total international spend and transaction count"
    )
    reward_points_earned: RewardPointsSummary = Field(
        description="Points earned this cycle and their INR redemption value"
    )
    mom_change_pct: Optional[float] = Field(
        default=None,
        description="Month-over-month spend change as a percentage (None if first month)"
    )
    summary_text: str = Field(
        description="LLM-generated 2-4 sentence narrative summary of spending"
    )
    tip: str = Field(
        description="LLM-generated personalised tip or product suggestion"
    )
    # --- internal / debug ---
    route_taken: str = Field(default="sql_query", description="Always 'sql_query' for this agent")
    sql_queries_executed: Optional[List[str]] = Field(
        default=None,
        description="All SQL queries run to build the context"
    )


# Keep AgentResponse for the knowledge_base route (policy / terms questions)
class AgentResponse(BaseModel):
    """
    Generic response used for knowledge_base (policy / terms) queries.
    Not used for the spend summarizer path.
    """
    query: str
    answer: str
    data_sources: Union[list, str]
    page_no: str
    document_name: str
    sql_query_executed: Optional[str] = None
    route_taken: str
