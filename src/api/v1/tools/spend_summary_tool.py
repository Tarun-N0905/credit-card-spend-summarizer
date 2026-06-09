import logging
from langchain_core.tools import tool
from src.api.v1.services.sql_service import query

logger = logging.getLogger(__name__)


@tool
def spend_summary_tool(nl_query: str) -> str:
    """
    Analyze and summarize credit card spending patterns from the business database.
    Use this for questions about total spend by category, monthly summaries,
    merchant-level breakdowns, EMI analysis, international transaction summaries,
    fee breakdowns, and reward points earned per period.
    Input should be a natural language question — SQL will be auto-generated.
    Returns a structured summary of the spending data.
    """
    try:
        sql_executed, results = query(nl_query)
        logger.info("spend_summary_tool: SQL executed: %s", sql_executed)
        if not results:
            return "No spending data found for this query."

        summary = (
            f"Spend Analysis Results\n"
            f"{'=' * 40}\n"
            f"SQL Executed:\n{sql_executed}\n\n"
            f"Data:\n{results}"
        )
        return summary
    except Exception as e:
        logger.error("spend_summary_tool failed: %s", e)
        return f"Spend summary failed: {e}"
