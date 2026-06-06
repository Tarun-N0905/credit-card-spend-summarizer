import logging
from langchain_core.tools import tool
from src.services.sql_service import query

logger = logging.getLogger(__name__)


@tool
def customer_data_tool(nl_query: str) -> str:
    """
    Query the customer business database for account-specific information such as
    credit card details, transaction history, reward points, outstanding balance,
    credit limit, billing statements, and customer profile data.
    Use this for any question about a specific customer's account or card data.
    Input should be a natural language question — SQL will be auto-generated.
    """
    try:
        sql_executed, results = query(nl_query)
        logger.info("customer_data_tool: SQL executed: %s", sql_executed)
        if not results:
            return "No data found for this query."
        return f"SQL Executed:\n{sql_executed}\n\nResults:\n{results}"
    except Exception as e:
        logger.error("customer_data_tool failed: %s", e)
        return f"Customer data lookup failed: {e}"
