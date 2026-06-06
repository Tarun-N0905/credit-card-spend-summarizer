import logging
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from src.core.settings import get_settings
from src.agents.prompts import NL2SQL_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


def _get_db() -> SQLDatabase:
    settings = get_settings()
    return SQLDatabase.from_uri(settings.cc_db_connection_string)


def generate_sql(nl_query: str) -> str:
    """
    Convert a natural language query into a SQL string using LLM + NL2SQL prompt.
    Returns the raw SQL string (no execution).
    """
    db = _get_db()
    schema = db.get_table_info()
    llm = _get_llm()

    chain = NL2SQL_PROMPT_TEMPLATE | llm
    result = chain.invoke({"query": nl_query, "schema": schema})

    sql = result.content.strip()
    # Strip markdown code fences if LLM wraps in ```sql ... ```
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip()


def execute_sql(sql: str) -> str:
    """
    Execute a SQL string on the cc_db (read-only business database).
    Returns query results as a formatted string.
    """
    db = _get_db()
    try:
        result = db.run(sql)
        logger.info("sql_service: executed SQL successfully")
        return result if result else "Query returned no results."
    except Exception as e:
        logger.error("sql_service: SQL execution failed: %s", e)
        return f"SQL execution error: {e}"


def query(nl_query: str) -> tuple[str, str]:
    """
    Full NL2SQL pipeline: natural language → SQL → execute → return results.

    Returns:
        (sql_executed, results_string)
    """
    try:
        sql = generate_sql(nl_query)
        logger.info("sql_service: generated SQL: %s", sql)
        results = execute_sql(sql)
        return sql, results
    except Exception as e:
        logger.error("sql_service: query pipeline failed: %s", e)
        return "", f"Error processing query: {e}"
