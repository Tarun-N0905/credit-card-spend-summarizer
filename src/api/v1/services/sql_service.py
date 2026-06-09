import logging
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from src.api.v1.core.settings import get_settings
from src.api.v1.agents.prompts import NL2SQL_PROMPT_TEMPLATE

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
    """Convert a natural language query into a SQL string using LLM + NL2SQL prompt."""
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


def execute_sql(sql: str) -> list[dict]:
    """Execute a SQL string on the cc_db and return a list of dicts (one per row)."""
    db = _get_db()
    try:
        result = db.run(sql)
        # SQLDatabase.run may return string; attempt parsing
        if isinstance(result, str):
            import json

            try:
                result = json.loads(result)
            except Exception:
                # Fallback: treat single-row string as one-row dict
                result = [{"result": result}]
        elif result is None:
            result = []
        logger.info("sql_service: executed SQL successfully")
        return result
    except Exception as e:
        logger.error("sql_service: SQL execution failed: %s", e)
        return []


def query(nl_query: str) -> tuple[str, list[dict]]:
    """Full NL2SQL pipeline: natural language → SQL → execute → return results."""
    try:
        sql = generate_sql(nl_query)
        logger.info("sql_service: generated SQL: %s", sql)
        results = execute_sql(sql)
        return sql, results
    except Exception as e:
        logger.error("sql_service: query pipeline failed: %s", e)
        return "", []
