import json
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.api.v1.core.settings import get_settings
from src.api.v1.agents.prompts import NL2SQL_PROMPT_TEMPLATE
from src.api.v1.services.sql_service import execute_sql


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_chat_model, temperature=0, api_key=s.openai_api_key
    )


def _run_nl2sql(enriched_query: str) -> tuple[str, list]:
    """
    Generate SQL from a natural-language query and execute it.
    Returns (sql_string, rows_list).  Never raises — returns ("", []) on error.
    """
    try:
        result = (NL2SQL_PROMPT_TEMPLATE | _get_llm()).invoke(
            {"query": enriched_query},
            config={
                "run_name": "nl2sql",
                "metadata": {
                    "node": "_run_nl2sql",
                    "query": enriched_query,
                },
            },
        )
        sql = result.content.strip()
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql).strip()
        print(f"[_run_nl2sql] generated SQL: {sql[:160]}")
        rows = execute_sql(sql)
        print(f"[_run_nl2sql] {len(rows)} row(s) returned")
        return sql, rows
    except Exception as e:
        print(f"[_run_nl2sql] failed: {e}")
        return "", []


@tool
def nl2sql_execute(question: str) -> str:
    """
    Convert a natural-language question about credit card account data into SQL,
    execute it against the database, and return the results as a JSON string.

    Use this for any question that requires a SINGLE query — transactions,
    balances, reward points, billing statement summaries, fee-waiver checks,
    top merchants, category spend, international transactions, etc.

    Examples:
    - "Show transactions for CC-881001 in March 2026"
    - "What is the current reward balance on CC-881001?"
    - "How much did I spend on food last month?"
    - "Am I on track for the annual fee waiver?"
    """
    sql, rows = _run_nl2sql(question)
    return json.dumps({"sql": sql, "results": rows}, default=str)


@tool
def nl2sql_execute_multi(question_a: str, question_b: str) -> str:
    """
    Run TWO independent natural-language questions as separate SQL queries and
    return both result sets as a combined JSON string.

    Use this when the user's question clearly requires TWO logically distinct
    data sets — for example:
    - Month-over-month comparisons ("this month vs last month")
    - Two different cards ("CC-001 vs CC-002")
    - Transactions AND rewards together as separate aggregates
    - Any "compare X and Y" question where X and Y need separate queries

    Pass each sub-question as question_a and question_b independently.
    Do NOT use this for questions that a single JOIN or CTE can answer.

    Examples:
    question_a = "Total spend on CC-881001 for March 2026"
    question_b = "Total spend on CC-881001 for February 2026"
    """
    sql_a, rows_a = _run_nl2sql(question_a)
    sql_b, rows_b = _run_nl2sql(question_b)
    return json.dumps(
        {
            "query_a": {"sql": sql_a, "results": rows_a},
            "query_b": {"sql": sql_b, "results": rows_b},
        },
        default=str,
    )


# Registry — imported by nodes.py and graph.py
SQL_TOOLS = [nl2sql_execute, nl2sql_execute_multi]
