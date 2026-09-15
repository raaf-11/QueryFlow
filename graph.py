from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from db import get_schema_description, rows_to_markdown, run_query
from tools import compare_results, extract_sql, get_llm, validate_sql

MAX_ATTEMPTS = 3
EMPTY_RESULT_IS_VALID = True
CROSS_CHECK_TEMPERATURE = 0.8


class AgentState(TypedDict):
    question: str
    schema: str

    sql: str
    error: str | None              # set by validator/executor, read on retry
    error_stage: str | None        # "validation" or "execution"

    columns: list[str]
    rows: list[dict[str, Any]]

    attempts: int                  # incremented once per generate_sql call
    max_attempts: int
    history: list[dict[str, Any]]  # one record per attempt

    status: Literal["pending", "success", "failed"]
    answer: str

    use_domain_notes: bool
    schema_mode: str               # "full" | "no_fk" | "names_only"

    cross_check: bool
    cross_check_sql: str
    cross_check_rows: list[dict[str, Any]]
    cross_check_error: str | None
    agreement: str                 # "agree" | "disagree" | "inconclusive" | "skipped"
    confidence: str                # "high" | "low" | "unverified"


DOMAIN_NOTES = """
DOMAIN NOTES
- Track.Milliseconds is a track's duration. There is no duration/length column.
- Invoice.Total is the amount of a sale. Revenue means SUM(Invoice.Total),
  or SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) at line-item level.
- A customer's name is split across Customer.FirstName and Customer.LastName.
- Genre.Name holds genre/style names; Track.Name holds song/track titles.
- Customer.SupportRepId links a customer to their Employee sales rep.
"""

GENERATOR_SYSTEM = """You are a careful SQLite analyst. You write one SQL query \
that answers the user's question against the schema below.

SCHEMA
{schema}
{domain_notes}
RULES
1. Output ONLY the SQL query. No prose, no markdown fences, no explanation.
2. Exactly one statement. SELECT only - never INSERT, UPDATE, DELETE or DDL.
3. Use only tables and columns that appear in the schema above, spelled exactly.
4. Join through the foreign keys shown in the schema.
5. Alias aggregates with a readable name, e.g. SUM(i.Total) AS TotalSpent.
6. If the question implies a ranking or "top N", use ORDER BY with LIMIT.
6a. For "the top one per group" questions (best X per country, most Y per genre),
   use RANK() OVER (PARTITION BY ...), not ROW_NUMBER(), and keep every tied
   row. If three customers tie for top spender in a country, all three are
   correct answers and dropping two of them loses real information. Use
   ROW_NUMBER() only when the question explicitly asks for exactly one row
   per group.
6b. Prefer the customer's own attribute over the transaction's copy of it when
   both exist: Customer.Country is where the customer is, Invoice.BillingCountry
   is where one invoice was billed. Use the former unless the question is
   clearly about billing.
7. If the question has no natural limit and could return thousands of rows,
   add a sensible LIMIT.
8. Select the columns needed to answer the question - not SELECT * - unless the
   question really is asking for whole records."""

RETRY_TEMPLATE = """Your previous attempt failed. Fix it.

FAILED SQL
{sql}

{stage} ERROR
{error}

Rewrite the query so it answers the original question and does not hit this
error again. Output ONLY the corrected SQL."""

CROSS_CHECK_SYSTEM = """You are a second analyst. Another analyst has already \
written a query for this question; you have not seen it, and you must not try \
to guess what they wrote.

Answer the question independently against the schema below.

SCHEMA
{schema}

RULES
1. Output ONLY the SQL query. No prose, no markdown fences.
2. Exactly one SELECT statement, read-only.
3. Use only tables and columns from the schema, spelled exactly.
4. Where a column's meaning or unit is not stated in the schema, choose the
   most literal reading of the question rather than the most convenient one.
5. Prefer a different formulation than the obvious one where both are correct -
   a subquery instead of a join, an aggregate instead of a window, and so on.
   The point is an independent route to the same answer."""

ANSWER_SYSTEM = """You turn SQL results into a short, direct answer.

Rules:
- Answer in 1-3 sentences of plain English. No markdown tables, no SQL.
- Use the exact values from the results. Never invent or estimate numbers.
- If the result set is empty, say plainly that there are none - an empty result
  is a real answer, not an error.
- If the results do not actually answer the question, say so.
- If told the two queries disagreed, do NOT state a number as fact. Say the
  question is ambiguous, give both readings, and say what would settle it."""


def inspect_schema(state: AgentState) -> dict:
    return {"schema": get_schema_description(state.get("schema_mode", "full"))}


def generate_sql(state: AgentState) -> dict:
    llm = get_llm()

    user_parts = [f"Question: {state['question']}"]

    # Retry pass: show the model its own failed SQL and the exact error.
    if state.get("error"):
        user_parts.append(
            RETRY_TEMPLATE.format(
                sql=state.get("sql", ""),
                stage=(state.get("error_stage") or "").upper(),
                error=state["error"],
            )
        )

    response = llm.invoke([
        SystemMessage(content=GENERATOR_SYSTEM.format(
            schema=state["schema"],
            domain_notes=DOMAIN_NOTES if state.get("use_domain_notes", True) else "",
        )),
        HumanMessage(content="\n\n".join(user_parts)),
    ])

    return {
        "sql": extract_sql(response.content),
        "attempts": state.get("attempts", 0) + 1,
        "error": None,
        "error_stage": None,
    }


def validate_sql_node(state: AgentState) -> dict:
    ok, message = validate_sql(state["sql"])

    if ok:
        return {"error": None, "error_stage": None}

    return {
        "error": message,
        "error_stage": "validation",
        "history": state["history"] + [{
            "attempt": state["attempts"],
            "sql": state["sql"],
            "stage": "validation",
            "error": message,
            "row_count": None,
        }],
    }


def execute_sql(state: AgentState) -> dict:
    try:
        columns, rows = run_query(state["sql"])
        return {"columns": columns, "rows": rows, "error": None, "error_stage": None}
    except Exception as e:
        # The error message is what the generator needs to fix itself, so it
        # becomes state rather than a crash.
        message = str(e).split("\n")[0].strip()
        return {"columns": [], "rows": [], "error": message, "error_stage": "execution"}


def check_result(state: AgentState) -> dict:
    error = state.get("error")

    if not error and not state.get("columns"):
        error = "Query executed but returned no columns at all."

    # Zero rows is a real answer ("no customers in Antarctica"), not a failure.
    if not error and not state.get("rows") and not EMPTY_RESULT_IS_VALID:
        error = "Query returned zero rows."

    if error:
        return {
            "status": "failed",
            "error": error,
            "error_stage": state.get("error_stage") or "execution",
            "history": state["history"] + [{
                "attempt": state["attempts"],
                "sql": state["sql"],
                "stage": state.get("error_stage") or "execution",
                "error": error,
                "row_count": None,
            }],
        }

    return {
        "status": "success",
        "error": None,
        "history": state["history"] + [{
            "attempt": state["attempts"],
            "sql": state["sql"],
            "stage": "success",
            "error": None,
            "row_count": len(state.get("rows", [])),
        }],
    }


def cross_check(state: AgentState) -> dict:
    if not state.get("cross_check"):
        return {"agreement": "skipped", "confidence": "unverified"}

    llm = get_llm(temperature=CROSS_CHECK_TEMPERATURE)
    sql = ""

    try:
        response = llm.invoke([
            SystemMessage(content=CROSS_CHECK_SYSTEM.format(schema=state["schema"])),
            HumanMessage(content=f"Question: {state['question']}"),
        ])
        sql = extract_sql(response.content)

        ok, message = validate_sql(sql)
        if not ok:
            return {
                "cross_check_sql": sql,
                "cross_check_error": f"validation: {message}",
                "agreement": "inconclusive",
                "confidence": "unverified",
            }

        _, rows = run_query(sql)
    except Exception as e:
        # A failed cross-check says nothing about the primary answer.
        return {
            "cross_check_sql": sql,
            "cross_check_error": str(e).split("\n")[0][:200],
            "agreement": "inconclusive",
            "confidence": "unverified",
        }

    # Compared both ways: compare_results is directional, and neither query
    # here is authoritative. One-way would flag ("Helena Holy") against
    # ("Helena", "Holy") as a disagreement in one order but not the other.
    _, forward = compare_results(state["rows"], rows)
    _, backward = compare_results(rows, state["rows"])
    agrees = forward or backward

    return {
        "cross_check_sql": sql,
        "cross_check_rows": rows,
        "cross_check_error": None,
        "agreement": "agree" if agrees else "disagree",
        "confidence": "high" if agrees else "low",
    }


def final_answer(state: AgentState) -> dict:
    if state.get("status") != "success":
        return {
            "status": "failed",
            "answer": (
                f"I could not answer this after {state['attempts']} attempts. "
                f"Last error ({state.get('error_stage')}): {state.get('error')}"
            ),
        }

    llm = get_llm()
    table = rows_to_markdown(state["columns"], state["rows"])

    parts = [
        f"Question: {state['question']}",
        f"SQL that was run:\n{state['sql']}",
        f"Results:\n{table}",
    ]

    if state.get("agreement") == "disagree":
        other = rows_to_markdown(
            list(state["cross_check_rows"][0].keys()) if state["cross_check_rows"] else [],
            state["cross_check_rows"],
        )
        parts.append(
            "WARNING: a second, independently written query for this same "
            "question returned a DIFFERENT answer. The question is ambiguous "
            "or a column's meaning is unclear.\n\n"
            f"Second query:\n{state['cross_check_sql']}\n\n"
            f"Second result:\n{other}\n\n"
            "Do not state either result as fact. Explain that the two readings "
            "disagree, give both, and say what information would settle it."
        )

    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content="\n\n".join(parts)),
    ])

    return {"answer": response.content.strip(), "status": "success"}


# Retry loop 1: bad SQL goes back to the generator with the validator's error,
# unless the attempt budget is spent.
def route_after_validation(state: AgentState) -> Literal["execute", "retry", "give_up"]:
    if not state.get("error"):
        return "execute"
    if state["attempts"] >= state["max_attempts"]:
        return "give_up"
    return "retry"


# Retry loop 2: same shape, but for execution failures. "success" here only
# means the query RAN; whether it is RIGHT is the cross-check node's job.
def route_after_result(state: AgentState) -> Literal["verify", "answer", "retry", "give_up"]:
    if state.get("status") == "success":
        return "verify" if state.get("cross_check") else "answer"
    if state["attempts"] >= state["max_attempts"]:
        return "give_up"
    return "retry"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("inspect_schema", inspect_schema)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql_node)
    g.add_node("execute_sql", execute_sql)
    g.add_node("check_result", check_result)
    g.add_node("cross_check", cross_check)
    g.add_node("final_answer", final_answer)

    g.add_edge(START, "inspect_schema")
    g.add_edge("inspect_schema", "generate_sql")
    g.add_edge("generate_sql", "validate_sql")

    g.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute": "execute_sql",
            "retry": "generate_sql",
            "give_up": "final_answer",
        },
    )

    g.add_edge("execute_sql", "check_result")

    g.add_conditional_edges(
        "check_result",
        route_after_result,
        {
            "verify": "cross_check",
            "answer": "final_answer",
            "retry": "generate_sql",
            "give_up": "final_answer",
        },
    )

    g.add_edge("cross_check", "final_answer")
    g.add_edge("final_answer", END)

    return g.compile()


AGENT = build_graph()


def run_agent(
    question: str,
    max_attempts: int = MAX_ATTEMPTS,
    use_domain_notes: bool = True,
    schema_mode: str = "full",
    cross_check: bool = False,
) -> dict:
    initial: AgentState = {
        "question": question,
        "schema": "",
        "sql": "",
        "error": None,
        "error_stage": None,
        "columns": [],
        "rows": [],
        "attempts": 0,
        "max_attempts": max_attempts,
        "history": [],
        "status": "pending",
        "answer": "",
        "use_domain_notes": use_domain_notes,
        "schema_mode": schema_mode,
        "cross_check": cross_check,
        "cross_check_sql": "",
        "cross_check_rows": [],
        "cross_check_error": None,
        "agreement": "skipped",
        "confidence": "unverified",
    }

    started = time.time()
    final = AGENT.invoke(
        initial,
        config={"recursion_limit": 40, "run_name": "text2sql-agent"},
    )
    final["elapsed_s"] = round(time.time() - started, 2)
    return final


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    q = " ".join(sys.argv[1:]) or "Which artist has the most albums?"
    result = run_agent(q, cross_check=True)

    print(f"\nQ: {q}")
    print(f"\nSQL:\n{result['sql']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nAttempts: {result['attempts']} | status: {result['status']} "
          f"| agreement: {result['agreement']} | confidence: {result['confidence']} "
          f"| {result['elapsed_s']}s")
    for h in result["history"]:
        flag = "OK" if h["stage"] == "success" else "FAIL"
        print(f"  attempt {h['attempt']}: {flag} ({h['stage']}) {h['error'] or ''}")
