"""
Two supporting pieces the graph leans on:

  * `validate_sql`  — static analysis with sqlglot, run BEFORE the query hits
                      the database. It catches syntax errors, non-SELECT
                      statements, and hallucinated table/column names.
  * `get_llm`       — the Groq chat model.

Catching a hallucinated column here rather than at execution time is the
difference between a useful error message ("no column Track.Duration; did you
mean Milliseconds?") and a cryptic one. Better error text means better retries.
"""

from __future__ import annotations

import os
import re
from difflib import get_close_matches

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from db import get_schema_map

# Statement types the agent must never produce. The database connection is
# already read-only, so this is defence in depth — but it also gives the
# generator a clean, correctable error instead of a driver-level failure.
_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
    exp.Alter, exp.TruncateTable,
)


def extract_sql(text: str) -> str:
    """
    Pull raw SQL out of an LLM response.

    Models wrap SQL in ```sql fences or add a sentence of preamble no matter how
    firmly you tell them not to, so we strip that here instead of burning a
    retry attempt on it.
    """
    text = text.strip()

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    # Drop any chatter before the first SELECT / WITH.
    m = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]

    return text.strip().rstrip(";").strip()


def _cte_names(tree: exp.Expression) -> set[str]:
    """Names defined by WITH clauses — these are not real tables."""
    return {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message). Empty message when valid.

    Checks, in order:
      1. parses as SQLite
      2. exactly one statement
      3. is a SELECT (read-only)
      4. every referenced table exists
      5. every referenced column exists on some referenced table
    """
    if not sql.strip():
        return False, "No SQL was produced."

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except ParseError as e:
        return False, f"SQL failed to parse: {e}"

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        return False, "No SQL statement found."
    if len(statements) > 1:
        return False, "Produce exactly one SQL statement, not multiple."

    tree = statements[0]

    for bad in _FORBIDDEN:
        if isinstance(tree, bad) or tree.find(bad):
            return False, (
                f"Only read-only SELECT queries are allowed; "
                f"found a {bad.__name__.upper()} statement."
            )

    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        return False, f"Expected a SELECT query, got {type(tree).__name__}."

    schema = get_schema_map()
    ctes = _cte_names(tree)

    # --- 4. table existence -------------------------------------------------
    referenced: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = tbl.name.lower()
        if not name or name in ctes:
            continue
        if name not in schema:
            suggestion = get_close_matches(name, schema.keys(), n=1, cutoff=0.5)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            return False, (
                f"Table '{tbl.name}' does not exist.{hint} "
                f"Valid tables: {', '.join(sorted(schema))}."
            )
        referenced.add(name)

    if not referenced and not ctes:
        return False, "The query does not reference any table."

    # --- 5. column existence ------------------------------------------------
    # Skipped when CTEs are present: a CTE invents its own output columns, and
    # resolving those properly needs a full binder. Being lenient there is much
    # better than rejecting correct queries.
    if ctes:
        return True, ""

    allowed = {c for t in referenced for c in schema[t]}
    # Aliases the query itself defines (SELECT ... AS foo, then ORDER BY foo).
    allowed |= {
        a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias
    }
    allowed |= {
        t.alias.lower() for t in tree.find_all(exp.Table) if t.alias
    }

    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name or name == "*" or name in allowed:
            continue
        suggestion = get_close_matches(name, sorted(allowed), n=1, cutoff=0.6)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        tables_txt = ", ".join(sorted(referenced))
        return False, (
            f"Column '{col.name}' does not exist on the referenced "
            f"table(s) ({tables_txt}).{hint}"
        )

    return True, ""


def get_llm(temperature: float = 0.0):
    """
    The Groq chat model used by both LLM nodes.

    temperature=0 by default so a retry only changes behaviour because the
    error feedback changed the prompt — not because of random sampling. That
    makes the self-correction loop honest and reproducible.
    """
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        max_retries=2,      # network-level retries only, unrelated to the agent loop
        timeout=60,
    )
