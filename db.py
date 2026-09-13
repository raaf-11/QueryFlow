"""
Database access layer.

Two jobs:
  1. Introspect the real schema with SQLAlchemy so the LLM never has to guess
     table or column names.
  2. Execute generated SQL in a read-only, row-capped way.

Everything here is deliberately plain: one engine, a few functions, no ORM
models. The agent only ever reads.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# Loaded here because every other module imports db, so this guarantees
# GROQ_API_KEY / LANGSMITH_* are present no matter which entrypoint you run.
load_dotenv()

def db_path() -> Path:
    """
    Where the database file lives.

    Read lazily rather than at import time so a caller can point at a different
    file (eval.py --db chinook_dirty.db) after this module is already imported.
    """
    return Path(os.getenv("CHINOOK_DB_PATH", "chinook.db")).resolve()

# Hard cap on rows pulled back into memory / into the LLM prompt.
MAX_ROWS = int(os.getenv("MAX_ROWS", "200"))

_engine: Engine | None = None


def get_engine() -> Engine:
    """Lazily create a single SQLAlchemy engine for the SQLite file."""
    global _engine
    if _engine is None:
        path = db_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Database not found at {path}. Run: python setup_db.py"
            )
        # SQLite URI mode lets us open the file read-only at the driver level,
        # which is a real guardrail rather than just a prompt instruction.
        uri = f"file:{path.as_posix()}?mode=ro"
        _engine = create_engine(
            "sqlite:///",
            creator=lambda: __import__("sqlite3").connect(uri, uri=True),
        )
    return _engine


@lru_cache(maxsize=1)
def get_schema_map() -> dict[str, set[str]]:
    """
    {lowercase_table_name: {lowercase_column_names}}.

    The SQL validator uses this to catch hallucinated tables/columns *before*
    the query ever touches the database.
    """
    insp = inspect(get_engine())
    return {
        t.lower(): {c["name"].lower() for c in insp.get_columns(t)}
        for t in insp.get_table_names()
    }


@lru_cache(maxsize=1)
def get_real_table_names() -> dict[str, str]:
    """{lowercase_name: ActualCaseName} — used to echo correct casing in errors."""
    return {t.lower(): t for t in inspect(get_engine()).get_table_names()}


@lru_cache(maxsize=4)
def get_schema_description(mode: str = "full") -> str:
    """
    A compact, LLM-friendly text rendering of the schema.

    Foreign keys matter more than anything else here — they are what let the
    model write correct JOINs instead of inventing linking columns.

    `mode` deliberately degrades what the model is told. This exists because
    with the full schema a strong model essentially never produces bad SQL
    against Chinook, so the retry loop never fires and cannot be measured.
    Degrading the context is how you create real failures to recover from:

      full        everything: columns, types, primary keys, foreign keys
      no_fk       columns but no foreign keys - the model must guess join keys
      names_only  table names only - the model must guess every column name

    Cached per mode; the schema never changes at runtime.
    """
    insp = inspect(get_engine())
    lines: list[str] = []

    for table in sorted(insp.get_table_names()):
        if mode == "names_only":
            lines.append(f"TABLE {table}")
            continue

        cols = insp.get_columns(table)
        pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])

        col_parts = []
        for c in cols:
            marker = " PK" if c["name"] in pk else ""
            col_parts.append(f"{c['name']} {c['type']}{marker}")

        lines.append(f"TABLE {table} ({', '.join(col_parts)})")

        if mode == "no_fk":
            continue

        for fk in insp.get_foreign_keys(table):
            local = ", ".join(fk["constrained_columns"])
            remote_cols = ", ".join(fk["referred_columns"])
            lines.append(
                f"  FK: {table}.{local} -> {fk['referred_table']}.{remote_cols}"
            )

    return "\n".join(lines)


def run_query(sql: str, max_rows: int = MAX_ROWS) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Execute a SELECT and return (column_names, rows_as_dicts).

    Raises whatever SQLAlchemy raises — the executor node catches it and feeds
    the message back to the query generator as retry context. That error text
    is the fuel for the self-correction loop, so we do not swallow it here.
    """
    with get_engine().connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, r)) for r in result.fetchmany(max_rows)]
    return columns, rows


def rows_to_markdown(columns: list[str], rows: list[dict[str, Any]], limit: int = 25) -> str:
    """Render a result set as a small markdown table for the final-answer prompt."""
    if not columns:
        return "(no columns)"
    if not rows:
        return "(query ran successfully and returned 0 rows)"

    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join("NULL" if r.get(c) is None else str(r.get(c)) for c in columns) + " |"
        for r in rows[:limit]
    ]
    out = "\n".join([head, sep, *body])
    if len(rows) > limit:
        out += f"\n\n({len(rows) - limit} more rows not shown)"
    return out
