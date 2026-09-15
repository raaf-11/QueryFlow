from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

load_dotenv()

def db_path() -> Path:
    
    return Path(os.getenv("CHINOOK_DB_PATH", "chinook.db")).resolve()

MAX_ROWS = int(os.getenv("MAX_ROWS", "200"))

_engine: Engine | None = None

def get_engine() -> Engine:
                                                                       
    global _engine
    if _engine is None:
        path = db_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Database not found at {path}. Run: python setup_db.py"
            )

        uri = f"file:{path.as_posix()}?mode=ro"
        _engine = create_engine(
            "sqlite:///",
            creator=lambda: __import__("sqlite3").connect(uri, uri=True),
        )
    return _engine

@lru_cache(maxsize=1)
def get_schema_map() -> dict[str, set[str]]:
     
    insp = inspect(get_engine())
    return {
        t.lower(): {c["name"].lower() for c in insp.get_columns(t)}
        for t in insp.get_table_names()
    }

@lru_cache(maxsize=1)
def get_real_table_names() -> dict[str, str]:
                                                                                   
    return {t.lower(): t for t in inspect(get_engine()).get_table_names()}

@lru_cache(maxsize=4)
def get_schema_description(mode: str = "full") -> str:
     
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
     
    with get_engine().connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, r)) for r in result.fetchmany(max_rows)]
    return columns, rows

def rows_to_markdown(columns: list[str], rows: list[dict[str, Any]], limit: int = 25) -> str:
                                                                                    
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
