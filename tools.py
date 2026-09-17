from __future__ import annotations

import os
import re
from difflib import get_close_matches

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from db import get_real_table_names, get_schema_map

_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
    exp.Alter, exp.TruncateTable,
)

def extract_sql(text: str) -> str:    
    text = text.strip()

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    m = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]

    return text.strip().rstrip(";").strip()

def _cte_names(tree: exp.Expression) -> set[str]:
                                                                    
    return {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

def validate_sql(sql: str) -> tuple[bool, str]:   
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

    if ctes:
        return True, ""

    allowed = {c for t in referenced for c in schema[t]}
                                                                              
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

        available = []
        for t in sorted(referenced):
            cols = sorted(schema[t])
            shown = ", ".join(cols[:20]) + ("..." if len(cols) > 20 else "")
            available.append(f"{get_real_table_names().get(t, t)}({shown})")

        return False, (
            f"Column '{col.name}' does not exist.{hint} "
            f"Available columns: {'; '.join(available)}"
        )

    return True, ""

def _norm_value(v) -> str:
                                                                       
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{round(float(v), 2):g}"
    return str(v).strip().lower()

def _norm_rows(rows: list[dict]) -> list[tuple]:     
    return sorted(tuple(sorted(_norm_value(v) for v in r.values())) for r in rows)

def _tokens(row: dict, whole_values: bool = True) -> set[str]: 
    out: set[str] = set()
    for v in row.values():
        n = _norm_value(v)
        if whole_values:
            out.add(n)
        parts = [p for p in n.replace(",", " ").split() if p]
        out.update(parts)
        if not whole_values and len(parts) == 0:
            out.add(n)
    return out

def compare_results(expected: list[dict], actual: list[dict]) -> tuple[bool, bool]:

    strict = _norm_rows(expected) == _norm_rows(actual)
    if strict:
        return True, True

    if len(expected) != len(actual):
        return False, False

    remaining = [_tokens(r) for r in actual]
    for e in expected:
        et = _tokens(e, whole_values=False)
        hit = next((i for i, at in enumerate(remaining) if et <= at), None)
        if hit is None:
            return False, False
        remaining.pop(hit)

    return False, True

def get_llm(temperature: float = 0.0):   
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        max_retries=2,                                                               
        timeout=60,
    )
