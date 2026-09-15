from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from db import get_schema_description              
from graph import MAX_ATTEMPTS, run_agent              

st.set_page_config(page_title="Ask the Chinook database", page_icon="?", layout="wide")

with st.sidebar:
    st.subheader("Settings")
    max_attempts = st.slider(
        "Max attempts", 1, 5, MAX_ATTEMPTS,
        help="Set this to 1 to disable self-correction and see the difference.",
    )

    cross_check = st.toggle(
        "Cross-check answers", value=True,
        help="Write the query a second time, independently, and compare results. "
             "Costs one extra call. Catches answers that are wrong without "
             "erroring - which the retry loop cannot.",
    )

    st.subheader("Try one of these")
    examples = [
        "Which artist has the most albums?",
        "What are the 5 longest tracks?",
        "Which 3 genres generated the most revenue?",
        "Who are our top 5 spending clients?",
        "How many customers do we have in Antarctica?",
        "List any tracks longer than 10 hours",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending = ex

    with st.expander("Database schema"):
        st.code(get_schema_description(), language="text")

st.title("Ask the Chinook database")
st.caption(
    "Ask in plain English. The agent writes SQL, checks it, runs it, and rewrites "
    "it from the error if something goes wrong."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_run(state: dict) -> None:
                                                            
    st.markdown(state["answer"])

    attempts = state["attempts"]
    left, right = st.columns([1, 4])
    left.metric("Attempts", attempts)
    if attempts > 1 and state["status"] == "success":
        right.success(f"Self-corrected: attempt 1 failed, attempt {attempts} worked.")
    elif state["status"] != "success":
        right.error(f"Gave up after {attempts} attempts.")
    right.caption(f"{state['elapsed_s']}s")

    if state.get("agreement") == "disagree":
        st.warning(
            "Two independently written queries returned different answers, so "
            "this question is ambiguous. Both readings are below - treat the "
            "number above as unconfirmed."
        )
        with st.expander("The two interpretations", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"First query — {len(state['rows'])} rows")
                st.code(state["sql"], language="sql")
            with c2:
                st.caption(f"Second query — {len(state['cross_check_rows'])} rows")
                st.code(state["cross_check_sql"], language="sql")
    elif state.get("agreement") == "agree":
        right.caption("Confirmed by an independent second query.")
    elif state.get("agreement") == "inconclusive":
        right.caption("Cross-check query failed; answer is unverified.")

    if attempts > 1 or state["status"] != "success":
        with st.expander(f"How it got here ({attempts} attempts)", expanded=True):
            for h in state["history"]:
                ok = h["stage"] == "success"
                st.markdown(
                    f"**Attempt {h['attempt']}** — "
                    + (f"ran, {h['row_count']} rows" if ok else f"failed at {h['stage']}")
                )
                st.code(h["sql"], language="sql")
                if not ok:
                    st.warning(h["error"])

    if state["status"] == "success":
        with st.expander("SQL that answered it", expanded=attempts == 1):
            st.code(state["sql"], language="sql")
        if state["rows"]:
            st.dataframe(pd.DataFrame(state["rows"]), use_container_width=True, hide_index=True)
        else:
            st.info("The query ran fine and matched no rows. That is the answer.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_run(m["state"])

question = st.chat_input("e.g. Which employee supports the most customers?")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Writing SQL, checking it, running it..."):
            try:
                state = run_agent(
                    question,
                    max_attempts=max_attempts,
                    cross_check=cross_check,
                )
            except Exception as e:
                st.error(f"Agent failed to run: {e}")
                st.stop()
        render_run(state)

    st.session_state.messages.append({"role": "assistant", "state": state})
