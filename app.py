"""
Streamlit UI for the essay grader.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from grader import (
    SUPPORTED_EXTS,
    column_keys,
    combine_reference_texts,
    empty_row,
    fieldnames_for,
    fill_row_from_result,
    grade_essay,
    read_text_from_bytes,
    results_to_csv_string,
)


# ---------- page setup -----------------------------------------------------

st.set_page_config(
    page_title="Essay grader",
    page_icon="📝",
    layout="wide",
)

# Initialise persistent state (Streamlit reruns the script on every interaction)
for key, default in [
    ("results", None),
    ("criteria", None),
    ("last_csv", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------- helpers --------------------------------------------------------

DEFAULT_CRITERIA_PATH = Path(__file__).parent / "criteria.json"
SUPPORTED_EXT_LIST = sorted(e.lstrip(".") for e in SUPPORTED_EXTS)
MODELS = ["gpt-4.1-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.o-mini", "gpt-4.1", "gpt-3.5-turbo"]


def load_default_criteria() -> dict | None:
    if DEFAULT_CRITERIA_PATH.exists():
        try:
            return json.loads(DEFAULT_CRITERIA_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not parse default criteria.json: {e}")
    return None


def summary_dataframe(results: list[dict], criteria: dict) -> pd.DataFrame:
    """Compact view: one row per student with marks but not full reasoning."""
    rows = []
    for r in results:
        row = {
            "Student": r["student_name"],
            "File": r["student_file"],
        }
        for c, (marks_col, _) in zip(criteria["criteria"], column_keys(criteria)):
            row[c["name"]] = r.get(marks_col, "")
        row["Total"] = r.get("total_marks", "")
        row["Status"] = r.get("status", "")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- sidebar --------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get a key at platform.openai.com/api-keys. Stored only in this browser session.",
    )

    model = st.selectbox(
        "Model",
        MODELS,
        index=0,
        help="gpt-4o-mini is ~10× cheaper. Use gpt-4o for tougher rubrics or longer essays.",
    )

    max_ref_chars = st.slider(
        "Max reference chars",
        min_value=10_000,
        max_value=200_000,
        value=80_000,
        step=10_000,
        help="Cap on combined reference materials sent to the model. Controls token cost.",
    )

    st.divider()

    st.subheader("Rubric")
    rubric_choice = st.radio(
        "Source",
        ["Default (criteria.json)", "Upload custom"],
        label_visibility="collapsed",
    )
    if rubric_choice == "Upload custom":
        rubric_upload = st.file_uploader("Upload criteria.json", type=["json"])
        if rubric_upload is not None:
            try:
                st.session_state.criteria = json.loads(rubric_upload.read().decode("utf-8"))
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.session_state.criteria = None
    else:
        st.session_state.criteria = load_default_criteria()

    criteria = st.session_state.criteria
    if criteria:
        st.success(
            f"**{criteria.get('assignment_name', 'Essay')}** "
            f"({len(criteria['criteria'])} criteria, max {criteria['max_marks']} marks)"
        )
        with st.expander("Show rubric"):
            for c in criteria["criteria"]:
                st.markdown(
                    f"**{c['name']}** — full: {c['full_marks']}, partial: {c['partial_marks']}"
                )
                st.caption(f"Full: {c['full_marks_description']}")
                st.caption(f"Partial: {c['partial_marks_description']}")
    else:
        st.warning("No rubric loaded.")


# ---------- main area ------------------------------------------------------

st.title("📝 Essay grader")
st.caption(
    "Upload a rubric, reference materials, and student essays. Click **Grade all**. "
    "Download a CSV of marks and reasoning."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Reference files")
    st.caption("Readings, lecture notes, case studies the students should be drawing on. Optional but recommended.")
    reference_files = st.file_uploader(
        "Upload reference files",
        type=SUPPORTED_EXT_LIST,
        accept_multiple_files=True,
        key="ref_uploader",
        label_visibility="collapsed",
    )
    if reference_files:
        st.caption(f"{len(reference_files)} reference file(s) loaded")

with col2:
    st.subheader("2. Student assignments")
    st.caption("One file per student. Filename becomes the student name in the CSV (e.g. `alice_smith.pdf`).")
    submission_files = st.file_uploader(
        "Upload student essays",
        type=SUPPORTED_EXT_LIST,
        accept_multiple_files=True,
        key="sub_uploader",
        label_visibility="collapsed",
    )
    if submission_files:
        st.caption(f"{len(submission_files)} student file(s) loaded")

st.divider()

# Validation for the grade button
missing = []
if not api_key:
    missing.append("OpenAI API key")
if not criteria:
    missing.append("rubric")
if not submission_files:
    missing.append("at least one student file")

c1, c2 = st.columns([3, 1])
with c1:
    grade_clicked = st.button(
        "Grade all assignments",
        type="primary",
        disabled=bool(missing),
        use_container_width=True,
    )
with c2:
    if st.button("Reset results", use_container_width=True):
        st.session_state.results = None
        st.session_state.last_csv = None
        st.rerun()

if missing:
    st.info("To enable grading, provide: " + ", ".join(missing) + ".")


# ---------- grading loop ---------------------------------------------------

if grade_clicked:
    # Build the reference text once
    ref_items = []
    if reference_files:
        with st.spinner("Reading reference files..."):
            for f in reference_files:
                try:
                    text = read_text_from_bytes(f.name, f.read())
                    ref_items.append((f.name, text))
                except Exception as e:
                    st.warning(f"Skipped reference `{f.name}`: {e}")
    reference_text = combine_reference_texts(ref_items, max_ref_chars)
    if reference_text:
        st.caption(f"Reference context: {len(reference_text):,} chars")

    client = OpenAI(api_key=api_key)

    progress = st.progress(0.0, text="Starting...")
    results: list[dict] = []
    errors = 0
    start = time.time()

    for i, f in enumerate(submission_files, 1):
        progress.progress((i - 1) / len(submission_files), text=f"Grading {f.name}...")
        row = empty_row(f.name, criteria)
        try:
            essay = read_text_from_bytes(f.name, f.read()).strip()
            if not essay:
                raise ValueError("essay text is empty after extraction")
            result = grade_essay(client, model, criteria, reference_text, essay)
            fill_row_from_result(row, result, criteria)
        except Exception as e:
            row["status"] = f"error: {e}"
            errors += 1
        results.append(row)

    progress.progress(1.0, text="Done")
    elapsed = time.time() - start

    st.session_state.results = results
    st.session_state.last_csv = results_to_csv_string(results, criteria)

    ok = len(results) - errors
    if errors == 0:
        st.success(f"Graded {ok} assignment(s) in {elapsed:.1f}s.")
    else:
        st.warning(f"Graded {ok} assignment(s) in {elapsed:.1f}s. {errors} failed (see status column).")


# ---------- results display ------------------------------------------------

if st.session_state.results and st.session_state.criteria:
    results = st.session_state.results
    criteria = st.session_state.criteria

    st.subheader("Results")

    # Summary table
    df = summary_dataframe(results, criteria)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Stats
    valid_totals = [r["total_marks"] for r in results
                    if r["status"] == "ok" and isinstance(r["total_marks"], (int, float))]
    if valid_totals:
        m1, m2, m3 = st.columns(3)
        m1.metric("Graded", f"{len(valid_totals)} / {len(results)}")
        m2.metric("Average", f"{sum(valid_totals) / len(valid_totals):.2f} / {criteria['max_marks']}")
        m3.metric("Range", f"{min(valid_totals)} – {max(valid_totals)}")

    # Download
    if st.session_state.last_csv:
        st.download_button(
            "⬇️ Download grades.csv",
            data=st.session_state.last_csv,
            file_name="grades.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # Detailed reasoning per student
    st.subheader("Reasoning per student")
    cols = column_keys(criteria)
    for r in results:
        header = f"{r['student_name']} — {r.get('total_marks', 'n/a')} / {criteria['max_marks']}"
        if r["status"] != "ok":
            header += "  ⚠️"
        with st.expander(header):
            if r["status"] != "ok":
                st.error(r["status"])
                continue
            for c, (marks_col, reasoning_col) in zip(criteria["criteria"], cols):
                st.markdown(f"**{c['name']}** — {r.get(marks_col, '')} / {c['full_marks']}")
                st.write(r.get(reasoning_col, ""))
            if r.get("overall_feedback"):
                st.markdown("**Overall feedback**")
                st.write(r["overall_feedback"])


# ---------- footer ---------------------------------------------------------

st.divider()
st.caption(
    "Tip: spot-check a few essays manually before trusting the grades. "
    "If the model's reasoning cites things that aren't in the essay, switch to a stronger model."
)
