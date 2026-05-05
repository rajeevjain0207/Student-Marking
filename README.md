# Essay grader

A small Python tool that grades student essays against a rubric using OpenAI and writes the results — per-criterion marks plus reasoning — to a CSV. Comes with a browser-based UI (Streamlit) and a command-line mode.

## What it does

Upload reference materials (the readings the students were assigned), upload student essays, hit grade. The system:

1. Extracts the text from each file (PDF, DOCX, TXT, MD).
2. Sends each essay to OpenAI along with the rubric and reference materials.
3. Asks the model to award full / partial / zero marks per criterion with a 1–3 sentence justification citing evidence from the essay.
4. Shows results inline and lets you download a CSV.

## Quick start (UI)

```bash
# 1. Set up
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the UI
streamlit run app.py
```

That opens `http://localhost:8501` in your browser. Then:

1. **Sidebar** — paste your OpenAI API key, pick a model, confirm the rubric.
2. **Section 1 (Reference files)** — upload the readings/lectures the assignment is based on (drag-and-drop multiple files at once).
3. **Section 2 (Student assignments)** — upload all student essays. Filename becomes the student name in the CSV (e.g. `alice_smith.pdf` → "Alice Smith").
4. Click **Grade all assignments**. A progress bar shows live status; results appear in a table with per-student expandable reasoning.
5. Click **Download grades.csv**.

The API key is held only in your browser session — nothing is persisted to disk by the app. (If you put it in a `.env` file, it pre-fills the field on launch.)

## Quick start (CLI)

The same logic also runs from the command line if you'd rather automate it:

```bash
cp .env.example .env
# edit .env and add your real key

# drop files into ./submissions and ./reference, then:
python grader.py --submissions ./submissions --reference ./reference
```

CSV lands in `output/grades.csv`. Full options:

```bash
python grader.py \
    --submissions ./submissions \
    --reference ./reference \
    --criteria ./criteria.json \
    --output ./output/grades.csv \
    --model gpt-4o-mini
```

## Reusing for other assignments

The rubric lives in `criteria.json`, separate from the code. To grade a different assignment, edit that file (or upload a new one through the UI's sidebar). Each criterion has:

```json
{
  "name": "Full descriptive name (used in the prompt to the model)",
  "short_name": "used_for_csv_columns",
  "full_marks": 0.5,
  "partial_marks": 0.25,
  "full_marks_description": "What earns full marks, specifically",
  "partial_marks_description": "What earns partial marks"
}
```

The script handles any number of criteria — just add more entries to the `criteria` array and update `max_marks`.

## Output

CSV with one row per student. Columns:

- `student_file`, `student_name`
- For each criterion: `<short_name>_marks`, `<short_name>_reasoning`
- `total_marks`, `max_marks`
- `overall_feedback` — 2–3 sentence summary
- `status` — `ok` or an error message if a file failed to parse

For Essay 6 the per-criterion columns are `uses_sources_*`, `personal_experience_*`, `critical_insight_*`, `addresses_prompt_*`.

## Project layout

```
essay_grader/
├── app.py                # Streamlit UI
├── grader.py             # Library + CLI entry point
├── criteria.json         # Default rubric (Essay 6)
├── requirements.txt
├── .env.example          # Optional: pre-fill API key on launch
├── .gitignore
├── reference/            # CLI input folder
├── submissions/          # CLI input folder
└── output/               # CLI output folder
```

## Cost

With `gpt-4o-mini`, each essay costs roughly a fraction of a cent — a class of 50 essays is well under $1. Switch to `gpt-4o` for ~10× the cost and noticeably tougher grading. The "Max reference chars" slider in the sidebar caps how much reference material is sent per call, controlling the input token bill.

## Sanity checks before you trust the grades

LLM grading is not infallible:

1. Spot-check 3–5 essays manually against the model's reasoning. If the reasoning cites evidence that isn't actually in the essay, switch to a stronger model.
2. Run the same batch twice. With `temperature=0.2` totals should be very stable; large swings mean the rubric is ambiguous and the full/partial descriptions need sharpening.
3. Treat the output as a first-pass — a recommended grade plus auditable reasoning that a human confirms or overrides, not a final verdict.

## Privacy

Student essays are sent to OpenAI's API. If your institution has data-handling rules about student work, check them. For sensitive deployments, consider OpenAI's zero-data-retention configuration, or swap the client in `grader.grade_essay` for a local model — the rest of the pipeline doesn't depend on OpenAI.
