templates/           Flask pages
tests/               Grading, parsing, and workflow tests
# AI Answer Sheet Grader

A local Flask application that extracts numbered questions and answers from academic answer sheets, compares student responses with teacher-confirmed model answers, and generates an editable, question-wise grading report. It uses local NLP techniques and does not call a paid or external LLM API. Suggested grades are estimates; teachers review and finalize them.

## Features

- Upload model and student answer sheets as selectable-text PDF, DOCX, or TXT files (12 MB request limit).
- Review and edit extracted questions and answers before grading; add or delete entries and set marks per model question.
- Match student responses by question number and flag unmatched responses for review.
- View semantic similarity, TF-IDF similarity, concept coverage, completeness, covered/missing concepts, and deterministic feedback per question.
- Override suggested marks in half-mark increments and enter student/subject information.
- Download question-wise grading reports as CSV or PDF.
- Use the included seven-question demo sheets in `sample_data/`.

## Technology Stack

- Python 3.10+
- Flask for the local web application
- scikit-learn and NumPy for TF-IDF and similarity calculations
- pypdf and python-docx for document text extraction
- ReportLab for PDF report generation
- pytest for tests
- Optional: Sentence Transformers with `all-MiniLM-L6-v2` for local sentence embeddings. The base installation uses the built-in fallback if this package or its model is unavailable.

## Project Structure

```text
app.py                         Flask routes and workflow
config.py                      Upload and grading configuration
nlp/                           Text normalization, scoring features, grading engine
parsers/                       PDF/DOCX/TXT extraction and answer parsing
reports/                       PDF report generation
templates/                     Flask HTML pages
static/css/                    Application styles
static/js/                     Review-form interactions
sample_data/                   Demo model and student answer sheets
tests/                          Grading, parsing, and workflow tests
requirements.txt               Core dependencies
requirements-transformers.txt  Optional embedding dependency
```

Local uploads, generated output, and the separate evaluation-sheet directory are excluded from Git by `.gitignore`.

## Installation

Python 3.10 or newer is recommended. Run these commands from the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional sentence embeddings (the first run may download model weights):

```bash
python -m pip install -r requirements-transformers.txt
```

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The app creates `uploads/` and `output/` as needed. Its workspace is held in process memory and is cleared when the server stops. This project is intended for local demonstration, not production deployment.

Flask uses a random session-signing key for each local process start. Set the `GRADER_SECRET_KEY` environment variable to provide a stable key when needed; do not commit secret values.

## Basic Workflow

1. Optionally enter student and subject details on the dashboard.
2. Upload `sample_data/model_answers.txt`; verify the detected questions, model answers, and per-question maximum marks.
3. Confirm the model answers.
4. Upload `sample_data/student_answers.txt`; review and correct response matching.
5. Grade, review feature scores and feedback, and enter any teacher-approved mark overrides.
6. Download the CSV or PDF report.

To run the test suite:

```bash
python -m pytest -q
```

## Grading Overview

The configurable weights in `config.py` are 65% semantic similarity, 20% concept coverage, 10% TF-IDF similarity, and 5% completeness. When available, the embedding backend uses cosine similarity over local `all-MiniLM-L6-v2` sentence embeddings, including sentence-level comparisons. Otherwise, the application uses normalized concept overlap and local TF-IDF/character-ngram similarity. Scores are bounded, converted to marks, and rounded to the nearest half mark. Empty answers receive zero; the teacher can edit marks before finalizing.

## Limitations

- PDF processing extracts selectable text only; scanned PDFs are not OCR processed.
- The parser supports common numbered answer-sheet formats. Verify the extracted content, especially for unusual layouts.
- Without Sentence Transformers and its model weights, semantic matching uses a simpler local fallback and may be less accurate for unseen paraphrases.
- The built-in lemmatizer and synonym mappings are lightweight educational approximations, not a full linguistic analyzer.
- There is no authentication or persistent database. Session workspaces disappear when the server stops.
- Automated scoring does not replace a teacher's judgment, particularly for ambiguous answers or high-stakes grading.