"""Flask application for local answer-sheet grading demonstrations."""

from __future__ import annotations

import csv
import io
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, BASE_DIR, MAX_UPLOAD_BYTES, UPLOAD_DIR
from nlp.grading_engine import GradingEngine
from parsers.answer_parser import parse_answer_sheet
from parsers.document_parser import DocumentParseError, extract_text
from reports.pdf_report import build_pdf


app = Flask(__name__)
app.secret_key = os.environ.get("GRADER_SECRET_KEY") or os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

for folder in (UPLOAD_DIR / "model", UPLOAD_DIR / "students", BASE_DIR / "output"):
    folder.mkdir(parents=True, exist_ok=True)

# Session data stays server-side: answer text never has to fit in a browser cookie.
WORKSPACES: dict[str, dict] = {}


def _workspace() -> dict:
    workspace_id = session.setdefault("workspace_id", uuid.uuid4().hex)
    return WORKSPACES.setdefault(workspace_id, {"metadata": {}, "model": [], "student": [], "results": []})


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(upload, category: str) -> Path:
    if not upload or not upload.filename:
        raise ValueError("Choose a file to upload.")
    if not _allowed(upload.filename):
        raise ValueError("Upload a PDF, DOCX, or TXT file.")
    safe_name = secure_filename(upload.filename)
    target = UPLOAD_DIR / category / f"{uuid.uuid4().hex[:10]}_{safe_name}"
    upload.save(target)
    return target


def _review_records(form, prefix: str, include_marks: bool) -> list[dict]:
    records = []
    for index, question in enumerate(form.getlist(f"{prefix}_question")):
        answer = form.getlist(f"{prefix}_answer")[index] if index < len(form.getlist(f"{prefix}_answer")) else ""
        number_values = form.getlist(f"{prefix}_number")
        try:
            number = int(number_values[index]) if index < len(number_values) else index + 1
        except ValueError:
            number = index + 1
        record = {"number": number, "question": question.strip(), "answer": answer.strip()}
        if include_marks:
            mark_values = form.getlist(f"{prefix}_marks")
            try:
                record["maximum_marks"] = max(0.0, float(mark_values[index])) if index < len(mark_values) else 5.0
            except ValueError:
                record["maximum_marks"] = 5.0
        records.append(record)
    return records


@app.get("/")
def index():
    workspace = _workspace()
    results = workspace.get("results", [])
    return render_template("index.html", workspace=workspace, results=results)


@app.post("/metadata")
def save_metadata():
    _workspace()["metadata"] = {
        "student_name": request.form.get("student_name", "").strip(),
        "subject": request.form.get("subject", "").strip(),
    }
    flash("Student and subject details saved.", "success")
    return redirect(url_for("index"))


@app.route("/upload-model", methods=["GET", "POST"])
def upload_model():
    if request.method == "POST":
        try:
            path = _save_upload(request.files.get("document"), "model")
            records = parse_answer_sheet(extract_text(path))
            if not records:
                raise ValueError("No numbered questions were detected. Use Q1., Question 1:, or 1. headings, then add records manually after parsing.")
            workspace = _workspace()
            metadata = workspace.setdefault("metadata", {})
            for field in ("student_name", "subject"):
                if field in request.form:
                    metadata[field] = request.form.get(field, "").strip()
            for record in records:
                record["maximum_marks"] = 5.0
            workspace["model"] = records
            workspace["model_filename"] = request.files["document"].filename
            flash(f"Detected {len(records)} question(s). Review the extracted text before confirming.", "success")
            return redirect(url_for("model_review"))
        except (ValueError, DocumentParseError) as error:
            flash(str(error), "error")
    return render_template("upload_model.html", workspace=_workspace())


@app.route("/model-review", methods=["GET", "POST"])
def model_review():
    workspace = _workspace()
    if request.method == "POST":
        records = _review_records(request.form, "model", True)
        records = [record for record in records if record["question"] or record["answer"]]
        if not records:
            flash("Add at least one question and model answer.", "error")
            return redirect(url_for("model_review"))
        if any(not record["answer"] for record in records):
            flash("At least one model answer is blank. Add or verify its text before continuing.", "error")
            return redirect(url_for("model_review"))
        workspace["model"] = records
        workspace["results"] = []
        flash("Model answers confirmed. Upload the student answer sheet next.", "success")
        return redirect(url_for("upload_student"))
    return render_template("model_review.html", records=workspace.get("model", []))


@app.route("/upload-student", methods=["GET", "POST"])
def upload_student():
    workspace = _workspace()
    if not workspace.get("model"):
        flash("Upload and confirm model answers first.", "error")
        return redirect(url_for("upload_model"))
    if request.method == "POST":
        try:
            path = _save_upload(request.files.get("document"), "students")
            records = parse_answer_sheet(extract_text(path))
            if not records:
                raise ValueError("No numbered question headings were detected. Check the document and use the manual entry option below.")
            workspace["student"] = records
            workspace["student_filename"] = request.files["document"].filename
            flash(f"Detected {len(records)} student response(s). Review and correct the text before grading.", "success")
            return redirect(url_for("student_review"))
        except (ValueError, DocumentParseError) as error:
            flash(str(error), "error")
    return render_template("upload_student.html")


@app.route("/student-review", methods=["GET", "POST"])
def student_review():
    workspace = _workspace()
    if request.method == "POST":
        workspace["student"] = _review_records(request.form, "student", False)
        if not workspace["student"]:
            flash("Add at least one student response record.", "error")
            return redirect(url_for("student_review"))
        return redirect(url_for("grade_answers"))
    return render_template("student_review.html", records=workspace.get("student", []))


@app.get("/grade")
def grade_answers():
    workspace = _workspace()
    model_records = workspace.get("model", [])
    if not model_records:
        flash("Confirm model answers before grading.", "error")
        return redirect(url_for("upload_model"))
    student_by_number = {int(record["number"]): record for record in workspace.get("student", [])}
    results = []
    used_numbers = set()
    engine = GradingEngine()
    for index, model in enumerate(model_records, start=1):
        number = int(model.get("number", index))
        student = student_by_number.get(number)
        if student:
            used_numbers.add(number)
        student_answer = student.get("answer", "") if student else ""
        report = engine.grade(
            model.get("question", ""), model.get("answer", ""), student_answer,
            model.get("maximum_marks", 5),
        )
        results.append({
            "number": number,
            "question": model.get("question", ""),
            "model_answer": model.get("answer", ""),
            "student_answer": student_answer,
            "matched": student is not None,
            **report,
        })
    extras = [record for record in workspace.get("student", []) if int(record["number"]) not in used_numbers]
    workspace["results"] = results
    workspace["unmatched_student"] = extras
    workspace["graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return redirect(url_for("results"))


@app.route("/results", methods=["GET", "POST"])
def results():
    workspace = _workspace()
    if request.method == "POST":
        for result in workspace.get("results", []):
            try:
                mark = float(request.form.get(f"marks_{result['number']}", result["marks_obtained"]))
            except ValueError:
                mark = result["marks_obtained"]
            result["marks_obtained"] = round(max(0.0, min(result["maximum_marks"], mark)) * 2) / 2
        workspace["finalized"] = True
        flash("Teacher marks saved and finalized.", "success")
        return redirect(url_for("results"))
    if not workspace.get("results"):
        flash("Grade a student answer sheet to view results.", "error")
        return redirect(url_for("index"))
    obtained = sum(item["marks_obtained"] for item in workspace["results"])
    maximum = sum(item["maximum_marks"] for item in workspace["results"])
    return render_template(
        "results.html", workspace=workspace, results=workspace["results"], obtained=obtained,
        maximum=maximum, percentage=(100 * obtained / maximum if maximum else 0),
    )


@app.get("/report.csv")
def download_csv():
    workspace = _workspace()
    if not workspace.get("results"):
        flash("There is no report to download yet.", "error")
        return redirect(url_for("index"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[
        "question_number", "question", "maximum_marks", "marks_obtained",
        "semantic_similarity", "tfidf_similarity", "concept_coverage", "completeness", "feedback",
    ])
    writer.writeheader()
    for item in workspace["results"]:
        writer.writerow({
            "question_number": item["number"], "question": item["question"],
            "maximum_marks": item["maximum_marks"], "marks_obtained": item["marks_obtained"],
            "semantic_similarity": round(item["semantic_similarity"], 4),
            "tfidf_similarity": round(item["tfidf_similarity"], 4),
            "concept_coverage": round(item["concept_coverage"], 4),
            "completeness": round(item["completeness"], 4), "feedback": item["feedback"],
        })
    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=answer-grading-report.csv"},
    )


@app.get("/report.pdf")
def download_pdf():
    workspace = _workspace()
    if not workspace.get("results"):
        flash("There is no report to download yet.", "error")
        return redirect(url_for("index"))
    payload = build_pdf(workspace)
    return send_file(io.BytesIO(payload), mimetype="application/pdf", as_attachment=True, download_name="answer-grading-report.pdf")


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The upload exceeds the 12 MB size limit.", "error")
    return redirect(request.referrer or url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)