import io
from pathlib import Path

import pytest
from pypdf import PdfReader

from app import WORKSPACES, app
from nlp.grading_engine import GradingEngine
from nlp.preprocessing import tokenize
from parsers.answer_parser import parse_answer_sheet
from parsers.document_parser import extract_text


@pytest.fixture
def engine():
    return GradingEngine()


def test_identical_answers_score_well(engine):
    answer = "A variable is a named memory location used to store a value."
    result = engine.grade("What is a variable?", answer, answer, 5)
    assert result["marks_obtained"] >= 4
    assert result["semantic_similarity"] > 0.95
    assert result["concept_coverage"] == 1


def test_paraphrase_receives_credit_without_embedding_download(engine):
    result = engine.grade(
        "What is a variable?",
        "A variable is a named memory location used to store a value.",
        "A variable is a named storage location in memory that holds data.",
        5,
    )
    assert result["marks_obtained"] >= 2.5
    assert result["concept_coverage"] >= 0.75


def test_acronym_paraphrase_in_sample_sheet_scores_as_relevant(engine):
    result = engine.grade(
        "What is Natural Language Processing?",
        "Natural Language Processing is a field of artificial intelligence that enables computers to process, understand, and generate human language.",
        "NLP is an AI area that helps computers work with and understand human languages.",
        5,
    )
    assert result["marks_obtained"] >= 2.5
    assert result["semantic_similarity"] >= 0.45


def test_unrelated_answer_scores_low(engine):
    result = engine.grade(
        "What is polymorphism?",
        "Polymorphism lets different classes share a common interface.",
        "Python was created by Guido van Rossum.",
        5,
    )
    assert result["marks_obtained"] <= 1
    assert result["overall_score"] < 0.25


def test_empty_answer_is_zero(engine):
    result = engine.grade("Define NLP", "NLP processes human language", "", 5)
    assert result["marks_obtained"] == 0
    assert result["feedback"] == "No answer was detected for this question."


def test_short_but_correct_answer_can_score_well(engine):
    result = engine.grade(
        "What is tokenization?",
        "Tokenization splits text into smaller units called tokens, such as words or sentences.",
        "Tokenization splits text into tokens.",
        5,
    )
    assert result["marks_obtained"] >= 2.5
    assert result["completeness"] > 0.55


def test_concept_overlap_uses_lemmatized_tokens():
    assert tokenize("Computers are processing variables") == ["computer", "process", "variable"]


def test_answer_parser_supports_multiple_heading_styles():
    parsed = parse_answer_sheet(
        "Question 1: What is NLP?\nAnswer 1: NLP processes language.\n\n"
        "2. Define tokenization.\nTokenization splits text into tokens."
    )
    assert len(parsed) == 2
    assert parsed[0] == {"number": 1, "question": "What is NLP?", "answer": "NLP processes language."}
    assert "Tokenization splits" in parsed[1]["answer"]


def test_score_is_bounded_and_rounded_to_half_marks(engine):
    result = engine.grade("Question", "Correct reference answer", "Correct reference answer", 3)
    assert 0 <= result["marks_obtained"] <= 3
    assert result["marks_obtained"] * 2 == int(result["marks_obtained"] * 2)


def test_text_document_parser(tmp_path: Path):
    file_path = tmp_path / "sheet.txt"
    file_path.write_text("Q1. Define NLP.\nAnswer: Process human language.", encoding="utf-8")
    assert "Define NLP" in extract_text(file_path)


def test_full_upload_grade_override_and_export_workflow():
    app.config.update(TESTING=True, SECRET_KEY="test-key")
    client = app.test_client()
    response = client.post("/upload-model", data={
        "student_name": "Asha Kumar", "subject": "Introduction to NLP",
        "document": (io.BytesIO(b"Q1. What is tokenization?\nModel: Tokenization splits text into tokens."), "model.txt"),
    }, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Review model answers" in response.data
    response = client.post("/model-review", data={
        "model_number": "1", "model_question": "What is tokenization?",
        "model_answer": "Tokenization splits text into tokens.", "model_marks": "5",
    }, follow_redirects=True)
    assert b"Upload student answers" in response.data
    response = client.post("/upload-student", data={
        "document": (io.BytesIO(b"Q1. What is tokenization?\nAnswer: It divides text into tokens."), "student.txt"),
    }, content_type="multipart/form-data", follow_redirects=True)
    assert b"Review student responses" in response.data
    response = client.post("/student-review", data={
        "student_number": "1", "student_question": "What is tokenization?",
        "student_answer": "It divides text into tokens.",
    }, follow_redirects=True)
    assert b"Grading analysis" in response.data
    response = client.post("/results", data={"marks_1": "4.5"}, follow_redirects=True)
    assert b"4.5" in response.data
    csv_response = client.get("/report.csv")
    assert csv_response.status_code == 200
    assert b"question_number,question,maximum_marks,marks_obtained" in csv_response.data
    pdf_response = client.get("/report.pdf")
    assert pdf_response.status_code == 200
    assert pdf_response.data.startswith(b"%PDF")
    pdf_text = " ".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_response.data)).pages)
    assert "Asha Kumar" in pdf_text
    assert "Introduction to NLP" in pdf_text
    workspace_id = client.get_cookie("session")
    assert workspace_id is not None
    WORKSPACES.clear()