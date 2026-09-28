"""Heuristic parser for common numbered academic answer-sheet layouts."""

import re


QUESTION_LINE = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*)?(\d{1,3})\s*(?:[.):\-]|\s+-)\s*(.*?)\s*$",
    re.IGNORECASE,
)
ANSWER_LINE = re.compile(r"^\s*(?:model\s+)?answer\s*(?:\d+)?\s*[:.)\-]?\s*(.*)$", re.IGNORECASE)
MODEL_LINE = re.compile(r"^\s*model\s*[:\-]\s*(.*)$", re.IGNORECASE)


def parse_answer_sheet(text: str) -> list[dict[str, str | int]]:
    """Return numbered question/answer records without fabricating missing text."""
    records: list[dict[str, str | int]] = []
    current: dict[str, str | int] | None = None
    mode = "answer"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        question_match = QUESTION_LINE.match(line)
        if question_match:
            if current:
                records.append(current)
            question_number = int(question_match.group(1))
            question_text = question_match.group(2).strip()
            current = {"number": question_number, "question": question_text, "answer": ""}
            mode = "answer" if question_text else "question"
            continue

        if current is None:
            continue

        answer_match = ANSWER_LINE.match(line)
        model_match = MODEL_LINE.match(line)
        if answer_match:
            mode = "answer"
            content = answer_match.group(1).strip()
            if content:
                current["answer"] = _append(current["answer"], content)
            continue
        if model_match:
            mode = "answer"
            content = model_match.group(1).strip()
            if content:
                current["answer"] = _append(current["answer"], content)
            continue

        if mode == "question":
            current["question"] = _append(current["question"], line)
            mode = "answer"
        else:
            current["answer"] = _append(current["answer"], line)

    if current:
        records.append(current)
    return records


def _append(existing: str | int, addition: str) -> str:
    return f"{existing}\n{addition}".strip() if existing else addition