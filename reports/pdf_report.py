"""Create compact, printable PDF grading reports with ReportLab."""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape


def build_pdf(workspace: dict) -> bytes:
    results = workspace.get("results", [])
    obtained = sum(item["marks_obtained"] for item in results)
    maximum = sum(item["maximum_marks"] for item in results)
    metadata = workspace.get("metadata", {})
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=8.5, leading=11, spaceAfter=4)
    story = [Paragraph("AI Answer Sheet Grading Report", styles["Title"]), Spacer(1, 8)]
    details = [
        ["Student", metadata.get("student_name") or "Not provided", "Subject", metadata.get("subject") or "Not provided"],
        ["Date", workspace.get("graded_at", ""), "Final score", f"{obtained:g} / {maximum:g}"],
        ["Percentage", f"{(obtained / maximum * 100 if maximum else 0):.1f}%", "Questions", str(len(results))],
    ]
    info = Table(details, colWidths=[0.8 * inch, 2.4 * inch, 0.9 * inch, 2.5 * inch])
    info.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f2")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d2cd")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 6)]))
    story.extend([info, Spacer(1, 12)])
    for item in results:
        story.append(Paragraph(f"Question {item['number']} | {item['marks_obtained']:g} / {item['maximum_marks']:g}", styles["Heading2"]))
        story.append(Paragraph(f"<b>Question:</b> {escape(item['question'])}", body))
        story.append(Paragraph(f"<b>Model answer:</b> {escape(item['model_answer'])}", body))
        story.append(Paragraph(f"<b>Student answer:</b> {escape(item['student_answer']) or 'No answer detected'}", body))
        story.append(Paragraph(
            f"<b>NLP:</b> Semantic {item['semantic_similarity']:.0%} | TF-IDF {item['tfidf_similarity']:.0%} | "
            f"Coverage {item['concept_coverage']:.0%} | Completeness {item['completeness']:.0%}", body,
        ))
        story.append(Paragraph(f"<b>Feedback:</b> {escape(item['feedback'])}", body))
        if item.get("missing_concepts"):
            story.append(Paragraph(f"<b>Missing concepts:</b> {escape(', '.join(item['missing_concepts']))}", body))
        story.append(Spacer(1, 7))
    story.append(Paragraph(
        "Automated grades are generated using NLP similarity and concept coverage. "
        "The teacher should review borderline or ambiguous answers before finalizing marks.", body,
    ))
    document.build(story)
    return output.getvalue()