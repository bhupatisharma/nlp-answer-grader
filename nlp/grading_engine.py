"""Transparent weighted grading and deterministic feedback generation."""

from config import FEATURE_WEIGHTS, SEMANTIC_FALLBACK_POWER
from nlp.completeness import completeness
from nlp.concept_extraction import concept_coverage, extract_concepts
from nlp.semantic_similarity import SemanticSimilarity
from nlp.tfidf_similarity import tfidf_similarity


class GradingEngine:
    def __init__(self, semantic_engine: SemanticSimilarity | None = None) -> None:
        self.semantic = semantic_engine or SemanticSimilarity()

    def grade(self, question: str, model_answer: str, student_answer: str, maximum_marks: float) -> dict:
        maximum = max(0.0, float(maximum_marks))
        if not student_answer.strip():
            semantic_score = tfidf_score = coverage = complete = overall = 0.0
            concepts = extract_concepts(model_answer, question)
            covered, missing = [], concepts
        else:
            semantic_score = self.semantic.compare(model_answer, student_answer)
            if not self.semantic.uses_embeddings and semantic_score > 0:
                semantic_score = semantic_score ** SEMANTIC_FALLBACK_POWER
            tfidf_score = tfidf_similarity(model_answer, student_answer)
            concepts = extract_concepts(model_answer, question)
            coverage, covered, missing = concept_coverage(concepts, student_answer, self.semantic)
            complete = completeness(model_answer, student_answer)
            weight_total = sum(FEATURE_WEIGHTS.values()) or 1.0
            overall = sum(
                FEATURE_WEIGHTS[name] * value
                for name, value in {
                    "semantic_similarity": semantic_score,
                    "tfidf_similarity": tfidf_score,
                    "concept_coverage": coverage,
                    "completeness": complete,
                }.items()
            ) / weight_total
            # Question relevance acts as a conservative guard against a fluent but unrelated answer.
            if question.strip():
                relevance = self.semantic.compare(question, student_answer)
                if relevance < 0.10 and self.semantic.compare(model_answer, student_answer) < 0.25:
                    overall = min(overall, 0.25)
        overall = max(0.0, min(1.0, overall))
        marks = round(overall * maximum * 2) / 2
        return {
            "semantic_similarity": semantic_score,
            "tfidf_similarity": tfidf_score,
            "concept_coverage": coverage,
            "completeness": complete,
            "overall_score": overall,
            "maximum_marks": maximum,
            "marks_obtained": min(maximum, marks),
            "concepts": concepts,
            "covered_concepts": covered,
            "missing_concepts": missing,
            "feedback": feedback_for(semantic_score, coverage, complete, student_answer, missing),
        }


def feedback_for(semantic: float, coverage: float, complete: float, response: str, missing: list[str]) -> str:
    if not response.strip():
        return "No answer was detected for this question."
    if semantic >= 0.78 and coverage >= 0.70:
        message = "Strong answer. The response covers the main concepts and is semantically close to the model answer."
    elif semantic >= 0.48 or coverage >= 0.45:
        message = "The answer covers some expected concepts but misses or weakly explains important points."
    else:
        message = "The response has limited similarity to the expected answer and may not address the main concepts."
    if semantic >= 0.65 and complete < 0.75:
        message = "The core idea is present, but the answer could explain it more fully."
    if missing:
        message += " Missing or weak concepts: " + ", ".join(missing[:5]) + "."
    return message