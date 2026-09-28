"""Extract model-answer concepts and identify covered or missing concepts."""

from collections import Counter

from config import CONCEPT_SIMILARITY_THRESHOLD
from nlp.preprocessing import STOPWORDS, canonical_concepts, tokenize


SYNONYMS = {
    "hold": {"store", "contain", "keep"},
    "store": {"hold", "contain", "keep"},
    "value": {"data", "datum", "information"},
    "data": {"value", "information"},
    "computer": {"machine", "system"},
    "understand": {"interpret", "process", "comprehend", "recognize"},
    "process": {"handle", "analyze", "understand", "work", "operate"},
    "enable": {"allow", "let", "permit", "help"},
    "allow": {"enable", "permit", "let", "help"},
    "field": {"area", "branch", "domain"},
    "dictionary": {"root", "base", "canonical"},
    "base": {"root", "dictionary", "canonical"},
    "form": {"word", "root", "lemma"},
    "lemma": {"base", "root", "word"},
    "root": {"base", "dictionary", "form", "lemma"},
    "word": {"form", "term"},
    "human": {"people", "person"},
}


def extract_concepts(model_answer: str, question: str = "", limit: int = 12) -> list[str]:
    question_terms = set(tokenize(question))
    terms = [term for term in tokenize(model_answer) if term not in question_terms and term not in STOPWORDS]
    counts = Counter(terms)
    # Frequency is an inexpensive relevance proxy; preserve first-seen order for ties.
    order = {term: index for index, term in enumerate(terms)}
    return sorted(counts, key=lambda term: (-counts[term], order[term]))[:limit]


def concept_coverage(
    concepts: list[str], student_answer: str, semantic_engine=None,
) -> tuple[float, list[str], list[str]]:
    if not concepts:
        return (1.0 if student_answer.strip() else 0.0), [], []
    student_terms = canonical_concepts(student_answer)
    present: list[str] = []
    missing: list[str] = []
    for concept in concepts:
        normalized_concept = next(iter(canonical_concepts(concept)), concept)
        covered = normalized_concept in student_terms
        if not covered and any(
            next(iter(canonical_concepts(synonym)), synonym) in student_terms
            for synonym in SYNONYMS.get(concept, set())
        ):
            covered = True
        if not covered and semantic_engine is not None and student_answer.strip():
            response_segments = [segment.strip() for segment in student_answer.replace(";", ".").split(".") if segment.strip()]
            response_segments.extend(student_answer.split(","))
            covered = max(
                (semantic_engine.compare(concept, segment) for segment in response_segments),
                default=0.0,
            ) >= CONCEPT_SIMILARITY_THRESHOLD
        (present if covered else missing).append(concept)
    return len(present) / len(concepts), present, missing