"""Length-aware answer completeness that does not make length the grade."""

from nlp.preprocessing import tokenize


def completeness(reference: str, response: str) -> float:
    student_words = len(tokenize(response))
    reference_words = len(tokenize(reference))
    if student_words == 0:
        return 0.0
    if reference_words == 0:
        return 1.0
    length_ratio = min(1.0, student_words / reference_words)
    # A short response starts above zero; coverage and similarity assess correctness.
    return min(1.0, 0.55 + 0.45 * length_ratio)