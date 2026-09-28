"""Local sentence-embedding similarity with a TF-IDF fallback."""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SEMANTIC_MODEL_NAME
from nlp.preprocessing import STOPWORDS, canonical_concepts, similarity_text


class SemanticSimilarity:
    def __init__(self) -> None:
        self._model = None
        self._attempted = False

    def _load_model(self):
        if self._attempted:
            return self._model
        self._attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(SEMANTIC_MODEL_NAME, device="cpu")
        except Exception:
            self._model = None
        return self._model

    @property
    def uses_embeddings(self) -> bool:
        return self._load_model() is not None

    def compare(self, reference: str, response: str) -> float:
        if not reference.strip() or not response.strip():
            return 0.0
        model = self._load_model()
        if model is not None:
            # Sentence embeddings map paraphrases near one another; cosine measures their angle.
            answers = model.encode([reference, response], normalize_embeddings=True)
            whole_answer = float(np.dot(answers[0], answers[1]))
            reference_sentences = _sentences(reference)
            response_sentences = _sentences(response)
            if len(reference_sentences) > 1 or len(response_sentences) > 1:
                vectors = model.encode(reference_sentences + response_sentences, normalize_embeddings=True)
                reference_vectors = vectors[:len(reference_sentences)]
                response_vectors = vectors[len(reference_sentences):]
                pair_scores = reference_vectors @ response_vectors.T
                # Best matching sentence in each direction rewards partial coverage without
                # requiring the student to preserve the model answer's sentence order.
                sentence_score = (pair_scores.max(axis=1).mean() + pair_scores.max(axis=0).mean()) / 2
                whole_answer = 0.65 * whole_answer + 0.35 * float(sentence_score)
            return _unit(whole_answer)
        return _fallback_similarity(reference, response)


def _fallback_similarity(reference: str, response: str) -> float:
    """Approximate semantic alignment with normalized concepts and TF-IDF offline."""
    # Canonical aliases provide modest synonym awareness without pretending to be embeddings.
    reference = similarity_text(reference)
    response = similarity_text(response)
    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS), analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
    )
    try:
        word_matrix = vectorizer.fit_transform([reference, response])
        word_score = float(cosine_similarity(word_matrix[0:1], word_matrix[1:2])[0, 0])
    except ValueError:
        word_score = 0.0
    character_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    try:
        char_matrix = character_vectorizer.fit_transform([reference, response])
        char_score = float(cosine_similarity(char_matrix[0:1], char_matrix[1:2])[0, 0])
    except ValueError:
        char_score = 0.0
    reference_concepts = canonical_concepts(reference)
    response_concepts = canonical_concepts(response)
    overlap = len(reference_concepts & response_concepts)
    precision = overlap / len(response_concepts) if response_concepts else 0.0
    recall = overlap / len(reference_concepts) if reference_concepts else 0.0
    concept_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return _unit(0.75 * concept_f1 + 0.20 * word_score + 0.05 * char_score)


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()] or [text]


def _unit(score: float) -> float:
    return max(0.0, min(1.0, float(score)))