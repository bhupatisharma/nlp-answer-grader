"""Interpretable lexical similarity with TF-IDF and cosine similarity."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.preprocessing import STOPWORDS, similarity_text


def tfidf_similarity(reference: str, response: str) -> float:
    if not reference.strip() or not response.strip():
        return 0.0
    # TF-IDF downweights common terms; cosine compares the resulting sparse vectors.
    reference = similarity_text(reference)
    response = similarity_text(response)
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([reference, response])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])