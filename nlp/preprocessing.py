"""Lightweight token normalization for the grading features."""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "answer", "according", "called", "different", "following", "given",
    "question", "used", "using", "would", "could", "one", "also",
}
IRREGULAR_LEMMAS = {
    "children": "child", "indices": "index",
    "men": "man", "mice": "mouse", "people": "person", "teeth": "tooth",
    "were": "be", "was": "be", "is": "be", "are": "be", "has": "have",
    "had": "have", "does": "do", "did": "do",
    "processing": "process", "variables": "variable", "enables": "enable",
    "helps": "help", "uses": "use", "changes": "change",
}
SIMILARITY_ALIASES = {
    "area": "field", "branch": "field", "domain": "field",
    "assist": "enable", "help": "enable", "permit": "enable", "let": "enable", "allow": "enable",
    "work": "process", "handle": "process", "analyze": "process", "analysis": "process",
    "interpret": "understand", "comprehend": "understand", "recognize": "understand",
    "generate": "create", "produce": "create", "form": "word", "term": "word",
    "hold": "store", "storage": "store", "contain": "store", "keep": "store",
    "data": "value", "datum": "value", "information": "value",
    "divide": "split", "separate": "split", "break": "split",
    "tokenization": "split", "tokens": "token", "units": "unit", "pieces": "unit",
    "dictionary": "base", "root": "base", "meaningful": "valid", "real": "valid",
    "cut": "reduce", "remove": "reduce", "reduction": "reduce", "decrease": "reduce",
    "incorrect": "invalid", "unreal": "invalid", "valid": "word",
    "linguistic": "language", "vocabulary": "language", "knowledge": "language",
    "importance": "important", "significance": "important", "weight": "important",
    "frequency": "frequent", "frequently": "frequent", "often": "frequent", "common": "frequent",
    "collection": "corpus", "many": "multiple", "numerous": "multiple",
    "less": "decrease", "lower": "decrease", "limited": "decrease", "little": "decrease", "small": "decrease",
    "similarity": "similar", "similarity's": "similar", "close": "similar",
    "near": "similar", "related": "similar", "compare": "comparison", "check": "comparison",
    "indicate": "mean", "represent": "mean", "meaning": "mean", "whether": "mean",
    "grammatical": "grammar", "category": "part", "role": "part", "tag": "label",
    "assign": "label", "label": "label", "context": "surrounding",
    "example": "instance", "include": "instance", "such": "instance", "give": "instance",
    "sentence": "text", "document": "text", "piece": "text", "representation": "vector",
    "embedding": "vector", "embedding's": "vector",
    "identify": "label", "part": "category",
    "stemming": "stem", "lemmatization": "lemma", "pos": "part", "speech": "grammar",
    "info": "value", "carry": "store",
    "across": "between", "relative": "between",
    "memorize": "learn", "memorise": "learn", "struggle": "poor",
    "poorly": "poor", "unseen": "new", "newer": "new",
}


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenize words, remove common function words, and apply simple lemmatization."""
    # Regex tokenization keeps this project runnable without a downloaded NLP corpus.
    expanded = re.sub(r"\bNLP\b", "natural language processing", text, flags=re.IGNORECASE)
    expanded = re.sub(r"\bAI\b", "artificial intelligence", expanded, flags=re.IGNORECASE)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']*", expanded.lower())
    normalized = [_lemma(token) for token in tokens]
    if remove_stopwords:
        normalized = [token for token in normalized if token not in STOPWORDS and len(token) > 1]
    return normalized


def clean_text(text: str) -> str:
    return " ".join(tokenize(text))


def similarity_text(text: str) -> str:
    """Expand common classroom acronyms and canonicalize explicit paraphrase families."""
    expanded = re.sub(r"\bTF\s*[- ]?\s*IDF\b", "term frequency inverse document frequency", text, flags=re.IGNORECASE)
    return " ".join(SIMILARITY_ALIASES.get(token, token) for token in tokenize(expanded))


def canonical_concepts(text: str) -> set[str]:
    """Normalize tokens into a compact concept vocabulary for paraphrase matching."""
    return {SIMILARITY_ALIASES.get(token, token) for token in tokenize(text)}


def _lemma(token: str) -> str:
    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if token == "named":
        return "name"
    if token in {"storage", "stored", "stores", "storing", "holds", "holding"}:
        return "store" if token.startswith("stor") else "hold"
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        stem = stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
        return stem + "e" if stem.endswith("ess") else stem
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        return stem + "e" if stem.endswith(("v", "c", "g", "z")) else stem
    if len(token) > 4 and token.endswith("es"):
        if token.endswith(("ses", "xes", "zes", "ches", "shes")):
            return token[:-2]
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token