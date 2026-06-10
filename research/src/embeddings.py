"""Local embedding helper. We tried Gemini gemini-embedding-001 first, but
the free-tier daily request quota (1000/day) is too tight to embed a full
80k-record corpus + ~1000 themes within one run.

We use TF-IDF cosine similarity instead — fully local, no API, no quota,
and adequate for theme deduplication, literature-anchor matching, and
quote grounding when paired with BM25. The trade-off is that TF-IDF is
purely lexical, so paraphrases that don't share terms get lower scores
than they would under a transformer embedding. We document this in the
manuscript Limitations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A single shared vectorizer is fitted lazily on first call. Subsequent
# embed() calls extend the vocabulary if necessary.
_VECTORIZER: TfidfVectorizer | None = None


def _vectorizer() -> TfidfVectorizer:
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
            max_features=50_000,
        )
    return _VECTORIZER


def fit(corpus: list[str]) -> None:
    """Fit the TF-IDF vocabulary on a representative corpus. Call once."""
    _vectorizer().fit(corpus)


def embed(texts: list[str], task_type: str | None = None,
          batch: int | None = None) -> np.ndarray:
    """Return (N, D) L2-normalized float32 TF-IDF vectors.

    `task_type` and `batch` are accepted for API compatibility with the
    earlier Gemini-based embed(); they are ignored. If the vectorizer
    hasn't been fit, the input texts are used as the fitting corpus.
    """
    vec = _vectorizer()
    if not hasattr(vec, "vocabulary_"):
        vec.fit(texts)
    mat = vec.transform(texts).astype(np.float32).toarray()
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)
