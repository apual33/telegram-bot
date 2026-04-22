"""
Local sentence-embeddings via sentence-transformers (all-MiniLM-L6-v2).
The model (~90 MB) is downloaded on first use and cached by the library.
"""
from __future__ import annotations

import json
import math

_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(text: str) -> list[float]:
    vec = _get_model().encode(text, convert_to_numpy=True)
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k(query: str, entries: list[dict], k: int = 5) -> list[dict]:
    """
    Rank `entries` by cosine similarity to `query`.
    Each entry must have an 'embedding' field (JSON string or list).
    Returns the top-k entries (without the embedding field), sorted by score desc.
    """
    query_vec = encode(query)
    scored = []
    for entry in entries:
        emb = entry["embedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        score = cosine_similarity(query_vec, emb)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, entry in scored[:k]:
        e = {k: v for k, v in entry.items() if k != "embedding"}
        e["score"] = round(score, 4)
        results.append(e)
    return results
