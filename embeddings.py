"""
Voyage embeddings via the Anthropic API (voyage-3 model).
"""
from __future__ import annotations

import json
import math

from anthropic import AsyncAnthropic


async def encode(text: str, client: AsyncAnthropic) -> list[float]:
    response = await client.beta.embeddings.create(
        model="voyage-3",
        input=[text],
        input_type="document",
    )
    return response.embeddings[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def top_k(query: str, entries: list[dict], client: AsyncAnthropic, k: int = 5) -> list[dict]:
    """
    Rank `entries` by cosine similarity to `query`.
    Each entry must have an 'embedding' field (JSON string or list).
    Returns the top-k entries (without the embedding field), sorted by score desc.
    """
    query_vec = await encode(query, client)
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
        e = {key: val for key, val in entry.items() if key != "embedding"}
        e["score"] = round(score, 4)
        results.append(e)
    return results
