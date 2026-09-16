"""In-memory BM25 sparse index, rebuilt from whatever is currently in Chroma.

Combining this with dense retrieval (see retrieval.py) is what makes the
search 'hybrid' — dense catches semantic similarity, BM25 catches exact
keyword/acronym matches (model names, dataset names, etc.) that embeddings
sometimes blur together.
"""

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_meta: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunk_meta = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.bm25 is None or not self.chunk_meta:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self.chunk_meta, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [{**meta, "bm25_score": float(score)} for meta, score in ranked]

    def save(self, path: str) -> None:
        Path(path).write_bytes(pickle.dumps({"chunk_meta": self.chunk_meta}))

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = pickle.loads(p.read_bytes())
        self.build(data["chunk_meta"])
        return True
