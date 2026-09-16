"""Thin wrapper around a persistent Chroma collection of paper chunks."""

from typing import Any

import chromadb

from app.schemas import Chunk


class ChromaManager:
    def __init__(self, persist_dir: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="paper_chunks")

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"paper_id": c.paper_id, "title": c.title, "section": c.section or ""}
                for c in chunks
            ],
        )

    def dense_search(
        self, query_embedding: list[float], top_k: int = 10, paper_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = {"paper_id": paper_id} if paper_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )
        return self._format(results)

    def all_chunks(self) -> list[dict[str, Any]]:
        """Pulls every stored chunk — used to (re)build the BM25 index."""
        data = self.collection.get()
        out = []
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            out.append({"chunk_id": cid, "text": doc, **meta})
        return out

    def list_papers(self) -> list[dict[str, Any]]:
        data = self.collection.get()
        papers: dict[str, dict[str, Any]] = {}
        for meta in data["metadatas"]:
            pid = meta["paper_id"]
            if pid not in papers:
                papers[pid] = {"paper_id": pid, "title": meta["title"], "num_chunks": 0}
            papers[pid]["num_chunks"] += 1
        return list(papers.values())

    @staticmethod
    def _format(results: dict[str, Any]) -> list[dict[str, Any]]:
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {"chunk_id": cid, "text": doc, "distance": dist, **meta}
            for cid, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
