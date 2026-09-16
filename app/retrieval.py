"""Retrieval layer: dense + sparse hybrid search, reciprocal rank fusion,
LLM-based reranking, and multi-query expansion.

This is the part of the project that actually demonstrates retrieval
engineering rather than 'call an LLM and hope'.
"""

import json
import logging
from typing import Any

from openai import OpenAI, OpenAIError

from app.bm25_index import BM25Index
from app.schemas import RerankResult
from app.vectorstore import ChromaManager

logger = logging.getLogger("research_assistant.retrieval")


class EmbeddingModel:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def encode(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
        except OpenAIError as exc:
            logger.exception("Embedding call failed")
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
        return response.data[0].embedding


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    """Merges multiple ranked lists (dense + sparse, or multiple queries)
    into one ranking using RRF, which is simple, has no score-scale issues
    between BM25 and cosine distance, and works well in practice.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            cid = item["chunk_id"]
            chunk_lookup[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_lookup[cid], "fusion_score": score} for cid, score in fused]


class MultiQueryExpander:
    """Asks the LLM to reformulate the user's question into a few different
    retrieval queries (keyword-y, concept-y, comparison-y), then we search
    with all of them and fuse the results. Catches relevant chunks that a
    single phrasing of the question would miss.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def expand(self, question: str, n: int = 3) -> list[str]:
        prompt = (
            "Generate {n} different search queries to retrieve passages from "
            "research papers that would help answer this question. Vary the "
            "phrasing (keyword-focused, concept-focused, comparison-focused). "
            "Return ONLY a JSON array of strings, nothing else.\n\n"
            f"Question: {question}"
        ).format(n=n)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            content = response.choices[0].message.content or "[]"
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            queries = json.loads(content)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [question] + queries
        except (OpenAIError, json.JSONDecodeError, ValueError):
            logger.warning("Multi-query expansion failed, falling back to original query")

        return [question]


class LLMReranker:
    """Reranks a shortlist of candidate chunks by relevance to the query
    using the chat model, instead of a locally-downloaded cross-encoder —
    keeps the stack lightweight while still doing real reranking.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        listing = "\n".join(
            f"[{c['chunk_id']}] {c['text'][:400]}" for c in candidates
        )
        prompt = (
            "Score each passage's relevance to the query on a 0.0-1.0 scale.\n"
            f"Query: {query}\n\nPassages:\n{listing}\n\n"
            'Respond ONLY with JSON: {"ranked": [{"chunk_id": "...", '
            '"relevance_score": 0.0}, ...]}, sorted most to least relevant.'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content or "{}"
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = RerankResult.model_validate_json(content)
            score_map = {r.chunk_id: r.relevance_score for r in parsed.ranked}
            by_id = {c["chunk_id"]: c for c in candidates}
            ranked = [
                {**by_id[cid], "relevance_score": score}
                for cid, score in score_map.items()
                if cid in by_id
            ]
            ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
            if ranked:
                return ranked[:top_k]
        except Exception:
            logger.warning("LLM rerank failed, falling back to fusion order")

        return candidates[:top_k]


class HybridRetriever:
    def __init__(
        self,
        chroma: ChromaManager,
        bm25: BM25Index,
        embedding_model: EmbeddingModel,
        reranker: LLMReranker,
        multi_query: MultiQueryExpander,
    ) -> None:
        self.chroma = chroma
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.multi_query = multi_query

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        paper_id: str | None = None,
        use_multi_query: bool = True,
        candidate_pool: int = 15,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        queries = self.multi_query.expand(question) if use_multi_query else [question]

        per_query_fused: list[list[dict[str, Any]]] = []
        for q in queries:
            dense = self.chroma.dense_search(
                self.embedding_model.encode(q), top_k=candidate_pool, paper_id=paper_id
            )
            sparse = self.bm25.search(q, top_k=candidate_pool)
            if paper_id:
                sparse = [c for c in sparse if c.get("paper_id") == paper_id]
            per_query_fused.append(reciprocal_rank_fusion([dense, sparse]))

        fused = reciprocal_rank_fusion(per_query_fused)[:candidate_pool]
        reranked = self.reranker.rerank(question, fused, top_k=top_k)
        return reranked, queries
