"""Pydantic models shared across the API and the RAG pipeline.

Keeping LLM output typed and validated (rather than freeform text) is one
of the main things that separates a toy RAG demo from something that looks
production-minded.
"""

from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str | None = None
    text: str


class IngestResponse(BaseModel):
    paper_id: str
    title: str
    num_chunks: int


class PaperSummary(BaseModel):
    paper_id: str
    title: str
    num_chunks: int


class Citation(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    snippet: str = Field(..., description="Short excerpt supporting the claim")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    paper_id: str | None = Field(
        None, description="Optionally restrict retrieval to a single paper"
    )


class ChatAnswer(BaseModel):
    """The structured shape we force the LLM to answer in."""

    answer: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ChatResponse(ChatAnswer):
    queries_used: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class RerankedChunk(BaseModel):
    chunk_id: str
    relevance_score: float


class RerankResult(BaseModel):
    ranked: list[RerankedChunk]
