"""Ties retrieval to a structured, cited LLM answer."""

import json
import logging

from openai import OpenAI, OpenAIError

from app.retrieval import HybridRetriever
from app.schemas import ChatAnswer, ChatResponse, Citation

logger = logging.getLogger("research_assistant.qa")

ANSWER_PROMPT = """You are a research assistant answering questions using ONLY the
provided passages from academic papers. Do not use outside knowledge. If the
passages don't contain the answer, say so honestly.

Question: {question}

Passages:
{context}

Respond ONLY with JSON matching this shape:
{{
  "answer": "...",
  "key_points": ["...", "..."],
  "citations": [{{"chunk_id": "...", "paper_id": "...", "title": "...", "snippet": "..."}}],
  "confidence": 0.0
}}

Every claim in "answer" must be traceable to a citation. "confidence" reflects
how well the passages actually support the answer (0.0-1.0).
"""


class QAEngine:
    def __init__(self, client: OpenAI, model: str, retriever: HybridRetriever) -> None:
        self.client = client
        self.model = model
        self.retriever = retriever

    def answer(
        self, question: str, top_k: int = 5, paper_id: str | None = None
    ) -> ChatResponse:
        chunks, queries_used = self.retriever.retrieve(
            question, top_k=top_k, paper_id=paper_id
        )

        if not chunks:
            return ChatResponse(
                answer="No relevant passages found. Try ingesting a paper first.",
                key_points=[],
                citations=[],
                confidence=0.0,
                queries_used=queries_used,
                retrieved_chunk_ids=[],
            )

        context = "\n\n".join(
            f"[{c['chunk_id']}] (paper: {c['title']}) {c['text'][:800]}" for c in chunks
        )
        prompt = ANSWER_PROMPT.format(question=question, context=context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = ChatAnswer.model_validate_json(content)
        except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
            logger.exception("QA generation failed")
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        return ChatResponse(
            **parsed.model_dump(),
            queries_used=queries_used,
            retrieved_chunk_ids=[c["chunk_id"] for c in chunks],
        )
