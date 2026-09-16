"""Ingestion: turn an uploaded PDF into normalized, overlapping text chunks
ready to embed and index.
"""

import logging
import re
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger("research_assistant.ingestion")


class PDFLoader:
    """Extracts and normalizes plain text from a PDF file, page by page
    so we can keep rough section/page context on each chunk.
    """

    @staticmethod
    def load_pages(path: str) -> list[str]:
        try:
            reader = PdfReader(path)
        except PdfReadError as exc:
            raise ValueError(f"Could not read PDF file: {exc}") from exc

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append(text)

        if not pages:
            raise ValueError(
                "No extractable text found in PDF (it may be a scanned image)."
            )
        return pages


class Chunker:
    """Sliding-window word chunker with overlap. Simple and dependency-free;
    swap for a semantic/sentence-aware splitter later if you want to push
    the retrieval-quality story further.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk_pages(self, pages: list[str], paper_id: str, title: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page_num, page_text in enumerate(pages, start=1):
            words = page_text.split()
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        paper_id=paper_id,
                        title=title,
                        section=f"page {page_num}",
                        text=chunk_text,
                    )
                )
                if end == len(words):
                    break
                start = end - self.overlap
        return chunks


def guess_title(pages: list[str], fallback: str) -> str:
    """Very light heuristic: first non-trivial line of the first page is
    usually the title for arXiv-style papers."""
    if not pages:
        return fallback
    first_line = pages[0].split(".")[0].strip()
    if 5 < len(first_line) < 200:
        return first_line
    return fallback
