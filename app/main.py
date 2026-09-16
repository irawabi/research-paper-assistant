"""FastAPI app: ingestion + hybrid RAG chat over a paper collection."""

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI

from app.config import settings
from app.bm25_index import BM25Index
from app.ingestion import Chunker, PDFLoader, guess_title
from app.qa_engine import QAEngine
from app.retrieval import EmbeddingModel, HybridRetriever, LLMReranker, MultiQueryExpander
from app.schemas import ChatRequest, ChatResponse, IngestResponse, PaperSummary
from app.vectorstore import ChromaManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("research_assistant")


class Services:
    def __init__(self) -> None:
        settings.require_llm_config()
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)

        self.chroma = ChromaManager(settings.chroma_db_dir)
        self.bm25 = BM25Index()
        if not self.bm25.load(settings.bm25_index_path):
            self.bm25.build(self.chroma.all_chunks())

        self.embedding_model = EmbeddingModel(client, settings.embed_model)
        self.chunker = Chunker()

        reranker = LLMReranker(client, settings.chat_model)
        multi_query = MultiQueryExpander(client, settings.chat_model)
        retriever = HybridRetriever(self.chroma, self.bm25, self.embedding_model, reranker, multi_query)
        self.qa_engine = QAEngine(client, settings.chat_model, retriever)

    def reindex_bm25(self) -> None:
        self.bm25.build(self.chroma.all_chunks())
        self.bm25.save(settings.bm25_index_path)


@lru_cache
def get_services() -> Services:
    return Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_services()  # fail fast on misconfiguration, warm up at startup
    yield


router = APIRouter()


@router.post("/papers/ingest/upload", response_model=IngestResponse)
async def ingest_pdf(
    title: str | None = Form(None),
    file: UploadFile = File(...),
) -> IngestResponse:
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=415, detail="file must be a PDF.")

    services = get_services()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            pages = PDFLoader.load_pages(tmp.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    paper_id = str(uuid.uuid4())
    resolved_title = title or guess_title(pages, fallback=file.filename or paper_id)
    return _ingest_pages(services, pages, paper_id, resolved_title)


def _ingest_pages(services: Services, pages: list[str], paper_id: str, title: str) -> IngestResponse:
    chunks = services.chunker.chunk_pages(pages, paper_id=paper_id, title=title)
    embeddings = [services.embedding_model.encode(c.text) for c in chunks]
    services.chroma.add_chunks(chunks, embeddings)
    services.reindex_bm25()
    return IngestResponse(paper_id=paper_id, title=title, num_chunks=len(chunks))


@router.get("/papers", response_model=list[PaperSummary])
def list_papers() -> list[PaperSummary]:
    return [PaperSummary(**p) for p in get_services().chroma.list_papers()]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    services = get_services()
    try:
        return services.qa_engine.answer(
            request.question, top_k=request.top_k, paper_id=request.paper_id
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


app = FastAPI(
    title="Research Paper Assistant",
    description=(
        "Hybrid (dense + BM25) RAG over ingested research papers, with "
        "LLM reranking, multi-query expansion, and structured, cited answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Research Paper Assistant API running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
