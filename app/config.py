"""Centralized, validated configuration read once at startup."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.chat_model = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

        self.chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self.bm25_index_path = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")

        self.chunk_size = 800
        self.chunk_overlap = 150

        self.app_host = "0.0.0.0"
        self.app_port = 8000

    def require_llm_config(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "Missing required environment variable: OPENROUTER_API_KEY. "
                "Set it in your environment or a .env file."
            )


settings = Settings()
