from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "agentic-rag-system"
    environment: str = "dev"

    data_dir: Path = Path("data")
    raw_pdfs_dir: Path = Path("data/raw_pdfs")
    parsed_dir: Path = Path("data/parsed")
    chunks_dir: Path = Path("data/chunks")
    metadata_path: Path = Path("data/metadata.json")

    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection: str = "papers"

    groq_api_key: str | None = None
    # openai/gpt-oss-120b: 500 tok/s, 131k ctx, strict structured outputs
    groq_model: str = "openai/gpt-oss-120b"

    # When True: never use mocks — all LLM calls go to real Groq API.
    # Set to False (default) to allow tests without a live API key.
    use_real_llm: bool = False

    retrieval_top_k: int = 6
    retrieval_bm25_top_k: int = 8
    retrieval_vector_top_k: int = 8


settings = Settings()
