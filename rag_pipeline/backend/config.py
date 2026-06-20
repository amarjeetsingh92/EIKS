"""
config.py — All settings for the RAG pipeline
==============================================
Instead of hardcoding values like API keys or port numbers throughout
the codebase, we centralise everything here. Settings are loaded from
a .env file (copy .env.example to .env to get started).

HOW IT WORKS:
  1. You put your secrets in a .env file in the project root.
  2. This file reads them automatically at startup.
  3. Every other file imports `settings` from here.

WHY pydantic-settings?
  - It validates types automatically (e.g. API_PORT must be an integer).
  - It gives clear error messages if a required value is missing.
  - It works with environment variables, .env files, and defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    All configurable settings for the pipeline.
    Each field maps to an environment variable of the same name (uppercased).

    Example: llm_provider maps to the env variable LLM_PROVIDER.
    """

    # ── LLM (Language Model) ─────────────────────────────────────────────────
    # The LLM is the AI that reads retrieved chunks and writes the final answer.
    # We support three providers; you set which one to use here.

    # Which LLM service to use. Must be one of: anthropic, openai, ollama
    llm_provider: str = Field("anthropic", description="anthropic | openai | ollama")

    # API key for Anthropic (Claude). Get yours at console.anthropic.com
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")

    # API key for OpenAI (GPT). Get yours at platform.openai.com
    openai_api_key: str = Field("", env="OPENAI_API_KEY")

    # Base URL for a locally-running Ollama server (no API key needed).
    ollama_base_url: str = Field("http://localhost:11434")

    # Model name for each provider:
    ollama_model:    str = Field("llama3")
    openai_model:    str = Field("gpt-4o")
    anthropic_model: str = Field("claude-opus-4-6")

    # Controls how "creative" the LLM is. 0.0 = deterministic, 1.0 = creative.
    # For factual RAG, keep this low (0.1–0.3).
    llm_temperature: float = Field(0.2)

    # Maximum number of tokens (roughly words) the LLM can write in its answer.
    llm_max_tokens: int = Field(1024)

    # ── Embeddings ───────────────────────────────────────────────────────────
    # Embeddings turn text into vectors of numbers so we can measure similarity.
    # "all-MiniLM-L6-v2" is fast, small, and good — a great starting point.

    # The sentence-transformers model to use for creating embeddings.
    # More options: https://www.sbert.net/docs/pretrained_models.html
    embedding_model: str = Field("all-MiniLM-L6-v2")

    # The number of dimensions in each embedding vector.
    # Must match what the model actually produces — all-MiniLM-L6-v2 → 384.
    embedding_dim: int = Field(384)

    # How many chunks to embed in one GPU/CPU batch. Larger = faster but more RAM.
    embedding_batch: int = Field(64)

    # ── Vector Store ─────────────────────────────────────────────────────────
    # The vector store holds all the embeddings and lets us search by similarity.

    # Which vector database to use.
    # "faiss"  — runs in-process, no server needed, great for development.
    # "qdrant" — needs a running Qdrant server, scales to millions of documents.
    vector_backend: str = Field("faiss", description="faiss | qdrant")

    # Where to save the FAISS index files on disk.
    faiss_index_path: str = Field("./data/faiss_index")

    # Where Qdrant is running (default: local Docker container).
    qdrant_url: str = Field("http://localhost:6333")

    # Name for the Qdrant collection (like a table name in a SQL database).
    qdrant_collection: str = Field("rag_documents")

    # How many chunks to retrieve per query. More = more context but slower LLM.
    retrieval_top_k: int = Field(5)

    # ── Chunking ─────────────────────────────────────────────────────────────
    # Chunking splits large documents into smaller pieces before embedding.
    # Smaller chunks = more precise retrieval. Larger chunks = more context.

    # Which chunking strategy to use:
    # "sliding_window" — simple, fast. Splits every N words with overlap.
    # "semantic"       — smarter, slower. Keeps sentences with similar meaning together.
    chunking_strategy: str = Field("sliding_window", description="sliding_window | semantic")

    # Maximum number of words per chunk.
    chunk_size: int = Field(512)

    # How many words to repeat between adjacent chunks.
    # Overlap ensures a sentence split across a boundary doesn't lose context.
    chunk_overlap: int = Field(64)

    # ── OCR ──────────────────────────────────────────────────────────────────
    # OCR (Optical Character Recognition) converts images/scans to text.
    # We use Tesseract, which must be installed separately (see README).

    # Path to the tesseract binary. Find it with: which tesseract
    tesseract_cmd: str = Field("/usr/bin/tesseract")

    # Language(s) to use for OCR. "eng" = English. For multiple: "eng+fra".
    ocr_language: str = Field("eng")

    # Resolution to render PDF pages at before OCR. Higher = better accuracy
    # but slower. 300 DPI is the standard for document OCR.
    ocr_dpi: int = Field(300)

    # ── API Server ───────────────────────────────────────────────────────────

    # The IP address to bind to. "0.0.0.0" means "listen on all interfaces".
    api_host: str = Field("0.0.0.0")

    # Port for the HTTP server. Browse to http://localhost:8000 after starting.
    api_port: int = Field(8000)

    # Number of worker processes. Keep at 1 for development.
    api_workers: int = Field(1)

    # Which frontend URLs are allowed to call the API.
    # ["*"] means "allow all" — fine for local dev, restrict in production.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")   # DEBUG | INFO | WARNING | ERROR

    class Config:
        # Tell pydantic-settings to look for a .env file in the current directory.
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Ignore any extra variables in .env that aren't defined above.
        extra = "ignore"


# ---------------------------------------------------------------------------
# Create a single shared instance.
# Every other module imports this instead of creating their own Settings().
# ---------------------------------------------------------------------------
settings = Settings()
