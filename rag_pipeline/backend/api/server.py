"""
server.py — FastAPI REST API
==============================
This file exposes the RAG pipeline as an HTTP API so the frontend
(and any other client) can use it without importing Python code.

WHAT IS FastAPI?
  FastAPI is a modern Python web framework. It automatically:
    - Validates request/response types using Pydantic models
    - Generates interactive docs at http://localhost:8000/docs
    - Handles serialisation (Python objects ↔ JSON)
    - Adds async support for high-throughput use cases

ENDPOINTS:
  POST /ingest/files    — upload PDF/image files and index them
  POST /ingest/urls     — scrape + index web URLs
  POST /query           — ask a question, get a cited answer
  GET  /index/stats     — how many chunks, which model, etc.
  DELETE /index         — wipe the entire index
  GET  /health          — simple liveness probe

HOW TO START:
  python run_server.py
  → API available at http://localhost:8000
  → Docs available at http://localhost:8000/docs

HOW THE FRONTEND USES IT:
  1. User drops a PDF → frontend calls POST /ingest/files
  2. User types a question → frontend calls POST /query
  3. Frontend displays answer + source chips from the response
"""

from __future__ import annotations
import tempfile
import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.rag.pipeline import get_pipeline, RAGPipeline
from backend.models import QueryResult, IngestStats


# ===========================================================================
# App setup
# ===========================================================================

app = FastAPI(
    title="Knowledge Retrieval Pipeline",
    description=(
        "Semantic search and RAG over unstructured and scanned documents. "
        "Visit /docs for the interactive API explorer."
    ),
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing) lets the frontend HTML file
# (served from a different origin than localhost:8000) call this API.
# In production, restrict allow_origins to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # ["*"] during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Dependency injection
# ===========================================================================
# FastAPI's Depends() system makes it easy to share a single pipeline
# instance across all requests. It also makes unit testing easy because
# you can swap in a mock pipeline during tests.

def pipeline_dep() -> RAGPipeline:
    """FastAPI dependency that returns the shared pipeline instance."""
    return get_pipeline()

# Type alias — used in function signatures below
Pipeline = Annotated[RAGPipeline, Depends(pipeline_dep)]


# ===========================================================================
# Request/Response schemas
# ===========================================================================
# These Pydantic models define what JSON the API expects and returns.
# FastAPI validates incoming requests against these schemas automatically.

class IngestURLsRequest(BaseModel):
    """Request body for POST /ingest/urls"""
    # List of web URLs to scrape and index.
    urls: list[str]
    # Optional: override the chunking strategy for this batch.
    # None = use the default from settings.
    chunking_strategy: str | None = None


class QueryRequest(BaseModel):
    """Request body for POST /query"""
    # The user's natural-language question.
    question: str
    # How many source chunks to retrieve. More = more context for the LLM.
    top_k: int = settings.retrieval_top_k
    # Optional metadata filter: only search chunks matching these key-value pairs.
    # Example: {"filename": "contract.pdf"}
    filter_metadata: dict | None = None


class IndexStats(BaseModel):
    """Response body for GET /index/stats"""
    total_chunks:    int
    vector_backend:  str
    embedding_model: str
    llm_provider:    str
    llm_model:       str


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/health")
def health():
    """
    Liveness probe — returns 200 OK if the server is running.
    The frontend polls this to show the green "connected" dot.
    """
    return {"status": "ok", "timestamp": time.time()}


@app.post("/ingest/files", response_model=IngestStats)
async def ingest_files(
    pipeline: Pipeline,
    files: list[UploadFile] = File(...),  # UploadFile = multipart/form-data upload
):
    """
    Upload one or more PDF or image files and index them.

    How it works:
      1. FastAPI streams the uploaded bytes into memory.
      2. We write them to a temporary directory on disk.
      3. We pass the file paths to the pipeline's indexing function.
      4. The temp directory is deleted automatically when we're done.

    Why a temp directory?
      Our loaders need file paths (not bytes), so we must save files to disk first.
      tempfile.TemporaryDirectory() creates a folder that auto-deletes on exit.
    """
    # Only allow safe file types to prevent malicious uploads.
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    # Use a temporary directory that auto-deletes when the 'with' block exits.
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths: list[str] = []

        for uploaded_file in files:
            # Extract the file extension from the original filename.
            ext = Path(uploaded_file.filename or "").suffix.lower()
            if ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}",
                )

            # Save the uploaded bytes to a temporary file.
            dest = os.path.join(tmp_dir, uploaded_file.filename)
            with open(dest, "wb") as out:
                # await f.read() reads the full file into memory.
                # For very large files, chunked streaming would be better.
                out.write(await uploaded_file.read())
            saved_paths.append(dest)

        # Run the full ingestion pipeline on the saved files.
        stats = pipeline.index_sources(saved_paths)

    # TemporaryDirectory is now deleted. stats contains everything we need.
    return stats


@app.post("/ingest/urls", response_model=IngestStats)
def ingest_urls(req: IngestURLsRequest, pipeline: Pipeline):
    """
    Scrape a list of URLs and index the extracted text.

    The web loader fetches each URL, strips boilerplate HTML (nav, footer, ads),
    and extracts the main content for indexing.
    """
    if not req.urls:
        raise HTTPException(status_code=400, detail="urls list cannot be empty")

    # Override chunking strategy if specified in this request.
    if req.chunking_strategy:
        from backend.ingestion.chunker import get_chunker
        pipeline.chunker = get_chunker(req.chunking_strategy)

    stats = pipeline.index_sources(req.urls)
    return stats


@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest, pipeline: Pipeline):
    """
    Ask a question and get a citation-grounded answer.

    Response includes:
      - answer:  The LLM's response with [Source N] citations
      - sources: The retrieved chunks (text + metadata + similarity score)
      - model:   Which LLM was used
      - latency_ms: How long the full pipeline took
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    result = pipeline.query(
        question        = req.question,
        top_k           = req.top_k,
        filter_metadata = req.filter_metadata,
    )
    return result


@app.get("/index/stats", response_model=IndexStats)
def index_stats(pipeline: Pipeline):
    """
    Return current index statistics and configuration.
    The frontend uses this to show the chunk counter and model info.
    """
    # Determine which model name to show based on the active provider.
    if settings.llm_provider == "anthropic":
        llm_model = settings.anthropic_model
    elif settings.llm_provider == "openai":
        llm_model = settings.openai_model
    else:
        llm_model = settings.ollama_model

    return IndexStats(
        total_chunks    = pipeline.count(),
        vector_backend  = settings.vector_backend,
        embedding_model = settings.embedding_model,
        llm_provider    = settings.llm_provider,
        llm_model       = llm_model,
    )


@app.delete("/index")
def clear_index(pipeline: Pipeline):
    """
    Wipe the entire vector index. This is irreversible.
    The frontend asks for confirmation before calling this endpoint.
    """
    pipeline.clear_index()
    return {"message": "Index cleared successfully."}
