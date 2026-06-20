"""
models.py — Core data models for the RAG pipeline
==================================================
Think of this file as the "blueprint" for every piece of data
that flows through the pipeline. Every stage (loading, chunking,
embedding, retrieval, answering) uses these same data structures
so they can talk to each other cleanly.

We use Pydantic models, which are like Python dataclasses but with
automatic type-checking and easy JSON serialisation/deserialisation.
"""

from __future__ import annotations
from enum import Enum          # Used to define fixed sets of string constants
from typing import Any
from pydantic import BaseModel, Field   # Data validation library
import uuid    # Generates unique IDs like "3f2a1c89-..."
import time    # Unix timestamps (seconds since 1970)


# ---------------------------------------------------------------------------
# DocumentSource — WHERE did a document come from?
# ---------------------------------------------------------------------------
# An "enum" (enumeration) is a set of named constants. Instead of writing
# the string "pdf_text" everywhere (and risking typos), we write
# DocumentSource.PDF_TEXT. Python will catch any misspelling at import time.

class DocumentSource(str, Enum):
    PDF_TEXT    = "pdf_text"     # Normal PDF with a text layer (copy-pasteable)
    PDF_SCANNED = "pdf_scanned"  # Scanned PDF — needs OCR to extract text
    IMAGE       = "image"        # JPEG, PNG, TIFF — also needs OCR
    WEB         = "web"          # Scraped from a URL
    UNKNOWN     = "unknown"      # Fallback when source can't be determined


# ---------------------------------------------------------------------------
# Document — one fully-loaded file or webpage
# ---------------------------------------------------------------------------
# Represents ONE fully-loaded document (e.g. an entire PDF or webpage).
# This is created by the loaders (loaders.py) and passed to the chunker.

class Document(BaseModel):
    """
    A raw document after text has been extracted from it.
    Created by the ingestion loaders, before any chunking.

    Example:
        doc = Document(
            source=DocumentSource.PDF_TEXT,
            origin="/home/user/report.pdf",
            content="The quarterly results show ...",
            page_count=12,
        )
    """

    # A random unique identifier — uuid4 means "universally unique, random".
    # default_factory means "call this function to create the default value".
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Where did this document come from? (PDF, scanned, image, web)
    source: DocumentSource

    # The original file path or URL — used for citations.
    origin: str   # e.g. "/home/user/report.pdf" or "https://example.com"

    # The extracted plain text — this is what we actually search over.
    content: str

    # How many pages the PDF had (0 for web pages / images).
    page_count: int = 0

    # Flexible key-value store for anything extra: filename, title, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)

    # When was this document ingested? Stored as a Unix timestamp (float).
    ingested_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Chunk — a small slice of a Document
# ---------------------------------------------------------------------------
# A Document is typically too long to embed as one unit — language models
# have context limits, and embedding an entire 50-page report as one vector
# loses too much detail. So we split each Document into smaller "chunks"
# of a few hundred words each. The chunker (chunker.py) does this.

class Chunk(BaseModel):
    """
    A piece of a Document, ready to be turned into an embedding vector.
    Each Chunk knows which Document it came from (doc_id) so we can
    cite the original source in answers later.

    Example: A 12-page PDF might produce 40 chunks of ~300 words each.
    """

    # Unique ID for this specific chunk.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Which Document did this chunk come from? Links back to Document.id.
    doc_id: str

    # Copied from the parent Document so you don't need to look it up later.
    origin: str
    source: DocumentSource

    # The actual text of this chunk — what gets embedded and searched.
    content: str

    # Position within the document: 0 = first chunk, 1 = second, etc.
    chunk_index: int

    # Which page this chunk came from (None for web content).
    page: int | None = None

    # Inherits metadata from parent Document (filename, URL, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# IndexedChunk — a Chunk that has been stored in the vector database
# ---------------------------------------------------------------------------

class IndexedChunk(Chunk):
    """
    A Chunk that has been embedded (converted to numbers) and stored
    in FAISS or Qdrant. We keep the raw vector here for debugging.
    """

    # The embedding vector — a list of floats like [0.12, -0.45, 0.88, ...]
    # The length depends on the model (384 for all-MiniLM-L6-v2).
    embedding: list[float]

    # The row number (FAISS) or point ID (Qdrant) in the vector store.
    vector_id: str | int


# ---------------------------------------------------------------------------
# RetrievedChunk — a search result
# ---------------------------------------------------------------------------
# When you submit a query, the vector store finds the most similar chunks.
# Each result is a RetrievedChunk — the text plus a similarity score.

class RetrievedChunk(BaseModel):
    """
    A chunk returned by similarity search, with how relevant it is.

    score: cosine similarity — 1.0 = identical, 0.0 = completely unrelated.
    rank:  0 = best match, 1 = second best, etc.
    """
    chunk: Chunk    # The actual text chunk
    score: float    # Similarity score (higher = more relevant)
    rank: int       # Position in the ranked list (0 = most relevant)


# ---------------------------------------------------------------------------
# QueryResult — the final answer from the full pipeline
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """
    The complete response to a user's question.
    Contains both the LLM's answer text AND the source chunks it used,
    so every claim in the answer can be traced back to a document.
    """
    question:   str                    # The original user question
    answer:     str                    # The LLM's cited answer
    sources:    list[RetrievedChunk]   # Chunks the answer was grounded in
    model:      str                    # Which LLM model produced the answer
    latency_ms: float                  # Total pipeline time in milliseconds
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# IngestStats — summary of an ingestion job
# ---------------------------------------------------------------------------

class IngestStats(BaseModel):
    """
    Returned after ingesting a batch of documents.
    Tells you what worked, what was skipped, and what failed.
    """
    total_documents: int        # Successfully loaded documents
    total_chunks:    int        # Chunks indexed into the vector store
    skipped:         int        # Sources that produced zero text
    errors:          list[str]  # Error messages for failed sources
    duration_s:      float      # Total wall-clock time in seconds
