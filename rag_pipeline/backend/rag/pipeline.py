"""
pipeline.py — The RAG Pipeline Orchestrator
=============================================
This is the heart of the system. RAGPipeline ties together every component
and provides two high-level operations:

  1. index_sources(sources)  — ingest documents and add them to the vector store
  2. query(question)         — answer a question using the indexed documents

FULL DATA FLOW:

  INDEX (done once, or whenever new documents arrive):
  ─────────────────────────────────────────────────────
  File / URL
    → DocumentIngestionPipeline.ingest_one()   [loaders.py]
    → Document (full text)
    → BaseChunker.chunk()                      [chunker.py]
    → list[Chunk] (small text windows)
    → EmbeddingEngine.embed_chunks()           [engine.py]
    → list[list[float]] (vectors)
    → VectorStore.add()                        [store.py]

  QUERY (done for every user question):
  ─────────────────────────────────────
  User question
    → EmbeddingEngine.embed_query()            [engine.py]
    → query_vector (a single embedding)
    → VectorStore.search()                     [store.py]
    → list[RetrievedChunk] (top-k results)
    → build_rag_prompt()                       [llm.py]
    → prompt string (question + context)
    → BaseLLM.complete()                       [llm.py]
    → answer string
    → QueryResult (answer + sources)

WHY THIS DESIGN?
  Each component is independent and testable. You can swap out:
  - The vector store (FAISS ↔ Qdrant)
  - The LLM (Claude ↔ GPT ↔ Ollama)
  - The chunking strategy
  without touching the rest of the pipeline.
"""

from __future__ import annotations
import time
from loguru import logger

from backend.config import settings
from backend.models import Document, QueryResult, IngestStats
from backend.ingestion.loaders   import DocumentIngestionPipeline
from backend.ingestion.chunker   import get_chunker
from backend.embeddings.engine   import get_engine
from backend.vectorstore.store   import VectorStore, get_vector_store
from backend.rag.llm             import BaseLLM, get_llm, build_rag_prompt


class RAGPipeline:
    """
    Full RAG pipeline — from raw documents to cited answers.

    All sub-components can be injected via the constructor for testing:

        # Default usage (everything from settings):
        pipeline = RAGPipeline()

        # Custom setup (e.g. for testing):
        pipeline = RAGPipeline(
            vector_store=FAISSVectorStore("./test_index"),
            llm=AnthropicLLM(),
            chunking_strategy="semantic",
        )
    """

    def __init__(
        self,
        vector_store:      VectorStore | None = None,
        llm:               BaseLLM     | None = None,
        chunking_strategy: str         | None = None,
    ):
        # Each component is either injected or created from defaults.
        # This pattern is called "Dependency Injection" and makes testing easy.

        # DocumentIngestionPipeline handles loading files and URLs.
        self.ingestion = DocumentIngestionPipeline()

        # Chunker splits documents into small windows. Strategy from .env or arg.
        self.chunker = get_chunker(chunking_strategy)

        # EmbeddingEngine turns text into vectors.
        self.embedder = get_engine()

        # Vector store holds indexed embeddings. Backend from .env or arg.
        self.store = vector_store or get_vector_store()

        # LLM generates the final answer from retrieved chunks.
        self.llm = llm or get_llm()

        logger.info(
            f"[RAGPipeline] Ready — "
            f"vector_store={type(self.store).__name__}, "
            f"llm={self.llm.model_id}, "
            f"chunks={self.count()}"
        )

    # ==========================================================================
    # INDEXING — build the searchable knowledge base
    # ==========================================================================

    def index_sources(self, sources: list[str]) -> IngestStats:
        """
        Full indexing pipeline for a list of file paths and/or URLs.

        This is the main entry point for adding new documents.
        Call this once for each batch of documents you want to search over.

        Args:
            sources: List of file paths (PDF, images) or URLs.

        Returns:
            IngestStats with counts, errors, and timing.

        Example:
            stats = pipeline.index_sources([
                "documents/annual_report.pdf",
                "scans/invoice_001.pdf",
                "https://docs.example.com/api",
            ])
            print(f"Indexed {stats.total_chunks} chunks")
        """
        t0 = time.time()

        # ── Step 1: Load documents ──────────────────────────────────────────
        # Detect source types, run OCR if needed, extract plain text.
        docs, stats = self.ingestion.ingest_bulk(sources)

        if not docs:
            logger.warning("[RAGPipeline] No documents were loaded.")
            return stats

        # ── Step 2: Chunk ───────────────────────────────────────────────────
        # Split each Document into small overlapping windows.
        chunks = self.chunker.chunk_many(docs)

        if not chunks:
            logger.warning("[RAGPipeline] Chunking produced zero chunks.")
            stats.total_chunks = 0
            return stats

        # ── Step 3: Embed ───────────────────────────────────────────────────
        # Convert each chunk's text to a vector of numbers.
        logger.info(f"[RAGPipeline] Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed_chunks(chunks)

        # ── Step 4: Store ───────────────────────────────────────────────────
        # Add vectors + chunk metadata to FAISS or Qdrant.
        self.store.add(chunks, embeddings)

        # Update stats with chunk count and final duration.
        stats.total_chunks = len(chunks)
        stats.duration_s   = round(time.time() - t0, 2)

        logger.success(
            f"[RAGPipeline] Indexed {stats.total_documents} docs → "
            f"{stats.total_chunks} chunks in {stats.duration_s}s"
        )
        return stats

    def index_documents(self, docs: list[Document]) -> int:
        """
        Index pre-loaded Document objects (skips the ingestion step).
        Useful when you've already loaded documents in your own code.

        Returns the number of chunks indexed.
        """
        chunks     = self.chunker.chunk_many(docs)
        embeddings = self.embedder.embed_chunks(chunks)
        self.store.add(chunks, embeddings)
        return len(chunks)

    # ==========================================================================
    # QUERYING — answer questions using indexed documents
    # ==========================================================================

    def query(
        self,
        question:        str,
        top_k:           int  = settings.retrieval_top_k,
        filter_metadata: dict | None = None,
    ) -> QueryResult:
        """
        Answer a question using Retrieval-Augmented Generation.

        The full flow:
          1. Embed the question as a vector.
          2. Search the vector store for the top_k most similar chunks.
          3. Build a citation-grounded prompt with those chunks as context.
          4. Send the prompt to the LLM.
          5. Return the answer with source references attached.

        Args:
            question:        The user's natural-language question.
            top_k:           How many chunks to retrieve (default: 5).
            filter_metadata: Optional dict to filter chunks by metadata.
                             Example: {"filename": "contract.pdf"}

        Returns:
            QueryResult with .answer (the LLM's response) and
            .sources (the chunks used to produce it).

        Example:
            result = pipeline.query("What are the payment terms?")
            print(result.answer)
            for source in result.sources:
                print(f"  [{source.rank+1}] {source.chunk.origin}")
        """
        t0 = time.time()

        # ── Step 1: Embed the question ──────────────────────────────────────
        # Turns the question into a vector we can compare against chunk vectors.
        query_vec = self.embedder.embed_query(question)

        # ── Step 2: Retrieve relevant chunks ───────────────────────────────
        # Find the top_k chunks whose vectors are most similar to the query.
        retrieved = self.store.search(
            query_vec,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Handle the case where the index is empty.
        if not retrieved:
            return QueryResult(
                question   = question,
                answer     = (
                    "No relevant documents found. "
                    "Please index some documents first using the 'Ingest' tab."
                ),
                sources    = [],
                model      = self.llm.model_id,
                latency_ms = round((time.time() - t0) * 1000, 1),
            )

        logger.debug(
            f"[RAGPipeline] Retrieved {len(retrieved)} chunks "
            f"(top score: {retrieved[0].score:.3f})"
        )

        # ── Step 3: Build the prompt ────────────────────────────────────────
        # Format the question + all retrieved chunks into a structured prompt
        # that instructs the LLM to cite sources and not hallucinate.
        prompt = build_rag_prompt(question, retrieved)

        # ── Step 4: Generate the answer ─────────────────────────────────────
        # Send the prompt to the configured LLM and wait for the response.
        answer = self.llm.complete(prompt)

        return QueryResult(
            question   = question,
            answer     = answer,
            sources    = retrieved,    # Chunks the answer was grounded in
            model      = self.llm.model_id,
            latency_ms = round((time.time() - t0) * 1000, 1),
        )

    # ==========================================================================
    # Utility methods
    # ==========================================================================

    def count(self) -> int:
        """How many chunks are currently indexed."""
        return self.store.count()

    def clear_index(self) -> None:
        """Wipe the entire vector store. Irreversible."""
        self.store.clear()
        logger.info("[RAGPipeline] Index cleared.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Most of the time you only want one RAGPipeline per process.
# get_pipeline() ensures we reuse the same instance instead of loading
# the embedding model and connecting to the vector store multiple times.

_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline instance, creating it if needed."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
