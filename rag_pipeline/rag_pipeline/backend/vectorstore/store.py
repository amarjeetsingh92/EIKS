"""
store.py — Vector Store Abstraction
=====================================
A vector store is a database optimised for storing and searching
high-dimensional vectors (embeddings). Unlike a SQL database that
finds rows by exact value match, a vector store finds the N rows
whose vectors are most similar to a query vector.

This is called "Approximate Nearest Neighbour" (ANN) search.

WE IMPLEMENT TWO BACKENDS:

  FAISSVectorStore (default)
  ──────────────────────────
  FAISS (Facebook AI Similarity Search) runs entirely in your Python process.
  No server to start. Saves the index to disk automatically.
  Great for development and datasets up to ~1 million chunks.
  Backed by flat files: index.faiss + metadata.pkl in ./data/faiss_index/

  QdrantVectorStore (production)
  ───────────────────────────────
  Qdrant is a standalone vector database server. Run it with Docker:
      docker run -p 6333:6333 qdrant/qdrant
  Supports filtering by metadata, horizontal scaling, and persistent storage.
  Better for large corpora (millions of documents) or multi-user deployments.

BOTH BACKENDS SHARE THE SAME INTERFACE (VectorStore ABC):
  store.add(chunks, embeddings)     → index new chunks
  store.search(query_vec, top_k=5)  → find most similar chunks
  store.delete(chunk_ids)           → remove chunks by ID
  store.count()                     → how many chunks are indexed
  store.clear()                     → wipe everything

Switch backends by setting VECTOR_BACKEND=qdrant in your .env file.
"""

from __future__ import annotations
import pickle       # Python's built-in serialisation (for saving chunk metadata)
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger

from backend.models import Chunk, RetrievedChunk
from backend.config import settings


# ===========================================================================
# VectorStore — the abstract interface both backends must implement
# ===========================================================================

class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    Both FAISSVectorStore and QdrantVectorStore inherit from this.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Add chunks and their pre-computed embeddings to the index.
        chunks[i] and embeddings[i] must correspond to the same text.
        """

    @abstractmethod
    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k: int = settings.retrieval_top_k,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the top_k chunks most similar to query_vec.
        Optionally filter results by metadata key-value pairs.
        Returns a ranked list (index 0 = most similar).
        """

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks from the index by their chunk.id values."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed chunks."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe the entire index. Irreversible."""


# ===========================================================================
# FAISSVectorStore — local, file-backed, no server needed
# ===========================================================================

class FAISSVectorStore(VectorStore):
    """
    FAISS-backed vector store that persists to disk automatically.

    INDEX TYPE: IndexFlatIP (Inner Product)
      "Flat" means exhaustive search — checks every vector for every query.
      "IP" = Inner Product. Since our vectors are unit-normalised,
      Inner Product equals Cosine Similarity.

      For large corpora (>500k chunks), consider IndexIVFFlat (approximate
      but much faster). For now, Flat is simpler and accurate.

    STORAGE FORMAT:
      index.faiss   → FAISS binary index (the vectors themselves)
      metadata.pkl  → Python pickle of list[Chunk] (parallel to the index)

    PARALLEL ARRAYS:
      FAISS assigns each vector an integer row ID (0, 1, 2, ...).
      self._chunks[i] stores the Chunk object for FAISS row i.
      So: after search(), scores[i] corresponds to self._chunks[indices[i]].
    """

    def __init__(self, index_path: str = settings.faiss_index_path):
        self.index_path = Path(index_path)
        # Create the directory if it doesn't exist yet.
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index_file = self.index_path / "index.faiss"
        self._meta_file  = self.index_path / "metadata.pkl"

        self._index = None     # The FAISS index object (None until first use)
        self._chunks: list[Chunk] = []   # Parallel array: _chunks[i] = chunk at FAISS row i

        # Try to load an existing index from disk.
        self._load_from_disk()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load a previously saved index and metadata, if they exist."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        if self._index_file.exists() and self._meta_file.exists():
            # Load the FAISS binary index.
            self._index = faiss.read_index(str(self._index_file))
            # Load the parallel chunk metadata.
            with open(self._meta_file, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info(f"[FAISS] Loaded existing index: {self.count()} chunks")
        else:
            logger.debug("[FAISS] No existing index found — will create on first add()")

    def _save_to_disk(self):
        """Persist the current index and chunk metadata to disk."""
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_file))
            with open(self._meta_file, "wb") as f:
                pickle.dump(self._chunks, f)

    def _create_index(self, dim: int):
        """Create a new empty FAISS IndexFlatIP for vectors of size `dim`."""
        import faiss
        # IndexFlatIP = exact inner product search (cosine for unit vectors).
        self._index = faiss.IndexFlatIP(dim)
        logger.debug(f"[FAISS] Created new IndexFlatIP (dim={dim})")

    # ── Interface implementation ──────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Add chunks to the FAISS index.

        Steps:
          1. Convert embeddings to a float32 numpy array (FAISS requirement).
          2. Create the index if this is the first batch.
          3. Append vectors to the FAISS index.
          4. Append chunk objects to the parallel _chunks list.
          5. Save to disk.
        """
        # FAISS requires float32 — Python lists default to float64.
        vecs = np.array(embeddings, dtype="float32")

        if self._index is None:
            # First batch — create the index with the right dimensionality.
            self._create_index(vecs.shape[1])

        # faiss.add() appends vectors. FAISS auto-assigns IDs 0, 1, 2, ...
        self._index.add(vecs)
        # Keep chunk metadata in parallel.
        self._chunks.extend(chunks)
        self._save_to_disk()

        logger.info(f"[FAISS] Added {len(chunks)} chunks. Total: {self.count()}")

    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k: int = settings.retrieval_top_k,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the top_k most similar chunks to a query vector.

        FAISS search() returns:
          scores:  shape (1, k) — similarity scores, highest first
          indices: shape (1, k) — FAISS row IDs of the matching chunks

        We over-fetch by 3x when metadata filtering is active, so we still
        get top_k results after filtering removes some.
        """
        if self._index is None or self.count() == 0:
            return []

        # FAISS expects a 2D array: shape (n_queries, dim). We have 1 query.
        q = np.array([query_vec], dtype="float32")

        # Over-fetch to allow post-search metadata filtering.
        fetch_k = min(top_k * 3 if filter_metadata else top_k, self.count())

        scores, indices = self._index.search(q, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for "not enough results" padding. Skip those.
            if idx < 0:
                continue

            chunk = self._chunks[idx]

            # Apply metadata filter: only include chunks matching ALL conditions.
            if filter_metadata:
                if not all(chunk.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(RetrievedChunk(
                chunk=chunk,
                score=float(score),
                rank=len(results),
            ))

            if len(results) >= top_k:
                break

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        """
        Remove chunks by ID.

        FAISS flat indices don't support deletion natively.
        Workaround: rebuild the index without the deleted chunks.
        This is O(n) but acceptable for typical use cases.
        """
        id_set   = set(chunk_ids)
        keep_idx = [i for i, c in enumerate(self._chunks) if c.id not in id_set]

        if not keep_idx:
            self.clear()
            return

        import faiss
        # Reconstruct the kept vectors from the existing index.
        kept_vecs = np.zeros((len(keep_idx), self._index.d), dtype="float32")
        for new_i, old_i in enumerate(keep_idx):
            self._index.reconstruct(old_i, kept_vecs[new_i])

        # Rebuild index and metadata with only the kept rows.
        self._index  = faiss.IndexFlatIP(self._index.d)
        self._index.add(kept_vecs)
        self._chunks = [self._chunks[i] for i in keep_idx]
        self._save_to_disk()

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Wipe the index and delete the disk files."""
        self._index  = None
        self._chunks = []
        for f in [self._index_file, self._meta_file]:
            if f.exists():
                f.unlink()   # Delete the file
        logger.info("[FAISS] Index cleared.")


# ===========================================================================
# QdrantVectorStore — production-ready, requires a Qdrant server
# ===========================================================================

class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store.

    Start Qdrant with Docker:
        docker run -p 6333:6333 qdrant/qdrant

    KEY CONCEPTS:
      Collection: Like a table in SQL. We use one collection per project.
      Point:      A single vector + its payload (metadata). ID is an integer.
      Payload:    The metadata attached to each point (chunk text, source, etc.)

    Qdrant handles persistence, replication, and filtering natively.
    You can run complex metadata filters alongside vector search:
      "find the top 5 most relevant chunks, but only from document X"
    """

    def __init__(
        self,
        url:        str = settings.qdrant_url,
        collection: str = settings.qdrant_collection,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("Run: pip install qdrant-client")

        from qdrant_client import QdrantClient

        self.collection = collection
        # Connect to the Qdrant server. timeout=30 for slow/remote servers.
        self.client = QdrantClient(url=url, timeout=30)
        # Create the collection if it doesn't already exist.
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the vector collection if it doesn't exist yet."""
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    # Must match our embedding model's output dimension.
                    size=settings.embedding_dim,
                    # Cosine distance for semantic similarity.
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"[Qdrant] Created collection '{self.collection}'")

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Upsert chunks into Qdrant as "points".
        Each point has:
          - id:      an integer (we use sequential IDs)
          - vector:  the embedding
          - payload: the chunk text and metadata (stored as JSON)

        "Upsert" = insert if not exists, update if exists. Safe to call
        multiple times with the same data.
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=i,
                vector=emb,
                payload={
                    # Store everything we need to reconstruct the Chunk object.
                    "chunk_id":    chunk.id,
                    "doc_id":      chunk.doc_id,
                    "origin":      chunk.origin,
                    "source":      chunk.source.value,  # .value = the string "pdf_text" etc.
                    "content":     chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata":    chunk.metadata,
                },
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"[Qdrant] Upserted {len(points)} points.")

    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k: int = settings.retrieval_top_k,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Search Qdrant and reconstruct Chunk objects from the payload."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build a Qdrant filter if metadata conditions were specified.
        qfilter = None
        if filter_metadata:
            qfilter = Filter(must=[
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ])

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=list(query_vec),
            limit=top_k,
            query_filter=qfilter,
        )

        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits):
            p = hit.payload   # The dict we stored in add()
            # Reconstruct the Chunk from what we stored in the payload.
            chunk = Chunk(
                id          = p["chunk_id"],
                doc_id      = p["doc_id"],
                origin      = p["origin"],
                source      = p["source"],
                content     = p["content"],
                chunk_index = p["chunk_index"],
                metadata    = p.get("metadata", {}),
            )
            results.append(RetrievedChunk(chunk=chunk, score=hit.score, rank=rank))

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete points matching any of the given chunk IDs."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))
            ]),
        )

    def count(self) -> int:
        return self.client.count(self.collection).count

    def clear(self) -> None:
        """Recreate the collection (wiping all data)."""
        from qdrant_client.models import Distance, VectorParams
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("[Qdrant] Collection cleared.")


# ===========================================================================
# Factory — pick the right backend from settings
# ===========================================================================

def get_vector_store(backend: str | None = None) -> VectorStore:
    """
    Return a VectorStore instance for the configured backend.

    Usage:
        store = get_vector_store()          # reads VECTOR_BACKEND from .env
        store = get_vector_store("qdrant")  # force a specific backend
    """
    b = backend or settings.vector_backend
    if b == "qdrant":
        return QdrantVectorStore()
    return FAISSVectorStore()
