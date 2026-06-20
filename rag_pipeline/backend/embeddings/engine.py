"""
engine.py — Embedding Engine
==============================
WHAT ARE EMBEDDINGS?
  An embedding is a way of converting text into a list of numbers (a vector).
  The model is trained so that semantically similar texts produce similar vectors.

  Example:
    "The dog chased the ball"   → [0.12, -0.45, 0.88, ...]   ← similar vectors
    "A puppy ran after a sphere" → [0.11, -0.43, 0.90, ...]
    "The stock market fell 3%"  → [-0.62, 0.31, -0.14, ...]  ← very different

  This lets us find relevant chunks by measuring vector similarity,
  not by keyword matching. That's why it's called "semantic search".

WHAT MODEL DO WE USE?
  By default: all-MiniLM-L6-v2 from sentence-transformers.
  - 384-dimensional vectors
  - Very fast, runs on CPU
  - Good quality for English text
  - ~90 MB download on first use

  You can swap in a different model via EMBEDDING_MODEL in .env.
  Bigger models (e.g. all-mpnet-base-v2) are more accurate but slower.

HOW COSINE SIMILARITY WORKS:
  We normalise all vectors to unit length (length = 1).
  For unit vectors, cosine similarity = dot product.
  Range: -1 (opposite) to +1 (identical). In practice, 0.7+ means "very similar".
"""

from __future__ import annotations
import numpy as np
from loguru import logger

from backend.config import settings
from backend.models import Chunk


class EmbeddingEngine:
    """
    Wraps sentence-transformers with lazy loading and batch processing.

    "Lazy loading" means the model is NOT downloaded/loaded when you create
    an EmbeddingEngine(). It's loaded the first time you call embed_texts().
    This keeps startup fast when the model isn't immediately needed.

    Usage:
        engine = EmbeddingEngine()

        # Embed a list of strings → numpy array of shape (n, 384)
        vecs = engine.embed_texts(["hello world", "foo bar"])

        # Embed a single query → shape (384,)
        q_vec = engine.embed_query("What are the payment terms?")

        # Embed chunks for indexing → list[list[float]] ready for vector store
        embeddings = engine.embed_chunks(chunks)
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        batch_size:  int = settings.embedding_batch,
    ):
        self.model_name = model_name
        self.batch_size  = batch_size
        self._model      = None   # Not loaded yet — lazy loading

    def _load(self):
        """
        Load the model from disk (or download it the first time).
        This is called automatically the first time you embed something.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Embeddings] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.success(
                f"[Embeddings] Ready — dim={self.dim}, "
                f"device={self._model.device}"
            )
        return self._model

    @property
    def dim(self) -> int:
        """The number of dimensions in each embedding vector (e.g. 384)."""
        return self._load().get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: Any list of strings to embed.

        Returns:
            numpy array of shape (len(texts), dim).
            Each row is the embedding vector for the corresponding text.

        Note: normalize_embeddings=True makes all vectors unit length.
              This lets us use dot product instead of full cosine similarity,
              which is faster and supported by FAISS's IndexFlatIP.
        """
        model = self._load()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            # Show a progress bar only for large batches (more than 32 texts).
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query string.

        This is called at query time (not during indexing) and is separate
        from embed_texts() for clarity — though internally it's the same model.

        Returns:
            numpy array of shape (dim,) — a single 1D vector.
        """
        # embed_texts returns shape (1, dim), [0] gives shape (dim,)
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Embed a list of Chunk objects for storage in the vector store.

        Extracts the text content from each chunk, embeds them in batches,
        and returns Python lists (not numpy arrays) because JSON serialisation
        and FAISS/Qdrant both expect plain Python lists.

        Returns:
            list of embedding vectors, parallel to the input chunks list.
            i.e. embeddings[3] is the vector for chunks[3].
        """
        texts      = [chunk.content for chunk in chunks]
        vecs_array = self.embed_texts(texts)      # numpy array (n, dim)
        return vecs_array.tolist()                 # Convert to list[list[float]]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# We only ever want ONE EmbeddingEngine per process — loading the model
# twice wastes ~200 MB of RAM. get_engine() ensures we reuse the same instance.

_engine: EmbeddingEngine | None = None

def get_engine() -> EmbeddingEngine:
    """Return the shared EmbeddingEngine instance, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
