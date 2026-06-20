"""
chunker.py — Chunking Strategies
==================================
WHY DO WE CHUNK?
  Embedding models have a maximum input size (usually ~512 tokens / ~380 words).
  If a document is longer, we must split it. Even when it fits, embedding a
  whole 50-page document as one vector is a bad idea — the vector becomes an
  average of everything and retrieval becomes imprecise.

  Good chunking is the single biggest lever for improving RAG quality.
  A well-chunked document produces focused, dense chunks that match user
  queries closely. A badly-chunked document splits sentences mid-thought
  and confuses the retriever.

TWO STRATEGIES:

  SlidingWindowChunker (default)
    ─────────────────────────────
    Splits text into fixed-size word windows with overlap.
    e.g. chunk_size=512, overlap=64 → every chunk shares 64 words with the next.
    Overlap prevents information being lost at chunk boundaries.

    ░░░░░░░░░░░░░░░░░░░░
         ░░░░░░░░░░░░░░░░░░░░
              ░░░░░░░░░░░░░░░░░░░░
    ← 512 words →
         ← 64 overlap →

    Fast, deterministic, great baseline. Use this unless you need better quality.

  SemanticChunker (higher quality, slower)
    ──────────────────────────────────────
    Embeds every sentence individually and measures cosine similarity between
    adjacent sentences. Starts a new chunk whenever similarity drops sharply —
    indicating a topic change.

    This keeps semantically related sentences together in the same chunk,
    which improves retrieval relevance and reduces hallucination.

    Slower because it embeds every sentence before chunking.
    Worth it for high-value documents where precision matters.
"""

from __future__ import annotations
import re
from abc import ABC, abstractmethod

from loguru import logger

from backend.models import Document, Chunk
from backend.config import settings


# ===========================================================================
# BaseChunker — shared interface
# ===========================================================================

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a single Document into a list of Chunks."""

    def chunk_many(self, docs: list[Document]) -> list[Chunk]:
        """
        Convenience method: chunk a list of Documents.
        Logs how many chunks each document produces.
        """
        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)
            name = doc.metadata.get("filename") or doc.origin
            logger.debug(f"[Chunker] {name} → {len(chunks)} chunks")
        return all_chunks


# ===========================================================================
# SlidingWindowChunker
# ===========================================================================

class SlidingWindowChunker(BaseChunker):
    """
    Splits text into fixed-size word windows with configurable overlap.

    Parameters:
        chunk_size:  Maximum number of words per chunk (default: 512).
        overlap:     Number of words shared between adjacent chunks (default: 64).
                     Must be less than chunk_size.

    Example with chunk_size=6, overlap=2:
        Input:  "A B C D E F G H I J"
        Chunk 0: "A B C D E F"
        Chunk 1: "E F G H I J"   ← shares E F with chunk 0
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        overlap:    int = settings.chunk_overlap,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap    = overlap
        # Step size = how far we advance the window each iteration.
        # With overlap, we move less than a full window each time.
        self._step = chunk_size - overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        # Split the document text into individual words.
        words = doc.content.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        i = 0   # Current word index (start of the current window)

        while i < len(words):
            # Slice words[i : i + chunk_size] — the current window.
            window  = words[i : i + self.chunk_size]
            content = " ".join(window).strip()

            if content:
                chunks.append(Chunk(
                    doc_id      = doc.id,
                    origin      = doc.origin,
                    source      = doc.source,
                    content     = content,
                    chunk_index = len(chunks),   # 0-based position in this doc
                    metadata    = {
                        **doc.metadata,
                        "chunk_strategy": "sliding_window",
                    },
                ))

            # Advance by step (not by chunk_size) so overlap is preserved.
            i += self._step

        return chunks


# ===========================================================================
# SemanticChunker
# ===========================================================================

class SemanticChunker(BaseChunker):
    """
    Groups sentences together while they share a similar topic.
    When the cosine similarity between adjacent sentences drops below
    a threshold, a new chunk begins.

    HOW COSINE SIMILARITY WORKS:
        Each sentence is converted to a vector. We measure the angle
        between adjacent sentence vectors. Vectors pointing in a similar
        direction (small angle, high cosine) → same topic.
        Vectors pointing in different directions (large angle, low cosine)
        → topic shift → start a new chunk.

    Parameters:
        model_name:        Embedding model (reuses the main embedding model).
        similarity_thresh: Drop below this → start a new chunk. Range [0, 1].
                           0.75 is a good starting point. Lower = bigger chunks.
        max_chunk_words:   Hard cap — never exceed this many words per chunk,
                           even if similarity stays high.
    """

    def __init__(
        self,
        model_name:        str   = settings.embedding_model,
        similarity_thresh: float = 0.75,
        max_chunk_words:   int   = settings.chunk_size,
    ):
        self.model_name        = model_name
        self.similarity_thresh = similarity_thresh
        self.max_chunk_words   = max_chunk_words
        self._model            = None   # Lazy-loaded on first use to save startup time

    def _get_model(self):
        """Load the embedding model (only once, then cached in self._model)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[SemanticChunker] Loading model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text on sentence-ending punctuation followed by whitespace.
        Returns non-empty sentences.

        This is a simple heuristic. For production use, consider
        spaCy or NLTK for more accurate sentence splitting.
        """
        # re.split with a lookbehind (?<=...) means "split after . ! or ?"
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        import numpy as np

        sentences = self._split_sentences(doc.content)
        if not sentences:
            return []

        # Embed all sentences at once (batch is much faster than one-by-one).
        model      = self._get_model()
        embeddings = model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,   # Unit vectors → cosine = dot product
        )

        chunks: list[Chunk] = []

        # current_sentences: sentences accumulated into the current chunk.
        # current_embeddings: their corresponding vectors (for averaging).
        current_sentences:  list[str]  = [sentences[0]]
        current_embeddings: list       = [embeddings[0]]

        def flush_chunk() -> None:
            """Save the current accumulated sentences as a new Chunk."""
            content = " ".join(current_sentences).strip()
            if content:
                chunks.append(Chunk(
                    doc_id      = doc.id,
                    origin      = doc.origin,
                    source      = doc.source,
                    content     = content,
                    chunk_index = len(chunks),
                    metadata    = {
                        **doc.metadata,
                        "chunk_strategy": "semantic",
                    },
                ))

        for i in range(1, len(sentences)):
            # Represent the current chunk by the MEAN of its sentence vectors.
            # This gives a "center of mass" topic vector.
            chunk_vec = np.mean(current_embeddings, axis=0)
            sent_vec  = embeddings[i]

            # Cosine similarity. Since vectors are unit-normalised,
            # this simplifies to the dot product.
            cosine_sim = float(np.dot(chunk_vec, sent_vec))

            # Count words in the current chunk to respect the hard cap.
            word_count = sum(len(s.split()) for s in current_sentences)

            # Decide whether to start a new chunk:
            new_topic    = cosine_sim < self.similarity_thresh
            chunk_too_big = word_count >= self.max_chunk_words

            if new_topic or chunk_too_big:
                # Topic shifted (or chunk is full) → save and start fresh.
                flush_chunk()
                current_sentences  = [sentences[i]]
                current_embeddings = [embeddings[i]]
            else:
                # Same topic → keep accumulating.
                current_sentences.append(sentences[i])
                current_embeddings.append(embeddings[i])

        # Don't forget the last chunk!
        flush_chunk()

        return chunks


# ===========================================================================
# Factory function — pick the right chunker from settings
# ===========================================================================

def get_chunker(strategy: str | None = None) -> BaseChunker:
    """
    Return a chunker based on the configured (or specified) strategy.

    Usage:
        chunker = get_chunker()                      # uses settings
        chunker = get_chunker("semantic")            # override
        chunks  = chunker.chunk_many(documents)
    """
    s = strategy or settings.chunking_strategy
    if s == "semantic":
        return SemanticChunker()
    # Default to sliding window — fast, reliable, good baseline.
    return SlidingWindowChunker()
