# Knowledge Retrieval Pipeline

<img width="1893" height="865" alt="image" src="https://github.com/user-attachments/assets/10064836-a65a-4f09-bdce-39aa0fbd2a5a" />


Scalable RAG system for unstructured and scanned documents.  
Combines OCR, semantic search, and citation-grounded question answering.

```
Documents (PDF, Scans, Images, Web)
         ↓
   [Ingestion Layer]  ← pdfplumber, Tesseract OCR, BeautifulSoup
         ↓
   [Chunking]         ← Sliding Window or Semantic
         ↓
   [Embeddings]       ← sentence-transformers (all-MiniLM-L6-v2)
         ↓
   [Vector Store]     ← FAISS (local) or Qdrant (scalable)
         ↓
   [RAG + LLM]        ← Anthropic / OpenAI / Ollama
         ↓
   Cited Answer with Source References
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd rag_pipeline

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR)

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set your API keys and preferred backends
```

### 4. Start the API server

```bash
python run_server.py
# API live at http://localhost:8000
# Docs at    http://localhost:8000/docs
```

### 5. Open the frontend

```bash
open frontend/index.html   # or serve with any static server
```

---

## CLI Usage

```bash
# Index files and URLs
python cli.py ingest report.pdf scanned_invoice.pdf https://example.com/article

# Ask a question
python cli.py query "What are the payment terms in the contract?"

# Check index stats
python cli.py stats

# Wipe the index
python cli.py clear
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/files` | Upload + index PDF / image files |
| `POST` | `/ingest/urls` | Scrape + index web URLs |
| `POST` | `/query` | Ask a question, get cited answer |
| `GET`  | `/index/stats` | Chunk count, backend info |
| `DELETE` | `/index` | Wipe the index |
| `GET`  | `/health` | Liveness probe |


---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Options |
|----------|---------|---------|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `ollama` |
| `VECTOR_BACKEND` | `faiss` | `faiss`, `qdrant` |
| `CHUNKING_STRATEGY` | `sliding_window` | `sliding_window`, `semantic` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |

---

## Vector Store Backends

### FAISS (default)
- No server needed, runs in-process
- Persists to `./data/faiss_index/`
- Good for up to ~1M chunks

### Qdrant (production)
```bash
# Start with Docker
docker run -p 6333:6333 qdrant/qdrant

# Switch in .env
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
```

---

## Chunking Strategies

**Sliding Window** (default) — fixed-size windows with overlap. Fast, deterministic.

**Semantic** — splits on sentence boundaries, starts a new chunk when cosine similarity drops. Better retrieval relevance at the cost of speed.

```bash
CHUNKING_STRATEGY=semantic
```

---

## Project Structure

```
rag_pipeline/
├── backend/
│   ├── config.py              # All settings (pydantic-settings)
│   ├── models.py              # Shared data models
│   ├── ingestion/
│   │   ├── loaders.py         # TextPDF, ScannedPDF, Image, Web loaders
│   │   └── chunker.py         # SlidingWindow + Semantic chunkers
│   ├── embeddings/
│   │   └── engine.py          # EmbeddingEngine (sentence-transformers)
│   ├── vectorstore/
│   │   └── store.py           # FAISS + Qdrant implementations
│   ├── rag/
│   │   ├── llm.py             # Anthropic / OpenAI / Ollama adapters
│   │   └── pipeline.py        # RAGPipeline orchestrator
│   └── api/
│       └── server.py          # FastAPI app
├── frontend/
│   └── index.html             # Single-file frontend
├── cli.py                     # CLI (ingest / query / stats / clear)
├── run_server.py              # uvicorn entry point
├── requirements.txt
└── .env.example
```
