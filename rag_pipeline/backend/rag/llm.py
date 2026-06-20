"""
llm.py — LLM Abstraction Layer
================================
This file provides a clean, swappable interface to different AI language models.
The RAG pipeline uses the LLM only at the LAST step — reading retrieved chunks
and writing a grounded answer. The retrieval step does NOT use the LLM.

SUPPORTED BACKENDS:
  Anthropic Claude  — cloud API, requires ANTHROPIC_API_KEY
  OpenAI GPT        — cloud API, requires OPENAI_API_KEY
  Ollama            — runs locally on your machine, no API key needed

Set LLM_PROVIDER in your .env file to choose. Default is "anthropic".

HOW RAG PROMPTING WORKS:
  1. We retrieve the top-k most relevant chunks from the vector store.
  2. We format them as numbered "Sources" in the prompt.
  3. We ask the LLM to answer ONLY using those sources, citing [Source N].
  4. The LLM cannot make things up because it's explicitly told to say
     "I don't have enough information" if the answer isn't in the context.

This citation-grounding is what reduces hallucination in RAG systems.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from loguru import logger

from backend.config import settings
from backend.models import RetrievedChunk


# ===========================================================================
# Prompt builder — shared by all LLM backends
# ===========================================================================

def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build a citation-grounded RAG prompt from a question and retrieved chunks.

    The prompt:
      1. Tells the LLM its role and strict constraints.
      2. Presents each chunk as a numbered "[Source N]" block.
      3. Asks the question at the end.

    By structuring sources as numbered references, the LLM naturally learns
    to write "[Source 2]" when using information from that chunk, which
    lets the frontend highlight which sources back up each claim.

    Args:
        question: The user's question.
        chunks:   Retrieved chunks in relevance order (best first).

    Returns:
        A complete prompt string ready to send to any LLM.
    """
    # Format each chunk as a labelled block.
    context_blocks = []
    for rc in chunks:
        # Build a human-readable source label for the prompt.
        # Prefer filename → URL → raw path, in that order.
        source_label = (
            rc.chunk.metadata.get("filename")
            or rc.chunk.metadata.get("url")
            or rc.chunk.origin
            or f"Source {rc.rank + 1}"
        )
        context_blocks.append(
            f"[Source {rc.rank + 1}: {source_label}]\n{rc.chunk.content}"
        )

    # Join all source blocks with a horizontal rule between them.
    context = "\n\n---\n\n".join(context_blocks)

    # The full prompt — note the strict instructions that prevent hallucination.
    return f"""You are a precise research assistant. Your job is to answer questions \
based ONLY on the provided source documents.

RULES YOU MUST FOLLOW:
- Answer only using information from the sources below.
- Cite your sources inline using [Source N] notation whenever you use information from them.
- If multiple sources support a point, cite all of them: [Source 1][Source 3].
- If the sources do not contain enough information to answer the question, say exactly:
  "I don't have enough information in the provided documents to answer this question."
- Do NOT use any external knowledge. Do NOT make things up (hallucinate).
- Be concise and factual. Avoid padding or filler sentences.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""


# ===========================================================================
# BaseLLM — the shared interface
# ===========================================================================

class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.
    All backends implement a single method: complete(prompt) → str.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return its text response.
        This is a synchronous (blocking) call — it waits for the full response.
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        A string identifying which model was used.
        Stored in QueryResult so users know what generated the answer.
        """


# ===========================================================================
# AnthropicLLM — Claude via Anthropic's API
# ===========================================================================

class AnthropicLLM(BaseLLM):
    """
    Calls Anthropic's Claude models via the official Python SDK.

    Setup:
      1. Sign up at console.anthropic.com
      2. Create an API key
      3. Add ANTHROPIC_API_KEY=your-key to .env

    Models (set ANTHROPIC_MODEL in .env):
      claude-opus-4-6    — most capable, slower, more expensive
      claude-sonnet-4-6  — balanced capability and speed (recommended)
      claude-haiku-4-5-20251001   — fastest, cheapest, good for simple questions
    """

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        import anthropic
        # The Anthropic client handles authentication, retries, and rate limiting.
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.debug(f"[LLM] Anthropic client ready, model: {self.model_id}")

    @property
    def model_id(self) -> str:
        return settings.anthropic_model

    def complete(self, prompt: str) -> str:
        """
        Send a message to Claude and return the response text.

        Anthropic's API uses a "messages" format (not a single prompt string):
          messages=[{"role": "user", "content": "..."}]

        temperature: 0.2 = fairly deterministic (good for factual RAG)
        max_tokens:  caps the response length (prevents runaway costs)
        """
        message = self._client.messages.create(
            model=self.model_id,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # message.content is a list of content blocks.
        # We want the text from the first (and usually only) text block.
        return message.content[0].text


# ===========================================================================
# OpenAILLM — GPT-4 via OpenAI's API
# ===========================================================================

class OpenAILLM(BaseLLM):
    """
    Calls OpenAI's GPT models via the official Python SDK.

    Setup:
      1. Sign up at platform.openai.com
      2. Create an API key
      3. Add OPENAI_API_KEY=your-key to .env
      4. Set LLM_PROVIDER=openai in .env
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)

    @property
    def model_id(self) -> str:
        return settings.openai_model

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


# ===========================================================================
# OllamaLLM — local models via Ollama
# ===========================================================================

class OllamaLLM(BaseLLM):
    """
    Calls a locally-running Ollama server. No API key or cloud account needed.

    Setup:
      1. Install Ollama from https://ollama.com
      2. Pull a model: ollama pull llama3
      3. Set LLM_PROVIDER=ollama in .env
      4. Optionally set OLLAMA_MODEL=llama3 (or mistral, gemma2, etc.)

    The Ollama server starts automatically when you run `ollama serve`.
    Default endpoint: http://localhost:11434
    """

    def __init__(self):
        # httpx is a modern HTTP client (like requests but async-compatible).
        import httpx
        self._http = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=120,  # Local models can be slow — give them 2 minutes
        )

    @property
    def model_id(self) -> str:
        return f"ollama/{settings.ollama_model}"

    def complete(self, prompt: str) -> str:
        # Ollama's /api/generate endpoint accepts a prompt and model name.
        # stream=False means we wait for the complete response.
        resp = self._http.post("/api/generate", json={
            "model":  settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        })
        resp.raise_for_status()
        return resp.json()["response"]


# ===========================================================================
# Factory — pick the right LLM from settings
# ===========================================================================

def get_llm(provider: str | None = None) -> BaseLLM:
    """
    Return an LLM instance for the configured provider.

    Usage:
        llm = get_llm()              # reads LLM_PROVIDER from .env
        llm = get_llm("openai")      # force a specific provider
        answer = llm.complete(prompt)
    """
    p = provider or settings.llm_provider
    if p == "anthropic":
        return AnthropicLLM()
    if p == "openai":
        return OpenAILLM()
    if p == "ollama":
        return OllamaLLM()
    raise ValueError(
        f"Unknown LLM provider: {p!r}. "
        f"Choose from: anthropic, openai, ollama"
    )
