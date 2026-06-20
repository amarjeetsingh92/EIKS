"""
loaders.py — Document Ingestion Layer
======================================
This file is responsible for reading raw files (PDFs, images, web pages)
and turning them into Document objects with extracted plain text.

SUPPORTED SOURCE TYPES:
  - Text PDFs   → pdfplumber extracts the text layer directly (fast, accurate)
  - Scanned PDFs → pdf2image converts pages to images, then Tesseract OCR reads them
  - Images       → Tesseract OCR reads JPEG/PNG/TIFF directly
  - Web pages    → requests fetches HTML, BeautifulSoup strips boilerplate

HOW THE PIPELINE USES THIS:
  DocumentIngestionPipeline.ingest_one(source)
      → detects source type automatically
      → calls the right loader
      → returns list[Document]

  DocumentIngestionPipeline.ingest_bulk(sources)
      → calls ingest_one for each source
      → collects results and errors
      → returns (list[Document], IngestStats)
"""

from __future__ import annotations
import io
import time
from abc import ABC, abstractmethod   # ABC = Abstract Base Class
from pathlib import Path              # Modern way to work with file paths
from typing import Iterator
from urllib.parse import urlparse     # Parse URL components

from loguru import logger   # Better logging than Python's built-in print/logging

from backend.models import Document, DocumentSource, IngestStats
from backend.config import settings


# ===========================================================================
# BaseLoader — the shared interface all loaders must follow
# ===========================================================================
# ABC (Abstract Base Class) lets us define a "contract": any class that
# inherits from BaseLoader MUST implement a load() method. This makes it
# easy to add new loaders later (e.g. for DOCX files) without changing the
# rest of the code.

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load a source (file path or URL) and return extracted Documents.
        Raises an exception if loading fails.
        Must be implemented by every concrete loader class.
        """


# ===========================================================================
# TextPDFLoader — for PDFs that already have a text layer
# ===========================================================================
# Many PDFs (reports, papers, contracts) are "born digital" — they have a
# hidden text layer you can select and copy. pdfplumber reads this layer
# directly without needing OCR. This is much faster and more accurate.

class TextPDFLoader(BaseLoader):
    """
    Extracts the text layer from a standard (non-scanned) PDF.
    Uses the pdfplumber library.

    When to use: PDFs created by Word, Google Docs, LaTeX, etc.
    NOT for: scanned PDFs (photographs of paper). Use ScannedPDFLoader instead.
    """

    def load(self, source: str) -> list[Document]:
        # Try importing pdfplumber — give a helpful error if it's missing.
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Run: pip install pdfplumber")

        path = Path(source)
        pages_text: list[str] = []

        # Open the PDF — pdfplumber handles the low-level PDF parsing.
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                # extract_text() returns None if a page has no text layer.
                text = page.extract_text() or ""
                pages_text.append(text.strip())

        # Join all pages with a blank line between them.
        # We filter out empty pages (e.g. blank separator pages).
        full_text = "\n\n".join(p for p in pages_text if p)

        # If we got nothing, the PDF is probably a scan — tell the user.
        if not full_text.strip():
            logger.warning(
                f"[TextPDF] No text layer found in {path.name}. "
                f"Try ScannedPDFLoader for OCR."
            )
            return []

        # Return a single Document containing the whole PDF's text.
        return [Document(
            source=DocumentSource.PDF_TEXT,
            origin=str(path.resolve()),   # Absolute path for reliable citations
            content=full_text,
            page_count=page_count,
            metadata={"filename": path.name, "pages": page_count},
        )]


# ===========================================================================
# ScannedPDFLoader — for scanned PDFs (photos of paper)
# ===========================================================================
# A "scanned PDF" is just a PDF full of images — it has no text layer.
# To extract text, we must:
#   1. Convert each page image to a Pillow Image object (pdf2image)
#   2. Feed each image to Tesseract OCR (pytesseract)
#   3. Collect the OCR output from every page
#
# This is much slower than TextPDFLoader but works on any scan.
# PREREQUISITE: Tesseract must be installed on your system (see README).

class ScannedPDFLoader(BaseLoader):
    """
    Extracts text from scanned PDFs using OCR (Optical Character Recognition).
    Requires: tesseract system binary + pdf2image + pytesseract Python packages.

    When to use: PDFs created by scanning paper documents.
    Quality depends on scan resolution — 300 DPI produces good results.
    """

    def load(self, source: str) -> list[Document]:
        try:
            from pdf2image import convert_from_path  # Converts PDF pages to images
            import pytesseract                        # Python wrapper for Tesseract
        except ImportError:
            raise ImportError("Run: pip install pdf2image pytesseract")

        # Tell pytesseract where the tesseract binary is.
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

        path = Path(source)
        logger.info(f"[ScannedPDF] Running OCR on {path.name}...")

        # Step 1: Convert PDF pages to PIL Images at the configured DPI.
        # Higher DPI = bigger images = slower but more accurate OCR.
        images = convert_from_path(str(path), dpi=settings.ocr_dpi)

        pages_text: list[str] = []
        for i, img in enumerate(images):
            # Step 2: Run Tesseract on each page image.
            # --psm 6 = "Assume a single uniform block of text"
            # Good default for most document pages.
            text = pytesseract.image_to_string(
                img,
                lang=settings.ocr_language,
                config="--psm 6",
            )
            pages_text.append(text.strip())
            logger.debug(f"  Page {i+1}/{len(images)} → {len(text)} chars extracted")

        # Step 3: Join all pages.
        full_text = "\n\n".join(p for p in pages_text if p)

        return [Document(
            source=DocumentSource.PDF_SCANNED,
            origin=str(path.resolve()),
            content=full_text,
            page_count=len(images),
            metadata={
                "filename": path.name,
                "pages": len(images),
                "ocr": True,  # Flag so the frontend can show the OCR badge
            },
        )]


# ===========================================================================
# ImageLoader — for standalone image files (JPEG, PNG, TIFF, etc.)
# ===========================================================================
# Works the same as ScannedPDFLoader but for a single image file.
# Useful for: photographs of documents, screenshots, scanned invoices, etc.

class ImageLoader(BaseLoader):
    """
    Runs Tesseract OCR on a standalone image file to extract text.
    Supports: JPG, JPEG, PNG, TIFF, TIF, BMP, WEBP.

    Tip: For small images (< 1000px on the longest side), we automatically
    upscale before OCR — this significantly improves accuracy on small text.
    """

    # Set of supported file extensions (lowercase).
    SUPPORTED = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

    def load(self, source: str) -> list[Document]:
        try:
            import pytesseract
            from PIL import Image   # Pillow — the standard Python imaging library
        except ImportError:
            raise ImportError("Run: pip install pytesseract Pillow")

        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

        path = Path(source)
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        # Open the image with Pillow.
        img = Image.open(path)
        w, h = img.size

        # Upscale small images. Tesseract struggles with text under ~12px.
        # If the longest dimension is < 1000px, scale up proportionally.
        if max(w, h) < 1000:
            scale = 1000 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
            logger.debug(f"[Image] Upscaled {w}x{h} → {img.size} before OCR")

        # Run OCR. --psm 6 works well for most document images.
        text = pytesseract.image_to_string(
            img,
            lang=settings.ocr_language,
            config="--psm 6",
        )

        return [Document(
            source=DocumentSource.IMAGE,
            origin=str(path.resolve()),
            content=text.strip(),
            page_count=1,
            metadata={
                "filename": path.name,
                "width": w, "height": h,
                "ocr": True,
            },
        )]


# ===========================================================================
# WebLoader — scrapes text from a URL
# ===========================================================================
# Fetches a webpage and strips out navigation, footers, scripts, and other
# boilerplate — keeping only the main readable content.
#
# TECHNIQUE:
#   1. requests.get() downloads the raw HTML
#   2. BeautifulSoup parses the HTML into a tree
#   3. We remove known noise tags (nav, footer, script, etc.)
#   4. We look for a <main> or <article> tag — most modern sites put the
#      main content there
#   5. We extract just the text, collapsing whitespace

class WebLoader(BaseLoader):
    """
    Scrapes readable text from a web URL.
    Strips navigation, footers, ads, and scripts to get the main content.

    Good for: articles, documentation, blog posts, Wikipedia pages.
    Less good for: JavaScript-heavy sites that load content dynamically.
                   (Use playwright-based crawling for those.)
    """

    # HTML tags that usually contain boilerplate, not real content.
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}

    def load(self, source: str) -> list[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup  # HTML parsing library
        except ImportError:
            raise ImportError("Run: pip install requests beautifulsoup4 lxml")

        logger.info(f"[WebLoader] Fetching {source}")

        # Fetch the page. timeout=20 prevents hanging forever on slow sites.
        # User-Agent header makes us look like a real browser.
        resp = requests.get(
            source,
            timeout=20,
            headers={"User-Agent": "RAGPipeline/1.0 (research tool)"},
        )
        resp.raise_for_status()   # Raises an exception if status code is 4xx/5xx

        # Parse the HTML. lxml is faster than Python's built-in html.parser.
        soup = BeautifulSoup(resp.text, "lxml")

        # Step 1: Remove boilerplate tags entirely from the tree.
        for tag in soup(self.SKIP_TAGS):
            tag.decompose()   # decompose() removes the tag and all its children

        # Step 2: Try to find the main content container.
        # Most well-structured pages have a <main> or <article> tag.
        # Fall back to <body> if neither exists.
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            logger.warning(f"[WebLoader] Could not find content in {source}")
            return []

        # Step 3: Extract text, separating elements with newlines.
        text = main.get_text(separator="\n", strip=True)

        # Collapse multiple blank lines into one.
        lines = [line for line in text.splitlines() if line.strip()]
        clean = "\n".join(lines)

        # Use the page title as the document title, fall back to the domain name.
        title = soup.title.string.strip() if soup.title else urlparse(source).netloc

        return [Document(
            source=DocumentSource.WEB,
            origin=source,
            content=clean,
            page_count=1,   # Web pages don't have "pages"
            metadata={"url": source, "title": title},
        )]


# ===========================================================================
# DocumentIngestionPipeline — the main entry point for ingestion
# ===========================================================================
# This class figures out WHICH loader to use for each source, calls it,
# and handles errors gracefully so one bad file doesn't stop the whole job.

class DocumentIngestionPipeline:
    """
    Auto-detecting ingestion pipeline.

    Usage:
        pipeline = DocumentIngestionPipeline()

        # Single file or URL:
        docs = pipeline.ingest_one("report.pdf")
        docs = pipeline.ingest_one("https://example.com/article")

        # Multiple sources at once:
        docs, stats = pipeline.ingest_bulk(["a.pdf", "b.png", "https://..."])
    """

    def __init__(self, auto_ocr_fallback: bool = True):
        """
        auto_ocr_fallback: If True, automatically retries a text PDF with
                           OCR if no text layer is found. Useful when you're
                           not sure whether a PDF is text or scanned.
        """
        self.auto_ocr_fallback = auto_ocr_fallback

        # Map of loader names → loader instances.
        # We instantiate them once here and reuse them across many files.
        self._loaders: dict[str, BaseLoader] = {
            "pdf_text":    TextPDFLoader(),
            "pdf_scanned": ScannedPDFLoader(),
            "image":       ImageLoader(),
            "web":         WebLoader(),
        }

    def _detect(self, source: str) -> str:
        """
        Figure out which loader to use based on the source type.
        Returns a key into self._loaders.
        """
        # URLs always go to the web loader regardless of any file extension.
        if source.startswith("http://") or source.startswith("https://"):
            return "web"

        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            return "pdf_text"           # We'll try text first; OCR is the fallback
        if ext in ImageLoader.SUPPORTED:
            return "image"

        raise ValueError(f"Cannot determine loader for source: {source!r}")

    def ingest_one(self, source: str) -> list[Document]:
        """
        Load a single file or URL. Returns a list of Documents
        (usually just one, but could be more for some source types).

        Raises exceptions on failure — use ingest_bulk for error handling.
        """
        kind   = self._detect(source)
        loader = self._loaders[kind]
        docs   = loader.load(source)

        # Automatic OCR fallback: if the PDF had no text layer, try OCR.
        if kind == "pdf_text" and not docs and self.auto_ocr_fallback:
            logger.info(f"[Ingestion] No text found, retrying with OCR: {source}")
            docs = self._loaders["pdf_scanned"].load(source)

        return docs

    def ingest_bulk(self, sources: list[str]) -> tuple[list[Document], IngestStats]:
        """
        Load multiple sources. Errors are caught per-source so one bad file
        doesn't abort the whole job.

        Returns:
            docs:  All successfully loaded Document objects.
            stats: Summary of what succeeded, was skipped, and failed.
        """
        all_docs: list[Document] = []
        errors:   list[str]      = []
        t0 = time.time()

        for src in sources:
            try:
                docs = self.ingest_one(src)
                all_docs.extend(docs)
                logger.success(f"✓ {src} → {len(docs)} document(s)")
            except Exception as e:
                # Log the error but keep going with the remaining sources.
                logger.error(f"✗ {src}: {e}")
                errors.append(f"{src}: {e}")

        stats = IngestStats(
            total_documents=len(all_docs),
            total_chunks=0,           # Will be filled in after chunking
            skipped=len(sources) - len(all_docs) - len(errors),
            errors=errors,
            duration_s=round(time.time() - t0, 2),
        )
        return all_docs, stats
