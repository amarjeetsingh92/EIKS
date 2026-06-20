"""
cli.py — Command-Line Interface
=================================
A terminal interface for the RAG pipeline.
Use this to ingest documents and ask questions without running the frontend.

USAGE:
    # Index a PDF, a scanned document, and a URL:
    python cli.py ingest report.pdf scan.pdf https://example.com/article

    # Ask a question:
    python cli.py query "What are the payment terms in the contract?"

    # Ask with more sources (default is 5):
    python cli.py query "Summarise the findings" --top-k 8

    # Show index statistics:
    python cli.py stats

    # Wipe the index:
    python cli.py clear

LIBRARIES USED:
    argparse — Python's built-in argument parser (parses "ingest", "query", etc.)
    rich     — Pretty terminal output with colours and tables
"""

import sys
import time
import argparse

# Rich is a Python library for beautiful terminal formatting.
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel

# Make sure Python can find the backend package when this script is run directly.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from backend.rag.pipeline import get_pipeline

# Create a single Console instance for all output.
console = Console()


# ===========================================================================
# Command handlers — one function per sub-command
# ===========================================================================

def cmd_ingest(args):
    """
    Handler for: python cli.py ingest <source1> <source2> ...

    args.sources is a list of file paths and/or URLs.
    """
    pipeline = get_pipeline()

    console.print(f"\n[bold cyan]Indexing {len(args.sources)} source(s)...[/]")
    console.print("[dim]This may take a while for large files or slow OCR.[/]\n")

    # Run the full ingestion pipeline.
    stats = pipeline.index_sources(args.sources)

    # Display results as a formatted table.
    table = Table(
        title="Ingest Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric",   style="dim", width=20)
    table.add_column("Value",    justify="right")

    table.add_row("Documents indexed", str(stats.total_documents))
    table.add_row("Chunks created",    str(stats.total_chunks))
    table.add_row("Sources skipped",   str(stats.skipped))
    table.add_row("Errors",            str(len(stats.errors)))
    table.add_row("Duration",          f"{stats.duration_s}s")

    console.print(table)

    # Print any errors in red.
    if stats.errors:
        console.print("\n[bold red]Errors:[/]")
        for err in stats.errors:
            console.print(f"  [red]✗ {err}[/]")


def cmd_query(args):
    """
    Handler for: python cli.py query "<question>" [--top-k N]
    """
    pipeline = get_pipeline()

    console.print(f"\n[bold cyan]Querying:[/] {args.question!r}\n")

    # Run the full RAG pipeline for this question.
    result = pipeline.query(args.question, top_k=args.top_k)

    # Display the answer in a green panel.
    console.print(Panel(
        result.answer,
        title="[bold green]Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # Show metadata below the answer.
    console.print(
        f"\n[dim]Model: {result.model}  |  "
        f"Latency: {result.latency_ms:.0f}ms  |  "
        f"Sources used: {len(result.sources)}[/]\n"
    )

    # List the source chunks used.
    if result.sources:
        console.print("[bold]Sources:[/]")
        for s in result.sources:
            # Prefer filename > URL > raw path for the source label.
            origin = (
                s.chunk.metadata.get("filename")
                or s.chunk.metadata.get("url")
                or s.chunk.origin
            )
            # Show the source label and similarity score.
            console.print(f"  [{s.rank+1}] [cyan]{origin}[/]  "
                          f"[dim](similarity: {s.score:.3f})[/]")
            # Show a preview of the chunk text (first 120 chars).
            preview = s.chunk.content[:120].replace("\n", " ")
            console.print(f"      [dim]{preview}...[/]")


def cmd_stats(args):
    """
    Handler for: python cli.py stats
    Shows index statistics and current configuration.
    """
    pipeline = get_pipeline()

    from backend.config import settings

    table = Table(title="Index Statistics", show_header=False)
    table.add_column("Setting", style="bold dim", width=22)
    table.add_column("Value")

    table.add_row("Chunks indexed",   str(pipeline.count()))
    table.add_row("Vector backend",   settings.vector_backend)
    table.add_row("Embedding model",  settings.embedding_model)
    table.add_row("LLM provider",     settings.llm_provider)
    table.add_row("Anthropic model",  settings.anthropic_model)
    table.add_row("Chunking",         settings.chunking_strategy)
    table.add_row("Chunk size",       str(settings.chunk_size))

    console.print(table)


def cmd_clear(args):
    """
    Handler for: python cli.py clear
    Wipes the entire vector index.
    """
    # Safety confirmation — this is irreversible.
    console.print("[yellow]Warning: This will delete all indexed chunks.[/]")
    confirm = input("Type 'yes' to confirm: ").strip().lower()

    if confirm == "yes":
        pipeline = get_pipeline()
        pipeline.clear_index()
        console.print("[bold yellow]Index cleared.[/]")
    else:
        console.print("[dim]Cancelled.[/]")


# ===========================================================================
# Argument parser setup
# ===========================================================================

def main():
    """
    Parse command-line arguments and dispatch to the right handler.

    argparse handles:
      - Showing help text (python cli.py --help)
      - Validating required arguments
      - Type conversion (--top-k 8 → int 8)
    """
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Knowledge Retrieval Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest report.pdf scan.pdf https://example.com
  python cli.py query "What are the key findings?"
  python cli.py query "Summarise the methodology" --top-k 8
  python cli.py stats
  python cli.py clear
        """,
    )

    # Sub-commands: ingest, query, stats, clear
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser(
        "ingest",
        help="Index files or URLs into the vector store",
    )
    p_ingest.add_argument(
        "sources",
        nargs="+",   # One or more
        help="File paths (PDF, PNG, JPG) or URLs to index",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser(
        "query",
        help="Ask a question and get a cited answer",
    )
    p_query.add_argument(
        "question",
        help='The question to answer, e.g. "What are the payment terms?"',
    )
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of source chunks to retrieve (default: 5)",
    )
    p_query.set_defaults(func=cmd_query)

    # ── stats ─────────────────────────────────────────────────────────────────
    p_stats = sub.add_parser(
        "stats",
        help="Show index statistics and configuration",
    )
    p_stats.set_defaults(func=cmd_stats)

    # ── clear ─────────────────────────────────────────────────────────────────
    p_clear = sub.add_parser(
        "clear",
        help="Wipe the entire index (irreversible)",
    )
    p_clear.set_defaults(func=cmd_clear)

    # Parse the arguments and call the appropriate handler function.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
