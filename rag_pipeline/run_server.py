"""
run_server.py — Start the API server
======================================
Run this file to start the HTTP API:

    python run_server.py

The server will be available at:
  http://localhost:8000        — API root
  http://localhost:8000/docs  — Interactive API documentation (Swagger UI)
  http://localhost:8000/redoc — Alternative docs (ReDoc)

WHAT IS uvicorn?
  uvicorn is an ASGI server — the component that handles incoming HTTP
  connections and routes them to your FastAPI app. FastAPI itself is just
  a framework; uvicorn is what actually "listens" on the port.

RELOAD MODE:
  reload=True means uvicorn watches your Python files for changes and
  restarts automatically. Great for development. Turn off in production.
"""

import uvicorn
from backend.config import settings

if __name__ == "__main__":
    print(f"\n🚀  Starting RAG Pipeline API server")
    print(f"   → http://{settings.api_host}:{settings.api_port}")
    print(f"   → Docs: http://localhost:{settings.api_port}/docs")
    print(f"   → LLM: {settings.llm_provider} ({settings.anthropic_model})")
    print(f"   → Vector store: {settings.vector_backend}\n")

    uvicorn.run(
        # "backend.api.server:app" tells uvicorn where to find the FastAPI app.
        # Format: "module.path:variable_name"
        "backend.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        # Auto-reload when files change. Set to False in production.
        reload=True,
        log_level=settings.log_level.lower(),
    )
