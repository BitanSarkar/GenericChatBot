"""
GenericChatBot — FastAPI web server

Routes:
  GET    /                  → serve index.html
  GET    /index/status      → { status, message, files, chunk_count }
  POST   /upload            → save files to refs/, trigger background re-index
  POST   /chat/stream       → SSE stream: token* → sources → done
  GET    /history           → full conversation history
  DELETE /history           → clear conversation history

SSE event types (all carry a JSON data payload):
  token   { text: "..." }
  sources { sources: [{source, chunk_index, reranker_score, rrf_score}] }
  error   { message: "..." }
  done    {}
"""

import json
import sys
import shutil
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, BackgroundTasks, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.retriever   import load_retriever, retrieve
from chatbot.generator   import build_prompt, stream_generate
from chatbot.memory      import add_turn, load_memory, format_for_prompt, clear_memory
from shared.redis_client import get_redis, INDEX_STATUS_KEY, INDEX_MESSAGE_KEY

REFS_DIR   = ROOT / "refs"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = Path(__file__).parent / "static"
REFS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="GenericChatBot")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Retriever singleton ───────────────────────────────────────────────────────
# Loaded once at startup (heavy: SentenceTransformer + ChromaDB + BM25 + CrossEncoder)
# Replaced atomically after each successful re-index.
_retriever = None   # tuple: (model, collection, bm25, store, reranker) | None


def _try_load_retriever() -> str | None:
    """Try to load retriever components. Returns error string or None on success."""
    global _retriever
    try:
        _retriever = load_retriever()
        return None
    except Exception as e:
        return str(e)


@app.on_event("startup")
async def _startup():
    r = get_redis()
    # Don't overwrite status if indexing survived a restart
    if r.get(INDEX_STATUS_KEY) == "running":
        r.set(INDEX_STATUS_KEY, "error")
        r.set(INDEX_MESSAGE_KEY, "Server restarted during indexing. Re-upload to try again.")

    err = _try_load_retriever()
    if err:
        if not r.exists(INDEX_STATUS_KEY):
            r.set(INDEX_STATUS_KEY, "idle")
            r.set(INDEX_MESSAGE_KEY, "No index yet — upload documents to begin.")
    else:
        r.set(INDEX_STATUS_KEY, "idle")
        r.set(INDEX_MESSAGE_KEY, "Ready.")


# ── Index status ──────────────────────────────────────────────────────────────

@app.get("/index/status")
def index_status():
    r = get_redis()
    files = sorted(
        str(p.relative_to(REFS_DIR))
        for p in REFS_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    chunk_count = 0
    if _retriever is not None:
        try:
            chunk_count = _retriever[1].count()   # collection.count()
        except Exception:
            pass
    return {
        "status":      r.get(INDEX_STATUS_KEY) or "idle",
        "message":     r.get(INDEX_MESSAGE_KEY) or "",
        "files":       files,
        "chunk_count": chunk_count,
    }


# ── Background indexer ────────────────────────────────────────────────────────

def _run_indexer():
    """
    Full re-index pipeline. Runs in a FastAPI BackgroundTask thread.

    Steps mirror indexer/run.py exactly — same functions, same outputs.
    Uses Redis for status updates so the UI can poll progress.
    """
    r = get_redis()
    try:
        r.set(INDEX_STATUS_KEY, "running")

        # Step 1 — Load
        r.set(INDEX_MESSAGE_KEY, "Step 1/4 — Loading documents from refs/...")
        from indexer.loader import load_all
        docs = load_all()
        (OUTPUT_DIR / "documents.json").write_text(json.dumps(docs, indent=2))

        # Step 2 — Chunk
        r.set(INDEX_MESSAGE_KEY, f"Step 2/4 — Chunking {len(docs)} document(s)...")
        from indexer.chunker import chunk_all
        chunks = chunk_all(docs)
        (OUTPUT_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2))

        # Step 2.5 — Contextualize (LLM, cache-aware)
        r.set(INDEX_MESSAGE_KEY, f"Step 3/4 — Contextualizing {len(chunks)} chunks (LLM, cache-aware)...")
        from indexer.contextualizer import contextualize_chunks
        chunks = contextualize_chunks(chunks, docs)
        (OUTPUT_DIR / "contextual_chunks.json").write_text(json.dumps(chunks, indent=2))

        # Step 3 — Embed
        r.set(INDEX_MESSAGE_KEY, f"Step 4/4 — Embedding {len(chunks)} chunks...")
        from sentence_transformers import SentenceTransformer
        from indexer.embedder import embed_chunks
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = embed_chunks(chunks, model)
        (OUTPUT_DIR / "embeddings.json").write_text(json.dumps(chunks, indent=2))

        # Step 4 — Store in ChromaDB
        r.set(INDEX_MESSAGE_KEY, "Storing vectors in ChromaDB...")
        from indexer.vectorstore import store as chroma_store
        count = chroma_store(chunks)

        # Reload retriever with new index
        r.set(INDEX_MESSAGE_KEY, "Loading retriever components...")
        err = _try_load_retriever()
        if err:
            raise RuntimeError(f"Retriever reload failed: {err}")

        r.set(INDEX_STATUS_KEY, "idle")
        r.set(INDEX_MESSAGE_KEY, f"Ready. {count} chunks indexed from {len(docs)} document(s).")

    except Exception as e:
        r.set(INDEX_STATUS_KEY, "error")
        r.set(INDEX_MESSAGE_KEY, f"Indexing failed: {e}")


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    r = get_redis()
    if r.get(INDEX_STATUS_KEY) == "running":
        raise HTTPException(status_code=409, detail="Indexing already in progress. Please wait.")

    saved = []
    for file in files:
        # webkitRelativePath is sent as the filename when uploading folders from the browser
        relative = file.filename or "unknown"
        dest = REFS_DIR / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(relative)

    background_tasks.add_task(_run_indexer)
    return {"saved": saved, "status": "indexing_started"}


# ── Chat (SSE) ────────────────────────────────────────────────────────────────

def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _chat_stream(question: str, mode: str) -> AsyncGenerator[str, None]:
    """
    Core chat generator — yields SSE-formatted strings.

    Flow:
      1. Retrieve relevant chunks (unless public mode)
      2. Build prompt with history + context
      3. Stream tokens from Ollama → SSE token events
      4. Persist both turns to Redis memory
      5. Send sources → done
    """
    # Guard: retriever required for private/hybrid
    if mode != "public" and _retriever is None:
        yield _sse("error", json.dumps({
            "message": "No documents indexed yet. Upload files first, or switch to Public mode."
        }))
        return

    matches = []
    if mode != "public":
        model, collection, bm25, store, reranker = _retriever
        matches = retrieve(question, collection, model, bm25, store, reranker, top_k=5)
        if not matches and mode == "private":
            yield _sse("error", json.dumps({
                "message": "This information is not in the indexed documents."
            }))
            return

    history_str = format_for_prompt()
    prompt      = build_prompt(question, matches, history_str, mode=mode)

    # Stream tokens
    full_answer = []
    try:
        for token in stream_generate(prompt):
            full_answer.append(token)
            yield _sse("token", json.dumps({"text": token}))
    except Exception as e:
        yield _sse("error", json.dumps({"message": f"Generation failed: {e}"}))
        return

    answer = "".join(full_answer)

    # Persist to Redis memory
    add_turn("user",      question)
    add_turn("assistant", answer)

    # Sources
    sources = [
        {
            "source":         m["source"],
            "chunk_index":    m["chunk_index"],
            "reranker_score": m["reranker_score"],
            "rrf_score":      m["rrf_score"],
        }
        for m in matches
    ]
    yield _sse("sources", json.dumps({"sources": sources}))
    yield _sse("done", "{}")


@app.post("/chat/stream")
async def chat_stream(request: Request):
    body     = await request.json()
    question = (body.get("question") or "").strip()
    mode     = body.get("mode", "hybrid")

    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if mode not in ("public", "private", "hybrid"):
        raise HTTPException(status_code=400, detail="mode must be public, private, or hybrid")

    return StreamingResponse(
        _chat_stream(question, mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if behind a proxy
        },
    )


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/history")
def get_history():
    return {"history": load_memory()}


@app.delete("/history")
def delete_history():
    clear_memory()
    return {"status": "cleared"}


# ── Root → UI ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
