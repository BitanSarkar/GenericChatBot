"""
Contextual Retrieval — Anthropic technique (Oct 2024)

For each chunk, calls the local LLM with:
  - the full source document (truncated to fit context window)
  - the chunk itself

The LLM writes 1-2 sentences situating the chunk within the document.
That context is prepended to the chunk text before embedding.

Why this helps:
  Without context, a chunk like "It delegates to the parent using super."
  has no idea it's about Java inheritance. The embedding is blind to origin.
  With context prepended, the embedding captures both the chunk AND where it
  came from — dramatically improving retrieval recall.

Parallelism:
  Each chunk's LLM call is a blocking HTTP request — pure I/O wait.
  ThreadPoolExecutor lets multiple calls be in-flight simultaneously.
  Start Ollama with OLLAMA_NUM_PARALLEL=4 for true concurrent inference:
      OLLAMA_NUM_PARALLEL=4 ollama serve

Incremental re-indexing (cache):
  Each chunk is hashed (MD5 of its raw text). The hash → context mapping
  is persisted to output/context_cache.json. On re-runs, cached chunks
  skip the LLM call entirely — only new/changed chunks pay the cost.

  This means adding one new document only contextualizes new chunks,
  not the 567 you already paid for.
"""

import json
import hashlib
import threading
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from shared.redis_client import get_redis, CACHE_KEY

OLLAMA_URL         = "http://localhost:11434/api/generate"
LLM_MODEL          = "qwen2.5:0.5b"
NUM_CTX            = 2048  # smaller KV cache = faster scheduling + less memory pressure
WORKERS            = 4     # parallel Ollama calls; match OLLAMA_NUM_PARALLEL

# Pre-truncate documents to this many words before passing to LLM.
# Done ONCE per document (not per chunk) to avoid repeated split() on large PDFs.
# With NUM_CTX=2048: ~1500 token budget for the doc → ~1000 words is safe.
DOC_TRUNCATE_WORDS = 1000


def _chunk_hash(chunk_text: str) -> str:
    """MD5 of the raw chunk text — used as cache key."""
    return hashlib.md5(chunk_text.encode()).hexdigest()


def _load_cache() -> dict:
    """Load entire context cache from Redis Hash into memory (fast HGETALL)."""
    return get_redis().hgetall(CACHE_KEY)


def _save_cache(cache: dict) -> None:
    """Persist the in-memory cache dict back to Redis Hash."""
    if cache:
        get_redis().hset(CACHE_KEY, mapping=cache)


def _build_context_prompt(document_text: str, chunk_text: str) -> str:
    # document_text is already truncated before this is called
    return f"""<document>
{document_text}
</document>

Here is a chunk from this document:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences that situate this chunk within the overall document. \
Mention the topic, section, or concept this chunk belongs to. \
Do not summarize the chunk itself — only provide context that would help \
someone searching for this information. Answer with only the context sentences."""


def _call_ollama(prompt: str) -> str:
    """Non-streaming Ollama call. Returns the generated text."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":   LLM_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "num_predict": 50,    # 1-2 sentences ≈ 30-40 tokens; 50 is a safe cap
                "num_ctx":     NUM_CTX,
                "temperature": 0.0,   # deterministic — same doc/chunk → same context
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _process_chunk(args: tuple) -> tuple[int, dict, str | None]:
    """
    Worker function — runs in a thread.

    Returns (original_index, enriched_chunk, new_cache_entry | None).
    new_cache_entry is (hash, context) when the LLM was called, None on cache hit.
    """
    idx, chunk, doc_text, cache = args
    chunk_text  = chunk["text"]
    chunk_hash  = _chunk_hash(chunk_text)

    if chunk_hash in cache:
        # Cache hit — reuse stored context, no LLM call
        context = cache[chunk_hash]
        chunk["original_text"] = chunk_text
        chunk["text"]          = f"{context}\n---\n{chunk_text}"
        return idx, chunk, None   # None = no new cache entry to write

    try:
        prompt  = _build_context_prompt(doc_text, chunk_text)
        context = _call_ollama(prompt)
        chunk["original_text"] = chunk_text
        chunk["text"]          = f"{context}\n---\n{chunk_text}"
        return idx, chunk, (chunk_hash, context)   # caller will write to cache
    except Exception as e:
        chunk["original_text"]  = chunk_text
        chunk["_context_error"] = str(e)
        return idx, chunk, None


def contextualize_chunks(
    chunks: list[dict],
    documents: list[dict],
    workers: int = WORKERS,
) -> list[dict]:
    """
    Enriches each chunk's 'text' field with LLM-generated context.

    Cache-aware: chunks whose hash exists in output/context_cache.json are
    skipped — their stored context is reused directly. Only new/changed chunks
    trigger an LLM call.

    Input:  chunks from chunker.py  +  documents from loader.py
    Output: same chunks (same order), with chunk['text'] enriched.
    """
    cache      = _load_cache()
    cache_lock = threading.Lock()   # guards writes from multiple threads

    # Pre-truncate each document ONCE — not once per chunk
    def _truncate(text: str) -> str:
        words = text.split()
        if len(words) > DOC_TRUNCATE_WORDS:
            return " ".join(words[:DOC_TRUNCATE_WORDS]) + "\n[document truncated...]"
        return text

    doc_lookup = {doc["source"]: _truncate(doc["text"]) for doc in documents}
    total      = len(chunks)

    # Count how many will be cache hits before we start
    hits = sum(1 for c in chunks if _chunk_hash(c["text"]) in cache)
    misses = total - hits
    print(f"Contextualizing {total} chunks  [workers={workers}]")
    print(f"  Cache hits : {hits}  (skipping LLM)")
    print(f"  LLM calls  : {misses}  (new/changed chunks)\n")

    work_items = [
        (i, chunk, doc_lookup.get(chunk["source"], ""), cache)
        for i, chunk in enumerate(chunks)
    ]

    results    = [None] * total
    completed  = 0
    failed     = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_process_chunk, item): item[0] for item in work_items}

        for future in as_completed(future_map):
            idx, enriched_chunk, cache_entry = future.result()
            results[idx] = enriched_chunk

            # Write new cache entry under lock — only one thread at a time
            if cache_entry:
                h, ctx = cache_entry
                with cache_lock:
                    cache[h] = ctx

            completed += 1
            error = enriched_chunk.get("_context_error")
            if error:
                failed += 1

            hit_marker = "[cache]" if cache_entry is None and not error else "[llm]  "
            with print_lock:
                status = enriched_chunk["text"][:80] if not error else f"[FAILED: {error}]"
                print(f"  [{completed:>4}/{total}] {hit_marker}  chunk {enriched_chunk['chunk_index']}  →  {status}")

    # Persist updated cache to Redis
    _save_cache(cache)
    print(f"\nCache saved → Redis:{CACHE_KEY}  ({len(cache)} entries)")

    print(f"Done. {total - failed}/{total} chunks enriched.")
    if failed:
        print(f"Warning: {failed} chunks kept as-is (see '_context_error' field).")

    return results


if __name__ == "__main__":
    chunks_path = ROOT / "output/chunks.json"
    docs_path   = ROOT / "output/documents.json"
    output_path = ROOT / "output/contextual_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError("Run chunker.py first to generate output/chunks.json")
    if not docs_path.exists():
        raise FileNotFoundError("Run loader.py first to generate output/documents.json")

    chunks    = json.loads(chunks_path.read_text())
    documents = json.loads(docs_path.read_text())

    print(f"Loaded {len(chunks)} chunks from {len(documents)} documents\n")

    chunks = contextualize_chunks(chunks, documents)

    output_path.write_text(json.dumps(chunks, indent=2))
    print(f"Saved to: {output_path.relative_to(ROOT)}")
