# GenericChatBot

A production-grade RAG (Retrieval-Augmented Generation) chatbot built entirely on local, open-source tooling. No cloud APIs required. Built as a learning project to deeply understand how modern RAG pipelines work — from chunking strategy to hybrid retrieval to cross-encoder re-ranking.

---

## What This Is

An interactive CLI chatbot that answers questions over your own documents. Drop PDFs or text files into `refs/`, run the indexer once, then chat. The retrieval pipeline implements the full modern RAG stack:

- **Contextual Retrieval** (Anthropic, 2024) — chunks enriched with LLM-generated context before embedding
- **Hybrid Search** — BM25 (keyword) + semantic (vector) combined via Reciprocal Rank Fusion
- **Cross-encoder Re-ranking** — final precision pass over retrieved candidates
- **Three retrieval modes** — public (LLM only), private (documents only), hybrid (both)
- **Persistent conversation memory** — full history across sessions

---

## Architecture

### Indexing Pipeline (run once)

```
refs/*.pdf / *.txt
       │
       ▼
 ┌─────────────┐
 │   loader    │  Extract text from PDFs and text files
 └──────┬──────┘
        │  documents.json
        ▼
 ┌─────────────┐
 │   chunker   │  Fixed-size sliding window (200 words, 40 overlap)
 └──────┬──────┘
        │  chunks.json
        ▼
 ┌──────────────────┐
 │  contextualizer  │  LLM generates 1-2 context sentences per chunk
 │  (qwen2.5:0.5b)  │  Prepended to chunk text before embedding
 │  + cache         │  MD5 hash cache — only new chunks pay LLM cost
 └──────┬───────────┘
        │  contextual_chunks.json
        ▼
 ┌─────────────┐
 │   embedder  │  all-MiniLM-L6-v2 → 384-dim vectors
 └──────┬──────┘
        │  embeddings.json
        ▼
 ┌─────────────┐
 │ vectorstore │  ChromaDB (persistent SQLite)
 └─────────────┘
        │
        ▼
  output/chroma_db/
```

### Query Pipeline (every turn)

```
User question
      │
      ├──────────────────────┬────────────────────────┐
      ▼                      ▼                        │
 Semantic search         BM25 search                  │
 (ChromaDB + embedding)  (keyword overlap)            │
 top-20 candidates       top-20 candidates            │
      │                      │                        │
      └──────────┬───────────┘                        │
                 ▼                                    │
           RRF merge                                  │
      (Reciprocal Rank Fusion)                        │
           top-20 candidates                          │
                 │                                    │
                 ▼                                    │
      Cross-encoder re-ranker                         │
      (ms-marco-MiniLM-L-6-v2)                        │
      scores each (query, chunk) pair jointly         │
      threshold gate: score < -3.0 → reject           │
           top-5 final chunks                         │
                 │                                    │
                 └──────────────┬─────────────────────┘
                                ▼
                      Prompt construction
                      (mode-aware: public / private / hybrid)
                                │
                                ▼
                      Ollama: dolphin-llama3
                      (streamed response)
                                │
                                ▼
                      output/memory.json
                      (full conversation persisted)
```

---

## Retrieval Modes

| Mode | Retrieval | LLM Behaviour | Use When |
|---|---|---|---|
| `public` | None | Completely unrestricted — answers from training knowledge only | General/public topics. Model knows it better than your docs. |
| `private` | Full pipeline | Strict — ONLY the indexed documents. Refuses if not found. | Internal docs, private data, anything post training cutoff. |
| `hybrid` | Full pipeline | Free — uses both context and training knowledge equally | Best of both worlds. Default mode. |

### Relevance Threshold (private + hybrid)

The cross-encoder scores each `(query, chunk)` pair. Scores from the MS-MARCO model:
- `> 0` — genuinely relevant
- `-3 to 0` — weakly related
- `< -3` — not relevant (threshold gate)

When the best chunk scores below `-3.0`, retrieval returns nothing. In `private` mode this means an immediate "not in documents" response — the LLM never sees the question.

---

## Why Each Technique

### Contextual Chunks
Standard chunking strips context at boundaries. A chunk like *"It delegates to the parent using super."* has no idea it's about Java inheritance — the embedding is blind to origin. Prepending LLM-generated context (*"This excerpt is from Chapter 5 on Inheritance..."*) makes the embedding capture both the concept and where it came from.

### Hybrid Search (BM25 + Semantic)
Pure semantic search misses exact keyword matches. BM25 finds `AbstractBeanFactory` exactly; semantic finds related concepts. Neither alone is best. RRF merges both ranked lists without needing to normalise scores (which would be comparing apples to oranges).

**RRF formula:** `score = 1/(rank_semantic + 60) + 1/(rank_bm25 + 60)`

### Cross-encoder Re-ranking
Bi-encoders encode query and chunk independently — the relevance signal *between* them is lost. A cross-encoder feeds both together through a transformer, so every query token can attend to every chunk token. Far more accurate, but too slow to run on all chunks. Solution: run it only on the 20 RRF candidates (~400ms extra per query).

---

## Project Structure

```
GenericChatBot/
├── indexer/
│   ├── run.py              # Orchestrates full indexing pipeline
│   ├── loader.py           # Load documents from refs/
│   ├── chunker.py          # Fixed-size sliding window chunking
│   ├── contextualizer.py   # LLM context enrichment + cache
│   ├── embedder.py         # Sentence-transformer embeddings
│   └── vectorstore.py      # ChromaDB storage
│
├── chatbot/
│   ├── main.py             # CLI entrypoint + conversation loop
│   ├── retriever.py        # Hybrid retrieval + RRF + re-ranking
│   ├── generator.py        # Prompt construction + Ollama generation
│   └── memory.py           # Conversation history (short + long term)
│
├── refs/                   # Drop your documents here
│   └── .refignore          # Exclude patterns (like .gitignore)
│
├── output/                 # Generated — do not edit manually
│   ├── documents.json
│   ├── chunks.json
│   ├── contextual_chunks.json
│   ├── embeddings.json
│   ├── context_cache.json  # Contextualizer cache (hash → context)
│   ├── memory.json         # Conversation history
│   └── chroma_db/          # ChromaDB vector store
│
└── config.yaml             # ChromaDB server config (optional)
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install chromadb sentence-transformers rank-bm25 pypdf requests
```

### Pull required Ollama models

```bash
ollama pull dolphin-llama3      # main chat LLM
ollama pull qwen2.5:0.5b        # contextualizer (lightweight, fast)
```

### Start Ollama with parallel inference

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

---

## Usage

### 1. Add your documents

Drop PDFs or text files into `refs/`. Supported formats: `.pdf`, `.txt`, `.md`, `.py`, `.java`, `.ts`, `.js`, `.html`, `.json`, `.yaml`, `.xml`

Use `.refignore` to exclude files (same syntax as `.gitignore`).

### 2. Index

```bash
cd GenericChatBot
source .venv/bin/activate
PYTHONUNBUFFERED=1 python -u indexer/run.py
```

This runs once (or when documents change). The contextualizer step takes the longest — one LLM call per chunk. Subsequent runs use the cache and are near-instant for unchanged chunks.

### 3. Chat

```bash
# Default: hybrid mode
python chatbot/main.py

# LLM answers from its own training knowledge only (no retrieval)
python chatbot/main.py --mode public

# Strict: only answer from your indexed documents
python chatbot/main.py --mode private

# Best of both worlds: context + LLM knowledge combined
python chatbot/main.py --mode hybrid
```

Type your question and press Enter. Type `exit` to quit (conversation saved automatically).

---

## Configuration

Key constants you may want to tune:

| File | Constant | Default | Effect |
|---|---|---|---|
| `contextualizer.py` | `WORKERS` | `4` | Parallel Ollama calls during indexing |
| `contextualizer.py` | `DOC_TRUNCATE_WORDS` | `1000` | Max doc words passed to LLM per chunk |
| `contextualizer.py` | `NUM_CTX` | `2048` | Ollama context window for contextualizer |
| `retriever.py` | `CANDIDATES_K` | `20` | Candidates per retriever before RRF |
| `retriever.py` | `RRF_K` | `60` | RRF damping constant |
| `retriever.py` | `RERANKER_THRESHOLD` | `-3.0` | Min cross-encoder score to consider relevant |
| `chatbot/main.py` | `TOP_K` | `5` | Final chunks injected into LLM prompt |

---

## Re-indexing

The contextualizer caches results by MD5 hash of each chunk's text in `output/context_cache.json`. This means:

- **Add a new document** → only new chunks pay the LLM cost. Existing 567 chunks: instant cache hits.
- **Edit an existing document** → only changed chunks (new hash) are re-contextualised.
- **No document changes** → entire contextualizer step is instant.

ChromaDB uses `upsert` — existing chunk IDs are updated in place, new ones are added.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM | `dolphin-llama3` via Ollama | Local, uncensored, no API cost |
| Contextualizer LLM | `qwen2.5:0.5b` via Ollama | Lightweight — only needs to write 2 sentences |
| Embedding model | `all-MiniLM-L6-v2` | 384D, fast, free, runs locally |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fine-tuned on 1M real search queries |
| Vector DB | ChromaDB | Local, no setup, SQLite-backed |
| Keyword search | `rank-bm25` | BM25Okapi, in-memory, zero config |
| PDF parsing | `pypdf` | Pure Python, no system deps |

---

## Roadmap

- [ ] Dockerize (indexer + chatbot as separate services)
- [ ] REST API wrapper (FastAPI)
- [ ] Web UI
- [ ] Groq API support for faster contextualisation
- [ ] Sentence-window retrieval
- [ ] Query expansion / HyDE
