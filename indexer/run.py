"""
Indexing pipeline — run this once whenever your docs in refs/ change.

Steps:
  1. loader         → output/documents.json
  2. chunker        → output/chunks.json
  2.5. contextualizer → output/contextual_chunks.json   ← NEW
  3. embedder       → output/embeddings.json
  4. vectorstore    → output/chroma_db/
"""
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
(ROOT / "output").mkdir(exist_ok=True)

# ── Step 1: Load ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1/4  Loading documents from refs/")
print("=" * 60)
from loader import load_all
docs = load_all()
(ROOT / "output/documents.json").write_text(json.dumps(docs, indent=2))
print(f"→ {len(docs)} documents saved\n")

# ── Step 2: Chunk ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2/5  Chunking documents")
print("=" * 60)
from chunker import chunk_all
chunks = chunk_all(docs)
(ROOT / "output/chunks.json").write_text(json.dumps(chunks, indent=2))
print(f"→ {len(chunks)} chunks saved\n")

# ── Step 2.5: Contextualize ────────────────────────────────────
print("=" * 60)
print("STEP 2.5/5  Contextualizing chunks (Anthropic technique)")
print("=" * 60)
from contextualizer import contextualize_chunks
chunks = contextualize_chunks(chunks, docs)
(ROOT / "output/contextual_chunks.json").write_text(json.dumps(chunks, indent=2))
print(f"→ contextual chunks saved\n")

# ── Step 3: Embed ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3/5  Embedding chunks")
print("=" * 60)
from sentence_transformers import SentenceTransformer
from embedder import embed_chunks
model  = SentenceTransformer("all-MiniLM-L6-v2")
chunks = embed_chunks(chunks, model)
(ROOT / "output/embeddings.json").write_text(json.dumps(chunks, indent=2))
print(f"→ embeddings saved\n")

# ── Step 4: Store ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4/5  Storing into ChromaDB")
print("=" * 60)
from vectorstore import store
count = store(chunks)
print(f"→ {count} chunks in ChromaDB\n")

print("=" * 60)
print("Indexing complete. You can now run: python chatbot/main.py")
print("=" * 60)
