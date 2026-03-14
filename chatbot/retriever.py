"""
Hybrid retriever — BM25 + Semantic + Reciprocal Rank Fusion (RRF)

How it works:
  1. Semantic search  — embed the query, find nearest vectors in ChromaDB (dense)
  2. BM25 search      — score all chunks by keyword overlap (sparse)
  3. RRF merge        — combine both ranked lists into one final ranking

Why two methods?
  Semantic catches paraphrasing and conceptual similarity.
  BM25 catches exact keyword matches that embeddings can miss.
  Neither alone is best — RRF gives you the benefits of both.

RRF formula:
  score(chunk) = 1/(rank_in_semantic + K) + 1/(rank_in_bm25 + K)
  K=60 is a constant that dampens the influence of top ranks.
  A chunk ranked #1 in both lists gets the highest combined score.
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb

ROOT         = Path(__file__).parent.parent
CHROMA_DIR   = str(ROOT / "output/chroma_db")
COLLECTION   = "documents"

# How many candidates each retriever fetches before RRF merges them.
# More candidates = better recall but slower. 20 per method is a good default.
CANDIDATES_K = 20

# RRF damping constant. 60 is the standard value from the original paper.
# Higher K = less weight on top ranks = more democratic merging.
RRF_K        = 60


def _rrf_merge(
    semantic_hits: list[dict],
    bm25_hits:     list[dict],
    top_k:         int,
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.

    Each hit must have a unique "id" field.
    The returned list is sorted by combined RRF score, descending.
    """
    scores  = {}   # id → cumulative RRF score
    payload = {}   # id → chunk data (for building the return value)

    for rank, hit in enumerate(semantic_hits):
        cid = hit["id"]
        scores[cid]  = scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        payload[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        scores[cid]  = scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        payload[cid] = hit   # safe — same data regardless of which list it came from

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {**payload[cid], "rrf_score": round(score, 6)}
        for cid, score in ranked
    ]


def retrieve(
    question:   str,
    collection,
    model:      SentenceTransformer,
    bm25:       BM25Okapi,
    store:      list[dict],           # parallel list to the BM25 index
    top_k:      int = 5,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 + semantic → RRF merge → top_k results.

    'store' is a list of chunk dicts built at load time (same order as BM25 index).
    Each dict has: id, source, chunk_index, text.
    """
    # ── 1. Semantic search ────────────────────────────────────────
    query_vector     = model.encode(question).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(CANDIDATES_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = [
        {
            "id":          f"{meta['source']}::chunk_{meta['chunk_index']}",
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance":    round(dist, 4),
            "text":        text,
        }
        for text, meta, dist in zip(
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
            semantic_results["distances"][0],
        )
    ]

    # ── 2. BM25 search ────────────────────────────────────────────
    # Tokenise the same way as at index time (simple whitespace split)
    tokenized_query = question.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    # Pair each score with its store entry, sort descending, take top candidates
    scored = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_hits = [
        {
            "id":          store[i]["id"],
            "source":      store[i]["source"],
            "chunk_index": store[i]["chunk_index"],
            "distance":    None,           # BM25 doesn't have a distance — use rrf_score
            "text":        store[i]["text"],
            "bm25_score":  round(score, 4),
        }
        for i, score in scored[:CANDIDATES_K]
        if score > 0   # skip chunks with zero keyword overlap
    ]

    # ── 3. RRF merge ─────────────────────────────────────────────
    return _rrf_merge(semantic_hits, bm25_hits, top_k)


def load_retriever():
    """
    Load everything needed for hybrid retrieval. Call once at startup.

    Returns:
      model      — SentenceTransformer for query encoding
      collection — ChromaDB collection for semantic search
      bm25       — BM25Okapi index built over all chunk texts
      store      — list of chunk dicts parallel to the BM25 index
    """
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION)

    # Fetch ALL chunks from ChromaDB to build the BM25 index.
    # This is a one-time cost at startup — BM25 lives in memory.
    print("Building BM25 index over all chunks...")
    all_data = collection.get(include=["documents", "metadatas"])

    store = [
        {
            "id":          f"{meta['source']}::chunk_{meta['chunk_index']}",
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "text":        text,
        }
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    # BM25 tokenises on whitespace — same as query time
    bm25 = BM25Okapi([chunk["text"].lower().split() for chunk in store])
    print(f"BM25 index ready. {len(store)} chunks.")

    return model, collection, bm25, store
