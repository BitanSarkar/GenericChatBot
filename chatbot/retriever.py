"""
Hybrid retriever — BM25 + Semantic + RRF + Cross-encoder Re-ranking

Full pipeline per query:
  1. Semantic search   — embed query, find nearest vectors in ChromaDB (dense)
  2. BM25 search       — score all chunks by keyword overlap (sparse)
  3. RRF merge         — combine both ranked lists → top-20 candidates
  4. Cross-encoder     — re-score each (query, chunk) pair jointly → final top-K

Stage 1+2+3 = fast, approximate (bi-encoder: query and chunk encoded separately)
Stage 4      = slow, precise    (cross-encoder: query and chunk encoded together,
                                 full cross-attention between every token pair)

Why does cross-attention win?
  A bi-encoder encodes query and chunk in isolation. The relevance signal
  that lives *between* them — e.g. "doesn't call super()" matching "automatically
  inserts a call to the superclass constructor" — gets lost.
  A cross-encoder sees both simultaneously, so every query token can attend to
  every chunk token. It reasons about relevance, not just similarity.

  Cost: must run on every (query, chunk) pair at query time.
  Solution: only run it on the 20 RRF candidates, not all 567 chunks.
  20 pairs × ~20ms = ~400ms extra — totally acceptable.
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

ROOT         = Path(__file__).parent.parent
CHROMA_DIR   = str(ROOT / "output/chroma_db")
COLLECTION   = "documents"

# Candidates fetched by each retriever before RRF merges them
CANDIDATES_K  = 20

# RRF damping constant (standard value from the 2009 paper)
RRF_K         = 60

# Cross-encoder model — fine-tuned on MS MARCO (1M real search query/passage pairs)
# Outputs a relevance score for each (query, chunk) pair
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum reranker score to consider a chunk relevant.
# MS-MARCO cross-encoder scores: >0 = relevant, <-5 = clearly irrelevant.
# If the best chunk doesn't clear this bar, retrieval returns nothing —
# private mode will then refuse to answer instead of hallucinating.
RERANKER_THRESHOLD = -3.0


def _rrf_merge(
    semantic_hits: list[dict],
    bm25_hits:     list[dict],
    top_k:         int,
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.
    Returns top_k results sorted by combined RRF score, descending.
    """
    scores  = {}
    payload = {}

    for rank, hit in enumerate(semantic_hits):
        cid = hit["id"]
        scores[cid]  = scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        payload[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        scores[cid]  = scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        payload[cid] = hit

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
    store:      list[dict],
    reranker:   CrossEncoder,
    top_k:      int = 5,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
      BM25 + semantic → RRF (top-20) → cross-encoder re-rank → top_k

    The reranker is the final arbiter — its scores override RRF order.
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
    tokenized_query = question.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    scored = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_hits = [
        {
            "id":          store[i]["id"],
            "source":      store[i]["source"],
            "chunk_index": store[i]["chunk_index"],
            "distance":    None,
            "text":        store[i]["text"],
            "bm25_score":  round(score, 4),
        }
        for i, score in scored[:CANDIDATES_K]
        if score > 0
    ]

    # ── 3. RRF merge → top-20 candidates ─────────────────────────
    # We pass CANDIDATES_K here, not top_k — re-ranker will do the final cut
    candidates = _rrf_merge(semantic_hits, bm25_hits, top_k=CANDIDATES_K)

    if not candidates:
        return []

    # ── 4. Cross-encoder re-ranking ───────────────────────────────
    # Feed every (query, chunk_text) pair to the cross-encoder simultaneously.
    # It scores them jointly — full cross-attention between query and chunk tokens.
    # reranker.predict() returns a raw logit per pair; higher = more relevant.
    pairs  = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)   # shape: (len(candidates),)

    # Attach reranker score and sort — this overrides RRF order
    for candidate, score in zip(candidates, scores):
        candidate["reranker_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
    top      = reranked[:top_k]

    # If the best chunk doesn't clear the threshold, nothing is relevant.
    # Return empty list — caller decides how to handle (refuse, fallback, etc.)
    if top and top[0]["reranker_score"] < RERANKER_THRESHOLD:
        return []

    return top


def load_retriever():
    """
    Load all components for hybrid retrieval + re-ranking. Call once at startup.

    Returns:
      model      — SentenceTransformer (bi-encoder for query + index-time embedding)
      collection — ChromaDB collection for semantic search
      bm25       — BM25Okapi index over all chunk texts
      store      — list of chunk dicts parallel to BM25 index
      reranker   — CrossEncoder for final re-ranking
    """
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION)

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

    bm25 = BM25Okapi([chunk["text"].lower().split() for chunk in store])
    print(f"BM25 index ready. {len(store)} chunks.")

    print(f"Loading re-ranker ({RERANKER_MODEL})...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("Re-ranker ready.")

    return model, collection, bm25, store, reranker
