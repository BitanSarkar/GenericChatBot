from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

ROOT       = Path(__file__).parent.parent
CHROMA_DIR = str(ROOT / "output/chroma_db")
COLLECTION = "documents"
TOP_K      = 20


def retrieve(question: str, collection, model: SentenceTransformer, top_k: int = TOP_K) -> list[dict]:
    query_vector = model.encode(question).tolist()

    results   = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for text, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        matches.append({
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance":    round(distance, 4),
            "text":        text,
        })
    return matches


def load_retriever():
    """Load model and ChromaDB collection. Call once at startup."""
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION)
    return model, collection
