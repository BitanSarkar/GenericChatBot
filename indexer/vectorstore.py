from pathlib import Path
import chromadb
import json

ROOT = Path(__file__).parent.parent

INPUT_PATH = ROOT / "output/embeddings.json"
CHROMA_DIR = str(ROOT / "output/chroma_db")
COLLECTION = "documents"


def store(chunks: list[dict]) -> int:
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION)

    collection.upsert(
        ids        = [f"{c['source']}::chunk_{c['chunk_index']}" for c in chunks],
        embeddings = [c["embedding"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        metadatas  = [{"source": c["source"], "extension": c["extension"], "chunk_index": c["chunk_index"]} for c in chunks],
    )
    return collection.count()


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Run embedder.py first to generate output/embeddings.json")

    chunks = json.loads(INPUT_PATH.read_text())
    print(f"Loaded {len(chunks)} chunks\n")

    count = store(chunks)
    print(f"Stored {count} chunks in collection '{COLLECTION}'")
    print(f"Location: output/chroma_db/")
