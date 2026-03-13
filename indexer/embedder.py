from pathlib import Path
from sentence_transformers import SentenceTransformer
import json

ROOT = Path(__file__).parent.parent

MODEL_NAME  = "all-MiniLM-L6-v2"
INPUT_PATH  = ROOT / "output/chunks.json"
OUTPUT_PATH = ROOT / "output/embeddings.json"


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts   = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    return chunks


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Run chunker.py first to generate output/chunks.json")

    chunks = json.loads(INPUT_PATH.read_text())
    print(f"Loaded {len(chunks)} chunks\n")

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Vector size: {model.get_sentence_embedding_dimension()} dimensions\n")

    chunks = embed_chunks(chunks, model)

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2))
    print(f"\nSaved to: {OUTPUT_PATH.relative_to(ROOT)}")
