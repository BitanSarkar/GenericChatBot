from pathlib import Path
import json

ROOT = Path(__file__).parent.parent

CHUNK_SIZE = 200
OVERLAP    = 40


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words  = text.split()
    if not words:
        return []

    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    return [
        {"source": doc["source"], "extension": doc["extension"], "chunk_index": i, "text": chunk}
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap))
    ]


def chunk_all(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}  →  {len(chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    input_path  = ROOT / "output/documents.json"
    output_path = ROOT / "output/chunks.json"

    if not input_path.exists():
        raise FileNotFoundError("Run loader.py first to generate output/documents.json")

    docs = json.loads(input_path.read_text())
    print(f"Read {len(docs)} documents\n")

    chunks = chunk_all(docs)

    output_path.write_text(json.dumps(chunks, indent=2))
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to: {output_path.relative_to(ROOT)}")
