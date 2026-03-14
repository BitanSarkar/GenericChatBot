"""
Chatbot — interactive question-answering loop over your indexed documents.
Make sure you have run:  python indexer/run.py  before starting this.

Usage:
  python main.py                  # hybrid mode (default)
  python main.py --mode public    # LLM answers from its own knowledge only
  python main.py --mode private   # strict: only answer from indexed documents
  python main.py --mode hybrid    # context-injected, LLM can supplement
"""
import argparse
from retriever import retrieve, load_retriever
from generator import build_prompt, generate
from memory    import load_memory, save_memory, add_turn, format_for_prompt

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GenericChatBot")
parser.add_argument(
    "--mode",
    choices=["public", "private", "hybrid"],
    default="hybrid",
    help=(
        "public  = LLM uses only its own training knowledge, no retrieval. "
        "private = answer strictly from indexed documents only. "
        "hybrid  = context injected, LLM can supplement (default)."
    ),
)
args = parser.parse_args()
MODE = args.mode

TOP_K = 5

# ── Startup ───────────────────────────────────────────────────────────────────
print(f"\nMode: {MODE.upper()}")
if MODE == "public":
    print("Retrieval disabled — LLM will answer from its own knowledge.\n")
else:
    print("Loading model and vector store...")
    model, collection, bm25, store, reranker = load_retriever()
    print(f"Ready. {collection.count()} chunks indexed.\n")

history = load_memory()
if history:
    print(f"Loaded {len(history)} turns from previous conversations.\n")
else:
    print("No previous memory found. Starting fresh.\n")

print("Type your question and press Enter. Type 'exit' to quit.\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    question = input("You: ").strip()

    if not question:
        continue
    if question.lower() in {"exit", "quit"}:
        save_memory(history)
        print("Memory saved. Bye.")
        break

    matches = []
    if MODE != "public":
        matches = retrieve(question, collection, model, bm25, store, reranker, top_k=TOP_K)
        if not matches:
            if MODE == "private":
                print("\nBot: This information is not in the indexed documents.\n")
            else:
                print("No relevant chunks found.\n")
            continue

    prompt = build_prompt(question, matches, format_for_prompt(history), mode=MODE)

    print("\nBot: ", end="")
    answer = generate(prompt)

    history = add_turn(history, "user",      question)
    history = add_turn(history, "assistant", answer)
    save_memory(history)

    if matches:
        print("\nSources:")
        for m in matches:
            print(f"  [reranker={m['reranker_score']}  rrf={m['rrf_score']}]  {m['source']}  chunk {m['chunk_index']}")
    print()
