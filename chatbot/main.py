"""
Chatbot — interactive question-answering loop over your indexed documents.
Make sure you have run:  python indexer/run.py  before starting this.
"""
from retriever import retrieve, load_retriever
from generator import build_prompt, generate
from memory    import load_memory, save_memory, add_turn, format_for_prompt

TOP_K = 5

print("Loading model and vector store...")
model, collection, bm25, store, reranker = load_retriever()
print(f"Ready. {collection.count()} chunks indexed.")

history = load_memory()
if history:
    print(f"Loaded {len(history)} turns from previous conversations.\n")
else:
    print("No previous memory found. Starting fresh.\n")

print("Type your question and press Enter. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if not question:
        continue
    if question.lower() in {"exit", "quit"}:
        save_memory(history)
        print("Memory saved. Bye.")
        break

    # Retrieve relevant chunks — hybrid BM25 + semantic, merged via RRF
    matches = retrieve(question, collection, model, bm25, store, reranker, top_k=TOP_K)

    if not matches:
        print("No relevant chunks found.\n")
        continue

    # Build prompt with conversation history + retrieved context
    conversation_so_far = format_for_prompt(history)
    prompt  = build_prompt(question, matches, conversation_so_far)

    # Generate and stream the answer
    print("\nBot: ", end="")
    answer = generate(prompt)

    # Persist this turn to memory
    history = add_turn(history, "user",      question)
    history = add_turn(history, "assistant", answer)
    save_memory(history)

    print("\nSources:")
    for m in matches:
        print(f"  [reranker={m['reranker_score']}  rrf={m['rrf_score']}]  {m['source']}  chunk {m['chunk_index']}")
    print()
