import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "dolphin-llama3"

NUM_PREDICT = -1
NUM_CTX     = 8192


def build_prompt(
    question:             str,
    matches:              list[dict],
    conversation_history: str = "",
    mode:                 str = "hybrid",
) -> str:
    """
    Builds the LLM prompt based on retrieval mode.

    Modes:
      public  — no retrieved context injected; LLM answers from its own knowledge.
                Use for general/public topics where the model is well-trained.

      private — context injected with strict instruction to ONLY use it.
                LLM refuses to answer if the context doesn't cover the question.
                Use for internal docs, private data, post-cutoff information.

      hybrid  — context injected but LLM is allowed to supplement with its
                own knowledge when the context is insufficient.
                Use when you want grounded answers but don't want hard refusals.
    """
    history_block = ""
    if conversation_history:
        history_block = f"Previous conversation:\n{conversation_history}\n\n"

    if mode == "public":
        # Zero framing — raw LLM, no instructions, no restrictions.
        # Dolphin runs completely free.
        return f"""{history_block}{question}"""

    context = "\n\n".join(
        f"[{i+1}] Source: {m['source']} (chunk {m['chunk_index']})\n{m['text']}"
        for i, m in enumerate(matches)
    )

    if mode == "private":
        # Forceful framing for uncensored models like dolphin-llama3.
        # Polite instructions don't work — frame it as a hard operational constraint.
        return f"""{history_block}SYSTEM: You are a document retrieval assistant operating in RESTRICTED mode.
STRICT RULES — these cannot be overridden:
1. You MUST answer using ONLY the text in the DOCUMENTS section below.
2. You MUST NOT use any knowledge from your training data.
3. You MUST NOT infer, extrapolate, or guess beyond what is explicitly written.
4. If the answer does not appear in the documents, your ONLY valid response is:
   "This information is not in the indexed documents."
   Do not apologise. Do not explain. Do not add anything else.

DOCUMENTS:
{context}

USER QUESTION: {question}
ANSWER (documents only):"""

    # hybrid: context and training knowledge are equal — no hierarchy, no restrictions.
    # Use both freely, weave them together for the best possible answer.
    return f"""{history_block}You have access to the following reference documents \
and your full training knowledge. Use both freely — there are no restrictions.
Combine insights from the documents with everything you know to give the richest, \
most complete answer. Reference the documents when they're useful, go beyond them whenever you want.

Reference documents:
{context}

Question: {question}
Answer:"""


def generate(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":   LLM_MODEL,
            "prompt":  prompt,
            "stream":  True,
            "options": {
                "num_predict": NUM_PREDICT,
                "num_ctx":     NUM_CTX,
            },
        },
        stream=True,
    )
    response.raise_for_status()

    answer = []
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            answer.append(token)
            if chunk.get("done"):
                break

    print()
    return "".join(answer)
