"""
LLM generator — calls Ollama and streams tokens.

  stream_generate(prompt)  →  yields tokens one by one  (used by web server SSE)
  generate(prompt)         →  blocks, returns full string (used by CLI main.py)
"""
import json
import requests
from typing import Generator

OLLAMA_URL  = "http://localhost:11434/api/generate"
LLM_MODEL   = "dolphin-llama3"
NUM_PREDICT = -1
NUM_CTX     = 8192


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    question:             str,
    matches:              list[dict],
    conversation_history: str = "",
    mode:                 str = "hybrid",
) -> str:
    """
    Builds the LLM prompt based on retrieval mode.

    public  — no context; LLM answers from its own training knowledge.
    private — context injected, strict: only answer from documents.
    hybrid  — context injected, LLM can supplement with its own knowledge.
    """
    history_block = ""
    if conversation_history:
        history_block = f"Previous conversation:\n{conversation_history}\n\n"

    if mode == "public":
        return f"{history_block}{question}"

    context = "\n\n".join(
        f"[{i+1}] Source: {m['source']} (chunk {m['chunk_index']})\n{m['text']}"
        for i, m in enumerate(matches)
    )

    if mode == "private":
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

    # hybrid
    return f"""{history_block}You have access to the following reference documents \
and your full training knowledge. Use both freely — there are no restrictions.
Combine insights from the documents with everything you know to give the richest, \
most complete answer. Reference the documents when they're useful, go beyond them whenever you want.

Reference documents:
{context}

Question: {question}
Answer:"""


# ── Streaming ─────────────────────────────────────────────────────────────────

def stream_generate(prompt: str) -> Generator[str, None, None]:
    """
    Yields tokens one by one as they arrive from Ollama.
    Used by the web server to build an SSE response.
    """
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

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


# ── Blocking (CLI) ────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    """
    Blocking wrapper around stream_generate — prints tokens to stdout as they
    arrive and returns the full answer string. Used by CLI main.py.
    """
    answer = []
    for token in stream_generate(prompt):
        print(token, end="", flush=True)
        answer.append(token)
    print()
    return "".join(answer)
