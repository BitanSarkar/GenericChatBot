import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "dolphin-llama3"


def build_prompt(question: str, matches: list[dict], conversation_history: str = "") -> str:
    context = "\n\n".join(
        f"[{i+1}] Source: {m['source']} (chunk {m['chunk_index']})\n{m['text']}"
        for i, m in enumerate(matches)
    )

    history_block = ""
    if conversation_history:
        history_block = f"""Previous conversation:
{conversation_history}

"""

    return f"""
{history_block}Context from documents:
{context}

Question: {question}
Answer:"""


NUM_PREDICT = -1   # max output tokens (-1 = unlimited)
NUM_CTX     = 8192   # context window: prompt + output tokens combined


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
