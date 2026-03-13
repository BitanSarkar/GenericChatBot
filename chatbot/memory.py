from pathlib import Path
from datetime import datetime, timezone
import json

ROOT        = Path(__file__).parent.parent
MEMORY_PATH = ROOT / "output/memory.json"
MAX_TURNS_IN_PROMPT = 10   # how many past turns (user + bot combined) to inject into each prompt


def load_memory() -> list[dict]:
    """Load full conversation history from disk. Returns empty list if none exists."""
    if not MEMORY_PATH.exists():
        return []
    return json.loads(MEMORY_PATH.read_text())


def save_memory(history: list[dict]) -> None:
    """Persist the full conversation history to disk."""
    MEMORY_PATH.parent.mkdir(exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(history, indent=2))


def add_turn(history: list[dict], role: str, content: str) -> list[dict]:
    """Append a single turn. role is 'user' or 'assistant'."""
    history.append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return history


def recent_turns(history: list[dict], n: int = MAX_TURNS_IN_PROMPT) -> list[dict]:
    """Return the last n turns — what gets injected into the prompt."""
    return history[-n:]


def format_for_prompt(history: list[dict]) -> str:
    """Format recent turns as a readable block for the LLM prompt."""
    turns = recent_turns(history)
    if not turns:
        return ""

    lines = []
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Bot"
        lines.append(f"{label}: {turn['content']}")

    return "\n".join(lines)
