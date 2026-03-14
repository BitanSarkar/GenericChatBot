"""
Conversation memory — backed by Redis List.

Each turn is a JSON object pushed to HISTORY_KEY:
  { "role": "user"|"assistant", "content": "...", "timestamp": "..." }

Redis List operations used:
  RPUSH  — append a turn  (O1)
  LRANGE — fetch turns     (On)
  DEL    — clear session   (O1)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Allow running this module directly or imported from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.redis_client import get_redis, HISTORY_KEY

MAX_TURNS_IN_PROMPT = 10   # rolling window injected into each LLM prompt


# ── Write ──────────────────────────────────────────────────────────────────────

def add_turn(role: str, content: str) -> None:
    """Append a single turn directly to Redis. No load-mutate-save cycle needed."""
    turn = {
        "role":      role,
        "content":   content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    get_redis().rpush(HISTORY_KEY, json.dumps(turn))


def clear_memory() -> None:
    """Delete the entire conversation history from Redis."""
    get_redis().delete(HISTORY_KEY)


# ── Read ───────────────────────────────────────────────────────────────────────

def load_memory() -> list[dict]:
    """Return all turns — full history (used by GET /history)."""
    entries = get_redis().lrange(HISTORY_KEY, 0, -1)
    return [json.loads(e) for e in entries]


def recent_turns(n: int = MAX_TURNS_IN_PROMPT) -> list[dict]:
    """Return the last n turns — what gets injected into the prompt."""
    entries = get_redis().lrange(HISTORY_KEY, -n, -1)
    return [json.loads(e) for e in entries]


def format_for_prompt() -> str:
    """Format the rolling window as a readable block for the LLM prompt."""
    turns = recent_turns()
    if not turns:
        return ""
    lines = []
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Bot"
        lines.append(f"{label}: {turn['content']}")
    return "\n".join(lines)
