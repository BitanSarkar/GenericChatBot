"""
Shared Redis client — single connection, used by:
  - web/server.py       (index status, chat history)
  - chatbot/memory.py   (conversation turns)
  - indexer/contextualizer.py  (context cache)
"""
import os
import redis

# ── Redis key constants ────────────────────────────────────────────────────────
HISTORY_KEY       = "chatbot:history"          # List  — conversation turns
CACHE_KEY         = "chatbot:context_cache"    # Hash  — md5 → LLM context
INDEX_STATUS_KEY  = "chatbot:index:status"     # String — idle | running | error
INDEX_MESSAGE_KEY = "chatbot:index:message"    # String — human-readable status

_client: "redis.Redis | None" = None


def get_redis() -> redis.Redis:
    """Return a singleton Redis client. Thread-safe (Redis-py is thread-safe)."""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )
    return _client
