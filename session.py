"""
session.py — Redis list-backed session store for multi-turn conversations.

Each session is stored as a Redis list of JSON-serialized message dicts.
Messages are appended in chronological order; the list is capped to
MAX_HISTORY turns (each turn = 1 user + 1 assistant message).

Functions
---------
session_key      : build the Redis key for a given session_id
get_history      : retrieve recent conversation turns for a session
append_messages  : persist a user/assistant turn and refresh TTL
"""

import json

import redis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REDIS_URL      = "redis://localhost:6379"
SESSION_TTL    = 86_400    # 24 hours in seconds
MAX_HISTORY    = 10        # max turns to inject into the prompt (20 messages)
SESSION_PREFIX = "session" # key format: session:{session_id}

_redis = redis.Redis.from_url(REDIS_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def session_key(session_id: str) -> str:
    """Return the Redis key for the given session_id."""
    return build_redis_session_key(session_id)

def build_redis_session_key(session_id: str) -> str:
    """Return the Redis key for the given session_id."""
    return f"{SESSION_PREFIX}:{session_id}"


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------
def get_history(session_id: str) -> list[dict]:
    """
    Return the last MAX_HISTORY turns for *session_id*.

    Fetches all messages from the Redis list and returns the most recent
    MAX_HISTORY * 2 messages (each turn = 1 user msg + 1 assistant msg).
    Returns an empty list if the session doesn't exist yet.
    """
    raw_messages = _redis.lrange(session_key(session_id), 0, -1)
    messages = [json.loads(m) for m in raw_messages]
    # Keep only the most recent MAX_HISTORY turns (2 messages per turn)
    return messages[-(MAX_HISTORY * 2):]


def append_messages(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Append a user/assistant turn to the session and refresh its TTL.

    Push both messages (as JSON dicts) onto the Redis list, then set
    the key to expire after SESSION_TTL seconds.

    Each message dict has the shape: {"role": "...", "content": "..."}
    """
    user_dict = { "role": "user", "content": user_msg }
    ai_assistant_dict = { "role": "assistant", "content": assistant_msg }
    _redis.rpush(build_redis_session_key(session_id), json.dumps(user_dict), json.dumps(ai_assistant_dict))
    _redis.expire(build_redis_session_key(session_id), SESSION_TTL)
