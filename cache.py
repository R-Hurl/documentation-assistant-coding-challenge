"""
cache.py — Semantic cache layer for the RAG pipeline.

Uses redisvl's SemanticCache to intercept queries that are close in vector
space to previously answered questions, avoiding redundant OpenAI API calls.

Functions
---------
check_cache        : look up a question; return cached answer or None
store_in_cache     : persist a question/answer pair
flush_cache        : clear all cached entries and reset metrics
get_metrics        : return hit/miss counts, latency averages, cost estimate
"""

import time

import redis
from dotenv import load_dotenv
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer

# load_dotenv must run before singletons so OpenAITextVectorizer can read the key
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.15           # cosine distance; lower = stricter match
REDIS_URL = "redis://localhost:6379"
CACHE_INDEX_NAME = "llmcache"
EMBED_MODEL = "text-embedding-3-small"
ESTIMATED_TOKENS_PER_RESPONSE = 600   # used for cost-savings estimate

# ---------------------------------------------------------------------------
# Module-level singletons (initialized once at import)
# ---------------------------------------------------------------------------
_vectorizer = OpenAITextVectorizer(model=EMBED_MODEL)

_cache = SemanticCache(
    name=CACHE_INDEX_NAME,
    redis_url=REDIS_URL,
    distance_threshold=SIMILARITY_THRESHOLD,
    vectorizer=_vectorizer,
)

_redis = redis.Redis.from_url(REDIS_URL)  # raw client for INCR on metric keys


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------
def check_cache(question: str) -> dict | None:
    """
    Look up *question* in the semantic cache.

    Returns a dict with keys ``answer`` and ``sources`` on a hit, or ``None``
    on a miss. Also increments the appropriate Redis metric counter.
    """
    results = _cache.check(prompt=question, num_results=1)
    if not results:
        _redis.incr("metrics:misses")
        return None

    hit = results[0]
    _redis.incr("metrics:hits")
    _redis.incrby("metrics:tokens_saved", ESTIMATED_TOKENS_PER_RESPONSE)
    return {
        "answer": hit["response"],
        "sources": hit.get("metadata", {}).get("sources", []),
    }


def store_in_cache(question: str, response: dict) -> None:
    """Store a question/answer pair in the semantic cache."""
    _cache.store(
        prompt=question,
        response=response["answer"],
        metadata={"sources": response["sources"]},
    )


def record_latency(metric_key: str, start: float) -> None:
    """Record elapsed milliseconds since *start* into the given Redis counter."""
    elapsed_ms = (time.monotonic() - start) * 1000
    _redis.incrby(metric_key, int(elapsed_ms))


def flush_cache() -> None:
    """
    Delete all cached entries and reset all metric counters.

    The index schema is preserved; only data keys are removed.
    """
    _cache.clear()
    _redis.delete(
        "metrics:hits",
        "metrics:misses",
        "metrics:latency_cached_ms_total",
        "metrics:latency_uncached_ms_total",
        "metrics:tokens_saved",
    )


def get_metrics() -> dict:
    """
    Return a snapshot of cache performance metrics.

    Computed fields
    ---------------
    hit_rate                  : fraction of requests served from cache
    avg_latency_cached_ms     : mean response time for cache hits
    avg_latency_uncached_ms   : mean response time for cache misses
    estimated_cost_saved_usd  : rough dollar savings from avoided LLM calls
    """
    hits      = int(_redis.get("metrics:hits") or 0)
    misses    = int(_redis.get("metrics:misses") or 0)
    tokens    = int(_redis.get("metrics:tokens_saved") or 0)
    lat_c     = int(_redis.get("metrics:latency_cached_ms_total") or 0)
    lat_u     = int(_redis.get("metrics:latency_uncached_ms_total") or 0)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    avg_cached   = lat_c / hits   if hits   > 0 else 0.0
    avg_uncached = lat_u / misses if misses > 0 else 0.0

    # text-embedding-3-small pricing: ~$0.02 / 1M tokens
    cost_saved = (tokens / 1_000_000) * 0.02

    return {
        "hits": hits,
        "misses": misses,
        "total_requests": total,
        "hit_rate": round(hit_rate, 4),
        "avg_latency_cached_ms": round(avg_cached, 2),
        "avg_latency_uncached_ms": round(avg_uncached, 2),
        "tokens_saved": tokens,
        "estimated_cost_saved_usd": round(cost_saved, 6),
    }
