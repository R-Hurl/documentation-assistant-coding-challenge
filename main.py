"""
main.py — FastAPI app exposing the RAG pipeline over HTTP.

Endpoints
---------
POST /ingest          : (re-)run the ingestion pipeline
POST /ask             : synchronous question → answer + sources
GET  /ask/stream      : streaming question → token-by-token SSE
GET  /metrics         : cache hit/miss stats and latency averages
DELETE /cache         : flush the semantic cache and reset metrics
"""

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import cache
import ingest
import rag

app = FastAPI(title="RTFM For Me", version="0.1.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/ingest")
def run_ingest():
    """Trigger the ingestion pipeline (idempotent)."""
    try:
        ingest.main()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest):
    """Check semantic cache first; fall back to full RAG pipeline on miss."""
    start = time.monotonic()
    cached_result = cache.check_cache(body.question)
    if cached_result is not None:
        cache.record_latency("metrics:latency_cached_ms_total", start)
        return {**cached_result, "cached": True}

    # Cache miss: run full RAG pipeline
    embedding = rag.embed_query(body.question)
    chunks    = rag.search_chunks(embedding)
    result    = rag.generate_answer(body.question, chunks)

    if result["answer"] != rag.OUT_OF_SCOPE_ANSWER:
        # Don't cache out-of-scope answers; they may mislead future similar queries.
        cache.store_in_cache(body.question, result)

    cache.record_latency("metrics:latency_uncached_ms_total", start)
    return {**result, "cached": False}


@app.get("/ask/stream")
async def ask_stream(question: str):
    """Stream a token-by-token answer via Server-Sent Events."""
    embedding = rag.embed_query(question)
    chunks = rag.search_chunks(embedding)
    return EventSourceResponse(rag.stream_answer(question, chunks))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
@app.get("/metrics")
def get_metrics():
    """Return semantic cache performance metrics."""
    return cache.get_metrics()


@app.delete("/cache")
def flush_cache():
    """Flush the semantic cache and reset all metrics."""
    cache.flush_cache()
    return {"status": "ok", "message": "Cache cleared."}
