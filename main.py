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
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import cache
import ingest
import rag
import session

app = FastAPI(title="RTFM For Me", version="0.1.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


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
    session_id = body.session_id or str(uuid.uuid4())
    history = session.get_history(session_id)

    start = time.monotonic()
    cached_result = cache.check_cache(body.question)
    if cached_result is not None:
        cache.record_latency("metrics:latency_cached_ms_total", start)
        session.append_messages(session_id, body.question, cached_result["answer"])
        return {**cached_result, "cached": True, "session_id": session_id}

    # Cache miss: run full RAG pipeline
    embedding = rag.embed_query(body.question)
    chunks    = rag.search_chunks(embedding)
    result    = rag.generate_answer(body.question, chunks, history)

    if result["answer"] != rag.OUT_OF_SCOPE_ANSWER:
        # Don't cache out-of-scope answers; they may mislead future similar queries.
        cache.store_in_cache(body.question, result)

    session.append_messages(session_id, body.question, result["answer"])
    cache.record_latency("metrics:latency_uncached_ms_total", start)
    return {**result, "cached": False, "session_id": session_id}


@app.get("/ask/stream")
async def ask_stream(question: str, session_id: str | None = None):
    """Stream a token-by-token answer via Server-Sent Events."""
    resolved_session_id = session_id or str(uuid.uuid4())
    history = session.get_history(resolved_session_id)

    embedding = rag.embed_query(question)
    chunks = rag.search_chunks(embedding)

    async def stream_and_save():
        full_answer = []
        async for token_json in rag.stream_answer(question, chunks, history):
            full_answer.append(token_json)
            yield token_json
        import json as _json
        answer_text = "".join(_json.loads(t)["token"] for t in full_answer)
        session.append_messages(resolved_session_id, question, answer_text)

    return EventSourceResponse(stream_and_save())


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
