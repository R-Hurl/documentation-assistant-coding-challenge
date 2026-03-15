"""
main.py — FastAPI app exposing the RAG pipeline over HTTP.

Endpoints
---------
POST /ingest          : (re-)run the ingestion pipeline
POST /ask             : synchronous question → answer + sources
GET  /ask/stream      : streaming question → token-by-token SSE
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

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
    """Embed the question, retrieve chunks, and return a grounded answer."""
    embedding = rag.embed_query(body.question)
    chunks = rag.search_chunks(embedding)
    return rag.generate_answer(body.question, chunks)


@app.get("/ask/stream")
async def ask_stream(question: str):
    """Stream a token-by-token answer via Server-Sent Events."""
    embedding = rag.embed_query(question)
    chunks = rag.search_chunks(embedding)
    return EventSourceResponse(rag.stream_answer(question, chunks))
