# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a "RTFM For Me" AI documentation assistant using RAG (Retrieval-Augmented Generation). The assistant answers questions about the fusion-cache library by searching relevant docs and generating grounded responses.

## Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis Stack (required for vector store, semantic cache, session storage)
docker compose up -d

# Ingest documentation into Redis (idempotent — safe to re-run)
python ingest.py
```

Environment variables are loaded from `.env` (contains `OPENAI_API_KEY`).

## Technology Stack

- **LLM & Embeddings**: OpenAI (`text-embedding-3-small` for embeddings, `gpt-*` for generation)
- **Framework**: LangChain (`langchain`, `langchain-openai`)
- **Vector Store**: Redis via `redisvl` — handles vector search, semantic caching, and session storage
- **Docs**: Markdown files in `doc-examples/fusion-cache/`

## Architecture

The system is built around three Redis-backed components:

1. **Vector Store** — fusion-cache docs are chunked, embedded, and stored; queried at runtime to retrieve relevant context
2. **Semantic Cache** — caches LLM responses by embedding similarity to avoid redundant API calls for similar questions
3. **Session/Memory Store** — persists conversation history per session and optionally long-term user context across sessions

The RAG pipeline: user query → embed query → vector search → retrieve top-k doc chunks → augment prompt → LLM → (cache result) → return answer.

## Ingestion Pipeline (`ingest.py`)

- Loads all `.md` files from `doc-examples/fusion-cache/`
- Chunks documents by paragraph boundaries (~500 tokens, ~50-token overlap), extracting nearest preceding `#`/`##` heading as metadata
- Embeds chunks via `text-embedding-3-small` (1536 dims, stored as `float32` bytes in Redis)
- Redis index name: `fusion-cache-docs`, key prefix: `doc`, storage: hash
- **Idempotent**: assigns deterministic IDs (`{stem}_chunk_{index}`), checks Redis before embedding — re-runs skip the OpenAI API entirely if all chunks exist

## Semantic Cache (`cache.py`)

- Redis index name: `llmcache`, similarity threshold: `0.15` cosine distance (lower = stricter)
- `check_cache` / `store_in_cache` / `flush_cache` — cache lifecycle; out-of-scope answers are never stored
- `record_latency(metric_key, start)` — call with `time.monotonic()` start; writes elapsed ms to Redis
- Metric keys: `metrics:hits`, `metrics:misses`, `metrics:latency_cached_ms_total`, `metrics:latency_uncached_ms_total`, `metrics:tokens_saved`
- `get_metrics()` returns derived fields: `hit_rate`, `avg_latency_*_ms`, `estimated_cost_saved_usd`
- All metric counters are reset on `flush_cache()`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | (Re-)run ingestion pipeline |
| `POST` | `/ask` | Question → answer + sources + `cached` bool |
| `GET` | `/ask/stream` | Token-by-token SSE stream |
| `GET` | `/metrics` | Cache hit/miss stats and latency averages |
| `DELETE` | `/cache` | Flush semantic cache and reset metrics |

## Key Dependency Pins

These versions are pinned to resolve compatibility issues — do not upgrade without testing:
- `openai==1.40.0` — newer versions break `langchain-openai==0.1.14`
- `httpx==0.27.2` — `>=0.28` removed the `proxies` kwarg that `openai` passes internally
