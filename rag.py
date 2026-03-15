"""
rag.py — RAG pipeline: embed → search → prompt → generate.

Functions
---------
embed_query       : embed a user question into a vector
search_chunks     : KNN vector search against the Redis index
build_prompt      : format retrieved chunks into a chat message list
generate_answer   : synchronous LLM call, returns answer + sources
stream_answer     : async token-by-token LLM stream
"""

import json
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from ingest import schema_dict

load_dotenv()

# ---------------------------------------------------------------------------
# Config (mirrors ingest.py)
# ---------------------------------------------------------------------------
REDIS_URL   = "redis://localhost:6379"
INDEX_NAME  = "fusion-cache-docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TOP_K       = 5  # Number of chunks to retrieve from Redis for each question
OUT_OF_SCOPE_ANSWER = "I don't know."

_openai_client = OpenAI()


# ---------------------------------------------------------------------------
# Step 1 — Embed
# ---------------------------------------------------------------------------
def embed_query(question: str) -> list[float]:
    """Return a 1536-dim embedding for *question* using text-embedding-3-small."""
    response = _openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=question,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step 2 — Search
# ---------------------------------------------------------------------------
def search_chunks(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Run KNN vector search against the Redis index and return the top-k chunks.

    Each returned dict has keys: text, source, heading.
    """
    index = SearchIndex.from_dict(schema_dict, redis_url=REDIS_URL)

    query = VectorQuery(
        vector=query_embedding,
        vector_field_name="embedding",
        return_fields=["text", "source", "heading"],
        num_results=top_k,
    )

    results = index.query(query)
    chunks = []
    for r in results:
        chunks.append({
            "text":    r.get("text", ""),
            "source":  r.get("source", ""),
            "heading": r.get("heading", ""),
        })
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Prompt
# ---------------------------------------------------------------------------
def build_prompt(question: str, chunks: list[dict]) -> list:
    """
    Format retrieved chunks into a [SystemMessage, HumanMessage] list.

    The system message instructs the model to answer from context only.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk["source"]
        if chunk["heading"]:
            label += f" › {chunk['heading']}"
        context_blocks.append(f"[{i}] ({label})\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    system_prompt = f"""You are a helpful documentation reading assistant with expertise with the FusionCache .NET library. Answer the question using only the provided context.
    Cite sources by their [N] reference numbers. If the answer is not in the context, say "I don't know."

    Context:
    {context}
    """

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}"),
    ]


# ---------------------------------------------------------------------------
# Step 4 — Generate (sync)
# ---------------------------------------------------------------------------
def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Call the LLM synchronously and return {"answer": str, "sources": list[str]}.
    """
    messages = build_prompt(question, chunks)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(messages)
    if OUT_OF_SCOPE_ANSWER.casefold() in response.content.strip().casefold():
        return {"answer": response.content, "sources": []}
    
    sources = list(dict.fromkeys(c["source"] for c in chunks if c["source"]))
    return {"answer": response.content, "sources": sources}


# ---------------------------------------------------------------------------
# Step 5 — Stream (async)
# ---------------------------------------------------------------------------
async def stream_answer(question: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """Yield JSON-encoded token events for SSE streaming."""
    messages = build_prompt(question, chunks)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)

    async for token in llm.astream(messages):
        text = token.content
        if text:
            yield json.dumps({"token": text})
