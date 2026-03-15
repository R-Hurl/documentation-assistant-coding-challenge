"""
ingest.py — Load, chunk, embed, and store fusion-cache docs in Redis.

Run: python ingest.py
Re-runs are idempotent: already-indexed chunks are skipped (no API calls).
"""

import os
import pathlib
import numpy as np
import redis as redis_lib
from dotenv import load_dotenv
from redisvl.index import SearchIndex
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCS_DIR = pathlib.Path("doc-examples/fusion-cache")
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "fusion-cache-docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
TOKEN_CHUNK_LENGTH = 500

# ---------------------------------------------------------------------------
# Redis index schema
# ---------------------------------------------------------------------------
schema_dict = {
    "index": {
        "name": INDEX_NAME,
        "prefix": "doc",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "text",        "type": "text"},
        {"name": "source",      "type": "tag"},
        {"name": "heading",     "type": "tag"},
        {"name": "chunk_index", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": EMBED_DIMS,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Chunking — TODO(human): implement this function
# ---------------------------------------------------------------------------
def chunk_document(text: str, source: str) -> list[dict]:
    """
    Split *text* into overlapping chunks suitable for embedding.

    Parameters
    ----------
    text   : full markdown content of one document
    source : filename stem (e.g. "AGentleIntroduction") used in metadata

    Returns
    -------
    List of dicts, each containing:
        text        : chunk text
        source      : source filename stem
        heading     : nearest preceding ## or # heading (empty string if none)
        chunk_index : integer position of this chunk within the document

    Implementation guide
    --------------------
    - Split on '\\n\\n' (paragraph boundaries) to respect natural doc structure
    - Target ~500 tokens per chunk (rough estimate: 1 token ≈ 4 chars / 0.75 words)
    - Merge short paragraphs into the current chunk before starting a new one
    - Add ~50-token overlap by appending the tail of the prior chunk
    - Track the nearest preceding '## ' or '# ' heading for each chunk
    """
    # TODO(human): implement chunking logic here
    paragraphs = text.split('\n\n')
    chunks = []
    current_paragraphs = []
    current_heading = ""
    chunk_index = 0
    overlap_text = ""

    for paragraph in paragraphs:
        if is_paragraph_heading(paragraph):
            current_heading = paragraph.strip()

        if not chunk_length_exceeded(current_paragraphs, paragraph):
            current_paragraphs.append(paragraph)
        else:
            chunk_text = (overlap_text + '\n\n' + '\n\n'.join(current_paragraphs)).strip()
            chunk = {"text": chunk_text, "source": source, "heading": current_heading, "chunk_index": chunk_index}
            chunks.append(chunk)
            overlap_text = chunk_text[-200:]
            chunk_index += 1
            current_paragraphs = [paragraph]
        
    if current_paragraphs:
        chunk_text = (overlap_text + '\n\n' + '\n\n'.join(current_paragraphs)).strip()
        chunk = {"text": chunk_text, "source": source, "heading": current_heading, "chunk_index": chunk_index}
        chunks.append(chunk)
    
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chunk_length_exceeded(paragraphs: list[str], new_paragraph: str) -> bool:
    """Return True if adding new_paragraph would exceed the target chunk length."""
    text = '\n\n'.join(paragraphs + [new_paragraph])
    return len(text) > TOKEN_CHUNK_LENGTH * 4  # Approximate char limit

def is_paragraph_heading(paragraph: str) -> bool:
    """Return True if the paragraph is a markdown heading."""
    return paragraph.startswith('##') or paragraph.startswith('#')

def load_docs(docs_dir: pathlib.Path) -> list[tuple[str, str]]:
    """Return list of (stem, text) for every .md file in docs_dir."""
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    print(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs


def build_all_chunks(docs: list[tuple[str, str]]) -> list[dict]:
    """Chunk all documents and assign deterministic IDs."""
    all_chunks = []
    for stem, text in docs:
        chunks = chunk_document(text, stem)
        for chunk in chunks:
            chunk["id"] = f"{stem}_chunk_{chunk['chunk_index']}"
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def find_new_chunks(index: SearchIndex, chunks: list[dict]) -> list[dict]:
    """Return only the chunks that don't yet exist in the index."""
    new = []
    for chunk in chunks:
        result = index.fetch(chunk["id"])
        if not result:
            new.append(chunk)
    return new


def embed_and_load(index: SearchIndex, chunks: list[dict]) -> None:
    """Embed chunks and load them into the Redis index."""
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(chunks)} chunks via OpenAI ({EMBED_MODEL})…")
    vectors = embedder.embed_documents(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = np.array(vec, dtype=np.float32).tobytes()
    index.load(chunks, id_field="id")
    print(f"Loaded {len(chunks)} chunks into Redis.")


def print_verification(index: SearchIndex, r: redis_lib.Redis, first_id: str) -> None:
    """Print index stats and a sample document to confirm ingestion."""
    info = index.info()
    print("\n── Index info ──────────────────────────────")
    print(f"  name      : {info.get('index_name')}")
    print(f"  docs      : {info.get('num_docs')}")
    print(f"  indexing  : {info.get('indexing')}")
    print("\n── Sample key: doc:{} ──────────────────────".format(first_id))
    key = f"doc:{first_id}"
    fields = r.hgetall(key)
    for k, v in fields.items():
        k_str = k.decode() if isinstance(k, bytes) else k
        if k_str == "embedding":
            v_str = f"<{len(v)} bytes>"
        else:
            v_str = v.decode() if isinstance(v, bytes) else str(v)
        # Truncate long values (embeddings) for readability
        print(f"  {k_str:<14}: {v_str[:80]}{'…' if len(v_str) > 80 else ''}")
    print("────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Connect to Redis
    r = redis_lib.from_url(REDIS_URL)
    r.ping()
    print(f"Connected to Redis at {REDIS_URL}")

    # Create (or reuse) the search index
    index = SearchIndex.from_dict(schema_dict, redis_url=REDIS_URL)
    index.create(overwrite=False)

    # Load and chunk all docs
    docs = load_docs(DOCS_DIR)
    all_chunks = build_all_chunks(docs)

    if not all_chunks:
        print("No chunks produced — check chunk_document implementation.")
        return

    # Idempotent upsert: only embed what's missing
    new_chunks = find_new_chunks(index, all_chunks)
    if not new_chunks:
        print("All chunks already indexed — skipping embedding.")
    else:
        print(f"{len(new_chunks)} new chunk(s) to embed (skipping {len(all_chunks) - len(new_chunks)} existing).")
        embed_and_load(index, new_chunks)

    # Verification
    print_verification(index, r, all_chunks[0]["id"])


if __name__ == "__main__":
    main()
