"""
RAG backend for the production API.

Reuses the Assignment 6 pipeline (chunk -> embed -> vector search -> grounded
answer), with two changes for containerised deployment:

- Embeddings come from Ollama's nomic-embed-text instead of sentence-transformers,
  which keeps torch out of the image (roughly 2 GB smaller).
- Vectors live in a separate ChromaDB container reached over HTTP, so API
  replicas share one index instead of each holding a private in-memory copy.
"""

import os
import re

import chromadb
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CORPUS_PATH = os.getenv("CORPUS_PATH", "corpus.txt")
COLLECTION_NAME = "production_rag"

SYSTEM_PROMPT = """Answer the question using ONLY the context below.
Answer in one or two complete sentences.
If the answer is not in the context, say exactly: "Not in context."

Context:
{context}

Question: {question}

Answer:"""

_collection = None


def _embed(text):
    """Embed a single string with Ollama."""

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embedding"]


def _chunk(text, chunk_size=400, overlap_sentences=1):
    """Split text into chunks that begin and end on sentence boundaries.

    An earlier version cut on a fixed word count. That decapitated sentences:
    the chunk retrieved for "What is deep learning?" started
    "is a type of machine learning based on artificial neural networks...",
    with the subject sliced off the front, and the model answered
    "Deep learning." - supplying the missing subject instead of the definition.

    Grouping whole sentences costs nothing and removes that failure entirely.
    One sentence of overlap keeps context across the seam.
    """

    # Split after ., ! or ? when followed by whitespace. Good enough for prose;
    # a production system would use a real sentence tokenizer.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return []

    chunks = []
    current = []
    length = 0

    for sentence in sentences:

        if current and length + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # Carry the tail sentences forward so meaning is not split at the seam.
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(s) + 1 for s in current)

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def init_index():
    """Create the collection and index the corpus if it is empty.

    Safe to call repeatedly: if the collection already holds documents it
    returns immediately.
    """

    global _collection

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    _collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if _collection.count() > 0:
        return _collection.count()

    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = _chunk(text)

    _collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=[_embed(chunk) for chunk in chunks],
    )

    return _collection.count()


def ensure_index():
    """Build the index on demand, retrying if startup lost the race.

    docker compose `depends_on` waits for the Chroma container to start, not
    for it to accept connections, so the API can boot first and fail to index.
    Retrying here means the service heals itself on the next request instead of
    staying broken until someone restarts it.
    """

    global _collection

    if _collection is not None:
        return True

    try:
        init_index()
        return True
    except Exception:
        return False


def index_ready():
    """True when the corpus is indexed and queryable."""

    try:
        return _collection is not None and _collection.count() > 0
    except Exception:
        return False


def retrieve(query, top_k=3):
    """Return the top_k most similar chunks for a query."""

    if not ensure_index():
        raise RuntimeError(
            "Vector index unavailable - Chroma is not reachable"
        )

    results = _collection.query(
        query_embeddings=[_embed(query)],
        n_results=top_k,
    )

    return results["documents"][0]


def build_prompt(query, top_k=3):
    """Retrieve context and render the grounded prompt."""

    chunks = retrieve(query, top_k=top_k)

    return SYSTEM_PROMPT.format(
        context="\n\n".join(chunks),
        question=query,
    ), chunks


def answer(query, top_k=3):
    """Non-streaming answer."""

    prompt, chunks = build_prompt(query, top_k=top_k)

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=600,
    )

    response.raise_for_status()

    return response.json().get("response", "").strip(), chunks


async def answer_stream(query, top_k=3):
    """Yield answer tokens as they are produced by the model."""

    import json

    prompt, _ = build_prompt(query, top_k=top_k)

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0},
            },
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():

                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")

                if token:
                    yield token

                if chunk.get("done"):
                    break


def health():
    """Report reachability of both dependencies."""

    status = {"ollama": False, "chroma": False, "index": False}

    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
        status["ollama"] = True
    except Exception:
        pass

    try:
        chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).heartbeat()
        status["chroma"] = True
    except Exception:
        pass

    # Reachability is not readiness. The API can talk to Chroma and still be
    # unable to answer because the corpus was never indexed - which is exactly
    # what happens when startup loses the race. Report that separately.
    if status["chroma"]:
        ensure_index()
        status["index"] = index_ready()

    return status
