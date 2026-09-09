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


def _chunk(text, chunk_size=300, overlap=50):
    """Split text into overlapping character windows on word boundaries."""

    words = re.findall(r"\S+", text)

    chunks = []
    step = max(1, chunk_size - overlap)

    # Approximate: 5 characters per word, matching Assignment 6's 300/50 split.
    words_per_chunk = max(1, chunk_size // 5)
    words_step = max(1, step // 5)

    for start in range(0, len(words), words_step):
        chunk = " ".join(words[start:start + words_per_chunk])
        if chunk:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break

    return chunks


def init_index():
    """Create the collection and index the corpus if it is empty."""

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


def retrieve(query, top_k=3):
    """Return the top_k most similar chunks for a query."""

    if _collection is None:
        raise RuntimeError("Index not initialised - call init_index() first")

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

    status = {"ollama": False, "chroma": False}

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

    return status
