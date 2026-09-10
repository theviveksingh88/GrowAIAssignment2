"""
Assignment 11 - Production LLM API with Evals & Observability
=============================================================

FastAPI service exposing the Assignment 6 RAG pipeline with:

- POST /chat      streaming (SSE) and non-streaming answers
- GET  /health    dependency health check
- GET  /metrics   summary of the request log
- logging middleware writing JSON Lines
- exact-match response cache

Run locally:
    uvicorn main:app --reload --port 8080

Run the full stack:
    docker compose up --build
"""

import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import rag_backend

LOG_PATH = os.getenv("LOG_PATH", "logs/requests.jsonl")

# Local inference is free; this rate is what the same traffic would cost on a
# hosted model, so the cost column stays meaningful when the backend changes.
COST_PER_1K_TOKENS = float(os.getenv("COST_PER_1K_TOKENS", "0.0002"))

CACHE_MAX_ENTRIES = 500

app = FastAPI(
    title="Production RAG API",
    description="Assignment 11 - FastAPI + Docker + DeepEval + observability",
    version="1.0.0",
)

# Exact-match cache. A dict is enough for one process; a multi-replica
# deployment would point this at Redis so replicas share the same cache.
_cache = {}


def estimate_tokens(text):
    """Rough token estimate: about 4 characters per token."""

    return max(1, len(text) // 4)


def estimate_cost(tokens):
    """Estimated cost of a call at the configured rate."""

    return round(tokens / 1000 * COST_PER_1K_TOKENS, 8)


def write_log(record):
    """Append one JSON Lines record to the request log."""

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# ============================================================
# Request / response models
# ============================================================

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    stream: bool = True
    top_k: int = Field(3, ge=1, le=10)
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    cached: bool
    latency_ms: float
    tokens_estimated: int
    cost_usd: float
    model: str


# ============================================================
# Logging middleware
# ============================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Record timestamp, latency, tokens, model and cache status per request.

    Streaming responses return from call_next before the body has been sent, so
    the log entry is deferred to a BackgroundTask. That runs after the response
    is fully flushed, by which point the streaming generator has populated
    request.state - otherwise every streamed request would log 0 tokens and a
    latency covering only the time to the first byte.
    """

    start = time.perf_counter()

    request.state.tokens = 0
    request.state.cached = False
    request.state.model = rag_backend.CHAT_MODEL

    response = await call_next(request)

    def emit():
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        tokens = getattr(request.state, "tokens", 0)

        write_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "model": getattr(request.state, "model", None),
            "tokens_estimated": tokens,
            "cost_usd": estimate_cost(tokens),
            "cached": getattr(request.state, "cached", False),
        })

    existing = response.background

    def run_all():
        if existing is not None:
            existing()
        emit()

    response.background = BackgroundTask(run_all)

    return response


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup():
    """Index the corpus once the vector database is reachable."""

    try:
        count = rag_backend.init_index()
        print(f"Vector index ready: {count} chunks")
    except Exception as error:
        # Do not crash the container if Chroma is still starting; /health
        # reports the real state and the first request retries.
        print(f"Index initialisation deferred: {error}")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    """Liveness plus dependency readiness."""

    status = rag_backend.health()

    # Every dependency must be up AND the index must be queryable. Without the
    # index check this endpoint returned "ok" while every /chat call failed.
    ready = all(status.values())

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "degraded",
            "dependencies": status,
            "cache_entries": len(_cache),
        },
    )


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    """Answer a question from the RAG pipeline, streaming by default."""

    cache_key = body.query.strip().lower()

    # ---- Cache hit ----
    if body.use_cache and cache_key in _cache:

        cache_start = time.perf_counter()

        cached_answer = _cache[cache_key]

        request.state.cached = True
        request.state.tokens = estimate_tokens(cached_answer)

        if not body.stream:
            return ChatResponse(
                answer=cached_answer,
                cached=True,
                latency_ms=round((time.perf_counter() - cache_start) * 1000, 3),
                tokens_estimated=estimate_tokens(cached_answer),
                cost_usd=estimate_cost(estimate_tokens(cached_answer)),
                model=rag_backend.CHAT_MODEL,
            )

        async def replay():
            yield f"data: {json.dumps({'token': cached_answer, 'cached': True})}\n\n"
            yield f"data: {json.dumps({'done': True, 'cached': True})}\n\n"

        return StreamingResponse(replay(), media_type="text/event-stream")

    # ---- Non-streaming ----
    if not body.stream:

        start = time.perf_counter()

        try:
            answer_text, _ = rag_backend.answer(body.query, top_k=body.top_k)
        except Exception as error:
            return JSONResponse(
                status_code=502,
                content={"error": "upstream model failed", "detail": str(error)},
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        tokens = estimate_tokens(answer_text)

        if body.use_cache:
            _remember(cache_key, answer_text)

        request.state.tokens = tokens

        return ChatResponse(
            answer=answer_text,
            cached=False,
            latency_ms=latency_ms,
            tokens_estimated=tokens,
            cost_usd=estimate_cost(tokens),
            model=rag_backend.CHAT_MODEL,
        )

    # ---- Streaming (SSE) ----
    async def event_stream():

        collected = []

        try:
            async for token in rag_backend.answer_stream(body.query, top_k=body.top_k):
                collected.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as error:
            yield f"data: {json.dumps({'error': str(error)})}\n\n"
            return

        full_answer = "".join(collected)

        if body.use_cache:
            _remember(cache_key, full_answer)

        # Populate state before the background log task runs.
        request.state.tokens = estimate_tokens(full_answer)

        yield f"data: {json.dumps({'done': True, 'cached': False})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _remember(key, value):
    """Store a cache entry, evicting the oldest when full."""

    if len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))

    _cache[key] = value


@app.get("/metrics")
def metrics():
    """Summarise the request log."""

    if not os.path.exists(LOG_PATH):
        return {"requests": 0}

    records = []

    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Only successful calls: a 422 never reached the cache or the model, so
    # counting it as a cache miss would understate the hit rate.
    chat_records = [
        r for r in records
        if r.get("path") == "/chat" and r.get("status_code") == 200
    ]

    cached = [r for r in chat_records if r.get("cached")]
    uncached = [r for r in chat_records if not r.get("cached")]

    def mean(rows, key):
        values = [r.get(key, 0) for r in rows]
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "requests_total": len(records),
        "chat_requests": len(chat_records),
        "cache_hits": len(cached),
        "cache_misses": len(uncached),
        "avg_latency_ms_cached": mean(cached, "latency_ms"),
        "avg_latency_ms_uncached": mean(uncached, "latency_ms"),
        "total_tokens_estimated": sum(r.get("tokens_estimated", 0) for r in records),
        "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in records), 6),
    }
