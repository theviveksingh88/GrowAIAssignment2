# Assignment 11 — Production LLM API with Evals & Observability

FastAPI service wrapping the Assignment 6 RAG pipeline, containerised with Docker,
evaluated with DeepEval, and instrumented with request logging, latency tracking and
per-call cost estimation.

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — `/chat` (SSE streaming), `/health`, `/metrics`, logging middleware, cache |
| `rag_backend.py` | Retrieval + generation; Ollama embeddings, ChromaDB over HTTP |
| `evals.py` | DeepEval suite — 10 cases, 3 metrics, local Ollama judge |
| `client.py` | Demo client: streaming output and cache speedup |
| `Dockerfile` | API image (~200 MB, no torch) |
| `docker-compose.yml` | API + ChromaDB |
| `corpus.txt` | Document served by the RAG pipeline |

## Prerequisites

Ollama runs on the **host**, not in a container:

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run the stack

```bash
docker compose up --build
```

- API → http://localhost:8080 (docs at `/docs`)
- ChromaDB → http://localhost:8000

## Try it

```bash
curl http://localhost:8080/health

# Streaming (Server-Sent Events)
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "stream": true}'

# Non-streaming
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "stream": false}'

curl http://localhost:8080/metrics
```

Or run the demo client, which shows streaming plus the cache speedup:

```bash
python client.py
```

## Run the evaluation suite

Run it **inside the container**, not on the host — the host is Python 3.14 and
`deepeval==2.1.6` requires <3.13, while an isolated venv pulls a `chromadb-client` on the v2 API
against this 0.6.3 v1 server.

```bash
docker compose start

docker exec assignment11-api pip install deepeval==2.1.6
docker exec assignment11-api pip install \
  "langchain>=0.3,<1.0" "langchain-core>=0.3,<0.4" "langchain-openai>=0.2,<0.3"

docker cp evals.py assignment11-api:/app/evals.py
docker exec assignment11-api python -u evals.py
```

`-u` matters: without it Python buffers piped stdout and a working run looks hung. The setup
survives `docker compose stop`/`start` but not `down`.

Metrics: **Answer Relevancy**, **Faithfulness**, **Contextual Precision**, threshold 0.7.
Two of the ten cases have no answer in the corpus, so a faithful system must decline rather
than invent one — a suite where every question is answerable cannot detect hallucination.

DeepEval judges with an LLM, defaulting to OpenAI. `evals.py` supplies a local Ollama judge
instead, so no API key is needed and no data leaves the machine. A small local judge is
slower and noisier than GPT-4-class judging — treat the scores as directional.

## Observability

Every request appends one line to `logs/requests.jsonl`:

```json
{"timestamp": "...", "method": "POST", "path": "/chat", "status_code": 200,
 "latency_ms": 1843.2, "model": "llama3.2", "tokens_estimated": 96,
 "cost_usd": 0.0000192, "cached": false}
```

Two details worth noting:

- **Streaming breaks naive middleware.** A streaming response returns from `call_next`
  before its body is sent, so logging there records only time-to-first-byte and zero tokens.
  The log write is deferred to a `BackgroundTask`, which runs after the body is flushed.
- **Cost is estimated, not measured.** Local inference is free; the rate in
  `COST_PER_1K_TOKENS` models what the same traffic would cost hosted, so the column stays
  meaningful if the backend changes. Token counts are a 4-chars-per-token approximation, not
  a real tokenizer.

## Design notes

- **Embeddings via Ollama, not sentence-transformers.** Keeps `torch` out of the image:
  ~200 MB instead of ~2.5 GB.
- **ChromaDB as a separate service.** API replicas share one index instead of each holding a
  private in-memory copy.
- **`host.docker.internal`** lets the container reach Ollama on the host. `extra_hosts` is
  set for Linux; Docker Desktop provides it already.
- **Cache is a process-local dict.** Correct for one replica. Multiple replicas need Redis,
  or each keeps its own cache and the hit rate drops.
