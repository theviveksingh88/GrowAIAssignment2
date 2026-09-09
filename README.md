# GrowAI — LLM Engineering Assignments

Hands-on assignments covering the LLM engineering stack end to end: tokenization and embeddings,
neural network fundamentals, local model benchmarking, prompt engineering, orchestration frameworks,
retrieval-augmented generation, ReAct agents, multi-agent systems with MCP, GraphRAG over a knowledge
graph, LoRA fine-tuning, and a containerised production API with evals and observability.

Everything runs **locally** — all LLM inference goes through [Ollama](https://ollama.com) on
`http://localhost:11434`, so no API keys or paid providers are required. The two exceptions are
noted where they apply: fine-tuning needs a Colab GPU, and the production API needs Docker.

---

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [Assignment1_tokenization.py](Assignment1_tokenization.py) | Tokenization & sentence embeddings |
| 2 | [Assignment2_NN_with_XOR.py](Assignment2_NN_with_XOR.py) | XOR neural network in PyTorch |
| 3 | [Assignment3_LLL_Benchmark.py](Assignment3_LLL_Benchmark.py) | Local model benchmarking with Ollama |
| 4 | [Assignment4_Prompt_Engg.ipynb](Assignment4_Prompt_Engg.ipynb) | Prompt engineering & structured output |
| 5 | [Assignment5_Multi_Step_Langchain.py](Assignment5_Multi_Step_Langchain.py) / [Assignment5_Multi_Step_llamaindex.py](Assignment5_Multi_Step_llamaindex.py) | LangChain vs. LlamaIndex chains |
| 6 | [Assignment6_Doc_QA.ipynb](Assignment6_Doc_QA.ipynb) | Hybrid RAG pipeline with reranking |
| 7 | [Assignment7_graphrag.ipynb](Assignment7_graphrag.ipynb) | GraphRAG knowledge explorer with Neo4j |
| 8 | [Assignment8_Agent_with-Custom_Tools.ipynb](Assignment8_Agent_with-Custom_Tools.ipynb) | ReAct agent with custom tools & memory |
| 9 | [Assignment9_multi_agent.py](Assignment9_multi_agent.py), [Assignment9_mcp_server.py](Assignment9_mcp_server.py), [Assignment9_mcp_client.py](Assignment9_mcp_client.py) | Multi-agent supervisor + MCP server/client |
| 10 | [Assignment10_finetune.ipynb](Assignment10_finetune.ipynb), [Assignment10_dataset.py](Assignment10_dataset.py), [Assignment10_validate.py](Assignment10_validate.py), [Assignment10_compare.py](Assignment10_compare.py), [Assignment10_Modelfile](Assignment10_Modelfile) | LoRA fine-tuning → GGUF → Ollama |
| 11 | [Assignment11_production_api/](Assignment11_production_api/) | Production FastAPI + Docker + evals |

Supporting files:

- [ai_document.txt](ai_document.txt) — source corpus for the RAG assignment
- [graph_document.txt](graph_document.txt) — corpus for GraphRAG (interconnected entities)
- [ollama_benchmark_results.csv](ollama_benchmark_results.csv) — output of Assignment 3
- `fastapi_dataset.json` — synthetic training data (generated, then filtered)
- `fastapi_dataset.raw.json` — the unfiltered generation output, kept for comparison

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -U \
  tiktoken transformers sentence-transformers torch \
  numpy pandas requests pydantic \
  chromadb rank-bm25 \
  langchain langchain-core langchain-ollama langchain-text-splitters \
  langgraph llama-index llama-index-llms-ollama \
  fastmcp jupyter

# Assignments 7, 10 and 11
pip install -U neo4j langchain-neo4j httpx
```

Assignment 10 installs its own dependencies inside Colab; Assignment 11 installs its own inside
Docker. Neither needs anything extra in this environment.

### 3. Install Ollama and pull the models

```bash
ollama serve

ollama pull qwen3:0.6b       # primary model for assignments 4-8
ollama pull tinyllama        # benchmark only
ollama pull phi3:mini        # benchmark only
ollama pull llama3.2         # assignments 7 and 10-11 (extraction, generation)
ollama pull nomic-embed-text # assignment 11 (embeddings without torch)
```

`qwen3:0.6b` is too small for reliable triple extraction and structured generation, so the later
assignments use `llama3.2`.

Hugging Face models (`Qwen/Qwen2-0.5B`, `all-MiniLM-L6-v2`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`) download automatically on first run.

---

## Assignment 1 — Tokenization & Embeddings

**Run:** `python Assignment1_tokenization.py`

Compares two tokenizers on the same string — OpenAI's `cl100k_base` via `tiktoken` and the
Hugging Face `Qwen/Qwen2-0.5B` tokenizer — then moves from tokens to *meaning*:

1. Encodes five sentences (including a Spanish translation of one of them) with the
   `all-MiniLM-L6-v2` sentence-transformer.
2. Uses the Spanish sentence as the query and ranks all five by cosine similarity
   (dot product on normalized embeddings).

**Takeaway:** the Spanish sentence and its English counterpart rank adjacent to each other,
demonstrating that embeddings capture cross-lingual semantics that token counts cannot.

---

## Assignment 2 — XOR Neural Network

**Run:** `python Assignment2_NN_with_XOR.py`

A minimal PyTorch model proving why non-linearity matters. XOR is not linearly separable, so a
single linear layer cannot solve it.

- Architecture: `Linear(2→4)` → `ReLU` → `Linear(4→1)` → `Sigmoid`
- Loss: `BCELoss`; Optimizer: `Adam`
- 5,000 training epochs, then all four input pairs are classified

Output shows raw sigmoid probabilities alongside the rounded classification for
`[0,0] → 0`, `[0,1] → 1`, `[1,0] → 1`, `[1,1] → 0`.

---

## Assignment 3 — Local Model Benchmark

**Run:** `python Assignment3_LLL_Benchmark.py` (requires all three models pulled)

Sends five prompts of increasing difficulty — factual recall, sentiment classification, multi-step
reasoning, code generation, and creative writing — to three local models and records the response
plus latency (Ollama returns `total_duration` in nanoseconds; the script converts to ms).

Results are written to [ollama_benchmark_results.csv](ollama_benchmark_results.csv), which includes a
`Manual Quality Rating (1-5)` column intended to be filled in by hand after reviewing the responses.

**Measured average latency (15 runs total):**

| Model | Avg response time |
|-------|------------------:|
| tinyllama | ~3,904 ms |
| phi3:mini | ~5,441 ms |
| qwen3:0.6b | ~10,774 ms |

`qwen3:0.6b` is the slowest despite being the smallest — it emits explicit reasoning tokens before
answering, which is also why it is the model chosen for the agent assignments.

---

## Assignment 4 — Prompt Engineering & Structured Output

**Run:** `jupyter notebook Assignment4_Prompt_Engg.ipynb`

Three tasks — sentiment classification, entity extraction, and summarization — each attempted with
three prompting strategies:

- **Zero-shot** — instruction only
- **Few-shot** — two worked examples before the real input
- **Chain-of-thought** — explicit numbered reasoning steps

The notebook closes with two things the raw prompting above cannot give you:

- **Structured output** — a Pydantic `Sentiment` model (`sentiment: str`, `confidence: float`) whose
  JSON schema is passed to Ollama's `format` parameter, then validated with
  `model_validate_json()`. This is the difference between "the model usually returns JSON" and
  "the model cannot return anything else."
- **A comparison table** of the three techniques' strengths and weaknesses.

---

## Assignment 5 — LangChain vs. LlamaIndex

**Run:** `python Assignment5_Multi_Step_Langchain.py` and `python Assignment5_Multi_Step_llamaindex.py`

The same three-step pipeline built twice in two different frameworks:

```
topic → explanation (for a 10-year-old) → 5 quiz questions → answer key
```

- **`Assignment5_Multi_Step_Langchain.py`** uses LangChain's LCEL pipe syntax:
  `prompt | model | StrOutputParser()`
- **`Assignment5_Multi_Step_llamaindex.py`** uses LlamaIndex's `PromptTemplate` + `llm.complete()`

Each stage's output feeds the next stage's prompt. Running both side by side makes the abstraction
trade-off concrete — LCEL composes declaratively, LlamaIndex stays closer to plain imperative calls.

---

## Assignment 6 — Hybrid RAG with Reranking

**Run:** `jupyter notebook Assignment6_Doc_QA.ipynb` (reads `ai_document.txt` from the working directory)

A full retrieval pipeline rather than naive top-k vector search:

1. **Chunking** — `RecursiveCharacterTextSplitter`, `chunk_size=300`, `chunk_overlap=50`
2. **Embedding** — `all-MiniLM-L6-v2`, normalized
3. **Vector store** — ChromaDB in-memory collection
4. **Keyword search** — BM25 (`rank_bm25`) over regex-tokenized chunks
5. **Fusion** — Reciprocal Rank Fusion, `score = Σ 1/(k + rank)` with `k=60`
6. **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair, keeping top 3
7. **Generation** — LangChain chain with a strict grounding instruction: answer only from context,
   otherwise reply exactly `"Not in context."`

Five test questions probe different retrieval paths — semantic phrasing, exact keyword matching
("artificial general intelligence"), and historical detail — and each answer is printed with its top
3 chunks plus their RRF and cross-encoder scores, so retrieval quality is inspectable rather than
assumed.

---

## Assignment 7 — GraphRAG Knowledge Explorer with Neo4j

**Run:** `jupyter notebook Assignment7_graphrag.ipynb` (needs Neo4j + `graph_document.txt`)

Combines a **Neo4j knowledge graph** with vector search to answer multi-hop questions that basic
RAG cannot.

Start Neo4j first — Community Edition, a free Aura instance, or Docker:

```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 neo4j:5-community
```

Pipeline:

1. **Paragraph chunking** of `graph_document.txt`
2. **Triple extraction** — `llama3.2` constrained by a JSON schema whose `relationship` field is an
   `enum`, producing `(entity1, relationship, entity2)` with entity types
3. **Cypher ingest** — `MERGE` nodes and relationships; labels and relationship types are validated
   against allow-lists before interpolation, since Cypher cannot parameterise them
4. **`graph_retrieval(query)`** — extract query entities, match them in the graph, traverse 1–2 hops
5. **RRF fusion** of graph facts and vector chunks
6. **Comparison** of pure vector RAG vs GraphRAG on 5 multi-hop questions

**On the corpus.** `graph_document.txt` deliberately separates facts about people from facts about
companies, so reasoning chains cross paragraph boundaries. This matters: in a first draft where each
chain sat inside one paragraph, vector search answered 4 of 5 questions on its own and GraphRAG
showed no advantage. All five questions were then verified against plain vector search — none have
the answer in their top-3 chunks.

**On hop depth.** The brief suggests 1–2 hops, but its own example question is a 3-hop chain
(`Zoho → Chennai → Freshworks → Girish Mathrubootham`), as are two of the five questions here.
`graph_retrieval` keeps the 2-hop default and takes `max_hops` as a parameter; the comparison runs at
3. Because deeper undirected traversal on a small graph reaches much of it, facts are ranked by
their **shallowest hop distance** and truncated — without that, context fills with distant facts and
answers get worse. A dedicated cell shows this trade-off at 1, 2 and 3 hops.

---

## Assignment 8 — ReAct Agent with Custom Tools

**Run:** `jupyter notebook Assignment8_Agent_with-Custom_Tools.ipynb`

A LangGraph `create_react_agent` wired to three custom tools:

| Tool | Behaviour |
|------|-----------|
| `calculate` | Evaluates math via an AST walk with an operator allow-list — no `eval()` |
| `define_word` | Looks up a word in a small hardcoded dictionary |
| `get_current_datetime` | Returns the system date and time |

Conversation state is persisted with `MemorySaver`, keyed by `thread_id`.

Eight tests exercise the full behaviour space: each tool individually, two direct questions the agent
should answer *without* reaching for a tool, a tool-call-plus-reasoning task, a memory test
("My name is Vivek." → "What is my name?"), and a final cell that prints every message in the
trajectory with its `tool_calls`, exposing the raw Thought → Action → Observation loop.

---

## Assignment 9 — Multi-Agent System & MCP

### Part A — Supervisor multi-agent

**Run:** `python Assignment9_multi_agent.py`

Three agents in a supervisor topology, where each specialist is exposed to the supervisor as a tool:

```
                 Supervisor Agent
                 /              \
        research_worker      analysis_worker
              |                    |
      Research Agent          Analysis Agent
              |                    |
    retrieve_information   compare_information
              |                    |
         Knowledge base       LLM comparison
```

The knowledge base holds three topics (artificial intelligence, machine learning, deep learning).
The supervisor's system prompt defines the routing rules, and three tests verify them:

1. Research only — "What is artificial intelligence?"
2. Analysis only — "Compare artificial intelligence and machine learning."
3. **Both workers** — research two topics, then compare them

### Part B — MCP server & client

Two terminals:

```bash
# Terminal 1
python Assignment9_mcp_server.py     # FastMCP, streamable-http on 127.0.0.1:8000

# Terminal 2
python Assignment9_mcp_client.py     # connects to http://127.0.0.1:8000/mcp
```

The server exposes two mock tools, `get_weather(city)` and `get_news(topic)`. The client lists the
available tools over the protocol and then calls both, demonstrating tool discovery and invocation
without the client having any hardcoded knowledge of the server's implementation.

---

## Assignment 10 — Fine-Tune & Deploy a Domain-Specific LLM

Four stages across two machines. Domain: **the FastAPI framework** — the base model knows it well
enough for a teacher model to generate correct data, answers are objectively checkable, and it
carries over into Assignment 11.

| Stage | Where | File |
|-------|-------|------|
| 1. Generate synthetic dataset | Local (Ollama) | [Assignment10_dataset.py](Assignment10_dataset.py) |
| 1b. Validate and filter it | Local | [Assignment10_validate.py](Assignment10_validate.py) |
| 2. LoRA fine-tune | **Colab GPU** | [Assignment10_finetune.ipynb](Assignment10_finetune.ipynb) |
| 3. Merge + quantize to GGUF | **Colab GPU** | same notebook |
| 4. Register + compare | Local (Ollama) | [Assignment10_Modelfile](Assignment10_Modelfile), [Assignment10_compare.py](Assignment10_compare.py) |

```bash
python Assignment10_dataset.py     # generate  -> fastapi_dataset.json
python Assignment10_validate.py    # filter, report which topics need refilling
python Assignment10_dataset.py     # refill the short topics
python Assignment10_validate.py    # confirm the pair count

# upload the JSON to Colab, run the notebook, download the .gguf
ollama create fastapi-expert -f Assignment10_Modelfile
python Assignment10_compare.py
```

Details worth noting:

- **LoRA targets attention projections only** (`q_proj`, `k_proj`, `v_proj`, `o_proj`) at rank 16;
  the notebook prints the trainable percentage and asserts it is under the 5% requirement.
- **Baselines are captured before training, not remembered.** The notebook records the base model's
  answers to the 5 evaluation questions while the adapter is still detached, then prints both side
  by side afterwards.
- **The Modelfile repeats the Alpaca prompt template.** A template mismatch between training and
  inference is the usual reason a fine-tune appears to do nothing.
- **The teacher model hallucinates APIs, and it had to be measured to be believed.**
  An audit of the first 51 generated pairs found **13 (25%) contained FastAPI APIs that do not
  exist** — `from fastapi import CORS`, `from fastapi import OpenAPI`, `EventSourceResponse`
  imported from `fastapi`, a `fastapi new` CLI command, and `@app.route`. Fine-tuning on those
  teaches the model to emit them *confidently*, which is worse than not fine-tuning at all.
  `Assignment10_validate.py` rejects pairs matching known-fake patterns and any pair whose fenced
  code block fails to parse; the generation prompt now names these specific mistakes, and the
  generator refills any topic left short.

  After one generate → validate → refill → validate cycle the shipped dataset is **62 pairs,
  0 contaminated, 16/16 topics covered, and all 28 code blocks parse** — against 51 pairs with 13
  contaminated before. Naming the specific hallucinations in the prompt eliminated the fake
  imports entirely on the second pass; the only remaining rejections were truncated code blocks.
  This is the practical lesson of synthetic data: the teacher model sets the quality ceiling, and
  nobody finds out where that ceiling is unless someone writes the check.
- **Generation is interruption-safe.** The run takes 15-30 minutes against a local model, so
  every call retries with backoff and the dataset is written after each topic. Re-running skips
  topics already in the output file. An earlier version saved only at the end and lost a whole
  run when Ollama dropped a connection.
- **Honest limitation:** ~60 synthetic pairs from a small teacher shift *style*, not capability.
  Fine-tuning teaches behaviour; RAG supplies facts. The notebook ends with that decision framework.

**This notebook cannot run on the Mac in this repo** — Unsloth requires CUDA. Use Colab with a T4.

---

## Assignment 11 — Production LLM API with Evals & Observability

**Run:** `cd Assignment11_production_api && docker compose up --build`

A containerised FastAPI service wrapping the Assignment 6 RAG pipeline. Full details in
[Assignment11_production_api/README.md](Assignment11_production_api/README.md).

- **`POST /chat`** — streaming via Server-Sent Events, or plain JSON with `"stream": false`
- **`GET /health`** — reports Ollama and ChromaDB reachability, 503 when degraded
- **`GET /metrics`** — cache hit rate and cached vs uncached latency from the request log
- **Docker Compose** — API + ChromaDB; Ollama stays on the host via `host.docker.internal`
- **DeepEval suite** — 10 cases, 3 metrics (Answer Relevancy, Faithfulness, Contextual Precision)
- **Logging middleware** — JSON Lines with timestamp, latency, estimated tokens, model, cost, cache

Three decisions that shaped the implementation:

- **Embeddings come from Ollama (`nomic-embed-text`), not sentence-transformers**, which keeps
  `torch` out of the image — roughly 200 MB instead of 2.5 GB.
- **Streaming breaks naive logging middleware.** A streaming response returns from `call_next`
  before its body is sent, so logging there would record zero tokens and only time-to-first-byte.
  The log write is deferred to a `BackgroundTask` that runs after the body is flushed.
- **DeepEval judges with an LLM, defaulting to OpenAI.** `evals.py` supplies a local Ollama judge
  instead, so the suite needs no API key and no data leaves the machine. Two of the ten cases have
  no answer in the corpus — a suite where everything is answerable cannot detect hallucination.

---

## Notes

- The first run of any assignment using Hugging Face models will be slow while weights download.
- `Assignment6_Doc_QA.ipynb` uses an **in-memory** ChromaDB client, so the collection is rebuilt on every kernel
  restart — re-run the notebook top to bottom.
- `Assignment3_LLL_Benchmark.py` must be run from this directory; it writes
  `ollama_benchmark_results.csv` relative to the current working directory.
- `qwen3:0.6b` is a very small model. Agent tool-selection and RAG grounding will occasionally be
  imperfect — that is expected at this size and is itself part of what these assignments demonstrate.
