"""
Assignment 10 - Step 1: Synthetic Instruction Dataset Generation
================================================================

Generates 50-100 instruction-response pairs for a narrow domain using a local
Ollama model, and saves them in Alpaca format for fine-tuning.

Domain: the FastAPI web framework.

Chosen because the base model has seen enough FastAPI to produce correct
answers, the output is objectively checkable (the code either runs or it does
not), and the same domain is used again in Assignment 11.

The run takes 15-30 minutes, so it is built to survive interruption:

- each Ollama call retries with backoff on transient connection errors
- the dataset is written after every topic, not once at the end
- re-running skips topics already present in the output file

Run:
    ollama serve
    ollama pull llama3.2
    python Assignment10_dataset.py

Output:
    fastapi_dataset.json     Alpaca format: instruction / input / output
"""

import json
import os
import time
from collections import Counter

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# A larger model generates the data that the smaller model will be trained on.
# This is the "teacher" in synthetic distillation.
TEACHER_MODEL = "llama3.2"

OUTPUT_PATH = "fastapi_dataset.json"

PAIRS_PER_TOPIC = 4

MAX_RETRIES = 3

# 16 topics x 4 pairs = 64 pairs, inside the 50-100 range the assignment asks for.
TOPICS = [
    "creating a basic FastAPI application and running it with uvicorn",
    "path parameters and type conversion",
    "query parameters and default values",
    "request bodies with Pydantic models",
    "response models and response_model_exclude",
    "status codes and returning custom responses",
    "dependency injection with Depends",
    "async endpoints and when to use async def vs def",
    "error handling with HTTPException",
    "middleware and CORS configuration",
    "background tasks",
    "streaming responses and Server-Sent Events",
    "file uploads with UploadFile",
    "automatic OpenAPI documentation",
    "testing FastAPI applications with TestClient",
    "structuring larger applications with APIRouter",
]

DATASET_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["instruction", "output"],
            },
        }
    },
    "required": ["pairs"],
}


def post_with_retry(payload):
    """POST to Ollama, retrying transient failures with backoff.

    Ollama can drop a connection when it unloads or reloads a model under
    memory pressure. Without a retry, one dropped connection ends the whole run.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=900)
            response.raise_for_status()
            return response

        except requests.RequestException as error:

            last_error = error

            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"    attempt {attempt} failed ({type(error).__name__}), "
                      f"retrying in {wait}s")
                time.sleep(wait)

    print(f"    giving up after {MAX_RETRIES} attempts: {last_error}")

    return None


def generate_pairs(topic, count):
    """Ask the teacher model for instruction-response pairs on one topic."""

    prompt = f"""You are creating training data for a FastAPI coding assistant.

Write {count} instruction-response pairs about: {topic}

Rules for each pair:
- "instruction" is a realistic question a developer would ask.
- "output" is a correct, complete answer.
- Include a short Python code example in the output where it helps.
- Keep each output under 150 words.
- Vary the phrasing of the instructions.
- Only use real FastAPI APIs. Do not invent functions.

Import rules - these are the mistakes to avoid, all seen in earlier runs:
- CORS middleware is `from fastapi.middleware.cors import CORSMiddleware`.
  There is no `CORS` object in the `fastapi` package.
- Server-Sent Events use `StreamingResponse` from `fastapi.responses`,
  or `EventSourceResponse` from `sse_starlette.sse`. `EventSourceResponse`
  is NOT importable from `fastapi`.
- OpenAPI schemas come from `app.openapi()`. There is no `OpenAPI` class
  to import from `fastapi`.
- The CLI has `fastapi dev` and `fastapi run`. There is no `fastapi new`.
- Route decorators are `@app.get`, `@app.post` and so on, never `@app.route`.
"""

    response = post_with_retry({
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": DATASET_SCHEMA,
        "options": {"temperature": 0.7},
    })

    if response is None:
        return []

    content = response.json()["message"]["content"]

    try:
        return json.loads(content).get("pairs", [])
    except json.JSONDecodeError:
        print("    could not parse response, skipping topic")
        return []


def load_existing():
    """Load a partial dataset from a previous run, if one exists."""

    if not os.path.exists(OUTPUT_PATH):
        return []

    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
            rows = json.load(file)
        print(f"Resuming: {len(rows)} pairs already in {OUTPUT_PATH}")
        return rows
    except (json.JSONDecodeError, OSError):
        return []


def save(rows):
    """Write the dataset, dropping duplicate instructions."""

    seen = set()
    deduped = []

    for row in rows:

        key = row["instruction"].lower().strip()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(deduped, file, indent=2, ensure_ascii=False)

    return deduped


def main():

    dataset = load_existing()

    # Count per topic rather than a simple "seen" set: a topic whose pairs were
    # dropped by the validator is short, and should be regenerated to refill it.
    topic_counts = Counter(row.get("topic") for row in dataset)

    print("=" * 70)
    print("SYNTHETIC DATASET GENERATION")
    print(f"Teacher model: {TEACHER_MODEL}")
    print("=" * 70)

    for i, topic in enumerate(TOPICS, 1):

        if topic_counts.get(topic, 0) >= PAIRS_PER_TOPIC:
            print(f"\n[{i}/{len(TOPICS)}] skipping (already complete): {topic}")
            continue

        print(f"\n[{i}/{len(TOPICS)}] {topic}", flush=True)

        start = time.time()

        pairs = generate_pairs(topic, PAIRS_PER_TOPIC)

        added = 0

        for pair in pairs:

            instruction = pair.get("instruction", "").strip()
            output = pair.get("output", "").strip()

            if not instruction or not output:
                continue

            dataset.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "topic": topic,
            })

            added += 1

        # Save after every topic so an interrupted run keeps its progress.
        deduped = save(dataset)

        print(f"    +{added} pairs, {len(deduped)} total "
              f"({time.time() - start:.0f}s)", flush=True)

    deduped = save(dataset)

    print("\n" + "=" * 70)
    print(f"Collected : {len(dataset)} pairs")
    print(f"After dedup: {len(deduped)} pairs")
    print(f"Saved to  : {OUTPUT_PATH}")
    print("=" * 70)

    if deduped:
        print("\nExample pair:\n")
        print("INSTRUCTION:", deduped[0]["instruction"])
        print("OUTPUT:", deduped[0]["output"][:300])


if __name__ == "__main__":
    main()
