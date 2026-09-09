"""
Assignment 10 - Step 4: Compare Base vs Fine-Tuned Model
========================================================

Runs the 5 evaluation questions against both the base model and the fine-tuned
model registered with Ollama, and prints the answers side by side with latency.

Prerequisites:
    ollama pull qwen2.5:1.5b                              # base model
    ollama create fastapi-expert -f Assignment10_Modelfile  # fine-tuned model

Run:
    python Assignment10_compare.py
"""

import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

BASE_MODEL = "qwen2.5:1.5b"
TUNED_MODEL = "fastapi-expert"

EVAL_QUESTIONS = [
    "How do I add a query parameter with a default value in FastAPI?",
    "How do I return a 404 error from a FastAPI endpoint?",
    "How do I use dependency injection in FastAPI?",
    "How do I stream a response from FastAPI using Server-Sent Events?",
    "How do I upload a file in FastAPI?",
]


def ask(model, prompt):
    """Send one prompt to a local Ollama model."""

    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=600,
        )

        response.raise_for_status()

        return response.json().get("response", "").strip(), time.time() - start

    except requests.HTTPError:
        return f"[model '{model}' not found - see prerequisites above]", 0.0
    except requests.RequestException as error:
        return f"[request failed: {error}]", 0.0


def main():

    print("=" * 78)
    print("BASE vs FINE-TUNED COMPARISON")
    print(f"Base : {BASE_MODEL}")
    print(f"Tuned: {TUNED_MODEL}")
    print("=" * 78)

    for i, question in enumerate(EVAL_QUESTIONS, 1):

        print("\n" + "=" * 78)
        print(f"QUESTION {i}: {question}")
        print("=" * 78)

        base_answer, base_time = ask(BASE_MODEL, question)
        tuned_answer, tuned_time = ask(TUNED_MODEL, question)

        print(f"\n--- BASE MODEL ({base_time:.1f}s) ---")
        print(base_answer[:600])

        print(f"\n--- FINE-TUNED MODEL ({tuned_time:.1f}s) ---")
        print(tuned_answer[:600])

    print("\n" + "=" * 78)
    print("What to look for:")
    print("  - Format consistency: does the tuned model answer in a stable shape?")
    print("  - Answer length: fine-tuning on short answers shortens responses.")
    print("  - Correctness: fine-tuning changes style far more than knowledge,")
    print("    so do not expect new facts the base model never had.")
    print("=" * 78)


if __name__ == "__main__":
    main()
