"""
Assignment 11 - DeepEval Test Suite
===================================

Evaluates the RAG pipeline on 10 test cases with 3 metrics:

- Answer Relevancy    : does the answer address the question?
- Faithfulness        : is the answer grounded in the retrieved context?
- Contextual Precision: did retrieval rank the relevant chunks highest?

DeepEval judges with an LLM. By default that is OpenAI, which needs an API key
and sends your data off-machine, so this suite plugs in a local Ollama judge
instead - the whole assignment stays offline and free.

Prerequisites:
    pip install -r requirements-dev.txt
    docker compose up -d chroma      # vector DB must be reachable
    ollama serve

Run:
    python evals.py
"""

import json

from pydantic import BaseModel

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

import httpx

import rag_backend


# ============================================================
# Local judge model
# ============================================================

class OllamaJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by a local Ollama model."""

    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def load_model(self):
        return self.model

    def generate(self, prompt, schema=None):
        """Generate a judgement, honouring DeepEval's pydantic schema when given."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }

        # DeepEval passes a pydantic model class when it needs structured output.
        # Ollama enforces it natively via the format parameter.
        if schema is not None:
            payload["format"] = schema.model_json_schema()

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=900,
        )

        response.raise_for_status()

        content = response.json()["message"]["content"]

        if schema is not None:
            return schema.model_validate_json(content)

        return content

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return f"Ollama ({self.model})"


# ============================================================
# Test cases
# ============================================================

# Ten questions over corpus.txt. The last two have no answer in the corpus,
# so a faithful system must decline rather than invent one - a suite where
# every question is answerable cannot detect hallucination.
TEST_CASES = [
    (
        "What is artificial intelligence?",
        "Intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans and other animals.",
    ),
    (
        "How do leading AI textbooks define the field?",
        "As the study of intelligent agents: systems that perceive their environment and take actions that maximize the achievement of goals.",
    ),
    (
        "When was artificial intelligence founded as an academic discipline?",
        "In 1956.",
    ),
    (
        "What are AI winters?",
        "Periods of disappointment and loss of funding that followed cycles of optimism in the AI field.",
    ),
    (
        "How do machine learning systems work?",
        "Instead of explicitly programming every rule, they learn patterns from data and use those patterns to make predictions or decisions on new data.",
    ),
    (
        "What is deep learning?",
        "A type of machine learning based on artificial neural networks with multiple layers.",
    ),
    (
        "What is natural language processing used for?",
        "Translation, chatbots, text classification, information retrieval, and speech assistants.",
    ),
    (
        "What is the difference between narrow and general artificial intelligence?",
        "Narrow AI performs specific tasks, while general AI is a hypothetical machine able to perform a wide range of intellectual tasks at a human-comparable level.",
    ),
    (
        "What challenges does artificial intelligence create?",
        "Privacy concerns, bias in automated decisions, security risks, job displacement, misinformation, and questions about accountability.",
    ),
    (
        "Who is the current chief executive of OpenAI?",
        "Not in context.",
    ),
]


def build_test_cases():
    """Run each question through the real pipeline and wrap the result."""

    rag_backend.init_index()

    cases = []

    for question, expected in TEST_CASES:

        print(f"  running: {question}")

        answer, chunks = rag_backend.answer(question)

        cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected,
                retrieval_context=chunks,
            )
        )

    return cases


def main():

    print("=" * 70)
    print("BUILDING TEST CASES (calling the RAG pipeline)")
    print("=" * 70)

    cases = build_test_cases()

    judge = OllamaJudge()

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False),
        FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model=judge, async_mode=False),
    ]

    print("\n" + "=" * 70)
    print(f"EVALUATING {len(cases)} CASES x {len(metrics)} METRICS")
    print("Judge:", judge.get_model_name())
    print("A local judge is slow - expect several minutes.")
    print("=" * 70)

    results = evaluate(test_cases=cases, metrics=metrics)

    with open("eval_results.json", "w", encoding="utf-8") as file:
        json.dump(
            {"summary": str(results)},
            file,
            indent=2,
        )

    print("\nSaved raw results to eval_results.json")


if __name__ == "__main__":
    main()
