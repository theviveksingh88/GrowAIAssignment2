"""
Assignment 10 - Step 1b: Validate the Synthetic Dataset
=======================================================

Synthetic data is only as good as the teacher model, and a small teacher
hallucinates APIs. In one run of Assignment10_dataset.py, 25% of the pairs
contained imports that do not exist - `from fastapi import CORS`,
`from fastapi import OpenAPI`, `EventSourceResponse` imported from `fastapi`,
and a `fastapi new` CLI command. Fine-tuning on those teaches the model to
produce them confidently, which is worse than not fine-tuning at all.

This script drops the bad pairs and reports which topics need refilling.

Workflow:
    python Assignment10_dataset.py     # generate
    python Assignment10_validate.py    # filter, report short topics
    python Assignment10_dataset.py     # refill the topics that lost pairs
    python Assignment10_validate.py    # confirm

Run:
    python Assignment10_validate.py
"""

import ast
import json
import os
import re
import shutil
from collections import Counter

DATASET_PATH = "fastapi_dataset.json"
BACKUP_PATH = "fastapi_dataset.raw.json"

TARGET_MIN_PAIRS = 50
PAIRS_PER_TOPIC = 4

# Each rule is (pattern, human-readable reason). All were observed in real output.
REJECT_RULES = [
    (r"from\s+fastapi\s+import\s+[^\n]*\bCORSS?\b",
     "CORS imported from fastapi (real: fastapi.middleware.cors.CORSMiddleware)"),
    (r"from\s+fastapi\s+import\s+[^\n]*\bOpenAPI\b",
     "OpenAPI imported from fastapi (real: app.openapi())"),
    (r"from\s+fastapi\s+import\s+[^\n]*\bEventSourceResponse\b",
     "EventSourceResponse imported from fastapi (real: sse_starlette.sse)"),
    (r"\bfastapi\s+new\b",
     "fastapi new (no such CLI command; real: fastapi dev / fastapi run)"),
    (r"@app\.route\b",
     "@app.route (FastAPI uses @app.get / @app.post / ...)"),
    (r"from\s+fastapi\s+import\s+[^\n]*\bMiddleware\b",
     "Middleware imported from fastapi (real: starlette.middleware)"),
]

CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def find_problems(pair):
    """Return a list of reasons this pair should be rejected."""

    text = pair["instruction"] + "\n" + pair["output"]

    problems = [reason for pattern, reason in REJECT_RULES
                if re.search(pattern, text, re.I)]

    # Any fenced Python block must at least parse.
    for block in CODE_BLOCK.findall(pair["output"]):
        try:
            ast.parse(block)
        except SyntaxError as error:
            problems.append(f"code block does not parse: {error.msg}")
            break

    return problems


def main():

    if not os.path.exists(DATASET_PATH):
        print(f"{DATASET_PATH} not found - run Assignment10_dataset.py first")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        rows = json.load(file)

    if not os.path.exists(BACKUP_PATH):
        shutil.copy(DATASET_PATH, BACKUP_PATH)
        print(f"Unfiltered copy saved to {BACKUP_PATH}")

    kept, rejected = [], []

    for row in rows:
        problems = find_problems(row)
        (rejected if problems else kept).append((row, problems))

    print("=" * 72)
    print("DATASET VALIDATION")
    print("=" * 72)
    print(f"Input pairs : {len(rows)}")
    print(f"Kept        : {len(kept)}")
    print(f"Rejected    : {len(rejected)}")

    if rejected:
        print("\nRejection reasons:")
        for reason, count in Counter(
            problem for _, problems in rejected for problem in problems
        ).most_common():
            print(f"  {count:2}  {reason}")

    clean = [row for row, _ in kept]

    with open(DATASET_PATH, "w", encoding="utf-8") as file:
        json.dump(clean, file, indent=2, ensure_ascii=False)

    counts = Counter(row.get("topic") for row in clean)

    short = [topic for topic, count in counts.items() if count < PAIRS_PER_TOPIC]

    print(f"\nTopics with fewer than {PAIRS_PER_TOPIC} pairs: {len(short)}")
    for topic in short:
        print(f"  {counts[topic]}/{PAIRS_PER_TOPIC}  {topic}")

    print("\n" + "=" * 72)

    if len(clean) < TARGET_MIN_PAIRS:
        print(f"{len(clean)} pairs is below the {TARGET_MIN_PAIRS} the assignment "
              f"requires.\nRun Assignment10_dataset.py again to refill short topics, "
              f"then re-run this script.")
    else:
        print(f"{len(clean)} clean pairs - meets the {TARGET_MIN_PAIRS}-pair minimum.")

    print("=" * 72)


if __name__ == "__main__":
    main()
