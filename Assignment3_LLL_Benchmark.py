import requests
import pandas as pd

# ----------------------------------------------------------
# ASSIGNMENT:
# Pull these models before running:
#
# ollama pull tinyllama
# ollama pull qwen3:0.6b
# ollama pull phi3:mini
# ----------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "tinyllama",
    "qwen3:0.6b",
    "phi3:mini"
]

# ----------------------------------------------------------
# Five prompts of varying difficulty
# ----------------------------------------------------------

PROMPTS = [
    {
        "type": "Simple Factual Question",
        "prompt": "What is the capital of Japan?"
    },
    {
        "type": "Sentiment Classification",
        "prompt": "Classify the sentiment of this sentence as Positive, Negative, or Neutral: 'The customer service was slow but the staff was very polite.'"
    },
    {
        "type": "Reasoning Problem",
        "prompt": "A train travels 60 km in 1 hour. How long will it take to travel 150 km at the same speed? Explain your reasoning."
    },
    {
        "type": "Code Generation",
        "prompt": "Write a Python function to check whether a string is a palindrome."
    },
    {
        "type": "Creative Writing",
        "prompt": "Write a short science-fiction story in about 100 words about humans living on Mars."
    }
]

results = []

print("=" * 80)
print("OLLAMA MODEL BENCHMARK")
print("=" * 80)

for model in MODELS:

    print(f"\nTesting Model: {model}")

    for item in PROMPTS:

        prompt = item["prompt"]

        print(f"  -> {item['type']}")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        try:

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get("response", "")

            # Ollama returns nanoseconds
            total_duration_ns = data.get("total_duration", 0)

            response_time_ms = round(total_duration_ns / 1_000_000, 2)

            # ---------------------------------------------------
            # Manual Quality Rating
            #
            # Review the response after execution and replace
            # these values (1-5).
            # ---------------------------------------------------

            quality = 0

            results.append({
                "Model": model,
                "Prompt Type": item["type"],
                "Prompt": prompt,
                "Response": answer,
                "Response Time (ms)": response_time_ms,
                "Manual Quality Rating (1-5)": quality
            })

            print(f"     Response Time : {response_time_ms} ms")

        except Exception as e:

            print("     ERROR:", e)

            results.append({
                "Model": model,
                "Prompt Type": item["type"],
                "Prompt": prompt,
                "Response": str(e),
                "Response Time (ms)": None,
                "Manual Quality Rating (1-5)": 0
            })

# ----------------------------------------------------------
# Save Complete Results
# ----------------------------------------------------------

df = pd.DataFrame(results)

df.to_csv("ollama_benchmark_results.csv", index=False)

# ----------------------------------------------------------
# Comparison Report
# ----------------------------------------------------------

summary = (
    df.groupby("Model")
      .agg(
          Average_Response_Time_ms=("Response Time (ms)", "mean")
      )
      .round(2)
      .reset_index()
)

print("\n")
print("=" * 80)
print("MODEL COMPARISON REPORT")
print("=" * 80)

print(summary)

print("\nDetailed results saved to:")
print("ollama_benchmark_results.csv")