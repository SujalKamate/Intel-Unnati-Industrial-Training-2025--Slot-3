import pandas as pd

# Load evaluation dataset
df = pd.read_csv("evaluation_data.csv")

# Convert Yes/No to numeric
df["language_match"] = df["language_match"].map({"Yes": 1, "No": 0})

# Metrics to evaluate
metrics = ["relevance", "correctness", "groundedness", "language_match"]

# Compute averages
results = {}
for m in metrics:
    results[m] = round(df[m].mean(), 2)

# Hallucination rate
hallucination_rate = round(
    1 - df["groundedness"].mean() / 5, 2
)

print("📊 Evaluation Results")
print("-" * 30)
for k, v in results.items():
    print(f"{k.capitalize()}: {v}")

print(f"Hallucination Rate: {hallucination_rate * 100}%")