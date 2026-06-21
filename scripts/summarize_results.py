import pandas as pd

INPUT = "experiments/ablation_results.csv"

df = pd.read_csv(INPUT)

summary = df.groupby("stage").agg({
    "valid": "mean",
    "struct_score": "mean",
    "align_score": "mean",
    "repairs": "mean"
})

summary["valid"] = summary["valid"] * 100

print("\nRESULT TABLE\n")
print(summary.round(3))