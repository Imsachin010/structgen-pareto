import json
import os
from statistics import mean

INPUT_PATH = "data/generated_outputs.json"
OUT_JSON = "experiments/ablation_summary.json"
OUT_MD = "experiments/ablation_summary.md"


def summarize_single_run(data):
    total = len(data)
    valid = 0
    structural_scores = []
    alignment_scores = []

    for entry in data:
        info = entry.get("info", {})

        if "error" not in info:
            valid += 1
            structural_scores.append(info.get("score", 0))
            alignment_scores.append(info.get("alignment_score", 0))

    return {
        "total_samples": total,
        "validity_rate": round(valid / total, 3) if total > 0 else 0,
        "avg_structural_score": round(mean(structural_scores), 3) if structural_scores else 0,
        "avg_alignment_score": round(mean(alignment_scores), 3) if alignment_scores else 0,
    }


def summarize_ablation(data):
    summary = {}

    for setting, results in data.items():
        total = len(results)
        valid = 0
        structural_scores = []
        alignment_scores = []

        for r in results:
            if r and "error" not in r:
                valid += 1
                structural_scores.append(r.get("score", 0))
                alignment_scores.append(r.get("alignment_score", 0))

        summary[setting] = {
            "total_samples": total,
            "validity_rate": round(valid / total, 3) if total > 0 else 0,
            "avg_structural_score": round(mean(structural_scores), 3) if structural_scores else 0,
            "avg_alignment_score": round(mean(alignment_scores), 3) if alignment_scores else 0,
        }

    return summary


def summarize():
    if not os.path.exists(INPUT_PATH):
        print("Input file not found.")
        return

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # Detect format
    if isinstance(data, list):
        summary = {"single_run": summarize_single_run(data)}
    elif isinstance(data, dict):
        summary = summarize_ablation(data)
    else:
        print("Unsupported JSON format.")
        return

    os.makedirs("experiments", exist_ok=True)

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown
    with open(OUT_MD, "w") as f:
        f.write("# Ablation Study Summary\n\n")
        f.write("| Setting | Validity Rate | Avg Structural | Avg Alignment |\n")
        f.write("|----------|--------------|----------------|---------------|\n")

        for setting, stats in summary.items():
            f.write(
                f"| {setting} | "
                f"{stats['validity_rate']} | "
                f"{stats['avg_structural_score']} | "
                f"{stats['avg_alignment_score']} |\n"
            )

    print("Summary generated successfully.")


if __name__ == "__main__":
    summarize()
