"""
generate_plots.py
Reads all experiment JSON files, computes metrics, and saves figures
to experiments/analysis/figures/
"""

import json
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – works outside notebooks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_DIR  = os.path.join(BASE_DIR, "experiments")
OUT_DIR  = os.path.join(EXP_DIR, "analysis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

RED   = "#E24B4A"
GREEN = "#1D9E75"
BLUE  = "#2C7BB6"
AMBER = "#F58518"

# ─────────────────────────────────────────────
# Helper: load and compute metrics from a JSON file
# ─────────────────────────────────────────────
def load_metrics(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total   = len(data)
    valid   = 0
    scores  = []
    aligns  = []
    repairs = 0
    failure_counts = {}

    for item in data:
        info = item.get("info", {})
        if "error" not in info:
            valid += 1
        scores.append(info.get("score", 0) or 0)
        aligns.append(info.get("alignment_score", 0) or 0)

        ra = info.get("repair_attempts", {})
        if isinstance(ra, dict):
            repairs += ra.get("json_repair_attempts",    0)
            repairs += ra.get("schema_repair_attempts",  0)

        for m in info.get("failure_modes", []):
            failure_counts[m] = failure_counts.get(m, 0) + 1

    return {
        "total":         total,
        "valid":         valid,
        "validity_rate": round(100 * valid / total, 1) if total else 0,
        "mean_score":    round(np.mean(scores), 3) if scores else 0,
        "mean_align":    round(np.mean(aligns), 3) if aligns else 0,
        "total_repairs": repairs,
        "failure_counts": failure_counts,
    }


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
files = {
    "Stage 0\n(Baseline)":        os.path.join(EXP_DIR, "stage0_baseline_ollama.json"),
    "Stage 1\n(Embedding RAG)":   os.path.join(EXP_DIR, "stage1_embedding_only.json"),
    "Stage 3\n(TF-IDF RAG)":      os.path.join(EXP_DIR, "stage3_tfidf_rag.json"),
}

metrics = {}
for label, path in files.items():
    if os.path.exists(path):
        metrics[label] = load_metrics(path)
        print(f"Loaded: {os.path.basename(path)}  →  {metrics[label]}")
    else:
        print(f"MISSING: {path}")

labels       = list(metrics.keys())
validity     = [metrics[l]["validity_rate"] for l in labels]
mean_scores  = [metrics[l]["mean_score"]    for l in labels]
mean_aligns  = [metrics[l]["mean_align"]    for l in labels]
repair_cnts  = [metrics[l]["total_repairs"] for l in labels]
colors_bars  = [RED, GREEN, GREEN]   # Stage 0 is baseline (red)

# ─────────────────────────────────────────────
# Figure 1 – Validity Rate bar chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, validity, color=colors_bars, width=0.45,
              edgecolor="white", linewidth=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel("JSON Validity Rate (%)", fontsize=11)
ax.set_title("Effect of Schema Enforcement on JSON Validity\n"
             "(Ollama LLaMA3 8B, N=100 queries)", fontsize=11)
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
for bar, val in zip(bars, validity):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5, f"{val}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# legend
patches = [mpatches.Patch(color=RED,   label="No enforcement (Stage 0)"),
           mpatches.Patch(color=GREEN, label="With enforcement (Stages 1 & 3)")]
ax.legend(handles=patches, fontsize=9, frameon=False, loc="upper left")
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "figure1_validity_rate.pdf")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────
# Figure 2 – Consistency Score vs Alignment Score (grouped bar)
# ─────────────────────────────────────────────
x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, mean_scores, w, label="Structural Consistency Score",
            color=BLUE,  edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + w/2, mean_aligns, w, label="Query Alignment Score",
            color=AMBER, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score (0–1)", fontsize=11)
ax.set_title("Structural Consistency & Query Alignment by Stage\n"
             "(Ollama LLaMA3 8B, N=100 queries)", fontsize=11)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            h + 0.02, f"{h:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "figure2_scores.pdf")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────
# Figure 3 – Repair Counts
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bars3 = ax.bar(labels, repair_cnts, color=[RED, BLUE, AMBER],
               width=0.45, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Total Repair Attempts", fontsize=11)
ax.set_title("LLM Repair Attempts per Stage\n"
             "(Ollama LLaMA3 8B, N=100 queries)", fontsize=11)
for bar, val in zip(bars3, repair_cnts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3, str(val),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(repair_cnts + [1]) * 1.3)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "figure3_repair_counts.pdf")
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.savefig(out3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ─────────────────────────────────────────────
# Figure 4 – Failure mode breakdown (Stage 0 only – others have very few)
# ─────────────────────────────────────────────
s0_failures = metrics.get("Stage 0\n(Baseline)", {}).get("failure_counts", {})
if s0_failures:
    fm_labels = list(s0_failures.keys())
    fm_vals   = list(s0_failures.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(fm_labels, fm_vals, color=[RED, AMBER, BLUE][:len(fm_labels)],
            edgecolor="white")
    ax.set_xlabel("Number of Queries", fontsize=11)
    ax.set_title("Stage 0 Failure Mode Breakdown\n"
                 "(enforce_schema=False, N=100)", fontsize=11)
    for i, val in enumerate(fm_vals):
        ax.text(val + 0.3, i, str(val), va="center", fontsize=10)
    ax.set_xlim(0, max(fm_vals) * 1.25)
    plt.tight_layout()
    out4 = os.path.join(OUT_DIR, "figure4_failure_modes.pdf")
    plt.savefig(out4, dpi=300, bbox_inches="tight")
    plt.savefig(out4.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")

print(f"\nAll figures saved to: {OUT_DIR}")
