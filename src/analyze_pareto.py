"""
analyze_pareto.py
-----------------
Loads JSONL logs from run_scale_experiment.py and produces:
  - Summary table (Table 2 equivalent for BDA paper)
  - Resource breakdown table (Table 3)
  - Derived efficiency table (Table 4)
  - Pareto frontier plot (Figure 1)
  - Marginal utility plot (Figure 2)
  - Alignment vs retrieval plot (Figure 3)
  - Repair distribution plot (Figure 4)

All figures saved to ./figures/, all tables to ./tables/ as CSV + LaTeX.

Usage:
    python analyze_pareto.py --logs logs/
    python analyze_pareto.py --logs logs/ --out-prefix bda_paper
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Plot style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

CONFIG_LABELS = {
    "C1": "C1: Raw Baseline",
    "C2": "C2: Schema+Repair",
    "C3": "C3: BGE RAG",
    "C4": "C4: TF-IDF RAG",
}
CONFIG_COLORS = {
    "C1": "#d62728",   # red
    "C2": "#1f77b4",   # blue
    "C3": "#2ca02c",   # green
    "C4": "#ff7f0e",   # orange
}
CONFIG_ORDER = ["C1", "C2", "C3", "C4"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_logs(log_dir: Path) -> pd.DataFrame:
    """Load all JSONL logs, take the latest file per config."""
    records = []
    for config_id in CONFIG_ORDER:
        files = sorted(log_dir.glob(f"{config_id}_*.jsonl"))
        if not files:
            print(f"[warn] No log found for {config_id}")
            continue
        latest = files[-1]
        print(f"Loading {latest}  ({latest.stat().st_size // 1024} KB)")
        with open(latest) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        raise ValueError(f"No log records found in {log_dir}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records across {df['config_id'].nunique()} configs.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_tokens"] = df["prompt_tokens"].fillna(0) + df["completion_tokens"].fillna(0)

    # Energy per valid sample (mJ)
    valid_mask = df["schema_valid"] == True
    df["energy_per_valid_mj"] = np.where(
        valid_mask & df["gpu_energy_mj"].notna(),
        df["gpu_energy_mj"],
        np.nan
    )

    # Latency per valid sample (ms)
    df["latency_per_valid_ms"] = np.where(valid_mask, df["latency_ms"], np.nan)

    # Tokens per valid sample
    df["tokens_per_valid"] = np.where(valid_mask, df["total_tokens"], np.nan)

    # Repair cost ratio = repair_count / (repair_count + 1)
    df["repair_cost_ratio"] = df["repair_count"] / (df["repair_count"] + 1)

    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Produce per-config summary statistics."""
    rows = []
    for config_id in CONFIG_ORDER:
        sub = df[df["config_id"] == config_id]
        if sub.empty:
            continue
        n = len(sub)
        n_valid = sub["schema_valid"].sum()

        row = {
            "config_id":          config_id,
            "config_label":       CONFIG_LABELS.get(config_id, config_id),
            "n_queries":          n,

            # Reliability
            "json_validity_pct":    100 * sub["json_valid"].mean(),
            "schema_validity_pct":  100 * sub["schema_valid"].mean(),
            "mean_repair_count":    sub["repair_count"].mean(),
            "total_repairs":        sub["repair_count"].sum(),
            "alignment_mean":       sub["alignment_score"].mean(),
            "alignment_std":        sub["alignment_score"].std(),
            "structural_mean":      sub["structural_score"].mean(),

            # Latency
            "latency_mean_ms":      sub["latency_ms"].mean(),
            "latency_median_ms":    sub["latency_ms"].median(),
            "latency_p95_ms":       sub["latency_ms"].quantile(0.95),

            # GPU energy
            "gpu_energy_mean_mj":   sub["gpu_energy_mj"].mean(),
            "gpu_energy_total_j":   sub["gpu_energy_mj"].sum() / 1000.0,
            "gpu_util_mean_pct":    sub["gpu_util_pct"].mean(),
            "gpu_mem_peak_mb":      sub["gpu_mem_mb"].max(),

            # CPU energy
            "cpu_energy_mean_mj":   sub["cpu_energy_mj"].mean(),

            # Tokens
            "total_tokens_mean":    sub["total_tokens"].mean(),
            "total_tokens_sum":     sub["total_tokens"].sum(),

            # Derived efficiency
            "energy_per_valid_mj":  (sub.loc[sub["schema_valid"], "gpu_energy_mj"].mean()
                                     if n_valid > 0 else np.nan),
            "latency_per_valid_ms": (sub.loc[sub["schema_valid"], "latency_ms"].mean()
                                     if n_valid > 0 else np.nan),
            "tokens_per_valid":     (sub.loc[sub["schema_valid"], "total_tokens"].mean()
                                     if n_valid > 0 else np.nan),
        }

        # Reliability gain per unit energy (Δvalidity / Δenergy) — computed later
        rows.append(row)

    agg = pd.DataFrame(rows).set_index("config_id")
    return agg


def compute_marginal_utility(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute incremental reliability gain and resource cost per step C1->C2->C3->C4."""
    rows = []
    configs = [c for c in CONFIG_ORDER if c in agg.index]
    for i in range(1, len(configs)):
        prev, curr = configs[i-1], configs[i]
        delta_validity = agg.loc[curr, "schema_validity_pct"] - agg.loc[prev, "schema_validity_pct"]
        delta_energy   = agg.loc[curr, "gpu_energy_mean_mj"]  - agg.loc[prev, "gpu_energy_mean_mj"]
        delta_latency  = agg.loc[curr, "latency_mean_ms"]     - agg.loc[prev, "latency_mean_ms"]
        delta_tokens   = agg.loc[curr, "total_tokens_mean"]   - agg.loc[prev, "total_tokens_mean"]

        efficiency = (delta_validity / delta_energy
                      if delta_energy and not np.isnan(delta_energy) and delta_energy != 0
                      else np.nan)

        rows.append({
            "transition":           f"{prev}->{curr}",
            "delta_validity_pp":    delta_validity,
            "delta_energy_mj":      delta_energy,
            "delta_latency_ms":     delta_latency,
            "delta_tokens":         delta_tokens,
            "reliability_per_mj":   efficiency,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PARETO FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pareto_frontier(agg: pd.DataFrame) -> list[str]:
    """
    Identify Pareto-optimal configs: maximize reliability, minimize energy.
    A config dominates another if it has ≥ reliability AND ≤ energy.
    """
    configs = [c for c in CONFIG_ORDER if c in agg.index]
    dominated = set()
    for i, ci in enumerate(configs):
        for j, cj in enumerate(configs):
            if i == j:
                continue
            ri = agg.loc[ci, "schema_validity_pct"]
            ei = agg.loc[ci, "gpu_energy_mean_mj"]
            rj = agg.loc[cj, "schema_validity_pct"]
            ej = agg.loc[cj, "gpu_energy_mean_mj"]
            if np.isnan(ri) or np.isnan(ei) or np.isnan(rj) or np.isnan(ej):
                continue
            # cj dominates ci if cj has >= reliability AND <= energy
            if rj >= ri and ej <= ei and (rj > ri or ej < ei):
                dominated.add(ci)
    pareto = [c for c in configs if c not in dominated]
    return pareto


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pareto(agg: pd.DataFrame, pareto: list[str], out: Path):
    """Figure 1: Reliability vs GPU energy scatter with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    configs = [c for c in CONFIG_ORDER if c in agg.index]
    xs = [agg.loc[c, "gpu_energy_mean_mj"] / 1000 for c in configs]
    ys = [agg.loc[c, "schema_validity_pct"]     for c in configs]

    for c, x, y in zip(configs, xs, ys):
        is_pareto = c in pareto
        ax.scatter(x, y,
                   color=CONFIG_COLORS[c],
                   s=120,
                   zorder=5,
                   edgecolors="black" if is_pareto else "none",
                   linewidths=1.5 if is_pareto else 0,
                   marker="*" if is_pareto else "o")
        
        # Custom offsets to prevent labels from overlapping each other or the points
        if c == "C1":
            offset = (-30, 10)  # above-left
        elif c == "C2":
            offset = (10, -15)  # below-right
        elif c == "C3":
            offset = (10, 10)   # above-right
        else: # C4
            offset = (-40, -15) # below-left

        ax.annotate(CONFIG_LABELS[c], (x, y),
                    textcoords="offset points", xytext=offset, fontsize=9)

    # Draw Pareto frontier line
    pareto_pts = sorted(
        [(agg.loc[c, "gpu_energy_mean_mj"] / 1000, agg.loc[c, "schema_validity_pct"])
         for c in pareto],
        key=lambda p: p[0]
    )
    if len(pareto_pts) > 1:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "k--", linewidth=1.2, alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Mean GPU Energy per Query (Joules)")
    ax.set_ylabel("Schema Validity Rate (%)")
    ax.set_title("Reliability–Energy Tradeoff: Pareto Frontier")

    star_patch  = mpatches.Patch(color="none", label="★ = Pareto-optimal")
    ax.legend(handles=[
        mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
        for c in configs
    ] + [star_patch], loc="center left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def fig_marginal_utility(marginal: pd.DataFrame, out: Path):
    """Figure 2: Marginal reliability gain and energy cost per transition."""
    if marginal.empty:
        print("[warn] No marginal data — skipping Figure 2.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    transitions = marginal["transition"]
    x = np.arange(len(transitions))

    bars1 = ax1.bar(x, marginal["delta_validity_pp"],
                    color=["#2ca02c" if v >= 0 else "#d62728"
                           for v in marginal["delta_validity_pp"]],
                    edgecolor="white", width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(transitions, rotation=15)
    ax1.set_ylabel("Δ Schema Validity (pp)")
    ax1.set_title("Marginal Reliability Gain")
    ax1.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars1, marginal["delta_validity_pp"]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3, f"{val:+.1f}pp",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, marginal["delta_energy_mj"],
                    color=["#d62728" if v >= 0 else "#2ca02c"
                           for v in marginal["delta_energy_mj"]],
                    edgecolor="white", width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(transitions, rotation=15)
    ax2.set_ylabel("Δ GPU Energy per Query (mJ)")
    ax2.set_title("Marginal Energy Cost")
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars2, marginal["delta_energy_mj"]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + abs(bar.get_height()) * 0.02,
                 f"{val:+.0f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Marginal Utility Analysis: Reliability vs Energy per Pipeline Step",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def fig_alignment_by_config(agg: pd.DataFrame, out: Path):
    """Figure 3: Alignment score across configs with std error bars."""
    configs = [c for c in CONFIG_ORDER if c in agg.index and
               not np.isnan(agg.loc[c, "alignment_mean"])]
    if not configs:
        print("[warn] No alignment data — skipping Figure 3.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(configs))
    means = [agg.loc[c, "alignment_mean"] for c in configs]
    stds  = [agg.loc[c, "alignment_std"]  for c in configs]
    colors = [CONFIG_COLORS[c] for c in configs]

    ax.bar(x, means, color=colors, edgecolor="white",
           width=0.5, yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=20, ha="right")
    ax.set_ylabel("Mean Query-Object Alignment Score")
    ax.set_title("Semantic Alignment by Configuration")
    ax.set_ylim(0.0, 1.0)

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.003, f"{m:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def fig_resource_breakdown(agg: pd.DataFrame, out: Path):
    """Figure 4: Stacked resource breakdown — latency and energy side by side."""
    configs = [c for c in CONFIG_ORDER if c in agg.index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ("latency_mean_ms",    "Mean Latency (ms)",              axes[0]),
        ("gpu_energy_mean_mj", "Mean GPU Energy/Query (Joules)", axes[1]),
        ("total_tokens_mean",  "Mean Total Tokens/Query",        axes[2]),
    ]

    for col, label, ax in metrics:
        if col == "gpu_energy_mean_mj":
            vals = [agg.loc[c, col] / 1000 for c in configs]
        else:
            vals = [agg.loc[c, col] for c in configs]
        
        colors = [CONFIG_COLORS[c] for c in configs]
        x      = np.arange(len(configs))
        bars   = ax.bar(x, vals, color=colors, edgecolor="white", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([c for c in configs], rotation=0)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{val:.0f}", ha="center", fontsize=9)

    plt.suptitle("Resource Consumption by Configuration", fontsize=11)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def fig_energy_variance(df: pd.DataFrame, out: Path):
    """Figure 5: Violin plot of GPU energy across configs."""
    configs = [c for c in CONFIG_ORDER if c in df["config_id"].unique()]
    data = [df[(df["config_id"] == c) & df["gpu_energy_mj"].notna()]["gpu_energy_mj"] / 1000 for c in configs]
    
    if not any(len(d) > 0 for d in data):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=False)
    
    for pc, c in zip(parts['bodies'], configs):
        pc.set_facecolor(CONFIG_COLORS[c])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    ax.set_xticks(np.arange(1, len(configs) + 1))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel("GPU Energy per Query (Joules)")
    ax.set_title("Energy Consumption Variance")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

def fig_latency_variance(df: pd.DataFrame, out: Path):
    """Figure 6: Box plot of latency across configs."""
    configs = [c for c in CONFIG_ORDER if c in df["config_id"].unique()]
    data = [df[(df["config_id"] == c) & df["latency_ms"].notna()]["latency_ms"] for c in configs]
    
    if not any(len(d) > 0 for d in data):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bplot = ax.boxplot(data, patch_artist=True)
    
    for patch, c in zip(bplot['boxes'], configs):
        patch.set_facecolor(CONFIG_COLORS[c])
        patch.set_alpha(0.7)
        
    ax.set_xticks(np.arange(1, len(configs) + 1))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=15, ha='right')
    ax.set_ylabel("Latency per Query (ms)")
    ax.set_title("Latency Distribution & Repair Spikes")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

def fig_repair_distribution(df: pd.DataFrame, out: Path):
    """Figure 7: Stacked bar chart of repair loop frequency."""
    configs = ["C2", "C3", "C4"]
    configs = [c for c in configs if c in df["config_id"].unique()]
    if not configs:
        return
        
    repair_counts = [0, 1, 2, 3] # Group 3+ as 3
    
    counts_dict = {c: [] for c in configs}
    for c in configs:
        sub = df[df["config_id"] == c]
        for rc in repair_counts:
            if rc == 3:
                count = len(sub[sub["repair_count"] >= 3])
            else:
                count = len(sub[sub["repair_count"] == rc])
            counts_dict[c].append(count)
            
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(configs))
    
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    labels = ["0 Repairs", "1 Repair", "2 Repairs", "3+ Repairs"]
    
    bottoms = np.zeros(len(configs))
    for i, rc in enumerate(repair_counts):
        vals = [counts_dict[c][i] for c in configs]
        ax.bar(x, vals, width=0.5, bottom=bottoms, label=labels[i], color=colors[i], edgecolor='white')
        bottoms += np.array(vals)
        
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel("Number of Queries")
    ax.set_title("Repair Loop Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def save_table(df: pd.DataFrame, name: str, table_dir: Path):
    """Save table as CSV and LaTeX."""
    csv_path = table_dir / f"{name}.csv"
    tex_path = table_dir / f"{name}.tex"
    df.to_csv(csv_path)
    df.to_latex(tex_path, float_format="%.3f", na_rep="—")
    print(f"Saved: {csv_path}  |  {tex_path}")


def build_paper_tables(agg: pd.DataFrame, marginal: pd.DataFrame,
                       pareto: list[str], table_dir: Path):
    """Build the three main tables for the paper."""

    # Table 2: Main reliability + resource summary
    t2_cols = [
        "n_queries",
        "json_validity_pct", "schema_validity_pct",
        "mean_repair_count", "alignment_mean", "structural_mean",
        "latency_mean_ms", "gpu_energy_mean_mj", "total_tokens_mean",
    ]
    t2 = agg[[c for c in t2_cols if c in agg.columns]].copy()
    t2.index.name = "Config"
    t2.columns = [
        "N", "JSON Valid%", "Schema Valid%",
        "Mean Repairs", "Align.", "Struct.",
        "Latency (ms)", "GPU Energy (mJ)", "Tokens",
    ][:len(t2.columns)]
    save_table(t2, "table2_main_results", table_dir)

    # Table 3: Resource deep-dive
    t3_cols = [
        "latency_mean_ms", "latency_p95_ms",
        "gpu_energy_mean_mj", "gpu_energy_total_j",
        "gpu_util_mean_pct", "gpu_mem_peak_mb",
        "cpu_energy_mean_mj", "total_tokens_mean",
    ]
    t3 = agg[[c for c in t3_cols if c in agg.columns]].copy()
    t3.index.name = "Config"
    save_table(t3, "table3_resource_breakdown", table_dir)

    # Table 4: Efficiency / derived metrics
    t4_cols = [
        "energy_per_valid_mj", "latency_per_valid_ms", "tokens_per_valid",
    ]
    t4 = agg[[c for c in t4_cols if c in agg.columns]].copy()
    t4["pareto_optimal"] = t4.index.map(lambda x: "Yes" if x in pareto else "No")
    t4.index.name = "Config"
    save_table(t4, "table4_efficiency", table_dir)

    # Table 5: Marginal utility
    if not marginal.empty:
        save_table(marginal.set_index("transition"),
                   "table5_marginal_utility", table_dir)

    print(f"\nAll tables saved in: {table_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs",       default="logs/",
                        help="Directory containing JSONL log files")
    parser.add_argument("--figures",    default="figures/")
    parser.add_argument("--tables",     default="tables/")
    parser.add_argument("--out-prefix", default="bda",
                        help="Prefix for output file names")
    args = parser.parse_args()

    log_dir   = Path(args.logs)
    fig_dir   = Path(args.figures);  fig_dir.mkdir(exist_ok=True)
    table_dir = Path(args.tables);   table_dir.mkdir(exist_ok=True)

    # -- Load & process --------------------------------------------------------
    df      = load_logs(log_dir)
    df      = compute_derived(df)
    agg     = aggregate_by_config(df)
    marginal= compute_marginal_utility(agg)
    pareto  = compute_pareto_frontier(agg)

    print(f"\nPareto-optimal configurations: {pareto}")

    # -- Print summary to console ---------------------------------------------
    print("\n-- Reliability Summary --------------------------------------")
    print(agg[["n_queries", "json_validity_pct", "schema_validity_pct",
               "mean_repair_count", "alignment_mean"]].to_string())

    print("\n-- Resource Summary -----------------------------------------")
    print(agg[["latency_mean_ms", "gpu_energy_mean_mj",
               "total_tokens_mean", "cpu_energy_mean_mj"]].to_string())

    print("\n-- Marginal Utility -----------------------------------------")
    print(marginal.to_string(index=False))

    print("\n-- Efficiency -----------------------------------------------")
    print(agg[["energy_per_valid_mj", "latency_per_valid_ms",
               "tokens_per_valid"]].to_string())

    # -- Figures ---------------------------------------------------------------
    p = args.out_prefix
    fig_pareto(agg, pareto,
               fig_dir / f"{p}_fig1_pareto.pdf")
    fig_marginal_utility(marginal,
               fig_dir / f"{p}_fig2_marginal.pdf")
    fig_alignment_by_config(agg,
               fig_dir / f"{p}_fig3_alignment.pdf")
    fig_resource_breakdown(agg,
               fig_dir / f"{p}_fig4_resources.pdf")
    fig_energy_variance(df,
               fig_dir / f"{p}_fig5_energy_variance.pdf")
    fig_latency_variance(df,
               fig_dir / f"{p}_fig6_latency_variance.pdf")
    fig_repair_distribution(df,
               fig_dir / f"{p}_fig7_repairs.pdf")

    # -- Tables ----------------------------------------------------------------
    build_paper_tables(agg, marginal, pareto, table_dir)

    print("\n✓ Analysis complete.")
    print(f"  Figures -> {fig_dir.resolve()}")
    print(f"  Tables  -> {table_dir.resolve()}")


if __name__ == "__main__":
    main()
