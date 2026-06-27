# StructGen Pareto: Reliability–Resource Tradeoffs in Synthetic Data Generation

This repository contains the source code, experimental pipelines, and analysis scripts for the **BDA 2026 paper**: *Reliability-Resource Tradeoffs in Structured Synthetic Data Generation*. 

This project extends previous baseline work by evaluating the exact resource costs (GPU energy, compute latency, and token overhead) incurred when forcing Large Language Models (LLMs) to adhere to strict structural constraints (e.g., JSON schemas) during synthetic data generation.

## 🗂️ Project Structure

The repository is organized to support automated, reproducible ML experiments:

```text
structgen-pareto/
├── data/                       # Datasets & schemas
│   ├── queries/                # Expanded queries for scale testing
│   ├── raw_queries.json        # Original seed queries
│   ├── complex_schema.json     # Highly nested schema (Generalizability Probe)
│   └── complex_queries.json    # Complex domain queries
├── logs/                       # JSONL logs from pipeline executions
├── figures/                    # Auto-generated plots (Pareto frontiers, latency dists)
├── tables/                     # Auto-generated LaTeX and CSV tables
├── docs/                       # Reports, summaries, and instrumentation documentation
│   ├── report.md
│   ├── final_numbers_summary.txt
│   ├── COMPLEX_RESULTS.txt
│   ├── ALIGNMENT_STATS.txt
│   ├── expansion_log.txt
│   └── instrumentation_docs.txt
├── src/                        # Main source code
│   ├── expand_queries.py       # LLM-based query augmentation script (100 -> 500)
│   ├── run_scale_experiment.py # Main experimentation pipeline (C1-C4)
│   ├── analyze_pareto.py       # Tradeoff analysis and visualization
│   └── instrument.py           # Resource profiling hooks
├── scripts/                    
│   └── run_baseline_stages.py  # Legacy baseline pipeline
└── requirements.txt            # Python dependencies
```

## ⚙️ Prerequisites & Environment Setup

This project requires a machine with an NVIDIA GPU (for `zeus-ml` energy instrumentation) and a local instance of [Ollama](https://ollama.com/) running `llama3:8b`. 

1. **Start the Ollama Backend:**
   Ensure the Ollama service is running and the model is pulled:
   ```bash
   ollama serve
   ollama pull llama3:8b
   ```

2. **Create and Activate Virtual Environment (Windows/PowerShell):**
   ```powershell
   python -m venv .structgen
   .\.structgen\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Note: CPU energy tracking via `pyRAPL` will automatically degrade gracefully and skip on non-Intel systems like AMD Ryzen. GPU energy tracking via NVML is the primary metric).*

## 🚀 Execution Pipeline

To reproduce the exact findings presented in the paper, execute the pipeline in the following chronological order. **All commands should be run from the root directory.**

### Phase 1: Data Preparation
We expand the original 100 seed queries to a robust scale of 500 queries using LLM augmentation to ensure statistical significance.
```powershell
python src/expand_queries.py --input data/raw_queries.json --output data/queries/expanded_queries.json --target 500
```

### Phase 2: Standard Schema Experiments ($N=495$)
Run the expanded queries through all 4 pipeline configurations. The script will automatically instrument GPU energy and latency for every query.
* **C1**: Raw Baseline (No schema enforcement)
* **C2**: Schema Enforcement + Python Repair Loop
* **C3**: Dense Embedding RAG (BGE-Small) + Repair
* **C4**: Sparse TF-IDF RAG + Repair

```powershell
python src/run_scale_experiment.py --queries data/queries/expanded_queries.json --config all --gpu 0
```
*(Note: This step takes several hours depending on GPU hardware).*

### Phase 3: Generalizability Probe (Complex Schema, $N=100$)
To test if the Pareto-optimality of C4 holds under extreme structural constraints, we run a secondary probe using a highly nested schema (`data/complex_schema.json`).

```powershell
python src/run_scale_experiment.py --queries data/complex_queries.json --schema data/complex_schema.json --config C1 --gpu 0 --no-resume
python src/run_scale_experiment.py --queries data/complex_queries.json --schema data/complex_schema.json --config C2 --gpu 0 --no-resume
python src/run_scale_experiment.py --queries data/complex_queries.json --schema data/complex_schema.json --config C4 --gpu 0 --no-resume
```
*Rename the output logs to `C1_complex.jsonl`, `C2_complex.jsonl`, etc., to prepare them for analysis.*

### Phase 4: Analysis & Plotting
Parse the generated logs to compute reliability-resource tradeoffs, perform Pareto analyses, and output all publication-ready `.tex` tables and `.pdf`/`.png` figures into the `tables/` and `figures/` directories.

```powershell
python src/analyze_pareto.py --logs logs/
```

## 📊 Key Findings

1. **The Cost of Hallucination:** Unstructured generation (C1) consumes massive energy (356 J/query) due to rambling and excess token generation. Enforcing a strict schema (C2) surprisingly *saves* 171 Joules per query while boosting validity to 97.6%.
2. **The Penalty of Repair Loops:** C2 introduces massive "tail latency" spikes (outliers taking 40+ seconds) when the LLM gets trapped in repair loops. 
3. **The Standard Pareto-Optimal Sweet Spot (C4):** Grounding the LLM with classical TF-IDF (C4) provides the exact same perfect validity (100%) as Dense Neural RAG (C3), but skips the heavy embedding compute tax. Under standard schemas, C4 achieves the lowest latency (7.9s) and lowest energy (150 J).
4. **The Generalizability Inversion:** When the schema becomes highly nested and complex, unstructured text retrieval (C4) fails to adequately guide the LLM, causing validity to drop (96.0%) and energy to spike (517 J) due to repair loop activation. Under complex constraints, the multi-step Schema+Repair pipeline (C2) reclaims the Pareto frontier (99.0% validity at 477 J).