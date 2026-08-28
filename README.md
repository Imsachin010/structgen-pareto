# StructGen-Pareto

> **Reliability–Resource Tradeoffs in Structured Synthetic Data Generation**

This repository contains the complete implementation, benchmark queries, experimental pipelines, GPU energy instrumentation, analysis scripts, and reproducibility artifacts for the paper **"Reliability–Resource Tradeoffs in Structured Synthetic Data Generation."**

The repository reproduces all experiments, figures, tables, and deployment analyses presented in the paper.

---

## Highlights

- 📊 **1,980** instrumented LLM inference runs
- ⚡ GPU energy profiling using **Zeus-ML**
- 📈 Reliability, latency, and energy tradeoff analysis
- 🔍 Four structured generation configurations (C1–C4)
- 📉 Automatic Pareto frontier construction
- 📋 Publication-ready LaTeX tables and figures
- 🔁 Fully reproducible experimental pipeline

---

## Repository Structure

```text
StructGen-Pareto/
│
├── data/
│   ├── queries/
│   ├── raw_queries.json
│   ├── complex_queries.json
│   └── complex_schema.json
│
├── src/
│   ├── expand_queries.py
│   ├── run_scale_experiment.py
│   ├── analyze_pareto.py
│   └── instrument.py
│
├── scripts/
│   └── run_baseline_stages.py
│
├── logs/
│
├── figures/
│
├── tables/
│
├── docs/
│
├── requirements.txt
│
└── README.md
```

---

# Experimental Pipeline

The experiments follow four sequential stages.

## Stage 1 — Query Expansion

Expand the original benchmark from 100 seed queries to approximately 500 evaluation queries.

```bash
python src/expand_queries.py \
    --input data/raw_queries.json \
    --output data/queries/expanded_queries.json \
    --target 500
```

---

## Stage 2 — Standard Benchmark

Run all four structured generation configurations.

```bash
python src/run_scale_experiment.py \
    --queries data/queries/expanded_queries.json \
    --config all \
    --gpu 0
```

Configurations:

| Configuration | Description |
|--------------|-------------|
| C1 | Raw Generation |
| C2 | Schema Enforcement + Repair |
| C3 | Embedding RAG + Repair |
| C4 | TF-IDF RAG + Repair |

---

## Stage 3 — Complex Schema Evaluation

Evaluate generalizability using a deeply nested schema.

```bash
python src/run_scale_experiment.py \
    --queries data/complex_queries.json \
    --schema data/complex_schema.json \
    --config C1 \
    --gpu 0 \
    --no-resume

python src/run_scale_experiment.py \
    --queries data/complex_queries.json \
    --schema data/complex_schema.json \
    --config C2 \
    --gpu 0 \
    --no-resume

python src/run_scale_experiment.py \
    --queries data/complex_queries.json \
    --schema data/complex_schema.json \
    --config C4 \
    --gpu 0 \
    --no-resume
```

---

## Stage 4 — Analysis

Generate all figures and publication-ready tables.

```bash
python src/analyze_pareto.py --logs logs/
```

Outputs are written to

- `figures/`
- `tables/`

---

# Hardware Requirements

Experiments were conducted using

- NVIDIA RTX 3050 6 GB GPU
- Ollama
- Llama 3 8B
- Python 3.11+

GPU energy measurements are collected using **Zeus-ML** through NVIDIA NVML.

---

# Installation

Clone the repository

```bash
git clone https://github.com/Imsachin010/StructGen-Pareto.git

cd StructGen-Pareto
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Pull the LLM

```bash
ollama pull llama3:8b
```

Start Ollama

```bash
ollama serve
```

---

# Reproducibility

This repository contains

- source code
- benchmark queries
- JSON schemas
- experiment configurations
- execution logs
- plotting scripts
- publication figures
- LaTeX tables

allowing all reported experiments to be reproduced.

---

# Paper Results

The repository reproduces the principal findings reported in the paper, including

- GPU energy measurements
- latency measurements
- schema validity
- JSON validity
- repair statistics
- Pareto frontiers
- marginal utility analysis
- deployment decision framework

---

# Citation

If you use this repository, please cite

```bibtex
@inproceedings{mishra2026structgen,
  title={Reliability--Resource Tradeoffs in Structured Synthetic Data Generation},
  author={Mishra, Sachin and Kannan, Aswin},
  booktitle={International Conference on Big Data Analytics (BDA)},
  year={2026}
}
```

(Update this entry once the proceedings become available.)

---

# License

This repository is released under the MIT License unless otherwise specified.

---

# Contact

**Sachin Mishra**

MS by Research (DSAI)

International Institute of Information Technology Bangalore

Email: sachin.mishra@iiitb.ac.in

---

## Acknowledgements

This work builds upon previous research on reliability-aware structured synthetic data generation and extends it with comprehensive resource profiling, empirical Pareto analysis, and deployment-oriented evaluation.