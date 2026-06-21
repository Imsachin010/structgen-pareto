# Structured Synthetic Data Generation — Research Pipeline

## Papers
### Paper 1: Previous Work (published)
"Reliability-Aware Structured Synthetic Data Generation..."
→ `scripts/run_baseline_stages.py`

### Paper 2: BDA 2026 (extension — resource tradeoffs)
"Reliability–Resource Tradeoffs..."
→ `expand_queries.py` → `run_scale_experiment.py` → `analyze_pareto.py`

---

## Recent Updates & Repository Restructuring

The repository has been fully restructured to support the new BDA 2026 scale experiments:
1. **Script Migration**: All BDA extension scripts (`expand_queries.py`, `instrument.py`, `run_scale_experiment.py`, `analyze_pareto.py`) have been moved to the repository root.
2. **Environment Setup**: Added support for the `.structgen` virtual environment and updated `requirements.txt` with hardware instrumentation dependencies (`zeus-ml`, `pynvml`, etc.). Note: `pyRAPL` is included but will automatically degrade gracefully on AMD Ryzen systems.
3. **Artifact Directories**: Created `data/queries/`, `logs/`, `tables/`, and `figures/` to automatically capture execution traces and analysis outputs.
4. **Cleanup**: Stale baseline evaluation scripts and figures were safely removed, and the baseline runner was renamed to `scripts/run_baseline_stages.py` to differentiate it from the new scale runner.

### Hardware Context (AMD Ryzen + RTX 2080Ti)
Because this system runs an AMD Ryzen CPU, Intel's RAPL interface (used by `pyRAPL`) is unavailable for CPU energy profiling. The `instrument.py` script will automatically detect this and safely skip CPU tracking. **We are fully relying on Zeus-ML (via NVML) to profile energy consumption on the NVIDIA RTX 2080Ti.**

## Setup Instructions

1. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv .structgen
   source .structgen/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Environment:**
   ```bash
   python -c "import ollama; import jsonschema; import sentence_transformers; print('core OK')"
   python -c "import pyRAPL; print('pyRAPL OK')"
   python -c "import zeus; print('zeus OK')"
   python -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK')"
   python -c "import pandas; import matplotlib; print('analysis OK')"
   ollama list
   ```

## Execution Pipeline (BDA 2026)

1. **Expand Queries (100 -> 500)**
   ```bash
   python expand_queries.py --input data/raw_queries.json --output data/queries/expanded_queries.json --target 500
   ```
2. **Run Scale Experiment**
   ```bash
   python run_scale_experiment.py --queries data/queries/expanded_queries.json
   ```
3. **Analyze Pareto & Generate Figures**
   ```bash
   python analyze_pareto.py --logs logs/
   ```