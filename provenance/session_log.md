# Session Log & Provenance

This document tracks the provenance of the experimental runs generated for the StructGen Pareto study, ensuring full reproducibility and auditability.

## 1. Original Dataset (June 2026)
**Date:** June 22-23, 2026
**Script:** rchive_bda/scripts/run_baseline_stages.py
**Data:** 
- Standard Schema (N=495)
- Complex Schema (N=100)
**Notes:** This data forms the foundational baseline metrics used in the original paper. The raw logs for the complex schema are stored in rchive_bda/logs/.

## 2. C1.5 Ablation (August 2026)
**Date:** August 23-24, 2026
**Script:** src/run_scale_experiment.py (with newly added C1_5_enforce_only config)
**Data:**
- Standard Schema (N=495)
- Complex Schema (N=100)
**Notes:** Independent session. Utilized the exact same JSON Schema definitions and identical generation prompt as the original run to strictly isolate the effect of disabling the structural repair loop. Logs are stored in esults/camera_ready_extension/ablation_c2_no_repair/.

## 3. Schema Complexity Curve (Tier B & C)
**Date:** August 24, 2026
**Script:** un_phase2.ps1 (batching src/run_scale_experiment.py)
**Data:**
- Tier B (Mild Complexity) Schema (N=100)
- Tier C (Moderate Complexity) Schema (N=100)
**Notes:** Independent session. New schemas were explicitly designed and introduced for this study to bridge the complexity gap between the Standard and Complex schemas.

---

### Important Note on Cross-Session Comparisons
**All comparisons made in the resulting paper are computed exclusively within the same session.** 
Because ambient room temperature, GPU idle states, and background OS processes fluctuate, absolute energy (Joules) and latency (ms) values may differ slightly across independent sessions. The N=20 sanity check conducted in August 2026 confirmed that the relative deltas (the Pareto curve relationships) remain identical to the original June 2026 run. However, to prevent cross-session confusion, absolute metrics are only compared against other metrics generated during the exact same experimental run.
