# Phase 1: C1.5 Ablation Results

The following table compares the baselines (C1 unconstrained, C2 Schema + Repair) against our new C1.5 ablation (Schema Enforce, NO Repair).

## Standard Schema (N=495)

| Configuration | JSON Validity | Mean GPU Energy (J) | Mean Latency (ms) |
| :--- | :--- | :--- | :--- |
| **C1** (No Schema, No Repair) | 0.0% | 356.5 | 13,552 |
| **C1.5** (Schema, No Repair) | 93.1% | 162.9 | 8,580 |
| **C2** (Schema, With Repair) | 97.6% | 185.3 | 9,437 |

## Complex Schema (N=100)

| Configuration | JSON Validity | Mean GPU Energy (J) | Mean Latency (ms) |
| :--- | :--- | :--- | :--- |
| **C1** (No Schema, No Repair) | 0.0% | 422.6 | 17,069 |
| **C1.5** (Schema, No Repair) | 97.0% | 458.0 | 18,873 |
| **C2** (Schema, With Repair) | 99.0% | 477.3 | 19,230 |
