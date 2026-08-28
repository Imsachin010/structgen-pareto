# Phase 2: Schema Complexity Curve Results

This table demonstrates the progression from Standard to Complex schemas, mapping how the optimal configuration shifts based on schema complexity.

| Schema Tier | Config | Schema Validity | GPU Energy (J) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | C1 (No Schema) | 0.0% | 356.5 | 13,552 |
| | C2 (Repair) | 97.6% | 185.3 | 9,437 |
| | C4 (RAG) | 100.0% | **150.9** | 7,962 |
| | | | | |
| **Tier B (Mild)** | C1 (No Schema) | 0.0% | 393.4 | 11,552 |
| | C2 (Repair) | 100.0% | 176.1 | 7,190 |
| | C4 (RAG) | 100.0% | **173.2** | 7,286 |
| | | | | |
| **Tier C (Mod)** | C1 (No Schema) | 0.0% | 388.7 | 11,620 |
| | C2 (Repair) | 100.0% | **236.6** | 8,732 |
| | C4 (RAG) | 98.0% | 279.9 | 10,146 |
| | | | | |
| **Complex** | C1 (No Schema) | 0.0% | 422.6 | 17,069 |
| | C2 (Repair) | 99.0% | **477.3** | 19,230 |
| | C4 (RAG) | 96.0% | 517.2 | 20,430 |

