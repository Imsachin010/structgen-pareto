import subprocess
import sys

def main():
    print("Running Phase 2: Schema Complexity Tiers")
    configs = ["C1", "C2", "C4"]
    
    print("--- Tier B (Mild) ---")
    for c in configs:
        cmd = [sys.executable, "src/run_scale_experiment.py", "--queries", "data/complex_queries.json", "--schema", "schemas/tier_b_schema.json", "--config", c]
        print(f"Executing: {' '.join(cmd)}")
        # subprocess.run(cmd)
        
    print("--- Tier C (Moderate) ---")
    for c in configs:
        cmd = [sys.executable, "src/run_scale_experiment.py", "--queries", "data/complex_queries.json", "--schema", "schemas/tier_c_schema.json", "--config", c]
        print(f"Executing: {' '.join(cmd)}")
        # subprocess.run(cmd)

if __name__ == "__main__":
    main()
