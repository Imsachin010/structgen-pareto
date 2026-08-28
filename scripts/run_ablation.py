import subprocess
import sys

def main():
    print("Running Phase 1: C2-Without-Repair Ablation (C1.5)")
    configs = ["C1", "C1.5", "C2"]
    
    print("--- Standard Schema ---")
    for c in configs:
        cmd = [sys.executable, "src/run_scale_experiment.py", "--queries", "data/raw_queries.json", "--schema", "schemas/standard_schema.json", "--config", c]
        print(f"Executing: {' '.join(cmd)}")
        # subprocess.run(cmd) # Commented out for repository structure definition
        
    print("--- Complex Schema ---")
    for c in configs:
        cmd = [sys.executable, "src/run_scale_experiment.py", "--queries", "data/complex_queries.json", "--schema", "schemas/complex_schema.json", "--config", c]
        print(f"Executing: {' '.join(cmd)}")
        # subprocess.run(cmd)

if __name__ == "__main__":
    main()
