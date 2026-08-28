import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate():
    csv_path = Path("results/camera_ready_extension/aggregated_results.csv")
    if not csv_path.exists():
        print("CSV not found. Run aggregate_results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Placeholder for Pareto figure generation logic
    print("Generating Pareto curve figures...")
    # plt.scatter(df['latency_ms'], df['gpu_energy_mj'])
    # plt.savefig("results/camera_ready_extension/pareto_curve.pdf")
    print("Done.")

if __name__ == "__main__":
    generate()
