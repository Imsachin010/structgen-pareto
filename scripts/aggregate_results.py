import json
import csv
from pathlib import Path

def aggregate():
    logs_dirs = [Path("logs"), Path("results/camera_ready_extension/schema_tier_new1/logs"), Path("results/camera_ready_extension/schema_tier_new2/logs")]
    out_csv = Path("results/camera_ready_extension/aggregated_results.csv")
    
    all_data = []
    
    for d in logs_dirs:
        if not d.exists(): continue
        for log_file in d.glob("*.jsonl"):
            with open(log_file, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    all_data.append(data)
                    
    if not all_data: return
    
    with open(out_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)
        
    print(f"Aggregated {len(all_data)} records into {out_csv}")

if __name__ == "__main__":
    aggregate()
