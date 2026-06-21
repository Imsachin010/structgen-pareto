import json
import os
import matplotlib.pyplot as plt

RESULTS_FILE = r"c:\Users\Imsac\Desktop\Project_X\X3\vlm_dataset_generation\experiments\nested_schema_results.json"
ANALYSIS_DIR = r"c:\Users\Imsac\Desktop\Project_X\X3\vlm_dataset_generation\experiments\analysis"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

OUT_MD = os.path.join(ANALYSIS_DIR, "nested_schema_report.md")
OUT_PLOT1 = os.path.join(ANALYSIS_DIR, "success_rate.png")
OUT_PLOT2 = os.path.join(ANALYSIS_DIR, "repair_counts.png")

def analyze_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    total = len(data)
    valid_count = 0
    failed_count = 0
    
    total_json_repairs = 0
    total_schema_repairs = 0
    
    # Track errors per layer
    json_parse_errors = 0
    schema_validation_errors = 0
    
    for item in data:
        info = item.get("info", {})
        if "error" not in info:
            valid_count += 1
        else:
            failed_count += 1
            
        repair_attempts = info.get("repair_attempts", {})
        total_json_repairs += repair_attempts.get("json_repair_attempts", 0)
        total_schema_repairs += repair_attempts.get("schema_repair_attempts", 0)
        
        for trace in repair_attempts.get("repair_trace", []):
            if trace.get("layer") == "json_parse":
                json_parse_errors += 1
            elif trace.get("layer") == "schema_validation":
                schema_validation_errors += 1

    # Plot 1: Success Rate
    plt.figure(figsize=(6, 4))
    plt.bar(["Valid Output", "Failed Output"], [valid_count, failed_count], color=["#4CAF50", "#F44336"])
    plt.title("Nested Schema Generation Success")
    plt.ylabel("Number of Queries")
    for i, v in enumerate([valid_count, failed_count]):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(OUT_PLOT1)
    
    # Plot 2: Repair Attempts
    plt.figure(figsize=(6, 4))
    plt.bar(["JSON Parse Repairs", "Schema Validation Repairs"], [total_json_repairs, total_schema_repairs], color=["#2196F3", "#FF9800"])
    plt.title("Repair Attempts by Layer")
    plt.ylabel("Number of Attempts")
    for i, v in enumerate([total_json_repairs, total_schema_repairs]):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(OUT_PLOT2)
    
    # Write Markdown Report
    with open(OUT_MD, "w") as f:
        f.write("# Nested Schema Execution Results\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Queries Executed:** {total}\n")
        f.write(f"- **Successful Outputs:** {valid_count} ({round(valid_count/total*100, 1)}%)\n")
        f.write(f"- **Failed Outputs:** {failed_count} ({round(failed_count/total*100, 1)}%)\n\n")
        
        f.write("## Repair Breakdown\n\n")
        f.write("This table breaks down exactly where the LLM stumbled and required a feedback repair loop.\n\n")
        f.write("| Repair Layer | Total Repair Attempts | Total Traced Unique Errors |\n")
        f.write("|--------------|-----------------------|----------------------------|\n")
        f.write(f"| **JSON Parse Malformations** | {total_json_repairs} | {json_parse_errors} |\n")
        f.write(f"| **Deep Schema Validations** | {total_schema_repairs} | {schema_validation_errors} |\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Validation Outcomes\n")
        f.write(f"![Success Rate]({os.path.abspath(OUT_PLOT1).replace(chr(92), '/')})\n\n")
        f.write("### Repair Frequencies\n")
        f.write(f"![Repair Attempts]({os.path.abspath(OUT_PLOT2).replace(chr(92), '/')})\n")

    print(f"Analysis complete. Results stored in {OUT_MD}")

if __name__ == "__main__":
    analyze_results()
