import json
import os
import time
import concurrent.futures
import ollama

# Paths
DATA_DIR = "data"
RAW_QUERIES = os.path.join(DATA_DIR, "raw_queries.json")
OUT_FILE = "experiments/stage0_baseline_ollama.json"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.prompt_templates import BASE_PROMPT
from pipeline.validator import validate_output

def generate_ollama(query):
    # Same prompt construction as Local Llama
    prompt = BASE_PROMPT.format(query=query)
    
    try:
        response = ollama.chat(
            model="llama3:8b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON. "
                        "Do not use markdown. "
                        "Do not invent names or brands. "
                        "Do not introduce entities not in the query or provided context. "
                        "Keep descriptions literal and minimal."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.3,
                "num_predict": 256,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        return str(e)

def process_single_query(query):
    raw_output = generate_ollama(query)
    
    # Baseline exactly matches no enforcement
    info = {
        "failure_modes": [],
        "query": query,
        "raw_output": raw_output
    }
    
    # Try parsing
    try:
        parsed = json.loads(raw_output)
    except Exception as e:
        info["error"] = "json_parse_error"
        info["failure_modes"].append("malformed_json")
        return {"query": query, "output": {}, "info": info}
        
    # Validate structure silently (without repairing) to check failure modes
    is_valid, error_msgs = validate_output(parsed)
    if not is_valid:
        info["error"] = "schema_validation_error"
        error_str = " ".join(error_msgs).lower()
        if "required property" in error_str:
            info["failure_modes"].append("missing_required_field")
        elif "is not of type" in error_str:
            info["failure_modes"].append("wrong_type")
        else:
            info["failure_modes"].append("schema_mismatch")
            
    return {"query": query, "output": parsed, "info": info}

def run_ollama_baseline():
    print("Loading queries...")
    with open(RAW_QUERIES, "r") as f:
        queries = json.load(f)
        
    print(f"Loaded {len(queries)} queries. Running via Local Ollama concurrently...")
    
    results = []
    start_time = time.time()
    
    # Run all 100 queries sequentially to avoid overloading local model
    for i, q in enumerate(queries):
        result = process_single_query(q)
        results.append(result)
        print(f"\rCompleted: {i+1}/{len(queries)}", end="")
        
    print(f"\nFinished in {time.time() - start_time:.2f} seconds.")
    
    # Save the exactly identical output format
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {OUT_FILE}")

if __name__ == "__main__":
    run_ollama_baseline()
