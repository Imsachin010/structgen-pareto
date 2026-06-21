import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.generator import generate_scene
from pipeline.validator import validate_output
from pipeline.complex_schema import COMPLEX_SCHEMA

OUT_DIR = "experiments"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_complex_queries():
    base = [
        "A busy city intersection at night with neon lights and multiple pedestrians and cars.",
        "A quiet forest path in autumn with falling leaves and a deer grazing.",
        "A high-tech laboratory with glowing screens, robotic arms, and scientists in white coats."
    ]
    return [{"id": i+1, "query": base[i % 3] + f" Variation {i+1}"} for i in range(30)]


def run_experiment():
    print("\nRunning Nested Schema Experiment...")
    queries = generate_complex_queries()
    
    results = []
    
    for item in queries:
        query = item["query"]
        
        # We need to monkey-patch or inject the schema enforcement for this test.
        # Since generator.py is wired for the basic schema, we will pass eval_schema=True and 
        # override the validation logic temporarily for the test, or rely on the custom prompt.
        
        REPAIR_PROMPT = """
The following JSON is structurally invalid based on the exact schema provided.

Validation errors:
{errors}

Fix the JSON to strictly adhere to this complex nested schema structure. 
Return ONLY valid JSON.

JSON:
{json_dump}
"""

        parsed, info = generate_scene(
            query=query,
            use_hyde=False,
            use_rag=False,
            enforce_schema=True,
            eval_schema=COMPLEX_SCHEMA,
            eval_repair_prompt=REPAIR_PROMPT
        )
        
        results.append({
            "query": query,
            "output": parsed,
            "info": info
        })
        
        print(f"Processed: {query[:40]}...  Valid: {'error' not in info}")

    out_file = os.path.join(OUT_DIR, "nested_schema_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Nested schema experiment completed. Results saved to {out_file}.")

if __name__ == "__main__":
    run_experiment()
