import json
import os
import re
import ollama

from pipeline.prompt_templates import BASE_PROMPT
from pipeline.hyde import generate_hyde
from pipeline.rag_module import SimpleRAG
from pipeline.validator import validate_output
from pipeline.evaluator import consistency_score, log_error, query_alignment_score
from pipeline.vlm_module import generate_image_caption
from pipeline.embedding_module import EmbeddingRetriever


DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "generated_outputs.json")
RAW_QUERIES = os.path.join(DATA_DIR, "raw_queries.json")

os.makedirs(DATA_DIR, exist_ok=True)

MAX_RETRIES = 2


# ----------------------------
# JSON Extraction Utility
# ----------------------------
def extract_json_block(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


# ----------------------------
# Deterministic Schema Enforcement
# ----------------------------
def enforce_minimum_schema(parsed):
    if "actions" not in parsed:
        parsed["actions"] = []

    for obj in parsed.get("objects", []):
        if "attributes" not in obj:
            obj["attributes"] = {}

        if "color" not in obj["attributes"]:
            obj["attributes"]["color"] = ""

        if "position" not in obj["attributes"]:
            obj["attributes"]["position"] = ""

    return parsed


import ollama

# ----------------------------
# LLM Adapter
# ----------------------------
def llm_generate(prompt):
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
        log_error("ollama_runtime_error", str(e))
        raise



# ----------------------------
# Main Generation Function
# ----------------------------
def generate_scene(
    query,
    image_path=None,
    use_hyde=True,
    use_rag=True,
    use_embedding_rag=False,
    rag=None,
    embedding_retriever=None,
    llm_fn=None,
    rag_top=3,
    rag_threshold=0.2,
    enforce_schema=True,
    eval_schema=None,
    eval_repair_prompt=None,
):
    llm_fn = llm_fn or llm_generate
    rag = rag or SimpleRAG.from_file(RAW_QUERIES)

    # ----------------------------
    # VLM Caption Integration
    # ----------------------------
    if image_path:
        caption = generate_image_caption(image_path)
        if caption:
            query = query + "\nImage context:\n" + caption

    # ----------------------------
    # HyDE Expansion
    # ----------------------------
    hyde_doc = generate_hyde(query, llm_fn) if use_hyde else None

    # ----------------------------
    # Retrieval (Embedding or TF-IDF)
    # ----------------------------
    retrieved = []

    if use_embedding_rag and embedding_retriever:
        raw_retrieved = embedding_retriever.retrieve(query, top_k=rag_top)
        retrieved = raw_retrieved

    elif use_rag and rag:
        raw_retrieved = rag.retrieve(query, top_k=rag_top)
        retrieved = [r for r in raw_retrieved if r["score"] >= rag_threshold]

    # ----------------------------
    # Build Context
    # ----------------------------
    context_parts = []

    if hyde_doc:
        context_parts.append(hyde_doc)

    if retrieved:
        context_parts += [r["doc"] for r in retrieved]

    context_text = "\n".join(context_parts)

    query_with_context = query + ("\nContext:\n" + context_text if context_text else "")

    if eval_schema:
        prompt = f"""
You are a structured data generator.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include backticks.

The JSON must follow this schema exactly:

{json.dumps(eval_schema, indent=2)}

User query:
{query_with_context}
"""
    else:
        prompt = BASE_PROMPT.format(query=query_with_context)

    # ----------------------------
    # JSON Parse + Repair Loop
    # ----------------------------
    repair_attempts = {
        "json_repair_attempts": 0,
        "schema_repair_attempts": 0,
        "repair_trace": []
    }
    
    parsed = None
    failure_modes = []
    
    for attempt in range(MAX_RETRIES + 1):

        raw_response = llm_fn(prompt)
        raw_response = extract_json_block(raw_response)

        try:
            parsed = json.loads(raw_response)
            break

        except Exception as e:
            if attempt == MAX_RETRIES:
                log_error(
                    "json_parse_error",
                    str(e),
                    output=raw_response,
                    metadata={"query": query},
                )
                failure_modes.append("malformed_json")
                return None, {"error": "json_parse_error", "details": str(e), "failure_modes": failure_modes}

            error_msg = str(e)
            repair_attempts["repair_trace"].append({
                "layer": "json_parse",
                "raw_output": raw_response,
                "error": error_msg
            })
            prompt = f"""
The following JSON is malformed.
Fix it and return ONLY valid JSON.

Malformed JSON:
{raw_response}
"""
            repair_attempts["json_repair_attempts"] += 1

    # ----------------------------
    # Escaping Enforcement
    # ----------------------------
    if not enforce_schema:
        # Check validation manually to record failure modes for baseline
        is_valid, errors = validate_output(parsed, schema=eval_schema) if parsed else (False, ["JSON Parse Failed"])
        if not is_valid:
            for err in errors:
                err_lower = err.lower()
                if "missing" in err_lower or "required" in err_lower:
                    failure_modes.append("missing_required_field")
                elif "type" in err_lower:
                    failure_modes.append("wrong_type")
                else:
                    failure_modes.append("other")
        
        # Deduplicate failure modes
        failure_modes = list(set(failure_modes))
        
        # ----------------------------
        # Evaluation (No Enforcement)
        # ----------------------------
        score, issues = consistency_score(parsed) if parsed else (0, [])
        alignment = query_alignment_score(query, parsed) if parsed else 0

        return parsed, {
            "score": score,
            "alignment_score": alignment,
            "issues": issues,
            "repair_attempts": repair_attempts,
            "failure_modes": failure_modes
        }

    # ----------------------------
    # Deterministic Enforcement
    # ----------------------------
    if not eval_schema:
        parsed = enforce_minimum_schema(parsed)

    # ----------------------------
    # Schema Validation + Repair
    # ----------------------------
    is_valid, errors = validate_output(parsed, schema=eval_schema)

    if not is_valid:

        for attempt in range(MAX_RETRIES):
            
            repair_attempts["repair_trace"].append({
                "layer": "schema_validation",
                "raw_output": json.dumps(parsed, indent=2),
                "error": "; ".join(errors)
            })

            if eval_repair_prompt:
                repair_prompt = eval_repair_prompt.format(errors=errors, json_dump=json.dumps(parsed, indent=2))
            else:
                if eval_schema:
                    repair_prompt = f"""
The following JSON is structurally invalid based on the exact schema provided.

Validation errors:
{errors}

Fix the JSON so it conforms exactly to this schema:
{json.dumps(eval_schema, indent=2)}

Return ONLY valid JSON.

JSON:
{json.dumps(parsed, indent=2)}
"""
                else:
                    repair_prompt = f"""
The following JSON is structurally invalid.

Validation errors:
{errors}

Fix the JSON so that:
- All required fields are present.
- Each object has 'name' and 'attributes'.
- Each 'attributes' contains 'color' and 'position'.
- 'actions' must be an array of strings.

Return ONLY valid JSON.

JSON:
{json.dumps(parsed, indent=2)}
"""

            raw_response = llm_fn(repair_prompt)
            raw_response = extract_json_block(raw_response)
            repair_attempts["schema_repair_attempts"] += 1

            try:
                parsed = json.loads(raw_response)
                if not eval_schema:
                    parsed = enforce_minimum_schema(parsed)
            except:
                continue

            is_valid, errors = validate_output(parsed, schema=eval_schema)
            if is_valid:
                break

        if not is_valid:
            log_error(
                "validation_error",
                "; ".join(errors),
                output=parsed,
                metadata={"query": query},
            )
            return parsed, {"error": "validation_error", "details": errors, "repair_attempts": repair_attempts}

    # ----------------------------
    # Evaluation
    # ----------------------------
    score, issues = consistency_score(parsed)
    alignment = query_alignment_score(query, parsed)

    return parsed, {
        "score": score,
        "alignment_score": alignment,
        "issues": issues,
        "repair_attempts": repair_attempts,
    }


# ----------------------------
# Demo Runner
# ----------------------------
def run_demo():
    if not os.path.exists(RAW_QUERIES):
        sample = [
            {"id": 1, "query": "A person walking a dog in the park"},
            {"id": 2, "query": "A red car parked near a tree"},
            {"id": 3, "query": "A group of kids playing football"},
        ]
        with open(RAW_QUERIES, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)

    rag = SimpleRAG.from_file(RAW_QUERIES)

    # Optional embedding retriever example
    documents = [
        "A person walking a dog in a park.",
        "A red car parked near a tree.",
        "Children playing football in a field."
    ]
    embedding_retriever = EmbeddingRetriever(documents)

    outputs = []

    with open(RAW_QUERIES, "r", encoding="utf-8") as f:
        queries = json.load(f)

    for item in queries:
        print(f"Generating for: {item['query']}")

        parsed, info = generate_scene(
            item["query"],
            image_path=None,                 # provide image path if testing VLM
            use_hyde=True,
            use_rag=False,
            use_embedding_rag=True,
            embedding_retriever=embedding_retriever,
            rag_top=3,
            rag_threshold=0.3,
        )

        outputs.append(
            {
                "query": item["query"],
                "output": parsed,
                "info": info,
            }
        )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"Demo done — outputs written to {OUT_FILE}")


if __name__ == "__main__":
    run_demo()
