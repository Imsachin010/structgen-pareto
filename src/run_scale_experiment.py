"""
run_scale_experiment.py
───────────────────────
Runs the four pipeline configurations (C1–C4) over 500 queries with
full resource instrumentation. Saves one JSONL log per configuration
to ./logs/. Designed to plug into your existing multi-module pipeline.

IMPORTANT — adapt the three import lines in Section "YOUR PIPELINE IMPORTS"
to match your actual module names. Everything else is self-contained.

Usage:
    # Run all configs:
    python run_scale_experiment.py --queries queries/expanded_queries.json

    # Run a single config (useful for overnight scheduling):
    python run_scale_experiment.py --queries queries/expanded_queries.json --config C2

    # Quick smoke test (5 queries):
    python run_scale_experiment.py --queries queries/expanded_queries.json --limit 5
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

from instrument import ResourceMonitor, ResourceRecord, estimate_tokens

# ═══════════════════════════════════════════════════════════════════════════════
# YOUR PIPELINE IMPORTS — adapt these three lines to your actual module names
# ═══════════════════════════════════════════════════════════════════════════════
# Example (replace with your real imports):
#
#   from pipeline.generator   import generate_structured        # LLM call
#   from pipeline.repair      import repair_loop                # repair loop
#   from pipeline.retrieval   import EmbeddingRetriever, TFIDFRetriever
#   from pipeline.evaluation  import compute_alignment, compute_structural
#   from pipeline.schema      import STANDARD_SCHEMA
#
# For now these are stubs — replace with your real imports before running.
# ═══════════════════════════════════════════════════════════════════════════════

import requests   # for direct Ollama calls in the stubs below

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3:8b"   # change to your exact Ollama model tag

# ── Schema (standard — same as baseline paper) ────────────────────────────────
STANDARD_SCHEMA = {
    "type": "object",
    "required": ["scene_description", "objects", "actions"],
    "additionalProperties": False,
    "properties": {
        "scene_description": {"type": "string"},
        "objects":           {"type": "array", "items": {"type": "string"}},
        "actions":           {"type": "array", "items": {"type": "string"}},
    },
}

# ── Retrieval corpus (your existing corpus path) ─────────────────────────────
CORPUS_PATH = "data/raw_queries.json"  

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS — replace these with imports from your real modules
# ═══════════════════════════════════════════════════════════════════════════════

import json as _json

def _ollama_generate(prompt: str, temperature: float = 0.3,
                     max_tokens: int = 256) -> tuple[str, int, int]:
    """Returns (response_text, prompt_tokens_est, completion_tokens_est)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "")
    p_tok = data.get("prompt_eval_count", estimate_tokens(prompt))
    c_tok = data.get("eval_count",        estimate_tokens(text))
    return text, p_tok, c_tok


def _build_generation_prompt(query: str, schema: dict,
                              context: str = "") -> str:
    schema_str = _json.dumps(schema, indent=2)
    ctx_block  = f"\nContext:\n{context}\n" if context else ""
    return (
        f"Generate a structured JSON object that describes the following scene query.\n"
        f"You MUST output ONLY valid JSON matching this schema exactly:\n"
        f"{schema_str}\n"
        f"{ctx_block}"
        f"Scene query: {query}\n\n"
        f"JSON output:"
    )


def _validate_output(text: str, schema: dict) -> tuple[bool, bool, dict | None]:
    """Returns (json_valid, schema_valid, parsed_obj)."""
    import jsonschema
    # Extract JSON from text
    text = text.strip()
    # Try to find JSON block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return False, False, None
    candidate = text[start:end]
    try:
        obj = _json.loads(candidate)
    except _json.JSONDecodeError:
        return False, False, None
    try:
        jsonschema.validate(obj, schema)
        return True, True, obj
    except jsonschema.ValidationError:
        return True, False, obj


def _repair_output(text: str, schema: dict, error_msg: str,
                   max_attempts: int = 2) -> tuple[str, int, int, int]:
    """
    Attempts to repair invalid output. Returns:
        (repaired_text, repair_count, extra_prompt_tokens, extra_completion_tokens)
    """
    repair_count = 0
    extra_p = 0
    extra_c = 0

    for _ in range(max_attempts):
        json_valid, schema_valid, obj = _validate_output(text, schema)
        if json_valid and schema_valid:
            break

        schema_str = _json.dumps(schema, indent=2)

        if not json_valid:
            repair_prompt = (
                f"The following text is not valid JSON. Fix it so it is valid JSON "
                f"matching this schema:\n{schema_str}\n\n"
                f"Invalid text:\n{text}\n\n"
                f"Fixed JSON only:"
            )
        else:
            repair_prompt = (
                f"The following JSON does not match the required schema.\n"
                f"Schema:\n{schema_str}\n"
                f"Error: {error_msg}\n"
                f"Fix the JSON:\n{text}\n\n"
                f"Fixed JSON only:"
            )

        text, p_tok, c_tok = _ollama_generate(repair_prompt)
        repair_count += 1
        extra_p += p_tok
        extra_c += c_tok

    return text, repair_count, extra_p, extra_c


def _load_corpus() -> list[str]:
    p = Path(CORPUS_PATH)
    if not p.exists():
        return []
    data = _json.loads(p.read_text())
    if isinstance(data, list):
        return [str(d) for d in data]
    return []


def _tfidf_retrieve(query: str, corpus: list[str], k: int = 3) -> str:
    if not corpus:
        return ""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    docs  = corpus + [query]
    vect  = TfidfVectorizer(stop_words="english")
    tfidf = vect.fit_transform(docs)
    sims  = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    top_k = np.argsort(sims)[-k:][::-1]
    return "\n".join(corpus[i] for i in top_k)


def _embedding_retrieve(query: str, corpus: list[str], k: int = 3) -> str:
    if not corpus:
        return ""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model    = SentenceTransformer("BAAI/bge-small-en-v1.5")
        c_embs   = model.encode(corpus,  convert_to_numpy=True, normalize_embeddings=True)
        q_emb    = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims     = (c_embs @ q_emb.T).squeeze()
        top_k    = np.argsort(sims)[-k:][::-1]
        return "\n".join(corpus[i] for i in top_k)
    except Exception as e:
        print(f"  [warn] embedding retrieval failed: {e}")
        return ""


def _compute_alignment(query: str, output_text: str) -> float | None:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model  = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embs   = model.encode([query, output_text],
                               convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        return None


def _compute_structural(obj: dict | None, schema: dict) -> float:
    if obj is None:
        return 0.0
    required = schema.get("required", [])
    if not required:
        return 1.0
    correct = sum(1 for f in required if f in obj)
    return correct / len(required)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_c1_baseline(query: str, schema: dict) -> dict:
    """C1: Raw LLM — no enforcement, no repair, no retrieval."""
    prompt  = f"Describe this scene as a JSON object: {query}\n\nJSON:"
    text, p_tok, c_tok = _ollama_generate(prompt)
    json_valid, schema_valid, obj = _validate_output(text, schema)
    return {
        "output":          text,
        "json_valid":      json_valid,
        "schema_valid":    schema_valid,
        "parsed":          obj,
        "repair_count":    0,
        "prompt_tokens":   p_tok,
        "completion_tokens": c_tok,
        "retrieval_context": "",
    }


def run_c1_5_enforce_only(query: str, schema: dict) -> dict:
    """C1.5: Schema enforcement — NO repair loop."""
    prompt  = _build_generation_prompt(query, schema)
    text, p_tok, c_tok = _ollama_generate(prompt)
    json_valid, schema_valid, obj = _validate_output(text, schema)
    return {
        "output":          text,
        "json_valid":      json_valid,
        "schema_valid":    schema_valid,
        "parsed":          obj,
        "repair_count":    0,
        "prompt_tokens":   p_tok,
        "completion_tokens": c_tok,
        "retrieval_context": "",
    }


def run_c2_enforcement(query: str, schema: dict) -> dict:
    """C2: Schema enforcement + multi-layer repair loop."""
    prompt  = _build_generation_prompt(query, schema)
    text, p_tok, c_tok = _ollama_generate(prompt)
    json_valid, schema_valid, obj = _validate_output(text, schema)

    repair_count = 0
    extra_p = extra_c = 0

    if not (json_valid and schema_valid):
        text, repair_count, extra_p, extra_c = _repair_output(
            text, schema, "validation failure"
        )
        json_valid, schema_valid, obj = _validate_output(text, schema)

    return {
        "output":            text,
        "json_valid":        json_valid,
        "schema_valid":      schema_valid,
        "parsed":            obj,
        "repair_count":      repair_count,
        "prompt_tokens":     p_tok + extra_p,
        "completion_tokens": c_tok + extra_c,
        "retrieval_context": "",
    }


def run_c3_embedding_rag(query: str, schema: dict, corpus: list[str]) -> dict:
    """C3: Schema enforcement + repair + embedding RAG."""
    context = _embedding_retrieve(query, corpus, k=3)
    prompt  = _build_generation_prompt(query, schema, context=context)
    text, p_tok, c_tok = _ollama_generate(prompt)
    json_valid, schema_valid, obj = _validate_output(text, schema)

    repair_count = 0
    extra_p = extra_c = 0

    if not (json_valid and schema_valid):
        text, repair_count, extra_p, extra_c = _repair_output(
            text, schema, "validation failure"
        )
        json_valid, schema_valid, obj = _validate_output(text, schema)

    return {
        "output":            text,
        "json_valid":        json_valid,
        "schema_valid":      schema_valid,
        "parsed":            obj,
        "repair_count":      repair_count,
        "prompt_tokens":     p_tok + extra_p,
        "completion_tokens": c_tok + extra_c,
        "retrieval_context": context,
    }


def run_c4_tfidf_rag(query: str, schema: dict, corpus: list[str]) -> dict:
    """C4: Schema enforcement + repair + TF-IDF RAG."""
    context = _tfidf_retrieve(query, corpus, k=3)
    prompt  = _build_generation_prompt(query, schema, context=context)
    text, p_tok, c_tok = _ollama_generate(prompt)
    json_valid, schema_valid, obj = _validate_output(text, schema)

    repair_count = 0
    extra_p = extra_c = 0

    if not (json_valid and schema_valid):
        text, repair_count, extra_p, extra_c = _repair_output(
            text, schema, "validation failure"
        )
        json_valid, schema_valid, obj = _validate_output(text, schema)

    return {
        "output":            text,
        "json_valid":        json_valid,
        "schema_valid":      schema_valid,
        "parsed":            obj,
        "repair_count":      repair_count,
        "prompt_tokens":     p_tok + extra_p,
        "completion_tokens": c_tok + extra_c,
        "retrieval_context": context,
    }


# Config dispatch table
CONFIG_MAP = {
    "C1":   ("Raw Baseline",          "none"),
    "C1.5": ("Schema Enforce (No Repair)", "none"),
    "C2":   ("Schema + Repair",       "none"),
    "C3":   ("Schema + Repair + BGE", "embedding"),
    "C4":   ("Schema + Repair + TFIDF","tfidf"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(config_id: str, queries: list[str], schema: dict,
               corpus: list[str], monitor: ResourceMonitor,
               log_path: Path, resume: bool = True) -> None:
    """Run one configuration over all queries, writing JSONL as we go."""

    config_name, retrieval_mode = CONFIG_MAP[config_id]
    print(f"\n{'='*60}")
    print(f"Config {config_id}: {config_name}  (retrieval={retrieval_mode})")
    print(f"Queries: {len(queries)} | Log: {log_path}")
    print(f"{'='*60}")

    # Resume support — find last completed query_id
    completed_ids: set[int] = set()
    if resume and log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["query_id"])
                except Exception:
                    pass
        if completed_ids:
            print(f"Resuming — {len(completed_ids)} queries already done.")

    with open(log_path, "a") as log_file:
        for qid, query in enumerate(queries):
            if qid in completed_ids:
                continue

            record = ResourceRecord(
                config_id      = config_id,
                query_id       = qid,
                schema_id      = "standard",
                retrieval_mode = retrieval_mode,
            )

            try:
                with monitor.measure(record):
                    if config_id == "C1":
                        result = run_c1_baseline(query, schema)
                    elif config_id == "C1.5":
                        result = run_c1_5_enforce_only(query, schema)
                    elif config_id == "C2":
                        result = run_c2_enforcement(query, schema)
                    elif config_id == "C3":
                        result = run_c3_embedding_rag(query, schema, corpus)
                    elif config_id == "C4":
                        result = run_c4_tfidf_rag(query, schema, corpus)
                    else:
                        raise ValueError(f"Unknown config: {config_id}")

                # Populate record with reliability results
                record.json_valid        = result["json_valid"]
                record.schema_valid      = result["schema_valid"]
                record.repair_count      = result["repair_count"]
                record.prompt_tokens     = result["prompt_tokens"]
                record.completion_tokens = result["completion_tokens"]

                # Compute alignment (only if output is schema-valid)
                if result["schema_valid"] and result["parsed"]:
                    obj_text = json.dumps(result["parsed"])
                    record.alignment_score  = _compute_alignment(query, obj_text)
                    record.structural_score = _compute_structural(
                        result["parsed"], schema
                    )

            except Exception as e:
                record.error = traceback.format_exc()
                print(f"  [error] query {qid}: {e}")

            # Write record immediately (crash-safe)
            log_file.write(json.dumps(record.to_dict()) + "\n")
            log_file.flush()

            # Progress print every 10 queries
            if (qid + 1) % 10 == 0:
                pct = 100 * (qid + 1) / len(queries)
                lat = f"{record.latency_ms:.0f}ms" if record.latency_ms else "?"
                gpu = (f"{record.gpu_energy_mj:.1f}mJ"
                       if record.gpu_energy_mj else "?")
                print(f"  [{qid+1:>4}/{len(queries)}] {pct:5.1f}%  "
                      f"valid={record.schema_valid}  "
                      f"repairs={record.repair_count}  "
                      f"latency={lat}  gpu_energy={gpu}")

    print(f"\nConfig {config_id} complete -> {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/queries/expanded_queries.json")
    parser.add_argument("--corpus",  default=CORPUS_PATH)
    parser.add_argument("--config",  default="all",
                        help="C1, C2, C3, C4, or 'all'")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limit to first N queries (for smoke tests)")
    parser.add_argument("--gpu",     type=int, default=0,
                        help="GPU device index for Zeus/NVML")
    parser.add_argument("--no-resume", action="store_true",
                        help="Restart from scratch (ignore existing logs)")
    parser.add_argument("--schema", default=None,
                        help="Path to JSON schema file. Defaults to STANDARD_SCHEMA.")
    args = parser.parse_args()

    if args.schema:
        SCHEMA = json.load(open(args.schema))
    else:
        SCHEMA = STANDARD_SCHEMA

    # ── Load queries ─────────────────────────────────────────────────────────
    raw_q = json.loads(Path(args.queries).read_text())
    queries = [q["query"] if isinstance(q, dict) else q for q in raw_q]
    if args.limit:
        queries = queries[:args.limit]
    print(f"Loaded {len(queries)} queries.")

    # ── Load corpus ──────────────────────────────────────────────────────────
    corpus = _load_corpus()
    print(f"Loaded {len(corpus)} corpus documents.")

    # ── Init monitor ─────────────────────────────────────────────────────────
    monitor = ResourceMonitor(gpu_index=args.gpu)

    # ── Determine configs to run ─────────────────────────────────────────────
    configs = (list(CONFIG_MAP.keys()) if args.config == "all"
               else [args.config.upper()])

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    for config_id in configs:
        log_path = log_dir / f"{config_id}_{ts}.jsonl"
        # If resuming, use the most recent existing log for this config
        if not args.no_resume:
            existing = sorted(log_dir.glob(f"{config_id}_*.jsonl"))
            if existing:
                log_path = existing[-1]
                print(f"Resuming existing log: {log_path}")

        run_config(
            config_id    = config_id,
            queries      = queries,
            schema       = SCHEMA,
            corpus       = corpus,
            monitor      = monitor,
            log_path     = log_path,
            resume       = not args.no_resume,
        )

    print("\nAll configurations complete.")
    print(f"Logs saved in: {log_dir.resolve()}")
    print("Next step: python analyze_pareto.py --logs logs/")


if __name__ == "__main__":
    main()
