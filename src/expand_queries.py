"""
expand_queries.py
─────────────────
Expands an existing raw_queries.json (100 queries) to 500 queries
using LLaMA3 8B via Ollama. Produces expanded_queries.json.

Usage:
    python expand_queries.py \
        --input raw_queries.json \
        --output queries/expanded_queries.json \
        --target 500

The script generates paraphrases + thematic variants so the expanded
set tests a range of phrasings and scene types, not just copies.
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests

# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3:8b"          # change to your exact model tag if different
TEMP       = 0.8               # higher than inference temp to maximise diversity
MAX_TOKENS = 256

# ── Expansion strategies ─────────────────────────────────────────────────────
STRATEGIES = [
    "paraphrase this scene description query using completely different wording",
    "create a more complex version of this scene query with additional objects or actions",
    "create a simpler, shorter version of this scene query",
    "create a query that describes a similar scene but in a different environment or context",
    "create a query that describes a scene with similar actions but different objects",
]


def ollama_generate(prompt: str, retries: int = 3) -> str | None:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMP, "num_predict": MAX_TOKENS},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"  [warn] attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def expand_query(original: str, strategy: str) -> str | None:
    prompt = (
        f"Task: {strategy}.\n\n"
        f"Original query: \"{original}\"\n\n"
        "Rules:\n"
        "- Output ONLY the new query sentence, nothing else.\n"
        "- Do not include quotes, labels, or explanations.\n"
        "- The query must describe a visual scene with objects and actions.\n"
        "- Keep it to 1-2 sentences.\n\n"
        "New query:"
    )
    result = ollama_generate(prompt)
    if result:
        # Clean up: strip quotes, labels, newlines
        result = result.strip().strip('"').strip("'")
        result = result.split("\n")[0].strip()
    return result


def deduplicate(queries: list[str], threshold: float = 0.85) -> list[str]:
    """Simple character-level dedup — fast, no embeddings needed here."""
    seen: list[str] = []
    for q in queries:
        q_lower = q.lower()
        is_dup = False
        for s in seen:
            # Jaccard similarity on word sets
            a = set(q_lower.split())
            b = set(s.lower().split())
            if len(a | b) == 0:
                continue
            sim = len(a & b) / len(a | b)
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen.append(q)
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="raw_queries.json",
                        help="Path to original queries JSON (list of strings or list of dicts with 'query' key)")
    parser.add_argument("--output", default="data/queries/expanded_queries.json")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load original queries ────────────────────────────────────────────────
    raw = json.loads(Path(args.input).read_text())
    if isinstance(raw, list) and len(raw) > 0:
        if isinstance(raw[0], dict):
            originals = [item.get("query", item.get("text", str(item))) for item in raw]
        else:
            originals = [str(q) for q in raw]
    else:
        raise ValueError("Input must be a JSON list of strings or dicts.")

    print(f"Loaded {len(originals)} original queries.")
    expanded = list(originals)  # start with originals

    needed = args.target - len(expanded)
    print(f"Need {needed} more queries to reach target of {args.target}.")

    # ── Generate expansions ──────────────────────────────────────────────────
    strategy_cycle = list(STRATEGIES) * ((needed // len(STRATEGIES)) + 2)
    random.shuffle(strategy_cycle)

    generated = 0
    attempt   = 0

    while generated < needed and attempt < needed * 3:
        # Pick a random original as seed
        seed_query = random.choice(originals)
        strategy   = strategy_cycle[attempt % len(strategy_cycle)]
        attempt   += 1

        print(f"  [{generated+1}/{needed}] strategy='{strategy[:40]}...'")
        new_q = expand_query(seed_query, strategy)

        if new_q and len(new_q) > 10:
            expanded.append(new_q)
            generated += 1
        else:
            print("  [warn] empty or too-short result, retrying with different seed.")

    # ── Deduplicate ──────────────────────────────────────────────────────────
    print(f"\nDeduplicating {len(expanded)} queries...")
    expanded = deduplicate(expanded)
    print(f"After dedup: {len(expanded)} unique queries.")

    # Trim or pad to exactly target
    if len(expanded) > args.target:
        expanded = expanded[:args.target]
    elif len(expanded) < args.target:
        print(f"[warn] Only reached {len(expanded)} unique queries (target={args.target}). "
              "Re-run with --target reduced or allow more retries.")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(expanded, indent=2))
    print(f"\nSaved {len(expanded)} queries -> {out_path}")

    # ── Summary stats ────────────────────────────────────────────────────────
    lengths = [len(q.split()) for q in expanded]
    print(f"Query length stats: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f} words")


if __name__ == "__main__":
    main()
