import json
import traceback
from sentence_transformers import SentenceTransformer
import numpy as np

records = [json.loads(l) for l in open('logs/C3_20260622_1841.jsonl') if l.strip()]
null_align = [r for r in records if r.get('alignment_score') is None]

if len(null_align) > 0:
    r = null_align[0]
    print(f"Testing Query ID {r['query_id']}")
    # get the actual text from raw queries
    queries = json.load(open('data/queries/expanded_queries.json'))
    query = queries[r['query_id']]
    # we don't have the parsed output saved directly in jsonl, wait, the jsonl only has structural_score and prompt_tokens. 
    # but the error happened inside _compute_alignment. 
    print("Wait, let's just try to load SentenceTransformer and see if it OOMs.")
    try:
        model  = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embs   = model.encode([query, "dummy"], convert_to_numpy=True, normalize_embeddings=True)
        print("Model loaded and encoded fine.")
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
