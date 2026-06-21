# pipeline/evaluator.py
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Lightweight semantic encoder
semantic_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

LOG_PATH = "data/logs/errors.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def consistency_score(output_json):
    """
    Basic consistency scoring. Expand with more checks later.
    Returns score in [0,1] and a list of found issues.
    """
    issues = []
    score = 0
    if "objects" in output_json and isinstance(output_json["objects"], list) and len(output_json["objects"])>0:
        score += 0.5
    else:
        issues.append("no objects")
    if "actions" in output_json and isinstance(output_json["actions"], list) and len(output_json["actions"])>0:
        score += 0.5
    else:
        issues.append("no actions")
    return score, issues

def log_error(error_type, message, output=None, metadata=None):
    metadata = metadata or {}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "time": datetime.utcnow().isoformat()+"Z",
        "error_type": error_type,
        "message": message,
        "output": output,
        "metadata": metadata
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def query_alignment_score(query, parsed):
    """
    Semantic alignment between query and generated structured output.
    Uses sentence embeddings instead of token overlap.
    """

    scene = parsed.get("scene_description", "")

    objects = " ".join(
        obj.get("name", "")
        for obj in parsed.get("objects", [])
    )

    actions = " ".join(parsed.get("actions", []))

    generated_text = f"{scene} {objects} {actions}"

    q_emb = semantic_model.encode([query])
    g_emb = semantic_model.encode([generated_text])

    score = cosine_similarity(q_emb, g_emb)[0][0]

    return float(score)
