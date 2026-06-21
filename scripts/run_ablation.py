from pipeline.generator import generate_scene
from pipeline.rag_module import SimpleRAG
import json

queries = [
    "A person walking a dog in the park",
    "A red car parked near a tree",
    "A group of kids playing football"
]

rag = SimpleRAG.from_file("data/raw_queries.json")

settings = [
    {"name": "no_rag", "use_rag": False},
    {"name": "top1", "use_rag": True, "rag_top_k": 1, "rag_threshold": 0.0},
    {"name": "top3", "use_rag": True, "rag_top_k": 3, "rag_threshold": 0.0},
    {"name": "thresholded", "use_rag": True, "rag_top_k": 3, "rag_threshold": 0.2}
]

results = {}

for setting in settings:
    results[setting["name"]] = []
    for q in queries:
        parsed, info = generate_scene(
            q,
            use_hyde=True,
            use_rag=setting.get("use_rag", True),
            rag=rag,
            rag_top_k=setting.get("rag_top_k", 3),
            rag_threshold=setting.get("rag_threshold", 0.0)
        )
        results[setting["name"]].append(info)

with open("experiments/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
