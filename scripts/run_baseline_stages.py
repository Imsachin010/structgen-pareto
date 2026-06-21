import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pipeline.generator import generate_scene
from pipeline.rag_module import SimpleRAG
from pipeline.embedding_module import EmbeddingRetriever

DATA_PATH = "data/raw_queries.json"
OUT_DIR = "experiments"
IMAGE_DIR = "images"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# Matched Image Mapping
# ----------------------------
def get_image_for_query(query):
    query_lower = query.lower()
    
    # 5 queries for dog.jpg
    if "dog" in query_lower:
        return os.path.abspath(os.path.join(IMAGE_DIR, "dog.jpg"))
    # 5 queries for car.jpg
    elif "car" in query_lower:
        return os.path.abspath(os.path.join(IMAGE_DIR, "car.jpg"))
    # 5 queries for football.jpg
    elif "football" in query_lower or "kids playing" in query_lower:
        return os.path.abspath(os.path.join(IMAGE_DIR, "football.jpg"))
    
    return None


def load_queries():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def run_stage(stage_name, use_embedding, use_rag, use_vlm, enforce_schema=True):
    print(f"\nRunning {stage_name}...")

    queries = load_queries()
    rag = SimpleRAG.from_file(DATA_PATH)

    documents = [
        "A person walking a dog in a park.",
        "A red car parked near a tree.",
        "Children playing football in a field."
    ]

    embedding_retriever = EmbeddingRetriever(documents)

    results = []

    for item in queries:
        query = item["query"]

        image_path = get_image_for_query(query) if use_vlm else None

        parsed, info = generate_scene(
            query=query,
            image_path=image_path,
            use_hyde=True,
            use_rag=use_rag,
            use_embedding_rag=use_embedding,
            rag=rag,
            embedding_retriever=embedding_retriever,
            rag_top=3,
            rag_threshold=0.3,
            enforce_schema=enforce_schema
        )

        results.append({
            "query": query,
            "output": parsed,
            "info": info
        })

    with open(os.path.join(OUT_DIR, f"{stage_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"{stage_name} saved.")


if __name__ == "__main__":

    # Stage 0 — No Enforcement Baseline
    # run_stage(
    #     stage_name="stage0_baseline",
    #     use_embedding=True,  # baseline standard
    #     use_rag=False,
    #     use_vlm=False,
    #     enforce_schema=False
    # )

    
    # Stage 1 — Embedding Only
    run_stage(
        stage_name="stage1_embedding_only",
        use_embedding=True,
        use_rag=False,
        use_vlm=False,
        enforce_schema=True
    )

    # Stage 2 — Embedding + Matched VLM
    # run_stage(
    #     stage_name="stage2_embedding_plus_vlm_matched",
    #     use_embedding=True,
    #     use_rag=False,
    #     use_vlm=True,
    #     enforce_schema=True
    # )

    # Stage 3 — TF-IDF Only
    run_stage(
        stage_name="stage3_tfidf_rag",
        use_embedding=False,
        use_rag=True,
        use_vlm=False,
        enforce_schema=True
    )

