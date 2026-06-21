import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
from concurrent.futures import ThreadPoolExecutor

from pipeline.generator import generate_scene
from pipeline.rag_module import SimpleRAG
from pipeline.embedding_module import EmbeddingRetriever

DATA_PATH = "data/raw_queries_100.json"
OUT_CSV = "experiments/ablation_results.csv"

BATCH_SIZE = 10
MAX_WORKERS = 3   # safe for 6GB GPU

os.makedirs("experiments", exist_ok=True)


def load_queries():
    with open(DATA_PATH) as f:
        return json.load(f)


def run_stage(query, stage_name, rag, embedding_retriever, use_embedding, use_rag):

    parsed, info = generate_scene(
        query=query,
        image_path=None,
        use_hyde=False,
        use_rag=use_rag,
        use_embedding_rag=use_embedding,
        rag=rag,
        embedding_retriever=embedding_retriever
    )

    valid = 0
    struct_score = 0
    align_score = 0
    repairs = 0

    if parsed is not None:
        valid = 1
        struct_score = info.get("score", 0)
        align_score = info.get("alignment_score", 0)
        repairs = info.get("repair_iterations", 0)

    return {
        "query": query,
        "stage": stage_name,
        "valid": valid,
        "struct_score": struct_score,
        "align_score": align_score,
        "repairs": repairs
    }


def run_query(item, rag, embedding_retriever):

    query = item["query"]

    print(f"Running: {query}")

    results = []

    results.append(
        run_stage(query, "embedding_only", rag, embedding_retriever, True, False)
    )

    results.append(
        run_stage(query, "embedding_plus_vlm", rag, embedding_retriever, True, False)
    )

    results.append(
        run_stage(query, "tfidf_rag", rag, embedding_retriever, False, True)
    )

    return results


def main():

    queries = load_queries()
    total_queries = len(queries)

    print(f"\nTotal queries: {total_queries}")
    print(f"Batch size: {BATCH_SIZE}\n")

    rag = SimpleRAG.from_file(DATA_PATH)

    documents = [q["query"] for q in queries]
    embedding_retriever = EmbeddingRetriever(documents)

    file_exists = os.path.exists(OUT_CSV)

    with open(OUT_CSV, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "stage",
                "valid",
                "struct_score",
                "align_score",
                "repairs"
            ]
        )

        if not file_exists:
            writer.writeheader()

        for start in range(0, total_queries, BATCH_SIZE):

            end = min(start + BATCH_SIZE, total_queries)

            batch = queries[start:end]

            print(f"\nBatch {start} → {end}")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                futures = [
                    executor.submit(run_query, item, rag, embedding_retriever)
                    for item in batch
                ]

                for future in futures:
                    for row in future.result():
                        writer.writerow(row)

    print(f"\nExperiment finished → {OUT_CSV}")


if __name__ == "__main__":
    main()