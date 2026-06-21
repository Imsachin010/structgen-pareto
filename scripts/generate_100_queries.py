import json
import os

base_queries = [
    "A person walking a dog in the park",
    "A red car parked near a tree",
    "A group of kids playing football"
]

queries = [{"id": i+1, "query": base_queries[i % 3] + f" scene {i+1}"} for i in range(100)]

with open("data/raw_queries.json", "w") as f:
    json.dump(queries, f, indent=2)

print("Generated 100 queries in data/raw_queries.json")
