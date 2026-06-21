import json
import random
import os

OUT_PATH = "data/raw_queries_100.json"

os.makedirs("data", exist_ok=True)

objects = [
    "dog", "cat", "bicycle", "car", "bus", "tree",
    "bench", "laptop", "phone", "backpack"
]

scenes = [
    "park", "street", "classroom", "office",
    "shopping mall", "playground", "parking lot"
]

actions = [
    "walking", "standing", "sitting", "running",
    "talking", "playing", "looking at"
]

queries = []

for i in range(100):
    obj = random.choice(objects)
    scene = random.choice(scenes)
    action = random.choice(actions)

    query = f"A person {action} near a {obj} in a {scene}"

    queries.append({
        "id": i + 1,
        "query": query
    })

with open(OUT_PATH, "w") as f:
    json.dump(queries, f, indent=2)

print("Generated 100 queries →", OUT_PATH)