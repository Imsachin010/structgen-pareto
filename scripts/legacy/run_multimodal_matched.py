import json
from pipeline.generator import generate_scene

DATA = "data/multimodal_pairs.json"

with open(DATA) as f:
    pairs = json.load(f)

results = []

for pair in pairs:

    image = pair["image"]

    for query in pair["queries"]:

        parsed, info = generate_scene(
            query=query,
            image_path=image,
            use_hyde=False
        )

        results.append(info["alignment_score"])

print("Average multimodal alignment:", sum(results)/len(results))