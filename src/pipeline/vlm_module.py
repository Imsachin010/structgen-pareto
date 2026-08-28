import ollama


def generate_image_caption(image_path, model="llava:7b"):
    """
    Uses Ollama vision model to extract a caption from image.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image concisely.",
                    "images": [image_path]
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 256
            }
        )

        return response["message"]["content"]

    except Exception as e:
        print("VLM error:", e)
        return None
