# pipeline/hyde.py
from pipeline.prompt_templates import HYDE_PROMPT
import json

def generate_hyde(query, llm_fn=None):
    """
    Generate a hypothetical document (HyDE). If llm_fn is provided, call it,
    otherwise return a simple heuristic augmentation.
    llm_fn(prompt:str) -> str
    """
    prompt = HYDE_PROMPT.format(query=query)
    if llm_fn:
        try:
            return llm_fn(prompt)
        except Exception as e:
            # fallback
            return f"Hypothetical expansion: {query} (no LLM available)"
    # simple heuristic HyDE:
    return f"A scene likely involving: {query}. Include objects and simple actions."
