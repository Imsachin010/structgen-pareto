# pipeline/prompt_templates.py
BASE_PROMPT = """
You are a structured data generator.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include backticks.

The JSON must follow this schema exactly:

{{
  "scene_description": "string",
  "objects": [
    {{
      "name": "string",
      "attributes": {{
        "color": "string",
        "position": "string"
      }}
    }}
  ],
  "actions": ["string"]
}}


User query:
{query}
"""


HYDE_PROMPT = """
Produce a short hypothetical document (1-2 lines) that elaborates on the query and may help the model produce the structured JSON.
Return plain text (not JSON).
Query:
{query}
"""
