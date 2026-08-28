# Prompt Documentation for Phase 1 Ablation (C1.5)

This document serves as proof that **C1.5 (Schema Enforcement, No Repair)** uses the exact same generation prompt as **C2 (Schema Enforcement + Repair)**, ensuring the ablation cleanly isolates the effect of the repair loop.

## 1. C1 (Raw Baseline) Prompt
*This is the unstructured baseline with no schema constraints.*
`	ext
Describe this scene as a JSON object: A red ball on a grassy field

JSON:
`

## 2. C1.5 (Enforcement Only) Prompt
*This enforces the schema structurally but does not catch or retry failures.*
`	ext
Generate a structured JSON object that describes the following scene query.
You MUST output ONLY valid JSON matching this schema exactly:
{
  "type": "object",
  "required": [
    "scene_description",
    "objects",
    "actions"
  ],
  "additionalProperties": false,
  "properties": {
    "scene_description": {
      "type": "string"
    },
    "objects": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "actions": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
}
Scene query: A red ball on a grassy field

JSON output:
`

## 3. C2 (Enforcement + Repair) Initial Prompt
*This is the exact prompt used by C2 on its first attempt. If this fails, C2 then loops to a secondary repair prompt (which C1.5 disables entirely).*
`	ext
Generate a structured JSON object that describes the following scene query.
You MUST output ONLY valid JSON matching this schema exactly:
{
  "type": "object",
  "required": [
    "scene_description",
    "objects",
    "actions"
  ],
  "additionalProperties": false,
  "properties": {
    "scene_description": {
      "type": "string"
    },
    "objects": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "actions": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
}
Scene query: A red ball on a grassy field

JSON output:
`

## Conclusion
As demonstrated above, the initial 	ext, p_tok, c_tok = _ollama_generate(prompt) call for **C1.5** and **C2** receives an identical, byte-for-byte matching string. The only difference in execution path is that C1.5 bypasses the _repair_output() validation block entirely.
