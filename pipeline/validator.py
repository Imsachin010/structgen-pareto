# pipeline/validator.py
import jsonschema
from jsonschema import Draft7Validator

SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                            "position": {"type": "string"}
                        },
                        "required": ["color", "position"]
                    }
                },
                "required": ["name", "attributes"]
            }
        },
        "actions": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["scene_description", "objects", "actions"],
    "additionalProperties": False
}

validator = Draft7Validator(SCHEMA)

def validate_output(output_json, schema=None):
    """
    Validate JSON and return (is_valid, errors_list)
    errors_list contains strings describing each validation error.
    """
    validation_schema = schema if schema else SCHEMA
    temp_validator = Draft7Validator(validation_schema) if schema else validator

    errors = []
    for err in temp_validator.iter_errors(output_json):
        # create human readable path + message
        path = ".".join([str(p) for p in err.absolute_path])
        if path:
            errors.append(f"{path}: {err.message}")
        else:
            errors.append(err.message)
    return (len(errors) == 0, errors)
