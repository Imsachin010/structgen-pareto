# A deeply nested complex schema for Stage 3 validation testing.
COMPLEX_SCHEMA = {
    "type": "object",
    "required": ["metadata", "scene"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["timestamp", "camera_info"],
            "properties": {
                "timestamp": {"type": "string"},
                "camera_info": {
                    "type": "object",
                    "required": ["resolution", "fps"],
                    "properties": {
                        "resolution": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "fps": {"type": "integer", "minimum": 1}
                    }
                }
            }
        },
        "scene": {
            "type": "object",
            "required": ["environment", "entities"],
            "properties": {
                "environment": {
                    "type": "object",
                    "required": ["lighting", "weather"],
                    "properties": {
                        "lighting": {"type": "string", "enum": ["sunny", "cloudy", "dark", "artificial"]},
                        "weather": {"type": "string"}
                    }
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "bounding_box"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "bounding_box": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "attributes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["key", "value", "confidence"],
                                    "properties": {
                                        "key": {"type": "string"},
                                        "value": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
