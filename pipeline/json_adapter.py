"""Keep OCR data separate from instructions, including empty/literal markers."""

import json

import dspy


class OCRJSONAdapter(dspy.JSONAdapter):
    def __init__(self):
        super().__init__(use_native_function_calling=False)

    def format_field_with_value(self, fields_with_values, role="user"):
        if role == "user":
            return json.dumps({field.name: value for field, value in fields_with_values.items()}, ensure_ascii=True)
        return super().format_field_with_value(fields_with_values, role=role)

    def format_user_message_content(self, signature, inputs, prefix="", suffix="", main_request=False):
        # Output requirements belong to the system message and API schema, not OCR data.
        data = {name: inputs[name] for name in signature.input_fields if name in inputs}
        if main_request and set(data) != set(signature.input_fields):
            raise ValueError("runtime JSON input is missing a required field")
        return "\n\n".join(part for part in (prefix, json.dumps(data, ensure_ascii=True), suffix) if part)
