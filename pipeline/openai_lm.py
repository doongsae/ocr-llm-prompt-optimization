"""Official OpenAI SDK transport behind DSPy's existing LM interface."""

import json
import os
import threading
import time
from functools import lru_cache

import dspy
from openai import OpenAI

from .config import API_BASE, MODEL
from .json_adapter import OCRJSONAdapter

CORRECTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ocr_correction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"corrected_text": {"type": "string"}},
            "required": ["corrected_text"],
            "additionalProperties": False,
        },
    },
}

_records = []
_lock = threading.Lock()


def require_key():
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in the launch environment, not in source files.")


@lru_cache(maxsize=1)
def sdk_client():
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is missing; no API request was made")
    return OpenAI(base_url=API_BASE, timeout=120.0, max_retries=2)


def reset_usage():
    with _lock:
        _records.clear()


def usage_summary():
    with _lock:
        records = list(_records)
    return {"completed_requests": len(records),
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["completion_tokens"] for r in records),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in records),
            "cached_tokens": sum(r["cached_tokens"] for r in records),
            "api_call_seconds_sum": sum(r["seconds"] for r in records),
            "by_role": {role: sum(r["role"] == role for r in records)
                        for role in sorted({r["role"] for r in records})}}


class LunaLM(dspy.BaseLM):
    def __init__(self, role, max_tokens, temperature):
        super().__init__(model=MODEL, max_tokens=max_tokens, temperature=temperature, cache=False)
        self.role = role

    def forward(self, prompt=None, messages=None, **kwargs):
        config = {**self.kwargs, **kwargs}
        for name in ("cache", "rollout_id"):
            config.pop(name, None)
        config["max_completion_tokens"] = config.pop("max_tokens")
        config["reasoning_effort"] = "none"
        if self.role == "student":
            # Enforce the wire contract even when DSPy's model registry is stale.
            config["response_format"] = CORRECTION_RESPONSE_FORMAT
        allowed = {"temperature", "max_completion_tokens", "reasoning_effort", "n", "response_format"}
        if set(config) - allowed:
            raise ValueError(f"unsupported OpenAI request options: {sorted(set(config) - allowed)}")
        started = time.monotonic()
        response = sdk_client().chat.completions.create(
            model=MODEL, messages=messages or [{"role": "user", "content": prompt}], **config)
        usage = response.usage
        if usage is None:
            raise RuntimeError("API response did not report token usage")
        with _lock:
            _records.append({
                "role": self.role, "seconds": time.monotonic() - started,
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
                "cached_tokens": getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0,
            })
        if not response.choices:
            raise RuntimeError("API response has no choices")
        for choice in response.choices:
            if choice.finish_reason != "stop":
                raise RuntimeError(f"API completion finish_reason={choice.finish_reason}; output cap={config['max_completion_tokens']}")
            if choice.message.refusal:
                raise RuntimeError("API explicitly refused the request")
            if choice.message.content is None:
                raise RuntimeError("API completion content is missing")
        if self.role == "student":
            for choice in response.choices:
                value = json.loads(choice.message.content)
                if not isinstance(value, dict) or set(value) != {"corrected_text"} or not isinstance(value["corrected_text"], str):
                    raise ValueError("API correction response violated the strict JSON contract")
        return response


def setup_lms(settings):
    student = LunaLM("student", settings["student_max_completion_tokens"], settings["student_temperature"])
    reflection = LunaLM("reflection", settings["reflection_max_completion_tokens"], settings["reflection_temperature"])
    dspy.configure(lm=student, adapter=OCRJSONAdapter())
    return student, reflection
