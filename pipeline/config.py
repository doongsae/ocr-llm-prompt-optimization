"""Frozen experiment settings; historical experiment files stay read-only."""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path):
    values = json.loads(path.read_text(encoding="utf-8"))
    seed = values["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32 - 1:
        raise ValueError("seed must be an integer from 0 to 4294967294")
    for name in ("gepa_error_ratio", "bootstrap_error_ratio"):
        value = values[name]
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError(f"{name} must be a fraction greater than 0 and at most 1")
    for name, default in (("openai_api_key", ""), ("data_root", "datasets"),
                          ("data_manifest", "data/frozen_rows.csv.gz")):
        value = values.setdefault(name, default)
        if not isinstance(value, str) or (name != "openai_api_key" and not value.strip()):
            raise ValueError(f"{name} must be a {'nonempty ' if name != 'openai_api_key' else ''}string")
    return values


USER_CONFIG = load_config(CONFIG_PATH if CONFIG_PATH.is_file() else ROOT / "config.example.json")
SEED = USER_CONFIG["seed"]


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def openai_api_key():
    return USER_CONFIG["openai_api_key"].strip() or os.environ.get("OPENAI_API_KEY", "").strip()


DATA_ROOT = resolve_path(USER_CONFIG["data_root"])
DATA_MANIFEST = resolve_path(os.environ.get("OCR_DATA_MANIFEST") or USER_CONFIG["data_manifest"])

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("DSPY_CACHEDIR", str(HERE / ".cache" / "dspy"))

MODELS = ["crnn", "vitstr", "trba", "abinet", "parseq", "svtrv2"]
GROUPS = {
    "common": ["IIIT5k", "SVT", "IC13_857", "IC15_1811", "SVTP", "CUTE80"],
    "unionhard": ["Curve", "Multi-Oriented", "Artistic", "Contextless", "Salient", "Multi-Words"],
    "general": ["General"],
    "ost": ["OST_weak", "OST_heavy"],
}
DATASETS = [dataset for datasets in GROUPS.values() for dataset in datasets]
STAGES = ("base", "gepa", "bootstrap")

MODEL = "gpt-6-luna"
API_BASE = "https://api.openai.com/v1"
VERSIONS = {"dspy": "3.1.3", "gepa": "0.0.26", "openai": "2.24.0"}
SETTINGS = {
    "model": MODEL,
    "api_base": API_BASE,
    "reasoning_effort": "none",
    "output_transport": "openai_strict_json_schema_with_dspy_json_adapter",
    "input_transport": "json_object_without_trailing_user_instructions",
    "student_temperature": 0.0,
    "reflection_temperature": 1.0,
    "student_max_completion_tokens": 32,
    "reflection_max_completion_tokens": 1200,
    "gepa_auto": "light",
    "reflection_minibatch_size": 3,
    "gepa_error_ratio": USER_CONFIG["gepa_error_ratio"],
    "gepa_seed": SEED,
    "bootstrap_error_ratio": USER_CONFIG["bootstrap_error_ratio"],
    "bootstrap_seed": SEED + 1,
    "bootstrap_max_demos": 12,
    "bootstrap_candidates": 24,
    "bootstrap_max_rounds": 1,
    "num_threads": 16,
    "metric": "normalized_exact_match_only",
    "selection": "native_optimizer_validation_mean_em",
    "reflection": "espo_c1_requirements_and_c2_diagnostic_revision_adapter",
}
