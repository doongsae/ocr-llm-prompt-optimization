"""Program provenance, completed-job verification, and grid freezing."""

import hashlib
import json

from . import config
from .storage import now, sha256_path, write_json


def fingerprint():
    paths = sorted(config.HERE.glob("*.py")) + sorted((config.HERE / "prompts").glob("*.txt"))
    source_hashes = {str(path.relative_to(config.ROOT)): sha256_path(path) for path in paths}
    value = {"settings": config.SETTINGS, "sources": source_hashes,
             "data_sha256": sha256_path(config.DATA_MANIFEST)}
    value["experiment_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def job_dir(model, group, stage):
    return config.HERE / "artifacts" / model / group / stage


def verify_job(model, group, stage):
    output = job_dir(model, group, stage)
    metadata = json.loads((output / "metadata.json").read_text())
    expected = {"status": "completed", "model": model, "group": group, "stage": stage,
                "experiment_sha256": fingerprint()["experiment_sha256"],
                "program_sha256": sha256_path(output / "program.json")}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"stale/incompatible completed job: {model}/{group}/{stage}")
    return metadata


def freeze_grid():
    jobs = [verify_job(model, group, stage) for model in config.MODELS for group in config.GROUPS for stage in config.STAGES]
    manifest = {"status": "frozen", "frozen_at_utc": now(),
                "experiment_sha256": fingerprint()["experiment_sha256"],
                "programs": [{"model": j["model"], "group": j["group"], "stage": j["stage"],
                              "training_experiment_sha256": j["experiment_sha256"],
                              "program_sha256": j["program_sha256"]} for j in jobs]}
    path = config.HERE / "manifests" / "programs.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["experiment_sha256"] != manifest["experiment_sha256"] or existing["programs"] != manifest["programs"]:
            raise ValueError("frozen manifest differs; refusing replacement")
    else:
        write_json(path, manifest)
    return path

