"""Compare native optimizer workflows with deterministic mocked API responses."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import contextlib
import hashlib
import io
import json
import socket
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, sys.argv[1])
if sys.argv[2] == "package":
    from pipeline import artifacts, config, openai_lm, training
    from pipeline.core import fresh_program
    owner = training
else:
    import run_experiment as owner
    import config
    import openai_lm
    artifacts = owner
    from data_contract import fresh_program

import dspy
from dspy.teleprompt import GEPA
from openai.types.chat import ChatCompletion

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()

def completion(text):
    return ChatCompletion(id="offline-fixture", created=0, model=config.MODEL,
                          object="chat.completion", choices=[{
                              "index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": text}}],
                          usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20})

requests = []
def respond(**request):
    requests.append(request)
    content = "\n".join(str(message["content"]) for message in request["messages"])
    if "Observed errors (JSON):" in content:
        errors = json.loads(content.split("Observed errors (JSON):\n", 1)[1].split("\n\nReturn", 1)[0])
        output = {"patterns": [{"name": "Observed error", "root_cause": "Character omission",
                               "error_ids": [row["error_id"] for row in errors],
                               "suggested_fix": "Correct observed omissions."}]}
    elif "Structured error diagnosis (JSON):" in content:
        output = {"new_instruction": "REVISED: Correct the OCR input. Output only corrected_text."}
    else:
        raw = json.loads(request["messages"][-1]["content"])["ocr_prediction"]
        output = {"corrected_text": "SPEED" if raw == "SPED" and "REVISED" in content else raw}
    return completion(json.dumps(output))

def small_gepa(**kwargs):
    kwargs.pop("auto")
    return GEPA(**kwargs, max_metric_calls=16)

examples = [dspy.Example(ocr_prediction=raw, corrected_text=gt, gt=gt,
                         model="trba", dataset="SVT", image_idx=str(index)).with_inputs("ocr_prediction")
            for index, (raw, gt) in enumerate([("SPED", "SPEED"), ("OPEN", "OPEN"), ("STOR", "STOR")])]
settings = {**config.SETTINGS, "num_threads": 1, "reflection_minibatch_size": 1,
            "bootstrap_candidates": 2, "bootstrap_max_demos": 2}
client = Mock()
client.chat.completions.create.side_effect = respond
fingerprint = {"settings": settings, "sources": {}, "data_sha256": "fixture",
               "experiment_sha256": "offline-workflow-fixture"}
paths = Path(sys.argv[3])
results = {}
path_owner = config if sys.argv[2] == "package" else owner
with contextlib.ExitStack() as stack:
    stack.enter_context(patch.object(path_owner, "HERE", paths))
    stack.enter_context(patch.object(path_owner, "SETTINGS", settings))
    stack.enter_context(patch.object(artifacts, "fingerprint", return_value=fingerprint))
    if owner is not artifacts:
        stack.enter_context(patch.object(owner, "fingerprint", return_value=fingerprint))
    stack.enter_context(patch.object(owner, "load_scope_examples", return_value=(examples, examples)))
    stack.enter_context(patch.object(owner, "GEPA", side_effect=small_gepa))
    stack.enter_context(patch.object(openai_lm, "sdk_client", return_value=client))
    stack.enter_context(patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")))
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    for stage in ("base", "gepa", "bootstrap"):
        requests.clear()
        metadata = owner.train_scope("trba", "common", stage)
        output = artifacts.job_dir("trba", "common", stage)
        stable = {key: metadata[key] for key in (
            "status", "model", "group", "stage", "train_rows", "biased_train_rows",
            "train_errors", "correct_train_kept", "sampling_fallback", "validation_rows", "contract")}
        usage = metadata["usage"]
        results[stage] = {
            "metadata_sha256": digest(stable),
            "program_sha256": digest(json.loads((output / "program.json").read_text())),
            "validation_sha256": digest(json.loads((output / "validation.json").read_text())),
            "validation_rows_sha256": hashlib.sha256((output / "validation_predictions.csv").read_bytes()).hexdigest(),
            "selection_sha256": digest(json.loads((output / "selection.json").read_text())) if stage != "base" else None,
            "sdk_requests_sha256": digest(requests), "mock_requests": len(requests),
            "usage": {key: value for key, value in usage.items() if key != "api_call_seconds_sum"},
        }
        artifacts.verify_job("trba", "common", stage)
print(json.dumps(results, sort_keys=True))
'''


def probe(path, mode, work):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    environment.update(LITELLM_LOCAL_MODEL_COST_MAP="True", DSPY_CACHEDIR=str(work / "cache"))
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(path), mode, str(work / "jobs")], env=environment,
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{mode} offline probe failed:\n" + "\n".join(result.stderr.splitlines()[-20:]))
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        before = probe(args.original.resolve(), "original", root / "before")
        after = probe(ROOT, "package", root / "after")
    mismatches = {stage: [key for key in before[stage] if before[stage][key] != after[stage][key]]
                  for stage in before}
    if any(mismatches.values()):
        raise RuntimeError("Workflow changed: " + json.dumps(mismatches))
    print(json.dumps({"status": "passed", "stages_checked": list(after),
                      "contracts_checked": list(after["gepa"]),
                      "mock_requests_per_stage": {stage: row["mock_requests"] for stage, row in after.items()},
                      "real_api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
