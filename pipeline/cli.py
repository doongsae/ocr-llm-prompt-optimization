"""Experiment commands, offline preflight, and explicit paid API smoke."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import dspy

from . import config
from .artifacts import fingerprint, freeze_grid
from .contracts import audit_data, verify_program
from .core import fresh_program
from .data import load_scope_examples
from .evaluation import run_test
from .metrics import facts
from .openai_lm import require_key, reset_usage, setup_lms, usage_summary
from .storage import now, write_json
from .training import train_scope


def preflight():
    versions = {name: importlib.metadata.version(name) for name in config.VERSIONS}
    if versions != config.VERSIONS:
        raise ValueError(f"dependency versions changed: {versions}")
    report = {"status": "passed", "checked_at_utc": now(), "versions": versions,
              "data": audit_data(), "provenance": fingerprint(),
              "api_key_present": bool(config.openai_api_key()),
              "api_request_performed": False,
              "runtime_contract": verify_program(fresh_program(), load_scope_examples("trba", "common")[0])}
    write_json(config.HERE / "status" / "preflight.json", report)
    return report


def api_smoke():
    require_key()
    reset_usage()
    _, reflection = setup_lms(config.SETTINGS)
    prediction = fresh_program()(ocr_prediction="OPEN")
    if not facts(dspy.Example(ocr_prediction="OPEN", gt="OPEN"), prediction)["output_valid"]:
        raise RuntimeError("correction smoke returned an invalid output")
    output = reflection(prompt='Return only a JSON object with a "ready" key set to true.',
                        response_format={"type": "json_object"})
    if json.loads(output[0]) != {"ready": True}:
        raise RuntimeError("reflection smoke returned invalid JSON")
    report = {"status": "passed", "at_utc": now(), "experiment_sha256": fingerprint()["experiment_sha256"],
              "student_output": prediction.corrected_text, "usage": usage_summary()}
    write_json(config.HERE / "status" / "api_smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "api-smoke", "train", "test", "all"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--model", choices=config.MODELS)
        subparser.add_argument("--group", choices=config.GROUPS)
    prepare = commands.add_parser("prepare", help="seed split, selected-image OCR, and frozen manifest")
    prepare.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    prepare.add_argument("--ocr-python", default=sys.executable)
    prepare.add_argument("--output-dir", type=Path, default=config.DATA_MANIFEST.parent)
    prepare.add_argument("--step", choices=["split", "ocr", "freeze", "all"], default="all")
    prepare.add_argument("--seed", type=int, default=config.SEED)
    prepare.add_argument("--train-target", type=int, default=500)
    prepare.add_argument("--val-target", type=int, default=500)
    prepare.add_argument("--test-per-dataset", type=int, default=300)
    prepare.add_argument("--general-train", type=int, default=120)
    prepare.add_argument("--general-val", type=int, default=300)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--batch-size", type=int, default=128)
    prepare.add_argument("--parseq-root", type=Path)
    prepare.add_argument("--openocr-root", type=Path)
    prepare.add_argument("--svtr-config", type=Path)
    prepare.add_argument("--svtr-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        command = [args.ocr_python, str(config.ROOT / "ocr" / "prepare.py"), args.step,
                   "--manifest", str(config.DATA_MANIFEST)]
        for name in ("data_root", "output_dir", "seed", "train_target", "val_target", "test_per_dataset",
                     "general_train", "general_val", "device", "batch_size", "parseq_root",
                     "openocr_root", "svtr_config", "svtr_checkpoint"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        subprocess.run(command, check=True)
        return
    if bool(args.model) != bool(args.group):
        parser.error("--model and --group must be provided together")
    if args.model and args.command != "train":
        parser.error("--model and --group are supported only by train")
    report = preflight()
    if args.command == "preflight":
        print(json.dumps(report, indent=2))
        return
    require_key()
    if args.command == "api-smoke":
        print(json.dumps(api_smoke(), indent=2))
        return
    smoke = config.HERE / "status" / "api_smoke.json"
    if not smoke.exists() or json.loads(smoke.read_text()).get("experiment_sha256") != fingerprint()["experiment_sha256"]:
        raise RuntimeError("run api-smoke successfully under this exact configuration first")
    if args.command in ("train", "all"):
        for model in ([args.model] if args.model else config.MODELS):
            for group in ([args.group] if args.group else config.GROUPS):
                for stage in config.STAGES:
                    train_scope(model, group, stage)
        if not args.model:
            freeze_grid()
    if args.command in ("test", "all"):
        run_test()

