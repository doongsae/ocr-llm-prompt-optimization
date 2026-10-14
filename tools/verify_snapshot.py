"""Compare historical and copied contracts offline without requesting credentials."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import dspy
if sys.argv[2] == "package":
    from pipeline import data as frozen
    from pipeline.config import SETTINGS
    from pipeline.core import fresh_program, filter_text
    from pipeline.data import build_error_biased_trainset_safe
    from pipeline.json_adapter import OCRJSONAdapter
    from pipeline.metrics import facts, bootstrap_metric
else:
    from config import SETTINGS
    from data_contract import frozen, fresh_program, build_error_biased_trainset_safe, filter_text
    from json_adapter import OCRJSONAdapter
    from metrics import facts, bootstrap_metric

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()

def examples_hash(examples):
    return digest([dict(example) for example in examples])

frame = frozen.load_data_manifest()
program = fresh_program()
scopes = []
for model in frozen.MODELS:
    for group in frozen.GROUPS:
        train, val = frozen.load_scope_examples("group", model, group)
        choices = {}
        for stage in ("gepa", "bootstrap"):
            biased, errors, corrects, fallback = build_error_biased_trainset_safe(
                train, SETTINGS[stage + "_error_ratio"], SETTINGS[stage + "_seed"])
            choices[stage] = [examples_hash(biased), errors, corrects, fallback]
        scopes.append([model, group, examples_hash(train), examples_hash(val), choices])

values = [None, "", "NA", "NULL", "OPEN", "Sp-ed!", "123", "A\nB", "a\"b\\c", "STOR"]
metrics = []
messages = []
for raw, gt, output in [("OPEN", "OPEN", "OPEN"), ("OPEN", "OPEN", "OPENS"),
                        ("SPED", "SPEED", "SPEED"), ("SPED", "SPEED", "SPED"),
                        ("SPEXX", "SPEED", "SPEEX"), ("", "", ""), ("", "OPEN", "")]:
    example = dspy.Example(ocr_prediction=raw, gt=gt, corrected_text=gt,
                          model="trba", dataset="SVT", image_idx="1").with_inputs("ocr_prediction")
    pred = dspy.Prediction(corrected_text=output)
    metrics.append([facts(example, pred), bootstrap_metric(example, pred),
                    bootstrap_metric(example, pred, trace=[])])
    messages.append(OCRJSONAdapter().format(program.correct.signature, [example], {"ocr_prediction": raw}))

print(json.dumps({"settings": SETTINGS, "signature": program.correct.signature.instructions,
                  "inputs": list(program.correct.signature.input_fields),
                  "outputs": list(program.correct.signature.output_fields),
                  "normalization": [filter_text(value) for value in values],
                  "frozen_rows": len(frame),
                  "frozen_rows_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                  "scopes": scopes, "metrics_sha256": digest(metrics),
                  "messages_sha256": digest(messages)}, sort_keys=True))
'''


def probe(path, manifest, cache, mode="original"):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    environment.update(OCR_DATA_MANIFEST=str(manifest), DSPY_CACHEDIR=str(cache),
                       LITELLM_LOCAL_MODEL_COST_MAP="True")
    output = subprocess.check_output([sys.executable, "-c", PROBE, str(path), mode], env=environment)
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original completed experiment directory")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        before = probe(args.original.resolve(), args.manifest.resolve(), Path(temporary) / "before")
        after = probe(ROOT, args.manifest.resolve(), Path(temporary) / "after", mode="package")
    mismatches = [key for key in before if before[key] != after[key]]
    if mismatches:
        raise RuntimeError("Snapshot changed these contracts: " + ", ".join(mismatches))
    print(json.dumps({"status": "passed", "frozen_rows": after["frozen_rows"],
                      "scopes_checked": len(after["scopes"]),
                      "contracts_checked": list(after), "api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
