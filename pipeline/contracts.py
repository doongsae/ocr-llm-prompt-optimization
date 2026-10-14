"""Validate split coverage, train-only demos, and runtime input boundaries."""

import dspy
from . import data
from .config import DATA_MANIFEST, DATASETS, GROUPS, MODELS
from .core import filter_text
from .json_adapter import OCRJSONAdapter
from .storage import sha256_path


def audit_data():
    frame = data.load_data_manifest()
    if set(frame.model) != set(MODELS) or set(frame.dataset) != set(DATASETS):
        raise ValueError("data model/dataset coverage changed")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("unexpected or missing split labels")
    keys = ["model", "dataset", "image_idx"]
    if frame.duplicated(keys).any():
        raise ValueError("an image occurs in more than one split or is duplicated")
    alignment = ["split", "dataset", "image_idx", "gt"]
    assignments = frame.groupby(["dataset", "image_idx"], sort=False)
    if (assignments["split"].nunique() > 1).any():
        raise ValueError("cross-model split alignment differs")
    if (assignments["gt"].nunique() > 1).any():
        raise ValueError("cross-model GT alignment differs")
    # Historical manifests can have model-local train keys; evaluation keys must match.
    reference = frame.loc[(frame.model == MODELS[0]) & (frame.split != "train"), alignment].sort_values(alignment).reset_index(drop=True)
    for model in MODELS:
        actual = frame.loc[frame.model == model, alignment].sort_values(alignment).reset_index(drop=True)
        if not actual.loc[actual.split != "train"].reset_index(drop=True).equals(reference):
            raise ValueError(f"split/image/GT alignment differs for {model}")
        coverage = set(zip(actual.split, actual.dataset))
        if coverage != {(split, ds) for split in ("train", "val", "test") for ds in DATASETS}:
            raise ValueError(f"split/dataset coverage is incomplete for {model}")
    if not frame["gt"].map(filter_text).equals(frame.gt_filtered):
        raise ValueError("stale normalized GT")
    if not frame.pred.map(filter_text).equals(frame.pred_filtered):
        raise ValueError("stale normalized OCR prediction")
    correct = (frame.gt_filtered == frame.pred_filtered).astype(int)
    if not correct.equals(frame.correct.astype(int)):
        raise ValueError("stale raw OCR correctness")
    scopes = []
    for model in MODELS:
        for group in GROUPS:
            train, val = data.load_scope_examples("group", model, group)
            if any(set(ex.inputs()) != {"ocr_prediction"} for ex in train + val):
                raise ValueError("metadata became an input field")
            scopes.append({"model": model, "group": group,
                           "train_rows": len(train), "validation_rows": len(val)})
    return {"status": "passed", "data_sha256": sha256_path(DATA_MANIFEST),
            "rows": len(frame), "split_rows_per_model": {
                model: frame.loc[frame.model == model, "split"].value_counts().to_dict() for model in MODELS},
            "training_scopes": len(scopes),
            "test_cells": len(MODELS) * len(DATASETS), "scopes": scopes}


def verify_program(program, train):
    if [name for name, _ in program.named_predictors()] != ["correct"]:
        raise ValueError("expected exactly one correction predictor")
    pred = program.correct
    if set(pred.signature.input_fields) != {"ocr_prediction"}:
        raise ValueError("unexpected runtime inputs")
    if set(pred.signature.output_fields) != {"corrected_text"}:
        raise ValueError("unexpected runtime outputs")
    allowed = {}
    for ex in train:
        allowed.setdefault(str(ex.ocr_prediction), set()).add(filter_text(ex.gt))
    for demo in pred.demos:
        if filter_text(demo.corrected_text) not in allowed.get(str(demo.ocr_prediction), set()):
            raise ValueError("demo is not an exact answer from this scope's train")
    if len(pred.demos) > 12:
        raise ValueError("demo cap exceeded")
    canary = dspy.Example(ocr_prediction="AUDIT_OCR", corrected_text="AUDIT_DEMO",
                         gt="SECRET_GT", model="SECRET_MODEL", dataset="SECRET_DATASET",
                         image_idx="SECRET_IMAGE").with_inputs("ocr_prediction")
    messages = OCRJSONAdapter().format(pred.signature, list(pred.demos) + [canary],
                                         {"ocr_prediction": "AUDIT_RUNTIME"})
    if any(secret in str(messages) for secret in
           ("SECRET_GT", "SECRET_MODEL", "SECRET_DATASET", "SECRET_IMAGE")):
        raise ValueError("non-input metadata leaked into runtime prompt")
    return {"input_fields": ["ocr_prediction"], "output_fields": ["corrected_text"],
            "demos": len(pred.demos), "exact_train_demos": True, "metadata_exposure": 0}
