"""Frozen OCR rows, scope-local examples, and deterministic train sampling."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import dspy
import pandas as pd

from .config import DATA_MANIFEST, DATASETS, GROUPS, MODELS
from .core import filter_text
DATA_COLUMNS = [
    "split",
    "model",
    "dataset",
    "image_idx",
    "gt",
    "gt_filtered",
    "pred",
    "pred_filtered",
    "correct",
]

def read_csv_clean(path: Path, **kwargs) -> pd.DataFrame:
    defaults = {
        "dtype": {"model": str, "dataset": str, "image_idx": str},
        "keep_default_na": False,
    }
    defaults.update(kwargs)
    frame = pd.read_csv(path, **defaults)
    if "image_idx" in frame:
        frame["image_idx"] = frame["image_idx"].astype(str)
    return frame


@lru_cache(maxsize=1)
def _cached_data_manifest() -> pd.DataFrame:
    if not DATA_MANIFEST.exists():
        raise FileNotFoundError(
            f"missing frozen data manifest: {DATA_MANIFEST}; run prepare or set data_manifest in config.json"
        )
    frame = read_csv_clean(DATA_MANIFEST)
    missing = [column for column in DATA_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"frozen data manifest missing columns: {missing}")
    frame = frame[DATA_COLUMNS].copy()
    for column in ["split", "model", "dataset", "image_idx", "gt", "pred"]:
        frame[column] = frame[column].astype(str)
    if frame.duplicated(["split", "model", "dataset", "image_idx"]).any():
        raise RuntimeError("duplicate keys in frozen data manifest")
    return frame


def load_data_manifest() -> pd.DataFrame:
    return _cached_data_manifest().copy()


def load_scope_frames(
    scope: str, model: str, name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    if scope == "group":
        if name not in GROUPS:
            raise ValueError(f"unknown group: {name}")
        datasets = GROUPS[name]
    elif scope == "cell":
        if name not in DATASETS:
            raise ValueError(f"unknown dataset: {name}")
        datasets = [name]
    else:
        raise ValueError(f"unknown scope: {scope}")

    frame = _cached_data_manifest()
    selected = frame[
        (frame["model"] == model) & frame["dataset"].isin(datasets)
    ]
    train = selected[selected["split"] == "train"].drop(columns="split").copy()
    val = selected[selected["split"] == "val"].drop(columns="split").copy()
    if train.empty or val.empty:
        raise RuntimeError(f"empty scope data: {scope}/{model}/{name}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def load_test_frame() -> pd.DataFrame:
    frame = _cached_data_manifest()
    test = frame[frame["split"] == "test"].drop(columns="split").copy()
    coverage = set(zip(test.model, test.dataset))
    if coverage != {(model, dataset) for model in MODELS for dataset in DATASETS}:
        raise RuntimeError("test model/dataset coverage is incomplete")
    return test.reset_index(drop=True)


def frame_to_examples(frame: pd.DataFrame) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    for row in frame.itertuples(index=False):
        examples.append(
            dspy.Example(
                ocr_prediction=str(row.pred),
                corrected_text=str(row.gt),
                gt=str(row.gt),
                gt_filtered=filter_text(row.gt),
                model=str(row.model),
                dataset=str(row.dataset),
                image_idx=str(row.image_idx),
            ).with_inputs("ocr_prediction")
        )
    return examples


def load_scope_examples(
    scope: str, model: str, name: str
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    train, val = load_scope_frames(scope, model, name)
    return frame_to_examples(train), frame_to_examples(val)


def build_error_biased_trainset_safe(
    examples: list[dspy.Example],
    target_error_ratio: float,
    seed: int,
    minimum_rows: int = 3,
) -> tuple[list[dspy.Example], int, int, str]:
    errors = [ex for ex in examples if filter_text(ex.ocr_prediction) != filter_text(ex.gt)]
    corrects = [ex for ex in examples if filter_text(ex.ocr_prediction) == filter_text(ex.gt)]
    rng = random.Random(seed)
    rng.shuffle(corrects)

    n_correct = int(len(errors) * (1.0 - target_error_ratio) / target_error_ratio)
    fallback = ""
    if not errors:
        n_correct = len(corrects)
        fallback = "zero_error_keep_all_correct"
    elif len(errors) + n_correct < minimum_rows:
        n_correct = min(len(corrects), max(n_correct, minimum_rows - len(errors)))
        fallback = "sparse_error_fill_to_minimum"
    elif n_correct > len(corrects):
        n_correct = len(corrects)
        fallback = "insufficient_correct_for_target_ratio"

    biased = errors + corrects[:n_correct]
    rng.shuffle(biased)
    if not biased:
        raise RuntimeError("scope has no usable training examples")
    return biased, len(errors), n_correct, fallback


