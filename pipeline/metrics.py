"""EM scoring, exact-trace admission and reflection feedback."""

import json

import dspy
from rapidfuzz.distance import Levenshtein

from .core import filter_text


def valid_output(value, original):
    return (isinstance(value, str) and (bool(value.strip()) or not str(original).strip())
            and not any(token in value for token in
                        ("\n", "\r", "<think>", "</think>", "<|channel")))


def facts(gold, pred):
    output = getattr(pred, "corrected_text", None)
    valid = valid_output(output, gold.ocr_prediction)
    gt = filter_text(gold.gt)
    return {
        "ocr_prediction": str(gold.ocr_prediction),
        "ground_truth": str(gold.gt),
        "corrected_text": output if isinstance(output, str) else None,
        "output_valid": valid,
        "raw_em": int(filter_text(gold.ocr_prediction) == gt),
        "em": int(valid and filter_text(output) == gt),
        "edit_distance_before": Levenshtein.distance(filter_text(gold.ocr_prediction), gt),
        "edit_distance_after": Levenshtein.distance(filter_text(output), gt) if valid else None,
    }


def exact_metric(gold, pred, trace=None):
    return float(facts(gold, pred)["em"])


def trace_is_exact(gold, pred, trace=None):
    return bool(facts(gold, pred)["em"])


def bootstrap_metric(gold, pred, trace=None):
    return trace_is_exact(gold, pred, trace) if trace is not None else exact_metric(gold, pred)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    record = facts(gold, pred)
    score = float(record["em"])
    if pred_name is None:
        return score
    return dspy.Prediction(score=score, feedback=json.dumps(record, ensure_ascii=True))

