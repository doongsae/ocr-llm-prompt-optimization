"""Strict row inference, cell accounting, and frozen test evaluation."""

import time

import dspy
import pandas as pd

from . import config, data
from .artifacts import freeze_grid, job_dir, verify_job
from .core import fresh_program
from .metrics import facts
from .openai_lm import reset_usage, setup_lms, usage_summary
from .storage import now, sha256_path, write_json


def summarize_rows(rows):
    if not rows:
        raise ValueError("empty evaluation")
    frame = pd.DataFrame(rows)
    cells = []
    for (model, dataset), group in frame.groupby(["model", "dataset"], sort=True):
        before = int(group.baseline_correct.sum())
        after = int(group.candidate_correct.sum())
        fixes = int(((group.baseline_correct == 0) & (group.candidate_correct == 1)).sum())
        harms = int(((group.baseline_correct == 1) & (group.candidate_correct == 0)).sum())
        if after - before != fixes - harms:
            raise ValueError("inconsistent fix/harm counts")
        cells.append({"model": model, "dataset": dataset, "rows": len(group),
                      "baseline_correct": before, "candidate_correct": after,
                      "fixes": fixes, "harms": harms, "net_correct": after - before,
                      "delta": (after - before) / len(group)})
    before = sum(c["baseline_correct"] for c in cells)
    after = sum(c["candidate_correct"] for c in cells)
    return {"rows": len(rows), "cells": len(cells), "baseline_correct": before,
            "candidate_correct": after, "baseline_accuracy": before / len(rows),
            "candidate_accuracy": after / len(rows), "fixes": sum(c["fixes"] for c in cells),
            "harms": sum(c["harms"] for c in cells), "net_correct": after - before,
            "wins": sum(c["net_correct"] > 0 for c in cells),
            "ties": sum(c["net_correct"] == 0 for c in cells),
            "losses": sum(c["net_correct"] < 0 for c in cells),
            "per_cell": cells}


def evaluate(program, examples):
    inputs = [dspy.Example(ocr_prediction=ex.ocr_prediction).with_inputs("ocr_prediction") for ex in examples]
    predictions = program.batch(inputs, num_threads=config.SETTINGS["num_threads"], max_errors=1,
                                disable_progress_bar=True)
    if len(predictions) != len(examples) or any(p is None for p in predictions):
        raise RuntimeError("inference did not produce all rows; no identity fallback")
    rows = []
    for ex, pred in zip(examples, predictions):
        record = facts(ex, pred)
        if not record["output_valid"]:
            raise RuntimeError("invalid final output; refusing to report a successful evaluation")
        rows.append({"model": ex.model, "dataset": ex.dataset, "image_idx": ex.image_idx,
                     "gt": ex.gt, "ocr_pred": ex.ocr_prediction, "llm_corrected": pred.corrected_text,
                     "baseline_correct": record["raw_em"], "candidate_correct": record["em"],
                     "dist_before": record["edit_distance_before"], "dist_after": record["edit_distance_after"]})
    return summarize_rows(rows), rows


def run_test():
    manifest = config.HERE / "manifests" / "programs.json"
    if not manifest.exists():
        raise RuntimeError("freeze all 24 scopes before any test inference")
    freeze_grid()
    output = config.HERE / "results"
    if output.exists():
        raise FileExistsError("test was already started; no automatic repeat")
    output.mkdir()
    started = time.monotonic()
    reset_usage()
    state = {"status": "running", "started_at_utc": now(), "manifest_sha256": sha256_path(manifest)}
    write_json(output / "status.json", state)
    try:
        frame = data.load_test_frame()
        setup_lms(config.SETTINGS)
        methods = {}
        all_cells = []
        for stage in config.STAGES:
            stage_start = time.monotonic()
            rows = []
            for model in config.MODELS:
                for group, datasets in config.GROUPS.items():
                    verify_job(model, group, stage)
                    program = fresh_program()
                    program.load(str(job_dir(model, group, stage) / "program.json"))
                    examples = data.frame_to_examples(frame[(frame.model == model) & frame.dataset.isin(datasets)])
                    _, scope_rows = evaluate(program, examples)
                    rows.extend(scope_rows)
            report = summarize_rows(rows)
            if report["cells"] != len(config.MODELS) * len(config.DATASETS) or report["rows"] != len(frame):
                raise ValueError("test coverage is incomplete")
            report["runtime_seconds"] = time.monotonic() - stage_start
            methods[stage] = report
            pd.DataFrame(rows).to_csv(output / f"{stage}_predictions.csv", index=False)
            all_cells.extend({"method": stage, **cell} for cell in report["per_cell"])
        pd.DataFrame(all_cells).to_csv(output / "main_table.csv", index=False)
        if sha256_path(manifest) != state["manifest_sha256"]:
            raise ValueError("program manifest changed during test")
        state.update(status="completed", methods=methods, usage=usage_summary(),
                     finished_at_utc=now(), runtime_seconds=time.monotonic() - started)
        write_json(output / "summary.json", state)
        lines = ["# OCR Post-Correction Results", "",
                 "| Method | EM (%) | Fixes | Harms | W/T/L | Test Seconds |",
                 "| --- | ---: | ---: | ---: | --- | ---: |"]
        for stage, report in methods.items():
            lines.append(f"| {stage} | {100*report['candidate_accuracy']:.4f} | {report['fixes']} | "
                         f"{report['harms']} | {report['wins']}/{report['ties']}/{report['losses']} | "
                         f"{report['runtime_seconds']:.2f} |")
        training = [verify_job(model, group, stage) for model in config.MODELS for group in config.GROUPS for stage in config.STAGES]
        lines += ["", f"Training wall-seconds sum: {sum(j['runtime_seconds'] for j in training):.2f}",
                  f"Test wall-seconds: {state['runtime_seconds']:.2f}", "", "## Cell Results", "",
                  "| Method | OCR | Dataset | Raw Correct | LLM Correct | Fixes | Harms | Delta (%) |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for cell in all_cells:
            lines.append(f"| {cell['method']} | {cell['model']} | {cell['dataset']} | {cell['baseline_correct']} | "
                         f"{cell['candidate_correct']} | {cell['fixes']} | {cell['harms']} | {100*cell['delta']:.4f} |")
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, usage=usage_summary(),
                     finished_at_utc=now(), runtime_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output / "status.json", state)

