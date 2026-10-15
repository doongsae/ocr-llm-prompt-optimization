"""Analyze GEPA+Bootstrap test outputs and verify their aggregate metrics."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHARSET = set("0123456789abcdefghijklmnopqrstuvwxyz")
DATASETS = [
    "IIIT5k", "SVT", "IC13_857", "IC15_1811", "SVTP", "CUTE80",
    "Curve", "Multi-Oriented", "Artistic", "Contextless", "Salient",
    "Multi-Words", "General", "OST_weak", "OST_heavy",
]
MODELS = ["crnn", "vitstr", "trba", "abinet", "parseq", "svtrv2"]
GROUPS = {
    "Common": DATASETS[:6],
    "U14M": DATASETS[6:13],
    "OST": DATASETS[13:],
}
CATEGORIES = [
    "correct_preserved", "error_fixed", "correct_harmed",
    "wrong_unchanged_literal", "wrong_changed_literal",
]


def normalize(text):
    return "".join(character for character in text.lower() if character in CHARSET)


def valid_output(output, original):
    return ((bool(output.strip()) or not original.strip())
            and not any(token in output for token in
                        ["\n", "\r", "<think>", "</think>", "<|channel"]))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(counter, outcomes=None, total_errors=None):
    result = {key: counter[key] for key in
              ["rows", "raw_correct", "raw_wrong", "ours_correct", *CATEGORIES,
               "wrong_unchanged_normalized", "wrong_changed_normalized",
               "changed_wrong_edit_better", "changed_wrong_edit_same",
               "changed_wrong_edit_worse", "invalid_output"]}
    count = counter["rows"]
    result.update(
        raw_accuracy_pct=100 * counter["raw_correct"] / count,
        raw_error_rate_pct=100 * counter["raw_wrong"] / count,
        ours_accuracy_pct=100 * counter["ours_correct"] / count,
        net_correct=counter["error_fixed"] - counter["correct_harmed"],
        delta_pp=100 * (counter["error_fixed"] - counter["correct_harmed"]) / count,
        error_fix_rate_pct=(100 * counter["error_fixed"] / counter["raw_wrong"]
                            if counter["raw_wrong"] else None),
        wrong_unchanged_literal_rate_pct=(
            100 * counter["wrong_unchanged_literal"] / counter["raw_wrong"]
            if counter["raw_wrong"] else None),
        wrong_changed_literal_rate_pct=(
            100 * counter["wrong_changed_literal"] / counter["raw_wrong"]
            if counter["raw_wrong"] else None),
        wrong_unchanged_normalized_rate_pct=(
            100 * counter["wrong_unchanged_normalized"] / counter["raw_wrong"]
            if counter["raw_wrong"] else None),
        wrong_changed_normalized_rate_pct=(
            100 * counter["wrong_changed_normalized"] / counter["raw_wrong"]
            if counter["raw_wrong"] else None),
        correct_harm_rate_pct=(100 * counter["correct_harmed"] / counter["raw_correct"]
                              if counter["raw_correct"] else None),
    )
    if total_errors is not None:
        result["raw_error_share_pct"] = 100 * counter["raw_wrong"] / total_errors if total_errors else 0.0
    if counter["raw_wrong"]:
        rates = sum(result[key] for key in
                    ["error_fix_rate_pct", "wrong_unchanged_literal_rate_pct",
                     "wrong_changed_literal_rate_pct"])
        if abs(rates - 100) >= 1e-8:
            raise ValueError("raw-error categories do not sum to 100 percent")
    if outcomes is not None:
        result.update(wins=sum(value > 0 for value in outcomes),
                      ties=sum(value == 0 for value in outcomes),
                      losses=sum(value < 0 for value in outcomes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    source, output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    if output_dir == source or source.is_relative_to(output_dir):
        parser.error("analysis output must not overwrite or contain its source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = source / "bootstrap_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    with (source / "main_table.csv").open(newline="", encoding="utf-8") as handle:
        cell_rows = [row for row in csv.DictReader(handle) if row["method"] == "bootstrap"]
    cells = {(row["model"], row["dataset"]): row for row in cell_rows}
    expected_cells = {(model, dataset) for model in MODELS for dataset in DATASETS}
    if len(cells) != len(cell_rows) or set(cells) != expected_cells:
        raise ValueError("Bootstrap main table has duplicate or missing model/dataset cells")
    summary = json.loads((source / "summary.json").read_text())["methods"]["bootstrap"]
    if {(row["model"], row["dataset"]) for row in predictions} != expected_cells:
        raise ValueError("Bootstrap predictions have incomplete model/dataset coverage")
    if len({(row["model"], row["dataset"], row["image_idx"])
            for row in predictions}) != len(predictions):
        raise ValueError("duplicate Bootstrap prediction keys")

    totals = Counter()
    grouped = defaultdict(Counter)
    dataset_counts = defaultdict(Counter)
    model_counts = defaultdict(Counter)
    cell_counts = defaultdict(Counter)
    classified = []
    for row in predictions:
        raw, output, gt = map(normalize, [row["ocr_pred"], row["llm_corrected"], row["gt"]])
        original_correct = int(row["baseline_correct"])
        candidate_correct = int(row["candidate_correct"])
        valid = valid_output(row["llm_corrected"], row["ocr_pred"])
        if original_correct != int(raw == gt) or candidate_correct != int(valid and output == gt):
            raise ValueError(f"stale correctness: {row['model']}/{row['dataset']}/{row['image_idx']}")
        if original_correct and candidate_correct:
            category = "correct_preserved"
        elif candidate_correct:
            category = "error_fixed"
        elif original_correct:
            category = "correct_harmed"
        elif row["ocr_pred"] == row["llm_corrected"]:
            category = "wrong_unchanged_literal"
        else:
            category = "wrong_changed_literal"
        update = Counter(rows=1, raw_correct=original_correct,
                         raw_wrong=1 - original_correct, ours_correct=candidate_correct)
        update[category] += 1
        update["invalid_output"] += int(not valid)
        normalized_category = category
        if not original_correct and not candidate_correct:
            normalized_category = ("wrong_unchanged_normalized" if raw == output
                                   else "wrong_changed_normalized")
            update[normalized_category] += 1
            if raw != output:
                before, after = int(row["dist_before"]), int(row["dist_after"])
                update[("changed_wrong_edit_better" if after < before
                        else "changed_wrong_edit_same" if after == before
                        else "changed_wrong_edit_worse")] += 1
        group = next(name for name, members in GROUPS.items() if row["dataset"] in members)
        for counter in [totals, grouped[group], dataset_counts[row["dataset"]],
                        model_counts[row["model"]], cell_counts[row["model"], row["dataset"]]]:
            counter.update(update)
        classified.append({**row, "benchmark_group": group,
                           "outcome_literal": category,
                           "outcome_normalized": normalized_category})

    for key, counter in cell_counts.items():
        expected = cells[key]
        for actual, field in [("rows", "rows"), ("raw_correct", "baseline_correct"),
                              ("ours_correct", "candidate_correct"),
                              ("error_fixed", "fixes"), ("correct_harmed", "harms")]:
            if counter[actual] != int(expected[field]):
                raise ValueError(f"cell metrics disagree with predictions: {key}/{field}")
        if sum(counter[name] for name in CATEGORIES) != counter["rows"]:
            raise ValueError(f"outcome categories do not cover every row: {key}")
    for actual, field in [("rows", "rows"), ("raw_correct", "baseline_correct"),
                          ("ours_correct", "candidate_correct"),
                          ("error_fixed", "fixes"), ("correct_harmed", "harms")]:
        if totals[actual] != summary[field]:
            raise ValueError(f"summary disagrees with predictions: {field}")

    delta = {key: counter["error_fixed"] - counter["correct_harmed"]
             for key, counter in cell_counts.items()}
    total_errors = totals["raw_wrong"]
    overall = statistics(totals, list(delta.values()), total_errors)
    for field in ("wins", "ties", "losses"):
        if overall[field] != summary[field]:
            raise ValueError(f"summary disagrees with cell outcomes: {field}")
    dataset_rows = [{"dataset": dataset, **statistics(dataset_counts[dataset],
                     [delta[model, dataset] for model in MODELS], total_errors)}
                    for dataset in DATASETS]
    group_rows = [{"group": group, **statistics(grouped[group],
                   [delta[model, dataset] for model in MODELS for dataset in members],
                   total_errors)}
                  for group, members in GROUPS.items()]
    model_rows = [{"model": model, **statistics(model_counts[model],
                   [delta[model, dataset] for dataset in DATASETS], total_errors)}
                  for model in MODELS]
    cell_rows = [{"model": model, "dataset": dataset,
                  **statistics(cell_counts[model, dataset], total_errors=total_errors)}
                 for model in MODELS for dataset in DATASETS]
    write_csv(output_dir / "classified_test.csv", classified)
    for name, rows in [("by_dataset", dataset_rows), ("by_group", group_rows),
                       ("by_model", model_rows), ("by_cell", cell_rows)]:
        write_csv(output_dir / f"{name}.csv", rows)
    result = {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "GEPA+Bootstrap", "test_split_modified": False,
        "fresh_inference_calls": 0, "verified_against_cell_metrics_and_summary": True,
        "correctness": "lowercase ASCII alphanumeric exact match with existing output validity check",
        "literal_unchanged": "the original OCR string exactly equals the output string",
        "normalized_unchanged": "the normalized OCR string equals the normalized output string",
        "rate_denominators": {
            "raw_error_rate_pct": "rows in the given slice",
            "raw_error_share_pct": "raw OCR errors across the complete supplied test",
            "error_fix_rate_pct": "raw OCR errors in the given slice",
            "wrong_unchanged_literal_rate_pct": "raw OCR errors in the given slice",
            "wrong_changed_literal_rate_pct": "raw OCR errors in the given slice",
            "correct_harm_rate_pct": "raw OCR correct predictions in the given slice",
        },
        "overall": overall, "by_group": group_rows, "by_dataset": dataset_rows,
        "by_model": model_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
