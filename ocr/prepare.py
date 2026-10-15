"""Select benchmark images before OCR and assemble the pipeline data manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OPENOCR_COMMIT = "45bfc0cc791c055739872e7255fe032cac1abe9c"
OPENOCR_ARCHIVE_SHA256 = "b9217f9cbdaab7589bda9d653cb1461c919a027e308b2d062d062c8ca4ea407f"
SVTR_CHECKPOINT_ID = "1P56B_PVYBF649zUqQFShuAZC1XuNmUcL"
SVTR_CHECKPOINT_SHA256 = "b65dd4675ef061a116390ba1590f538b69ab18795a55fd30a14bd98af1338f49"
SVTR_CONFIG_PATH = "configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml"

DATASET_PATHS = {
    "IIIT5k": "test/IIIT5k", "SVT": "test/SVT", "IC13_857": "test/IC13_857",
    "IC15_1811": "test/IC15_1811", "SVTP": "test/SVTP", "CUTE80": "test/CUTE80",
    "Curve": "u14m/curve", "Multi-Oriented": "u14m/multi_oriented",
    "Artistic": "u14m/artistic", "Contextless": "u14m/contextless",
    "Salient": "u14m/salient", "Multi-Words": "u14m/multi_words",
    "General": "u14m/general", "OST_weak": "OST/weak", "OST_heavy": "OST/heavy",
}
MODELS = ("crnn", "vitstr", "trba", "abinet", "parseq", "svtrv2")
SPLITS = ("train", "val", "test")
SELECT_COLUMNS = ("split", "dataset", "image_idx", "gt")
PRED_COLUMNS = ("model", "dataset", "image_idx", "gt", "gt_filtered", "pred", "pred_filtered", "correct")
DATA_COLUMNS = ("split", *PRED_COLUMNS)


def normalized_text(text):
    return "".join(ch for ch in text.lower() if ch in "0123456789abcdefghijklmnopqrstuvwxyz")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_selection(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(SELECT_COLUMNS):
            raise ValueError("invalid selection columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        row["image_idx"] = int(row["image_idx"])
        key = (row["dataset"], row["image_idx"])
        if row["split"] not in SPLITS or row["dataset"] not in DATASET_PATHS or row["image_idx"] < 0:
            raise ValueError(f"invalid selected key: {key}")
        if key in seen:
            raise ValueError(f"duplicate or overlapping split key: {key}")
        seen.add(key)
    if not rows:
        raise ValueError("empty image selection")
    return rows


def svtr_label_supported(gt, max_text_length):
    normalized = unicodedata.normalize("NFKD", "".join(gt.split())).encode("ascii", "ignore").decode()
    return 1 <= len(normalized) <= max_text_length


def read_catalogue(data_root, datasets, max_text_length, include_svtr=False):
    import lmdb

    catalogue, counts = {}, {}
    for dataset in datasets:
        path = Path(data_root) / DATASET_PATHS[dataset]
        if not (path / "data.mdb").is_file():
            raise FileNotFoundError(f"missing LMDB dataset: {path}")
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin() as txn:
                count = int(txn.get(b"num-samples"))
                eligible = []
                rejected_svtr = 0
                for image_idx in range(count):
                    raw = txn.get(f"label-{image_idx + 1:09d}".encode())
                    if raw is None:
                        raise ValueError(f"missing GT: {dataset}/{image_idx}")
                    gt = raw.decode("utf-8")
                    if 1 <= len(normalized_text(gt)) <= max_text_length:
                        if include_svtr and not svtr_label_supported(gt, max_text_length):
                            rejected_svtr += 1
                            continue
                        eligible.append({"dataset": dataset, "image_idx": image_idx, "gt": gt})
            catalogue[dataset] = eligible
            counts[dataset] = {"source_rows": count, "eligible_rows": len(eligible),
                               "unsupported_svtr_labels": rejected_svtr}
            print(f"[labels] {dataset}: {len(eligible)}/{count} eligible", flush=True)
        finally:
            env.close()
    return catalogue, counts


def select_splits(catalogue, seed, train_target, val_target, test_per_dataset,
                  general_train, general_val):
    import numpy as np

    rng = np.random.RandomState(seed)
    test_rng = random.Random(seed)
    non_general_size = sum(len(rows) for ds, rows in catalogue.items() if ds != "General")
    selected = []
    for dataset, candidates in catalogue.items():
        count = len(candidates)
        if dataset == "General":
            train_n, val_n = general_train, general_val
        else:
            train_n = max(2, int(train_target * count / non_general_size)) if non_general_size else 0
            val_n = max(2, int(val_target * count / non_general_size)) if non_general_size else 0
        if count < train_n + val_n + 1:
            raise ValueError(f"{dataset}: {count} eligible images cannot supply train={train_n}, val={val_n} and test")
        # Keep General independent of the non-General split RNG.
        split_rng = np.random.RandomState(seed) if dataset == "General" else rng
        shuffled = split_rng.permutation(count).tolist()
        remaining = sorted(shuffled[train_n + val_n:])
        test_rng.shuffle(remaining)
        partitions = {"train": shuffled[:train_n], "val": shuffled[train_n:train_n + val_n],
                      "test": remaining[:test_per_dataset]}
        for split, indices in partitions.items():
            selected.extend({"split": split, **candidates[i]} for i in sorted(indices))
    return selected


def prepare_split(args):
    selection = args.output_dir / "selection.csv"
    if args.manifest.exists():
        return reuse_manifest(args.manifest)
    if selection.exists():
        rows = read_selection(selection)
        print(f"[reuse] split: {selection}", flush=True)
        return {"status": "reused", "selected_images": len(rows),
                "selection_sha256": file_hash(selection),
                "split_counts": dict(Counter(row["split"] for row in rows))}
    started = time.monotonic()
    catalogue, counts = read_catalogue(args.data_root, args.datasets, args.max_text_length,
                                       include_svtr="svtrv2" in args.models)
    rows = select_splits(catalogue, args.seed, args.train_target, args.val_target,
                         args.test_per_dataset, args.general_train, args.general_val)
    write_csv(selection, rows, SELECT_COLUMNS)
    read_selection(selection)
    report = {"seed": args.seed, "data_root": str(args.data_root), "datasets": args.datasets,
              "train_target_non_general": args.train_target, "val_target_non_general": args.val_target,
              "test_per_dataset": args.test_per_dataset, "general_train": args.general_train,
              "general_val": args.general_val, "max_text_length": args.max_text_length,
              "source_counts": counts,
              "joint_svtr_label_filter": "svtrv2" in args.models,
              "selected_images": len(rows), "selection_sha256": file_hash(selection),
              "split_counts": dict(Counter(row["split"] for row in rows)),
              "counts_by_dataset": {ds: dict(Counter(row["split"] for row in rows if row["dataset"] == ds))
                                    for ds in catalogue},
              "runtime_seconds": time.monotonic() - started}
    write_json(args.output_dir / "split.json", report)
    return report


def prediction_row(model, sample, prediction):
    gt_filtered = normalized_text(sample["gt"])
    pred_filtered = normalized_text(prediction)
    return {"model": model, "dataset": sample["dataset"], "image_idx": sample["image_idx"],
            "gt": sample["gt"], "gt_filtered": gt_filtered, "pred": prediction,
            "pred_filtered": pred_filtered, "correct": int(gt_filtered == pred_filtered)}


def validate_predictions(rows, selected, model):
    expected = {(row["dataset"], int(row["image_idx"])): row for row in selected}
    actual = {}
    for row in rows:
        key = (row["dataset"], int(row["image_idx"]))
        if key in actual or key not in expected or row["model"] != model:
            raise ValueError(f"unexpected/duplicate OCR row: {model}/{key}")
        if row["gt"] != expected[key]["gt"]:
            raise ValueError(f"GT alignment failure: {model}/{key}")
        corrected = prediction_row(model, expected[key], row["pred"])
        if any(str(row[col]) != str(corrected[col]) for col in ("gt_filtered", "pred_filtered", "correct")):
            raise ValueError(f"stale normalized values: {model}/{key}")
        actual[key] = corrected
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        raise ValueError(f"incomplete OCR coverage: {model}; missing={len(missing)}, first={missing[:5]}")
    return actual


def save_predictions(output_dir, model, rows, selected, provenance, started):
    validate_predictions(rows, selected, model)
    directory = Path(output_dir) / "predictions"
    path = directory / f"{model}.csv"
    rows.sort(key=lambda row: (list(DATASET_PATHS).index(row["dataset"]), int(row["image_idx"])))
    write_csv(path, rows, PRED_COLUMNS)
    report = {"model": model, "rows": len(rows), "selection_sha256": file_hash(Path(output_dir) / "selection.csv"),
              "prediction_sha256": file_hash(path), "provenance": provenance,
              "runtime_seconds": time.monotonic() - started}
    write_json(directory / f"{model}.json", report)
    print(f"[OCR] {model}: {len(rows)} selected images completed", flush=True)
    return report


def load_predictions(output_dir, model, selected):
    path = output_dir / "predictions" / f"{model}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = validate_predictions(rows, selected, model)
    digest = file_hash(path)
    selection_digest = file_hash(output_dir / "selection.csv")
    metadata = path.with_suffix(".json")
    if metadata.exists():
        report = json.loads(metadata.read_text(encoding="utf-8"))
        if report["selection_sha256"] != selection_digest or report["prediction_sha256"] != digest:
            raise ValueError(f"stale/modified OCR output: {model}")
    else:
        report = {"model": model, "rows": len(rows), "selection_sha256": selection_digest,
                  "prediction_sha256": digest, "runtime_seconds": 0,
                  "provenance": {"source": "existing_csv", "checkpoint": "not_recorded"}}
    return mapping, report


def reuse_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(DATA_COLUMNS):
            raise ValueError("existing manifest has invalid columns")
        rows = sum(1 for _ in reader)
    if not rows:
        raise ValueError("existing manifest is empty")
    print(f"[reuse] manifest: {path}", flush=True)
    return {"status": "reused", "manifest": str(path), "rows": rows, "manifest_sha256": file_hash(path)}


def download_asset(url, destination, expected_sha256, drive=False):
    if destination.is_file():
        if file_hash(destination) != expected_sha256:
            raise ValueError(f"cached OCR asset hash mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {destination.name}", flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        if drive:
            import gdown

            if not gdown.download(id=url, output=str(temporary), quiet=False, use_cookies=False):
                raise RuntimeError("official SVTRv2-B checkpoint download failed")
        else:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if file_hash(temporary) != expected_sha256:
            raise ValueError(f"downloaded OCR asset hash mismatch: {destination.name}")
        os.replace(temporary, destination)
    return destination


def resolve_svtr_assets(args):
    cache = ROOT / ".cache" / "ocr"
    root = getattr(args, "openocr_root", None)
    if root is None:
        root = cache / f"OpenOCR-{OPENOCR_COMMIT}"
        if not root.exists():
            archive = download_asset(f"https://codeload.github.com/Topdu/OpenOCR/zip/{OPENOCR_COMMIT}",
                                     cache / f"OpenOCR-{OPENOCR_COMMIT}.zip", OPENOCR_ARCHIVE_SHA256)
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                with zipfile.ZipFile(archive) as source:
                    source.extractall(directory)
                os.replace(Path(directory) / root.name, root)
    config = getattr(args, "svtr_config", None) or root / SVTR_CONFIG_PATH
    if not (root / "openrec").is_dir() or not config.is_file():
        raise FileNotFoundError("OpenOCR source or SVTRv2 inference config is missing")
    checkpoint = getattr(args, "svtr_checkpoint", None)
    if checkpoint is None:
        checkpoint = download_asset(SVTR_CHECKPOINT_ID, cache / "svtrv2-b-u14m-filter.pth",
                                    SVTR_CHECKPOINT_SHA256, drive=True)
    if not checkpoint.is_file():
        raise FileNotFoundError("SVTRv2 checkpoint is missing")
    args.openocr_root, args.svtr_config, args.svtr_checkpoint = root, config, checkpoint


def run_ocr(args):
    selected = read_selection(args.output_dir / "selection.csv")
    pending = []
    for model in args.models:
        if (args.output_dir / "predictions" / f"{model}.csv").exists():
            load_predictions(args.output_dir, model, selected)
            print(f"[reuse] OCR: {model}", flush=True)
        else:
            pending.append(model)
    if "svtrv2" in pending:
        resolve_svtr_assets(args)
    directory = Path(__file__).resolve().parent
    for model in pending:
        command = [sys.executable, str(directory / ("eval_svtrv2.py" if model == "svtrv2" else "eval_parseq_selected.py")),
                   "--data-root", str(args.data_root), "--output-dir", str(args.output_dir),
                   "--device", args.device, "--batch-size", str(args.batch_size)]
        if model == "svtrv2":
            command += ["--openocr-root", str(args.openocr_root), "--config", str(args.svtr_config),
                        "--checkpoint", str(args.svtr_checkpoint)]
        else:
            command += ["--model", model]
            if args.parseq_root:
                command += ["--parseq-root", str(args.parseq_root)]
        subprocess.run(command, check=True)


def freeze_manifest(args):
    if args.manifest.exists():
        return reuse_manifest(args.manifest)
    selected = read_selection(args.output_dir / "selection.csv")
    digest = file_hash(args.output_dir / "selection.csv")
    output = []
    reports = []
    for model in args.models:
        mapping, report = load_predictions(args.output_dir, model, selected)
        for split in SPLITS:
            for sample in selected:
                if sample["split"] == split:
                    output.append({"split": split, **mapping[(sample["dataset"], sample["image_idx"])]})
        reports.append(report)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.manifest.name}.", dir=args.manifest.parent)
    os.close(fd)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DATA_COLUMNS)
            writer.writeheader()
            writer.writerows(output)
        os.replace(temporary, args.manifest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    result = {"status": "completed", "manifest": str(args.manifest), "manifest_sha256": file_hash(args.manifest),
              "selected_images": len(selected), "rows": len(output), "models": args.models,
              "selection_sha256": digest, "ocr_runtime_seconds_sum": sum(r["runtime_seconds"] for r in reports),
              "ocr_provenance_by_model": {r["model"]: r["provenance"] for r in reports}}
    write_json(args.output_dir / "preparation.json", result)
    return result


def main():
    config_path = ROOT / "config.json"
    defaults = json.loads((config_path if config_path.is_file() else ROOT / "config.example.json").read_text(encoding="utf-8"))
    data_root = Path(defaults.get("data_root", "datasets")).expanduser()
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "ocr", "freeze", "all"])
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_PATHS, default=list(DATASET_PATHS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--train-target", type=int, default=500)
    parser.add_argument("--val-target", type=int, default=500)
    parser.add_argument("--test-per-dataset", type=int, default=300)
    parser.add_argument("--general-train", type=int, default=120)
    parser.add_argument("--general-val", type=int, default=300)
    parser.add_argument("--max-text-length", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parseq-root", type=Path)
    parser.add_argument("--openocr-root", type=Path)
    parser.add_argument("--svtr-config", type=Path)
    parser.add_argument("--svtr-checkpoint", type=Path)
    args = parser.parse_args()
    for name in ("data_root", "output_dir", "manifest", "parseq_root", "openocr_root", "svtr_config", "svtr_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.manifest = args.manifest or args.output_dir / "frozen_rows.csv.gz"
    if not args.manifest.name.endswith(".csv.gz"):
        parser.error("the generated manifest must use a .csv.gz filename")
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate models/datasets")
    if any(getattr(args, name) <= 0 for name in ("train_target", "val_target", "test_per_dataset", "general_train", "general_val", "max_text_length", "batch_size")):
        parser.error("sample counts, max length and batch size must be positive")
    if args.max_text_length > 25:
        parser.error("the supplied recognizers support at most 25 characters")
    if args.manifest.exists():
        print(json.dumps(reuse_manifest(args.manifest), indent=2), flush=True)
        return
    if args.command in ("split", "all"):
        print(json.dumps(prepare_split(args), indent=2), flush=True)
    if args.command in ("ocr", "all"):
        run_ocr(args)
    if args.command in ("freeze", "all"):
        print(json.dumps(freeze_manifest(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
