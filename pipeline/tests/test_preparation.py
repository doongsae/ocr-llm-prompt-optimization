"""Seed selection, selected-image export, and the pipeline handoff without API calls."""

import importlib.util
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ocr import prepare
from pipeline import cli, config, contracts, data, openai_lm


def catalogue(size=30):
    return {ds: [{"dataset": ds, "image_idx": i, "gt": f"WORD{i}"} for i in range(size)]
            for ds in prepare.DATASET_PATHS}


def selected_rows():
    return prepare.select_splits(catalogue(), 42, 28, 28, 3, 3, 3)


class SplitTests(unittest.TestCase):
    def test_same_seed_same_selection_and_different_seed_changes_it(self):
        args = (catalogue(), 42, 28, 28, 3, 3, 3)
        first = prepare.select_splits(*args)
        self.assertEqual(first, prepare.select_splits(*args))
        self.assertNotEqual(first, prepare.select_splits(catalogue(), 43, 28, 28, 3, 3, 3))

    def test_all_datasets_covered_and_splits_disjoint(self):
        rows = selected_rows()
        self.assertEqual({row["dataset"] for row in rows}, set(prepare.DATASET_PATHS))
        keys = {(row["dataset"], row["image_idx"]) for row in rows}
        self.assertEqual(len(keys), len(rows))
        for ds in prepare.DATASET_PATHS:
            self.assertEqual({row["split"] for row in rows if row["dataset"] == ds}, set(prepare.SPLITS))

    def test_proportional_allocation_and_general_are_not_error_conditioned(self):
        pool = {ds: rows for ds, rows in catalogue().items() if ds in {"SVT", "IIIT5k", "General"}}
        pool["IIIT5k"] += [{"dataset": "IIIT5k", "image_idx": i, "gt": "X"} for i in range(30, 90)]
        rows = prepare.select_splits(pool, 42, 12, 12, 3, 5, 6)
        counts = {ds: sum(row["split"] == "train" and row["dataset"] == ds for row in rows) for ds in pool}
        self.assertEqual(counts, {"IIIT5k": 9, "SVT": 3, "General": 5})
        self.assertTrue(all("pred" not in row and "correct" not in row for row in rows))

    def test_small_test_pool_is_used_without_overlap(self):
        pool = {"SVT": catalogue(8)["SVT"]}
        rows = prepare.select_splits(pool, 42, 2, 2, 300, 3, 3)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 4)

    def test_insufficient_dataset_fails(self):
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            prepare.select_splits({"SVT": catalogue(4)["SVT"]}, 42, 2, 2, 3, 3, 3)

    def test_selection_reader_rejects_cross_split_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.csv"
            rows = [{"split": split, "dataset": "SVT", "image_idx": 0, "gt": "X"} for split in ("train", "test")]
            prepare.write_csv(path, rows, prepare.SELECT_COLUMNS)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                prepare.read_selection(path)


class ManifestTests(unittest.TestCase):
    def populate(self, directory):
        output = Path(directory)
        selected = selected_rows()
        prepare.write_csv(output / "selection.csv", selected, prepare.SELECT_COLUMNS)
        for model in prepare.MODELS:
            rows = [prepare.prediction_row(model, sample, "NaN" if i == 0 else sample["gt"])
                    for i, sample in enumerate(selected)]
            prepare.save_predictions(output, model, rows, selected, {"fixture": True}, time.monotonic())
        return Namespace(output_dir=output, manifest=output / "frozen_rows.csv.gz", models=list(prepare.MODELS)), selected

    def test_all_six_exports_freeze_and_pass_existing_pipeline_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args, selected = self.populate(directory)
            report = prepare.freeze_manifest(args)
            frame = pd.read_csv(args.manifest, keep_default_na=False,
                                dtype={"image_idx": str})
            self.assertEqual(report["rows"], len(selected) * 6)
            self.assertIn("NaN", set(frame.pred))
            with patch.object(data, "_cached_data_manifest", return_value=frame), \
                    patch.object(contracts, "DATA_MANIFEST", args.manifest):
                audit = contracts.audit_data()
                self.assertEqual(audit["training_scopes"], 24)
                self.assertEqual(audit["test_cells"], 90)
                self.assertEqual(len(data.load_test_frame()), 15 * 3 * 6)

    def test_stale_or_modified_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            path = args.output_dir / "predictions" / "crnn.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "stale/modified"):
                prepare.freeze_manifest(args)
            self.assertFalse(args.manifest.exists())

    def test_missing_prediction_and_wrong_gt_are_rejected(self):
        selected = selected_rows()
        rows = [prepare.prediction_row("crnn", sample, sample["gt"]) for sample in selected]
        with self.assertRaisesRegex(ValueError, "incomplete OCR"):
            prepare.validate_predictions(rows[:-1], selected, "crnn")
        rows[0]["gt"] = "CHANGED"
        with self.assertRaisesRegex(ValueError, "GT alignment"):
            prepare.validate_predictions(rows, selected, "crnn")

    def test_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            prepare.freeze_manifest(args)
            digest = prepare.file_hash(args.manifest)
            self.assertEqual(prepare.freeze_manifest(args)["status"], "reused")
            self.assertEqual(prepare.file_hash(args.manifest), digest)

    def test_existing_split_is_reused_without_label_scan_or_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            path = args.output_dir / "selection.csv"
            before = (prepare.file_hash(path), path.stat().st_mtime_ns)
            with patch.object(prepare, "read_catalogue") as scan:
                self.assertEqual(prepare.prepare_split(args)["status"], "reused")
            scan.assert_not_called()
            self.assertEqual(before, (prepare.file_hash(path), path.stat().st_mtime_ns))

    def test_complete_model_exports_are_reused_without_loading_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            before = {p: (prepare.file_hash(p), p.stat().st_mtime_ns) for p in (args.output_dir / "predictions").iterdir()}
            with patch.object(prepare.subprocess, "run") as inference:
                prepare.run_ocr(args)
            inference.assert_not_called()
            self.assertEqual(before, {p: (prepare.file_hash(p), p.stat().st_mtime_ns) for p in before})

    def test_only_missing_model_is_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            (args.output_dir / "predictions" / "crnn.csv").unlink()
            args.data_root = args.output_dir
            args.device = "cpu"
            args.batch_size = 2
            args.parseq_root = None
            with patch.object(prepare.subprocess, "run") as inference:
                prepare.run_ocr(args)
            inference.assert_called_once()
            command = inference.call_args.args[0]
            self.assertEqual(command[command.index("--model") + 1], "crnn")

    def test_valid_existing_csv_without_metadata_is_reused_without_fake_weight_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            (args.output_dir / "predictions" / "crnn.json").unlink()
            with patch.object(prepare.subprocess, "run") as inference:
                prepare.run_ocr(args)
            inference.assert_not_called()
            report = prepare.freeze_manifest(args)
            self.assertEqual(report["ocr_provenance_by_model"]["crnn"]["checkpoint"], "not_recorded")

    def test_all_command_reuses_manifest_before_any_split_or_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            prepare.freeze_manifest(args)
            before = (prepare.file_hash(args.manifest), args.manifest.stat().st_mtime_ns)
            arguments = ["prepare", "all", "--data-root", "/tmp/nonexistent_benchmark", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), patch.object(prepare, "prepare_split") as split, \
                    patch.object(prepare, "run_ocr") as inference, patch("builtins.print"):
                prepare.main()
            split.assert_not_called()
            inference.assert_not_called()
            self.assertEqual(before, (prepare.file_hash(args.manifest), args.manifest.stat().st_mtime_ns))

    def test_cross_model_split_misalignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            prepare.freeze_manifest(args)
            frame = pd.read_csv(args.manifest, keep_default_na=False, dtype={"image_idx": str})
            row = frame.index[frame.model == "svtrv2"][0]
            frame.loc[row, "split"] = "test"
            with patch.object(data, "_cached_data_manifest", return_value=frame), \
                    self.assertRaisesRegex(ValueError, "alignment"):
                contracts.audit_data()

    def test_model_local_historical_train_keys_remain_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.populate(directory)
            prepare.freeze_manifest(args)
            frame = pd.read_csv(args.manifest, keep_default_na=False, dtype={"image_idx": str})
            row = frame.index[(frame.model == "svtrv2") & (frame.dataset == "General") & (frame.split == "train")][0]
            frame.loc[row, "image_idx"] = "100000"
            with patch.object(data, "_cached_data_manifest", return_value=frame), \
                    patch.object(contracts, "DATA_MANIFEST", args.manifest):
                self.assertEqual(contracts.audit_data()["training_scopes"], 24)


@unittest.skipUnless(importlib.util.find_spec("lmdb"), "OCR LMDB dependency is not installed")
class LMDBTests(unittest.TestCase):
    def test_label_scan_filters_gt_and_never_requires_full_image_inference(self):
        import lmdb

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / prepare.DATASET_PATHS["SVT"]
            path.mkdir(parents=True)
            env = lmdb.open(str(path), map_size=1024 * 1024)
            with env.begin(write=True) as txn:
                labels = ["GOOD", "!!!", "A" * 26, "NaN", "", "0"]
                txn.put(b"num-samples", str(len(labels)).encode())
                for idx, label in enumerate(labels, 1):
                    txn.put(f"label-{idx:09d}".encode(), label.encode())
            env.close()
            pool, counts = prepare.read_catalogue(directory, ["SVT"], 25)
            self.assertEqual([row["image_idx"] for row in pool["SVT"]], [0, 3, 5])
            self.assertEqual(counts["SVT"], {"source_rows": 6, "eligible_rows": 3, "unsupported_svtr_labels": 0})

    def test_joint_support_rule_excludes_long_punctuation_without_fixed_ids(self):
        self.assertTrue(prepare.svtr_label_supported("HELLO WORLD", 25))
        self.assertFalse(prepare.svtr_label_supported("WORD" + "." * 24, 25))


class PreparationCLITests(unittest.TestCase):
    def test_pipeline_prepare_needs_no_model_or_python_paths(self):
        with patch.object(sys, "argv", ["pipeline", "prepare"]), \
                patch.object(cli.subprocess, "run") as run:
            cli.main()
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--data-root") + 1], str(config.DATA_ROOT))
        self.assertNotIn("--openocr-root", command)
        self.assertNotIn("--svtr-config", command)
        self.assertNotIn("--svtr-checkpoint", command)

    def test_pipeline_prepare_forwards_parameters_without_api_or_training(self):
        arguments = ["pipeline", "prepare", "--data-root", "/tmp/benchmark", "--seed", "51",
                     "--ocr-python", "ocr-python", "--step", "split"]
        with patch.object(sys, "argv", arguments), patch.object(cli.subprocess, "run") as run, \
                patch.object(cli, "preflight") as preflight, patch.object(cli, "require_key") as key:
            cli.main()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["ocr-python", str(config.ROOT / "ocr" / "prepare.py"), "split"])
        self.assertEqual(command[command.index("--seed") + 1], "51")
        self.assertEqual(command[command.index("--manifest") + 1], str(config.DATA_MANIFEST))
        preflight.assert_not_called()
        key.assert_not_called()

    def test_pipeline_prepare_seed_defaults_to_config(self):
        with patch.object(sys, "argv", ["pipeline", "prepare", "--data-root", "/tmp/benchmark"]), \
                patch.object(config, "SEED", 73), patch.object(cli.subprocess, "run") as run:
            cli.main()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--seed") + 1], "73")

    def test_ocr_starts_each_model_in_a_separate_sequential_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare.write_csv(root / "selection.csv", selected_rows(), prepare.SELECT_COLUMNS)
            args = Namespace(output_dir=root, models=["crnn", "svtrv2"], data_root=root,
                             device="cpu", batch_size=2, openocr_root=root,
                             svtr_config=root / "cfg.yml", svtr_checkpoint=root / "best.pth", parseq_root=None)
            (root / "openrec").mkdir()
            args.svtr_config.touch()
            args.svtr_checkpoint.touch()
            with patch.object(prepare.subprocess, "run") as run:
                prepare.run_ocr(args)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertTrue(commands[0][1].endswith("eval_parseq_selected.py"))
            self.assertTrue(commands[1][1].endswith("eval_svtrv2.py"))


class ConfigTests(unittest.TestCase):
    def test_user_config_sets_seed_and_both_optimizer_ratios(self):
        values = {"seed": 73, "gepa_error_ratio": 0.8, "bootstrap_error_ratio": 0.5}
        spec = importlib.util.spec_from_file_location("config_fixture", config.HERE / "config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "read_text", return_value=json.dumps(values)):
            spec.loader.exec_module(module)
        self.assertEqual(module.SEED, 73)
        self.assertEqual(module.SETTINGS["gepa_seed"], 73)
        self.assertEqual(module.SETTINGS["bootstrap_seed"], 74)
        self.assertEqual(module.SETTINGS["gepa_error_ratio"], 0.8)
        self.assertEqual(module.SETTINGS["bootstrap_error_ratio"], 0.5)

    def test_config_template_is_used_without_a_local_file(self):
        spec = importlib.util.spec_from_file_location("config_template_fixture", config.HERE / "config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "is_file", return_value=False):
            spec.loader.exec_module(module)
        self.assertEqual(module.USER_CONFIG["openai_api_key"], "")
        self.assertEqual(module.DATA_ROOT, config.ROOT / "datasets")

    def test_config_key_and_relative_paths_are_loaded_without_environment_changes(self):
        values = {"seed": 42, "gepa_error_ratio": 0.75, "bootstrap_error_ratio": 0.6,
                  "openai_api_key": " config-fixture ", "data_root": "sample-data",
                  "data_manifest": "sample-output/rows.csv.gz"}
        spec = importlib.util.spec_from_file_location("config_paths_fixture", config.HERE / "config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "read_text", return_value=json.dumps(values)), patch.dict(os.environ, {}, clear=True):
            spec.loader.exec_module(module)
            self.assertEqual(module.openai_api_key(), "config-fixture")
            self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertEqual(module.DATA_ROOT, config.ROOT / "sample-data")
        self.assertEqual(module.DATA_MANIFEST, config.ROOT / "sample-output/rows.csv.gz")
        self.assertNotIn("openai_api_key", module.SETTINGS)

    def test_key_environment_fallback_and_manifest_override_remain_supported(self):
        with patch.dict(config.USER_CONFIG, {"openai_api_key": ""}), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "environment-fixture"}):
            self.assertEqual(config.openai_api_key(), "environment-fixture")
        with patch.dict(config.USER_CONFIG, {"openai_api_key": "configured-fixture"}), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "environment-fixture"}):
            self.assertEqual(config.openai_api_key(), "configured-fixture")
        spec = importlib.util.spec_from_file_location("config_environment_fixture", config.HERE / "config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"OCR_DATA_MANIFEST": "/tmp/override-fixture.csv.gz"}):
            spec.loader.exec_module(module)
        self.assertEqual(module.DATA_MANIFEST, Path("/tmp/override-fixture.csv.gz"))

    def test_sdk_receives_configured_key_without_an_api_request(self):
        openai_lm.sdk_client.cache_clear()
        try:
            with patch.dict(config.USER_CONFIG, {"openai_api_key": "configured-fixture"}), \
                    patch.dict(os.environ, {}, clear=True), patch.object(openai_lm, "OpenAI") as client:
                openai_lm.sdk_client()
                self.assertEqual(client.call_args.kwargs["api_key"], "configured-fixture")
                client.return_value.chat.completions.create.assert_not_called()
        finally:
            openai_lm.sdk_client.cache_clear()

    def test_invalid_seed_or_ratio_is_rejected_before_any_work(self):
        valid = {"seed": 42, "gepa_error_ratio": 0.75, "bootstrap_error_ratio": 0.6}
        for key, value in (("seed", -1), ("seed", True), ("gepa_error_ratio", 75),
                           ("gepa_error_ratio", 0), ("bootstrap_error_ratio", 1.1),
                           ("openai_api_key", None), ("data_root", ""), ("data_manifest", [])):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**valid, key: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    config.load_config(path)


class OCRAssetTests(unittest.TestCase):
    def test_valid_download_is_atomically_cached_and_reused_offline(self):
        payload = b"official-asset-fixture"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.zip"
            with patch.object(prepare, "urlopen", return_value=io.BytesIO(payload)) as download:
                self.assertEqual(prepare.download_asset("https://example.invalid/asset", path, digest), path)
            download.assert_called_once()
            before = path.stat().st_mtime_ns
            with patch.object(prepare, "urlopen") as download:
                prepare.download_asset("https://example.invalid/asset", path, digest)
            download.assert_not_called()
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_bad_or_partial_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.zip"
            with patch.object(prepare, "urlopen", return_value=io.BytesIO(b"wrong")), \
                    self.assertRaisesRegex(ValueError, "downloaded OCR asset hash mismatch"):
                prepare.download_asset("https://example.invalid/asset", path, "expected")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_corrupt_cache_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.zip"
            path.write_bytes(b"changed")
            with patch.object(prepare, "urlopen") as download, \
                    self.assertRaisesRegex(ValueError, "cached OCR asset hash mismatch"):
                prepare.download_asset("https://example.invalid/asset", path, "expected")
            download.assert_not_called()

    def test_default_svtr_assets_are_resolved_without_user_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / ".cache" / "ocr"
            source = cache / f"OpenOCR-{prepare.OPENOCR_COMMIT}"
            archive = root / "upstream.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(f"{source.name}/openrec/__init__.py", "")
                handle.writestr(f"{source.name}/{prepare.SVTR_CONFIG_PATH}", "official fixture")
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint fixture")
            cache.mkdir(parents=True)
            args = Namespace()
            with patch.object(prepare, "ROOT", root), \
                    patch.object(prepare, "download_asset", side_effect=[archive, checkpoint]) as download:
                prepare.resolve_svtr_assets(args)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(args.openocr_root, source)
            self.assertEqual(args.svtr_config, source / prepare.SVTR_CONFIG_PATH)
            self.assertEqual(args.svtr_checkpoint, checkpoint)
            self.assertTrue(args.svtr_config.is_file())

    def test_explicit_local_assets_never_download_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openrec").mkdir()
            config_file, checkpoint = root / "inference.yml", root / "best.pth"
            config_file.write_text("custom fixture", encoding="utf-8")
            checkpoint.write_bytes(b"custom fixture")
            args = Namespace(openocr_root=root, svtr_config=config_file, svtr_checkpoint=checkpoint)
            with patch.object(prepare, "download_asset") as download:
                prepare.resolve_svtr_assets(args)
            download.assert_not_called()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("lmdb", "torch", "torchvision", "PIL")),
                     "OCR dependencies are not installed")
class SelectedOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib

        # Keep native operator registration outside temporary sys.modules fixtures.
        importlib.import_module("torchvision")

    def load_svtr_backend(self):
        with patch.dict(sys.modules, {"prepare": prepare}):
            spec = importlib.util.spec_from_file_location("svtr_batch_fixture", config.ROOT / "ocr/eval_svtrv2.py")
            backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend)
        return backend

    def test_svtr_batches_cover_every_sample_including_exact_multiples(self):
        import numpy as np

        backend = self.load_svtr_backend()
        ratios = np.array([1] * 8 + [2] * 4 + [6] * 3 + [12] * 2)
        dataset = SimpleNamespace(wh_ratio=ratios, wh_ratio_sort=np.argsort(ratios))
        for size in (1, 2, 4, 8, 128):
            with self.subTest(batch_size=size):
                batches = list(backend.evaluation_batches(dataset, {"scales": [[128, 32]]}, size))
                indices = [item[2] for batch in batches for item in batch]
                self.assertEqual(sorted(indices), list(range(len(ratios))))
                self.assertTrue(all(0 < len(batch) <= size for batch in batches))

    def test_svtr_batches_preserve_official_ratio_shapes_and_adaptive_sizes(self):
        import numpy as np

        backend = self.load_svtr_backend()
        dataset = SimpleNamespace(wh_ratio=np.array([1] * 4 + [6] * 3 + [12] * 2),
                                  wh_ratio_sort=np.arange(9))
        batches = list(backend.evaluation_batches(dataset, {"scales": [[128, 32]]}, 4))
        self.assertEqual([[item[2] for item in batch] for batch in batches],
                         [[0, 1, 2, 3], [4, 5], [6], [7], [8]])
        self.assertEqual([(batch[0][0], batch[0][1]) for batch in batches],
                         [(32, 32), (192, 32), (192, 32), (384, 32), (384, 32)])

    def test_parseq_reads_only_selected_images_and_preserves_true_ids(self):
        import lmdb
        import torch
        from PIL import Image

        with patch.dict(sys.modules, {"prepare": prepare}):
            spec = importlib.util.spec_from_file_location("selected_export_fixture",
                                                        config.ROOT / "ocr" / "eval_parseq_selected.py")
            backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / prepare.DATASET_PATHS["SVT"]
            path.mkdir(parents=True)
            selected = [{"split": split, "dataset": "SVT", "image_idx": idx, "gt": f"WORD{idx}"}
                        for split, idx in (("train", 5), ("test", 1))]
            image = io.BytesIO()
            Image.new("RGB", (80, 32), "white").save(image, format="PNG")
            env = lmdb.open(str(path), map_size=1024 * 1024)
            with env.begin(write=True) as txn:
                txn.put(b"num-samples", b"10")
                for sample in selected:
                    idx = sample["image_idx"] + 1
                    txn.put(f"image-{idx:09d}".encode(), image.getvalue())
                    txn.put(f"label-{idx:09d}".encode(), sample["gt"].encode())
            env.close()
            prepare.write_csv(root / "selection.csv", selected, prepare.SELECT_COLUMNS)

            class Model:
                hparams = {"img_size": [32, 128]}
                tokenizer = SimpleNamespace(decode=lambda logits: (["ANSWER"] * len(logits), None))

                def __call__(self, images):
                    return torch.zeros((len(images), 1))

            weights = SimpleNamespace(_WEIGHTS_URL={"crnn": "https://example.invalid/crnn.pt"})
            args = Namespace(model="crnn", data_root=root, output_dir=root, device="cpu", batch_size=2, parseq_root=None)
            with patch.object(backend, "load_model", return_value=Model()), \
                    patch.object(backend, "file_hash", return_value="fixture"), \
                    patch.dict(sys.modules, {"strhub.models.utils": weights}):
                backend.evaluate_model(args)
            rows = pd.read_csv(root / "predictions" / "crnn.csv", keep_default_na=False)
            self.assertEqual(set(rows.image_idx), {1, 5})
            self.assertEqual(set(rows.pred), {"ANSWER"})


if __name__ == "__main__":
    unittest.main()
