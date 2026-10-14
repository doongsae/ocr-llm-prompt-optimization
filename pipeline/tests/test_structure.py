"""Module boundaries and CLI sequencing without data or API inference."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import cli, config


class StructureTests(unittest.TestCase):
    def test_core_operations_have_one_owning_module(self):
        owners = {}
        for path in config.HERE.glob("*.py"):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.FunctionDef):
                    owners.setdefault(node.name, []).append(path.stem)
        expected = {"fresh_program": "core", "filter_text": "core", "facts": "metrics",
                    "load_scope_examples": "data", "write_json": "storage", "sha256_path": "storage",
                    "optimize_instruction": "training", "optimize_demonstrations": "training",
                    "train_scope": "training", "evaluate": "evaluation", "run_test": "evaluation",
                    "fingerprint": "artifacts", "verify_job": "artifacts", "main": "cli"}
        for function, module in expected.items():
            with self.subTest(function=function):
                self.assertEqual(owners.get(function), [module])

    def test_package_uses_relative_imports_without_path_mutation(self):
        local_modules = {path.stem for path in config.HERE.glob("*.py")}
        for path in config.HERE.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                with self.subTest(module=path.name, line=getattr(node, "lineno", 0)):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        self.assertNotIn(node.module.split(".")[0], local_modules)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                        self.assertNotEqual((node.value.id, node.attr), ("sys", "path"))
        self.assertFalse((config.HERE / "run_experiment.py").exists())
        self.assertFalse((config.HERE / "data_contract.py").exists())
        self.assertFalse((config.HERE / "frozen_data.py").exists())
        self.assertFalse((config.HERE / "optimization.py").exists())
        self.assertFalse((config.HERE / "checks.py").exists())

    def test_package_help_works_without_private_data_or_credentials(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OCR_DATA_MANIFEST"}}
        result = subprocess.run([sys.executable, "-m", "pipeline", "--help"], cwd=config.ROOT,
                                env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight,api-smoke,train,test,all", result.stdout)


class CLISequenceTests(unittest.TestCase):
    def run_cli(self, directory, arguments, calls, smoke_fingerprint="fixture"):
        smoke = Path(directory) / "status" / "api_smoke.json"
        smoke.parent.mkdir(parents=True)
        smoke.write_text(json.dumps({"experiment_sha256": smoke_fingerprint}), encoding="utf-8")
        with patch.object(config, "HERE", Path(directory)), \
                patch.object(sys, "argv", ["pipeline", *arguments]), \
                patch.object(cli, "preflight", return_value={"status": "passed"}), \
                patch.object(cli, "require_key"), \
                patch.object(cli, "fingerprint", return_value={"experiment_sha256": "fixture"}), \
                patch.object(cli, "train_scope", side_effect=lambda *job: calls.append(job)), \
                patch.object(cli, "freeze_grid", side_effect=lambda: calls.append("freeze")), \
                patch.object(cli, "run_test", side_effect=lambda: calls.append("test")):
            cli.main()

    def test_all_trains_24_scopes_in_order_before_freeze_and_test(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            self.run_cli(directory, ["all"], calls)
        expected = [(model, group, stage) for model in config.MODELS
                    for group in config.GROUPS for stage in config.STAGES]
        self.assertEqual(len(expected), 72)
        self.assertEqual(calls, expected + ["freeze", "test"])

    def test_scope_filter_trains_three_stages_without_full_grid_freeze(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            self.run_cli(directory, ["train", "--model", "trba", "--group", "common"], calls)
        self.assertEqual(calls, [("trba", "common", stage) for stage in config.STAGES])

    def test_stale_smoke_stops_before_training(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            self.run_cli(directory, ["all"], calls, smoke_fingerprint="incompatible")
        self.assertEqual(calls, [])

    def test_preflight_does_not_require_api_key_or_start_training(self):
        with patch.object(sys, "argv", ["pipeline", "preflight"]), \
                patch.object(cli, "preflight", return_value={"status": "passed"}), \
                patch.object(cli, "require_key") as key, \
                patch.object(cli, "train_scope") as train, \
                patch("builtins.print"):
            cli.main()
        key.assert_not_called()
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
