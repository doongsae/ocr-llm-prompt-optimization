"""Publication checks, documentation links and reference tables."""

import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

from analysis import analyze_errors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import check_publication as publication


class PublicationTests(unittest.TestCase):
    def test_config_template_is_allowed_only_with_an_empty_key(self):
        self.assertEqual(publication.inspect("config.example.json", lambda: b'{"openai_api_key": ""}'), [])
        self.assertIn("example config must contain an empty API key",
                      publication.inspect("config.example.json", lambda: b'{"openai_api_key": "not-empty"}'))

    def test_private_names_are_rejected_without_reading_contents(self):
        for name in (".openai_key.key", ".env", ".env.example", ".env.local", "weights.pth", "config.json"):
            with self.subTest(name=name):
                reader = Mock()
                self.assertTrue(publication.inspect(name, reader))
                reader.assert_not_called()

    def test_worktree_check_skips_local_config_without_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("not publication data", encoding="utf-8")
            (root / "config.example.json").write_text('{"openai_api_key": ""}', encoding="utf-8")
            with patch.object(publication, "ROOT", root):
                self.assertEqual([name for name, _ in publication.files(False)], ["config.example.json"])

    def test_generated_staged_file_is_rejected(self):
        reader = Mock()
        self.assertTrue(publication.inspect("pipeline/results/predictions.csv", reader))
        reader.assert_not_called()

    def test_synthetic_credentials_are_detected_without_exposing_values(self):
        for token in ("sk-" + "A" * 48, "sk-" + "proj-" + "A" * 48,
                      "ghp_" + "A" * 40, "AKIA" + "A" * 16):
            with self.subTest(prefix=token[:3]):
                payload = f'key = "{token}"\n'.encode()
                issues = publication.inspect("example.py", lambda: payload)
                self.assertIn("potential credential detected; value suppressed", issues)
                self.assertNotIn(token, str(issues))

    def test_word_containing_key_prefix_is_not_a_credential(self):
        payload = b"description = 'risk-lowercase-hyphenated-prose-example'\n"
        self.assertEqual(publication.inspect("example.py", lambda: payload), [])

    def test_symlink_is_rejected_without_reading_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.txt").write_text("fixture", encoding="utf-8")
            (root / "link.txt").symlink_to(root / "target.txt")
            reader = Mock()
            with patch.object(publication, "ROOT", root):
                self.assertTrue(publication.inspect("link.txt", reader))
            reader.assert_not_called()

    def test_binary_and_invalid_python_are_rejected(self):
        self.assertEqual(publication.inspect("example.txt", lambda: b"\xff"),
                         ["unexpected binary file"])
        self.assertEqual(publication.inspect("example.py", lambda: b"def broken(:\n"),
                         ["syntax error at line 1"])

    def test_local_archives_are_not_publishable_and_machine_paths_are_rejected(self):
        reader = Mock()
        self.assertTrue(publication.inspect("archive/example.py", reader))
        reader.assert_not_called()
        payload = b'root = "/home/' + b'fixture/project"\n'
        self.assertIn("machine-specific source path",
                      publication.inspect("example.py", lambda: payload))

    def test_large_file_is_rejected(self):
        self.assertIn("file exceeds 10 MiB publication limit",
                      publication.inspect("example.txt", lambda: b"x" * (10 * 1024 * 1024 + 1)))

    def test_index_inspection_reads_index_not_worktree(self):
        def fake_git(*arguments):
            if arguments == ("ls-files", "-z"):
                return b"README.md\0"
            self.assertEqual(arguments, ("show", ":README.md"))
            return b"staged content"

        with patch.object(publication, "git", side_effect=fake_git):
            entries = list(publication.files(staged=True))
            self.assertEqual([name for name, _ in entries], ["README.md"])
            self.assertEqual(entries[0][1](), b"staged content")


class DocumentationTests(unittest.TestCase):
    def test_local_documentation_links_resolve(self):
        paths = [ROOT / "README.md", ROOT / "ocr/README.md", ROOT / "reference_results/README.md",
                 *sorted((ROOT / "docs").glob("*.md"))]
        for path in paths:
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
                link = urlsplit(target)
                if not link.scheme and link.path:
                    with self.subTest(document=path.name, target=target):
                        self.assertTrue((path.parent / unquote(link.path)).is_file())

    def test_reference_distributions_agree(self):
        fields = ("rows", "raw_wrong", "error_fixed", "correct_harmed",
                  "wrong_unchanged_literal", "wrong_changed_literal")
        expected = dict(zip(fields, (26826, 4335, 921, 279, 2719, 695)))
        for path in sorted((ROOT / "reference_results").glob("by_*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with self.subTest(table=path.name):
                self.assertEqual({field: sum(int(row[field]) for row in rows) for field in fields}, expected)


class ErrorAnalysisTests(unittest.TestCase):
    def make_results(self, directory, all_correct=False):
        source = Path(directory) / "source"
        source.mkdir()
        cases = [("OPEN", "OPEN", "OPEN"), ("SPED", "SPEED", "SPEED"),
                 ("OPEN", "OPEN", "OPENS"), ("SPED", "SPEED", "SPED")]
        predictions, cells = [], []
        for model in analyze_errors.MODELS:
            for dataset in analyze_errors.DATASETS:
                raw, gt, corrected = cases[0 if all_correct else len(predictions) % len(cases)]
                before, after = int(raw == gt), int(corrected == gt)
                predictions.append({"model": model, "dataset": dataset, "image_idx": "0",
                                    "ocr_pred": raw, "gt": gt, "llm_corrected": corrected,
                                    "baseline_correct": before, "candidate_correct": after,
                                    "dist_before": 1 - before, "dist_after": 1 - after})
                cells.append({"method": "bootstrap", "model": model, "dataset": dataset, "rows": 1,
                              "baseline_correct": before, "candidate_correct": after,
                              "fixes": int(not before and after), "harms": int(before and not after)})
        fields = ("rows", "baseline_correct", "candidate_correct", "fixes", "harms")
        summary = {field: sum(row[field] for row in cells) for field in fields}
        summary.update(wins=sum(row["candidate_correct"] > row["baseline_correct"] for row in cells),
                       ties=sum(row["candidate_correct"] == row["baseline_correct"] for row in cells),
                       losses=sum(row["candidate_correct"] < row["baseline_correct"] for row in cells))
        analyze_errors.write_csv(source / "bootstrap_predictions.csv", predictions)
        analyze_errors.write_csv(source / "main_table.csv", cells)
        (source / "summary.json").write_text(json.dumps({"methods": {"bootstrap": summary}}))
        return source, predictions, cells, summary

    def run_analysis(self, source):
        return subprocess.run([sys.executable, "-O", str(ROOT / "analysis/analyze_errors.py"),
                               "--source-dir", str(source), "--output-dir", str(source.parent / "output")],
                              capture_output=True, text=True, timeout=30)

    def test_analysis_accepts_new_sample_counts_and_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, predictions, _, summary = self.make_results(directory)
            result = self.run_analysis(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
        self.assertEqual(report["overall"]["rows"], len(predictions))
        self.assertEqual(report["overall"]["error_fixed"], summary["fixes"])
        self.assertEqual(report["overall"]["correct_harmed"], summary["harms"])
        for field in ("wins", "ties", "losses"):
            self.assertEqual(report["overall"][field], summary[field])

    def test_analysis_handles_zero_raw_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, _ = self.make_results(directory, all_correct=True)
            result = self.run_analysis(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
        self.assertEqual(report["overall"]["raw_wrong"], 0)
        self.assertEqual(report["overall"]["raw_error_share_pct"], 0.0)
        self.assertIsNone(report["overall"]["error_fix_rate_pct"])

    def test_analysis_rejects_inconsistent_inputs_even_with_optimized_python(self):
        for corruption in ("rows", "wins", "duplicate_cell", "duplicate_prediction",
                           "missing_cell", "missing_prediction", "correctness"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                source, predictions, cells, summary = self.make_results(directory)
                if corruption in ("rows", "wins"):
                    summary[corruption] += 1
                    (source / "summary.json").write_text(json.dumps({"methods": {"bootstrap": summary}}))
                elif corruption.endswith("cell"):
                    cells = cells + [cells[0]] if corruption == "duplicate_cell" else cells[1:]
                    analyze_errors.write_csv(source / "main_table.csv", cells)
                else:
                    if corruption == "duplicate_prediction":
                        predictions.append(predictions[0])
                    elif corruption == "missing_prediction":
                        predictions.pop()
                    else:
                        predictions[0]["candidate_correct"] = 0
                    analyze_errors.write_csv(source / "bootstrap_predictions.csv", predictions)
                result = self.run_analysis(source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ValueError", result.stderr)
                self.assertFalse((source.parent / "output/classified_test.csv").exists())


if __name__ == "__main__":
    unittest.main()
