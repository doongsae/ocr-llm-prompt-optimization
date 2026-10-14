"""Publication checks, documentation links and reference tables."""

import csv
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

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


if __name__ == "__main__":
    unittest.main()
