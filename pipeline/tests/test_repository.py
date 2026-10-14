"""Offline checks for the repository packaging utilities."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import check_publication as publication
import collect_archive as archive


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
                self.assertIsNotNone(archive.SECRET.search(payload.decode()))

    def test_word_containing_key_prefix_is_not_a_credential(self):
        payload = b"description = 'risk-lowercase-hyphenated-prose-example'\n"
        self.assertEqual(publication.inspect("example.py", lambda: payload), [])
        self.assertIsNone(archive.SECRET.search(payload.decode()))

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


class SourceArchiveTests(unittest.TestCase):
    def test_source_allowlist_excludes_private_generated_and_vendor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("run.py", "config.yaml", "prompts/diagnose.txt", "requirements.txt",
                     ".openai_key.key", "weights.pth", "predictions.csv", "data/private.py",
                     "results/generated.py", "parseq/vendor.py")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            (root / "linked.py").symlink_to(root / "run.py")
            selected = {path.relative_to(root).as_posix() for path in archive.source_files(root)}
            self.assertEqual(selected, {"run.py", "config.yaml", "prompts/diagnose.txt", "requirements.txt"})

    def test_inventory_digest_matches_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.py"
            content = b"print('fixture')\r\n"
            path.write_bytes(content)
            self.assertEqual(archive.sha256(path), hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
