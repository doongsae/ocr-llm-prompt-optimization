"""Copy research source files only; never follow symlinks or copy experiment data."""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git", ".cache", ".claude", ".codex", "__pycache__", "node_modules",
    "parseq", "OpenOCR", "data", "datasets", "test", "evaluation", "u14m",
    "OST", "wordart_test", "benchmark_bctr", "ltb", "eval_results", "results",
    "logs", "status", "manifests", "artifacts", "attempts", "provenance",
    "checkpoints", "weights", ".venv", "venv",
}
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".ini"}
SECRET = re.compile(r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(source):
    for parent, directories, files in os.walk(source, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if name not in EXCLUDED and not name.startswith(".")
                                and not (Path(parent) / name).is_symlink())
        for name in sorted(files):
            path = Path(parent) / name
            if path.is_symlink() or name.startswith("."):
                continue
            if path.suffix in SUFFIXES or (path.suffix == ".txt" and
                    (name.startswith("requirements") or path.parent.name == "prompts")):
                yield path
    exporter = source / "OpenOCR/tools/eval_rec_all_en_with_preds.py"
    if exporter.is_file() and not exporter.is_symlink():
        yield exporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Otherwise inventory only")
    args = parser.parse_args()
    source = args.source.resolve()
    destination = HERE / "archive" / "original_sources"
    if not source.is_dir() or HERE == source or HERE.is_relative_to(source):
        parser.error("source must be an existing, separate research directory")
    paths = list(source_files(source))
    unsafe = [str(path.relative_to(source)) for path in paths
              if SECRET.search(path.read_text(encoding="utf-8", errors="replace"))]
    if unsafe:
        raise RuntimeError("Potential credentials found; no files copied: " + ", ".join(unsafe))
    records = []
    for path in paths:
        relative = path.relative_to(source)
        target = destination / relative
        digest = sha256(path)
        if args.copy:
            if target.exists() and sha256(target) != digest:
                raise FileExistsError(f"refusing to overwrite changed archive file: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            if sha256(target) != digest:
                raise RuntimeError(f"copy hash mismatch: {relative}")
        records.append({"source": relative.as_posix(),
                        "archived_path": target.relative_to(HERE).as_posix(),
                        "sha256": digest, "bytes": path.stat().st_size})
    result = {"copied": args.copy, "source_files": len(records),
              "total_bytes": sum(item["bytes"] for item in records), "files": records}
    if args.copy:
        (HERE / "docs/source_inventory.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
