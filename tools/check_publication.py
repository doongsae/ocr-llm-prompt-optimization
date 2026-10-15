"""Offline inspection of publishable files or the actual Git index contents."""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".cache", ".pytest_cache",
                 "data", "datasets", "weights", "checkpoints", "artifacts", "status",
                 "manifests", "results", "logs", "archive"}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".p12", ".pt", ".pth", ".ckpt", ".safetensors",
                      ".bin", ".mdb", ".lmdb", ".zip", ".pdf", ".log"}
SECRET = re.compile(r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
MACHINE_PATH = re.compile(r"/(?:data\d*|home)/[A-Za-z0-9_.-]+/")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def files(staged):
    if staged:
        for raw in git("ls-files", "-z").split(b"\0"):
            if raw:
                name = raw.decode()
                yield name, lambda name=name: git("show", f":{name}")
    else:
        for parent, directories, names in os.walk(ROOT, followlinks=False):
            directories[:] = sorted(name for name in directories if name not in EXCLUDED_DIRS)
            for name in sorted(names):
                path = Path(parent) / name
                if path == ROOT / "config.json":
                    continue
                if path.name == "PUBLICATION_CHECK.json":
                    continue
                yield path.relative_to(ROOT).as_posix(), path.read_bytes


def inspect(name, content):
    path = Path(name)
    failures = []
    if name == "config.json" or path.suffix in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
        return ["forbidden secret/data/binary filename"]
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return ["generated/private directory is staged"]
    if (ROOT / path).is_symlink():
        return ["symlink is not permitted in publication snapshot"]
    data = content()
    if len(data) > 10 * 1024 * 1024:
        failures.append("file exceeds 10 MiB publication limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return failures + ["unexpected binary file"]
    if SECRET.search(text):
        failures.append("potential credential detected; value suppressed")
    if name == "config.example.json" and json.loads(text).get("openai_api_key") != "":
        failures.append("example config must contain an empty API key")
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            failures.append(f"syntax error at line {exc.lineno}")
    if MACHINE_PATH.search(text):
        failures.append("machine-specific source path")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    inspected = 0
    failures = []
    for name, read in files(args.staged):
        inspected += 1
        issues = inspect(name, read)
        if issues:
            failures.append({"path": name, "issues": issues})
    report = {"status": "failed" if failures else "passed", "files_checked": inspected,
              "git_index_checked": args.staged, "failures": failures,
              "limitations": "Pattern checks are not a guarantee against all secrets. License review remains required."}
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
