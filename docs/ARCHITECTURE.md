# Active Pipeline Architecture

The first snapshot separated current code from history. The subsequent
2026-10-14 refactor also separates responsibilities inside the current code.
The old 374-line `run_experiment.py` no longer exists in the active package.
The compact version keeps optimizer functions together with training, and
preflight/smoke checks together with the CLI. Separate `optimization.py` and
`checks.py` files have been removed rather than left as forwarding wrappers.
No facade re-exports the old runner's functions, and no active module edits
`sys.path` to import historical folders or sibling files.

## Module Ownership

| Module | Responsibility |
| --- | --- |
| `__main__.py`, `cli.py` | Command parsing, sequencing, offline preflight, explicit paid API smoke |
| `config.py` | Paths, model/dataset groups, stages, pinned settings, cache defaults |
| `core.py` | OCR signature, correction program, factory, text normalization |
| `data.py` | Frozen rows, scope-local examples, deterministic biased sampling |
| `contracts.py` | Split coverage/disjointness, train-demo validity, runtime metadata boundaries |
| `metrics.py` | Normalized EM, Boolean trace admission, factual textual feedback |
| `openai_lm.py`, `json_adapter.py` | SDK requests, JSON input/output transport, usage accounting |
| `reflection.py`, `prompts/` | ESPO-inspired diagnosis/revision proposal hook |
| `training.py` | Native GEPA/Bootstrap, candidate records, scope lifecycle, validation, completed/failed status |
| `evaluation.py` | Batched predictions, fix/harm accounting, frozen full test and tables |
| `artifacts.py` | Fingerprints, completed-job verification, frozen program manifest |
| `storage.py` | Atomic JSON writes, SHA-256, UTC timestamps |

`contracts.py` no longer acts as a barrel for data loaders, program factories,
or storage utilities. Callers import each function from its owner. Metrics do
not import data loading or experiment orchestration. Optimizers do not dispatch
CLI commands. Training and evaluation do not import the CLI. Program signatures
and prompts retain their original content.

## Execution Flow

```text
python -m pipeline
  cli.main
    prepare command -> ocr/prepare.py [active or optional OCR environment; no API]
      LMDB labels -> seed split -> selected-image OCR -> frozen_rows.csv.gz
    cli.preflight [training/evaluation commands]
    matching API-smoke fingerprint check
    training.train_scope (model -> group -> stage)
      data.load_scope_examples / deterministic sampling
      training.optimize_instruction           [GEPA stage]
      training.optimize_demonstrations        [Bootstrap stage]
      contracts.verify_program
      evaluation.evaluate                     [validation]
      artifacts + storage                     [persist completed program]
    artifacts.freeze_grid                     [all scopes complete]
    evaluation.run_test                       [frozen programs only]
```

There are still 24 scopes, 72 stage artifacts, and 90 test cells per stage.
The three stages remain initial prompt, GEPA, and GEPA+Bootstrap. Sampling,
seed order, candidate acceptance, validation selection, prompts, SDK options,
output checks, row order, and thread count are unchanged. The command changed
from a file path to `python -m pipeline` because imports are now package-relative.

Unused `group_for_dataset` and `scope_data_sha256` helpers have been removed.
Completed jobs are reused only when their experiment fingerprint and program
hash match. The obsolete `resume_plan` branch that could admit a different code
fingerprint is removed; split, output, demo, program-hash, and freeze checks remain.

## What Is Not Refactored

`archive/` remains 933 byte-identical original research files. It is a local
historical record, not a cleaned collection of 933 supported implementations.
No archived abandoned optimizer is imported by the active package. OCR
preparation is connected through the existing CLI and can use the active Python
environment or an optional separate OCR environment. It exports the same
manifest schema consumed by `data.py`.
OCR-specific image preprocessing is kept in the two backend scripts, not
mixed into GEPA/Bootstrap. Historical analysis remains a read-only utility.

Source fingerprints necessarily change when modules move. Old trained artifacts
are not copied into the new run directories or relabeled as compatible. No
claim is made that future API responses will be identical to historical ones.
See [Validation](VALIDATION.md) for the offline equivalence evidence.
