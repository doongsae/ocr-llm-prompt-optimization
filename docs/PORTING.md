# Snapshot Scope and Porting

## Supported Code

The active `pipeline/` copies the completed Luna EM-only GEPA+Bootstrap code.
The serializer and reflection prompt text are copied unchanged. Metrics, SDK
transport, and reflection retain their original logic with package-relative
imports; the shared API-key guard moved into the SDK module. Settings, train sampling, candidate selection,
scope boundaries, runtime inputs, inference limits, and LLM/output failure handling are
preserved. No abandoned A/B, contrastive, delayed-pair, or TRAS optimizer is
made the active method by this reorganization.

The original loader depended on two historical modules outside its own folder.
`core.py` retains the original signature, corrector, normalization, seed
initialization, and program factory. `data.py` retains the original data-loading
and sampling functions while removing unused local-server settings and helpers.
`contracts.py` owns only data/program validation, rather than re-exporting
loaders, storage functions, and the factory from other modules.

`config.py` resolves the manifest through `OCR_DATA_MANIFEST`, local
`config.json`, or a repository-relative default. `artifacts.py` hashes the
current package modules instead of
historical module paths. The old runner is organized into command/check,
training/optimization, evaluation, and provenance modules. Execution is now
`python -m pipeline`, with ordinary relative imports and no runtime path edits.
The full-manifest integration test skips
when private data is unavailable. Analysis and OCR-export paths are explicit
CLI arguments. These are portability changes, not a claim that API generations
will be bit-identical across separate calls or future provider model revisions.

The final simplification merges `optimization.py` into `training.py` and
`checks.py` into `cli.py`, preserving their function bodies. Uncalled dataset
group and scope-hash helpers are removed. The old cross-fingerprint
`resume_plan` compatibility branch is deliberately removed: same-fingerprint
completed jobs remain reusable, but different code/configuration/data or an
altered program is rejected without rewriting metadata. This is a provenance
policy simplification, not a change to optimizer rewards or candidate selection.

## Preparation Integration

The later preparation integration connects `python -m pipeline prepare` to
seed splitting, selected-image inference for all six recognizers, and manifest
assembly. The full-benchmark exporter was replaced by the selected-only
backend. General's 80/40 correctness quota is not used for new splits; the
optimizer's separate train sampling is unchanged. Data audits now validate
coverage and aligned evaluation keys rather than requiring historical totals.
The original historical manifest remains loadable. See [OCR Preparation](../ocr/README.md).

## Deliberately Excluded

No secret file is read or copied. No dataset images, GT-bearing prediction
tables, frozen data manifest, weights, caches, or original trained artifacts
are included. Historical checkpoint-resume metadata is not fabricated for the
new source fingerprint. No live smoke, real-LLM training/test inference, or OCR
GPU export has been performed as part of preparing this snapshot.

Original multi-process monitoring, quota-specific resume tools, and cost
estimators tied to historical output directories remain in the local archive,
not the supported portable runner. The portable runner preserves sequential
scope execution and the original 16-way row evaluation. A new portable
multi-process launcher would be a separate change, not a silently claimed copy.

## Validation

`tools/verify_snapshot.py` compares the copied pipeline with the source in
isolated processes: settings, original instruction/field contract, all frozen
rows, every scope's train/validation order, biased train ordering, factual
metrics, and formatted messages on controlled examples. It performs no API
calls. `tools/verify_workflow.py` also compares deterministic mocked native
initial-prompt/GEPA/Bootstrap execution against the original runner, including
SDK request sequences, chosen programs/candidates, validation rows and metrics,
and token accounting. The offline unit suite mocks the SDK and exercises optimizer handoff,
failure handling, parsing, rewards, and provenance. Publication checks inspect
file names, common credential patterns, binary/size limits, syntax, and paths.
