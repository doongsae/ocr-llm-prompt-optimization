# OCR Preparation

## Flow

Read LMDB image IDs and GT -> seed split -> six recognizers on selected images
only -> `frozen_rows.csv.gz` -> existing prompt pipeline.

`prepare.py` owns splitting, sequential worker orchestration and manifest
assembly. `eval_parseq_selected.py` handles the five PARSeq-distributed models;
`eval_svtrv2.py` handles the official OpenOCR model. No recognizer is trained.
The old full-benchmark exporter is not needed by this flow.

Install the upstream OCR dependencies in an OCR environment (torch,
torchvision, timm 0.9.16, pytorch-lightning, lmdb, numpy, Pillow and PyYAML).
Keep this separate from the pinned DSPy environment. Use official
[PARSeq](https://github.com/baudm/parseq) and
[OpenOCR](https://github.com/Topdu/OpenOCR) code and the existing pretrained
weights. Dataset layout and checkpoint sources are in
[Data Protocol](../docs/DATA_PROTOCOL.md). Datasets and weights are not bundled.

## Execution

Run from the repository root. The default manifest path is
`data/frozen_rows.csv.gz`; the same `OCR_DATA_MANIFEST` setting controls both
preparation and subsequent prompt training.

```bash
python -m pipeline prepare \
  --ocr-python <OCR_ENV_PYTHON> \
  --data-root <BENCHMARK_DIRECTORY> \
  --openocr-root <OPENOCR_CHECKOUT> \
  --svtr-config <SVTRV2_INFER_YAML> \
  --svtr-checkpoint <SVTRV2_BEST_PTH> \
  --device cuda:0

python -m pipeline preflight
python -m pipeline api-smoke
python -m pipeline all
```

Only `api-smoke` and `all` use a paid LLM API. PARSeq code/weights may be
downloaded by torch.hub if they are not cached; `--parseq-root` selects an
existing local checkout. OpenOCR/config/checkpoint paths must exist locally.
Models run sequentially in separate processes to release GPU memory between
recognizers. Missing records or GT mismatches stop the run; they do not become
copy predictions or silently reduce the final evaluation.

`--step split` prepares or reuses the split, `--step ocr` runs only missing
model exports and `--step freeze` assembles or reuses the manifest. A complete
manifest skips all preparation; existing files are not overwritten. To
prepare a different seed, change `config.json` and set a new
`OCR_DATA_MANIFEST` path and output directory rather than replacing an active
experiment's data.

## Sampling

- Seed comes from the root `config.json` (default 42). IDs are generated from the current LMDB labels, not
  recovered from old experiment CSVs or hardcoded exclusion lists.
- Non-General: Train and Validation targets are 500 each across the 14
  datasets, allocated proportionally with `max(2, floor(target * n / N))`.
  Actual totals can be below 500 due to rounding.
- General: random Train 120 and Validation 300, with no 80-error/40-correct
  quota and no recognizer-specific split.
- Test: randomly sample up to 300 remaining images per dataset. Train,
  Validation and Test keys are disjoint and shared by all six models.
- Eligible GT has normalized ASCII-alphanumeric length 1-25. With SVTRv2,
  its existing whitespace-removal/NFKD-ASCII label length must also be 1-25.
  Unsupported counts are recorded before splitting, without model outputs.

Sample counts and an optional split-only seed override are CLI options. The
optimizer's separate error ratios are set in `config.json`; they do not
control this image split.
This creates a new split, not a reproduction of historical fixed-image results.

## Outputs

The output directory contains `selection.csv`, `split.json`, per-model
`predictions/*.csv` and metadata, `preparation.json`, and the compressed
manifest. Reports include split counts, source eligibility, selection/output
hashes, checkpoint provenance and runtime. The manifest uses the existing
pipeline's columns, model names and text normalization. All such data outputs
belong in the Git-ignored `data/` directory, not in the public source tree.
