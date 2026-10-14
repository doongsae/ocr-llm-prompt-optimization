# Automatic Prompt Optimization for OCR Post-Correction

## Framework

DSPy automates instruction and few-shot tuning for LLM-based OCR
post-correction. OCR and LLM weights stay frozen: optimization changes the
prompt, not model parameters. Each program is trained independently for one
OCR model and dataset group.

```text
Benchmark images + GT -> seed split -> selected-image OCR -> frozen manifest
  -> GEPA instruction search -> Bootstrap few-shot search -> freeze -> Test
```

| Stage | What happens |
| --- | --- |
| GEPA | Run the correction LLM on Train OCR text; compare its output with GT. The reflection LLM uses factual feedback to propose instruction revisions. Validation mean EM selects the instruction. |
| Bootstrap | Keep that instruction fixed and search demonstration sets using labeled Train pairs and successful generated traces. Generated traces must match GT; Validation mean EM selects the final set. |
| Test | Freeze all programs, then compare initial prompt, GEPA, and GEPA+Bootstrap against raw OCR on the same Test images. |

At inference, the only LLM input is the target model's OCR string; its output
is one corrected string. GT is used for training feedback and scoring.
Correction and reflection both use `gpt-6-luna` with `reasoning_effort="none"`.

Reward is normalized Exact Match: lowercase ASCII-alphanumeric strings must
match completely (1), otherwise 0. Edit distance is a feedback/analysis field,
not a partial reward. GEPA uses ESPO-inspired diagnosis/revision templates;
see [Reflection Sources](docs/REFLECTION_SOURCES.md) for the adaptation scope.

## Models and Data

| OCR models | Checkpoint source | OCR training data |
| --- | --- | --- |
| CRNN, ViTSTR, TRBA, ABINet, PARSeq | [PARSeq release](https://github.com/baudm/parseq/releases/tag/v1.0.0) | Real 9 datasets |
| SVTRv2-B | [OpenOCR](https://github.com/Topdu/OpenOCR) | Union14M-L-Filter |

Benchmark images come from [OpenOCR-Data](https://huggingface.co/datasets/topdu/OpenOCR-Data).
There are 15 datasets: Common 6, UnionHard 6, General 1, OST 2.
Optimization is independent for 6 models x 4 groups = 24 scopes;
evaluation covers 6 models x 15 datasets = 90 cells.

Split images before OCR; run recognizers only on selected images. Train and
Validation each target 500 non-General images, allocated proportionally.
General uses random Train 120 / Validation 300.
Test uses up to 300 remaining images per dataset. All six models share the
split. See [Data Protocol](docs/DATA_PROTOCOL.md) for dataset names, rounding,
label eligibility and OCR-training overlap limitations.

## Setup

Python 3.10+. Install dependencies and create the local configuration:

```bash
python -m pip install -r requirements.txt -r ocr/requirements.txt
cp config.example.json config.json
```

Set `openai_api_key` to your [OpenAI API key](https://platform.openai.com/api-keys)
and `data_root` to the downloaded benchmark directory in `config.json`:

| Setting | Meaning | Default |
| --- | --- | --- |
| `openai_api_key` | OpenAI API key | Empty; required for LLM calls |
| `data_root` | Benchmark LMDB directory | `datasets` |
| `data_manifest` | Prepared OCR table | `data/frozen_rows.csv.gz` |
| `seed` | Split and GEPA seed; Bootstrap uses seed + 1 | `42` |
| `gepa_error_ratio` | Target raw-OCR-error fraction in the GEPA Train pool | `0.75` |
| `bootstrap_error_ratio` | Target raw-OCR-error fraction in the Bootstrap Train pool | `0.60` |

Paths are relative to the repository root unless absolute. Error ratios retain
raw OCR errors and subsample correct rows; they are not reward weights or final
demo proportions. `config.json` is Git-ignored; keep real keys out of
`config.example.json`.

## Execution

```bash
python -m pipeline prepare
python -m pipeline preflight
python -m pipeline api-smoke
python -m pipeline all
```

`prepare` defaults to split -> OCR -> manifest. Existing manifest, split and
completed OCR results are reused; only missing model exports are run.
To generate a new split, change `seed` and use a new `data_manifest` path.
`all` runs prompt training and Test, not OCR.

Official OCR code, inference configuration and pretrained weights are downloaded
automatically and cached. `api-smoke` and `all` make paid API calls.
See [OCR Preparation](ocr/README.md) for cache paths and optional overrides.

## Results and Checks

Historical Test: Raw OCR 83.8403%; GEPA+Bootstrap 86.2335%,
921 fixes / 279 harms, 62 wins / 6 ties / 22 losses over 90 cells.
[Reference tables](reference_results/README.md) use the historical split, which
was repeatedly inspected. They are not results from a new seed.

```bash
python -m unittest discover -s pipeline/tests -v
python tools/check_publication.py
```

[Validation](docs/VALIDATION.md)
