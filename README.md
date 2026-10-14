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
is one corrected string. GT is used for training feedback and scoring, not as
a runtime input. There is no ensemble, verifier, image input to the LLM, or
cross-model prediction input. Correction and reflection both use `gpt-6-luna`.

Reward is normalized Exact Match: lowercase ASCII-alphanumeric strings must
match completely (1), otherwise 0. Edit distance is a feedback/analysis field,
not a partial reward. GEPA uses ESPO-inspired diagnosis/revision templates,
not the full ESPO optimizer.

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
General uses random Train 120 / Validation 300, without an error/correct quota.
Test uses up to 300 remaining images per dataset. All six models share the new
split. See [Data Protocol](docs/DATA_PROTOCOL.md) for dataset names, rounding,
label eligibility and OCR-training overlap limitations.

## Configuration

Edit the root `config.json`:

```json
{
  "seed": 42,
  "gepa_error_ratio": 0.75,
  "bootstrap_error_ratio": 0.60
}
```

`seed` controls new splits and GEPA; Bootstrap uses `seed + 1`.
Ratios are fractions: `0.75` = 75%. They control raw-OCR-error proportions in
optimizer training pools by retaining errors and subsampling correct rows,
not reward weights or final few-shot composition.

## Execution

Python 3.10+. Install the prompt-pipeline environment:

```bash
python -m pip install -r requirements.txt
```

Prepare data using a separate OCR environment and existing upstream assets:

```bash
python -m pipeline prepare \
  --ocr-python <OCR_ENV_PYTHON> --data-root <BENCHMARK_DIRECTORY> \
  --openocr-root <OPENOCR_CHECKOUT> --svtr-config <SVTRV2_INFER_YAML> \
  --svtr-checkpoint <SVTRV2_BEST_PTH>

python -m pipeline preflight
python -m pipeline api-smoke
python -m pipeline all
```

`prepare` defaults to split -> OCR -> manifest. Existing manifest, split and
completed OCR results are reused; only missing model exports are run. Invalid
or mismatched rows are not silently accepted. A changed seed does not replace
an existing split: use a new `OCR_DATA_MANIFEST` path for a new split.
`all` runs prompt training and Test, not OCR.

Default data is `data/frozen_rows.csv.gz`; `OCR_DATA_MANIFEST` changes this path.
Set `OPENAI_API_KEY` before `api-smoke` / `all`, which make paid API calls.
Data and generated outputs are Git-ignored. OCR setup details:
[OCR Preparation](ocr/README.md).

## Results and Checks

Historical Test: Raw OCR 83.8403%; GEPA+Bootstrap 86.2335%,
921 fixes / 279 harms, 62 wins / 6 ties / 22 losses over 90 cells.
[Reference tables](reference_results/README.md) are old results, not results
from a new seed. The historical benchmark was repeatedly inspected.

```bash
python -m unittest discover -s pipeline/tests -v
python tools/check_publication.py
```

[Validation](docs/VALIDATION.md) | [Publication](docs/GITHUB_PREPARATION.md)
