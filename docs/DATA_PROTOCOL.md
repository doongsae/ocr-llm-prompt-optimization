# Data Protocol

## Dataset and OCR Sources

Benchmark distribution: <https://huggingface.co/datasets/topdu/OpenOCR-Data>.
The distribution includes benchmark LMDBs and filtered Union14M training data.

| Evaluation group | Datasets |
| --- | --- |
| Common | IIIT5k, SVT, IC13_857, IC15_1811, SVTP, CUTE80 |
| U14M | Curve, Multi-Oriented, Artistic, Contextless, Salient, Multi-Words, General |
| OST | OST_weak, OST_heavy |

Prompt optimization separates U14M into UnionHard (six datasets) and General.
Each OCR model has one program per group: 24 programs and 90 evaluation cells.
Runtime prompts use only the target OCR string and the scope's Train demos.

The five PARSeq-distributed recognizers use the authors' Real-trained releases:

| OCR model | Weight | Distribution |
| --- | --- | --- |
| CRNN | crnn-679d0e31.pt | PARSeq release v1.0.0 |
| ViTSTR | vitstr-26d0fcf4.pt | PARSeq release v1.0.0 |
| TRBA | trba-cfaed284.pt | PARSeq release v1.0.0 |
| ABINet | abinet-1d1e373e.pt | PARSeq release v1.0.0 |
| PARSeq | parseq-bb5792a6.pt | PARSeq release v1.0.0 |
| SVTRv2-B | best.pth, u14m_filter/svtrv2_smtr_gtc_rctc | OpenOCR |

Sources: <https://github.com/baudm/parseq/releases/tag/v1.0.0>,
<https://github.com/baudm/parseq/blob/main/Datasets.md>,
<https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md>.

Real consists of COCO-Text, RCTW17, Uber-Text, ArT, LSVT, MLT19, ReCTS,
TextOCR, and OpenVINO. SVTRv2-B uses Union14M-L-Filter.

## Manifest

The manifest is a processed OCR-output table with these columns:

```text
split,model,dataset,image_idx,gt,gt_filtered,pred,pred_filtered,correct
```

Split labels are `train`, `val`, and `test`. `image_idx` is a zero-based,
dataset-local LMDB image identifier. Prompt optimization loads this table
without rerunning OCR, preserving row order. Strings are read with pandas
`keep_default_na=False`.

## Seed-Based Split

`prepare` reads labels before OCR and samples without consulting model outputs.
All six models share the generated Train/Validation/Test keys.

| Split | Sampling |
| --- | --- |
| Non-General Train and Validation | Target 500 each across 14 datasets; per-dataset count is `max(2, floor(target * n / N))` |
| General Train / Validation | Uniform random 120 / 300 images |
| Test | Up to 300 images per dataset from the remaining pool |

Here `n` is the dataset's eligible count and `N` the total eligible non-General
count. Integer rounding can make totals differ from 500. The seed is read from
`config.json`; saved splits are reused. A new split requires a new manifest
path. See [OCR Preparation](../ocr/README.md) for sampling overrides.

Eligible GT has normalized ASCII-alphanumeric length 1-25. With SVTRv2 enabled,
the GT must also have length 1-25 after whitespace removal, NFKD normalization
and ASCII conversion. Excluded-label counts are recorded in `split.json`.

Under `data_root`, LMDB directories follow this layout:

```text
test/{IIIT5k,SVT,IC13_857,IC15_1811,SVTP,CUTE80}/data.mdb
u14m/{curve,multi_oriented,artistic,contextless,salient,multi_words,general}/data.mdb
OST/{weak,heavy}/data.mdb
```

Braces list alternative child directory names.

## Historical Reference Split

The published reference tables use this pre-existing manifest, not a newly
generated seed-based split:

| Dataset | Train / model | Validation / model | Test / model |
| --- | ---: | ---: | ---: |
| IIIT5k | 75 | 75 | 300 |
| SVT | 16 | 16 | 300 |
| IC13_857 | 21 | 21 | 300 |
| IC15_1811 | 45 | 45 | 300 |
| SVTP | 16 | 16 | 300 |
| CUTE80 | 7 | 7 | 273 |
| Curve | 60 | 60 | 300 |
| Multi-Oriented | 34 | 34 | 300 |
| Artistic | 22 | 22 | 300 |
| Contextless | 19 | 19 | 300 |
| Salient | 39 | 39 | 300 |
| Multi-Words | 20 | 20 | 298 |
| General | 120 | 300 | 300 |
| OST_weak | 60 | 60 | 300 |
| OST_heavy | 60 | 60 | 300 |
| Total | 614 | 794 | 4471 |

The historical non-General Train and Validation splits each have 494 rows
after rounding. Historical General Train contains 80 errors and 40 raw-correct
examples per OCR model. Its keys can therefore differ across models, whereas
new preparation samples the same 120 images for all models. Reproducing these
reference results requires the historical manifest.

## Limits on Leakage Claims

Within each OCR model, `(dataset, image_idx)` is disjoint across our three splits.
Train GT may appear in demonstrations and reflection evidence. Validation/test
GT is used for scoring but is not passed as a correction input. This does not
prove global image-content deduplication or absence of pretraining contamination.

The SVTRv2 public checkpoint uses Common benchmark evaluation during checkpoint
selection. PARSeq's Real training corpus can overlap the Union14M benchmark;
actual overlap in our sampled rows has not been fully audited or removed.
Union14M-L-Filter removes benchmark overlap from the SVTRv2 OCR training side
according to the authors' procedure, not from our sampled test retroactively.
OST derives from Common benchmark imagery and should not be described as having
no possible source-image relationship to other benchmarks.

The historical test has been repeatedly observed; it is not a blind holdout.
The source manifest and dataset images are excluded from Git.
