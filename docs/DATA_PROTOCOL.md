# Data Protocol

## Dataset and OCR Sources

Benchmark distribution: <https://huggingface.co/datasets/topdu/OpenOCR-Data>.
The dataset repository also distributes filtered Union14M OCR training data;
it is inaccurate to describe the entire repository as test-only.

| Evaluation group | Datasets |
| --- | --- |
| Common | IIIT5k, SVT, IC13_857, IC15_1811, SVTP, CUTE80 |
| U14M | Curve, Multi-Oriented, Artistic, Contextless, Salient, Multi-Words, General |
| OST | OST_weak, OST_heavy |

The completed prompt experiment keeps UnionHard and General separate during
training. It therefore uses 24 independent programs, not 18. Runtime prompts
contain only the target OCR string and train demonstrations from its own scope.

The five PARSeq-distributed recognizers are author-reproduced models trained on
the Real collection, not the original model authors' MJ+ST checkpoints:

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

## Our Frozen Prompt-Optimization Split

The manifest is a processed OCR-output table with these columns:

```text
split,model,dataset,image_idx,gt,gt_filtered,pred,pred_filtered,correct
```

Split labels are `train`, `val`, and `test`. `image_idx` is a zero-based,
dataset-local image identifier, not a global image ID. No OCR is rerun by the
prompt optimizer. Rows and sampling order are preserved when this manifest is
loaded. Strings are read with pandas `keep_default_na=False`.

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

Non-General train and validation allocations came from a 500-row target for
each split, proportionally allocated with integer rounding (494 actual rows).
General train was added separately: 80 raw errors and 40 raw-correct examples
per model. Test is a fixed sample of up to 300 rows per dataset; it is not the
full original benchmark. Use the existing manifest to reproduce the historical
protocol, rather than inventing a new split from this description.

## New Seed-Based Preparation

`python -m pipeline prepare` samples from LMDB image IDs and labels before
running OCR. It retains non-General proportional allocation and up-to-300
test sampling. General train is now a uniform random 120 images, not the
historical 80-error/40-correct recipe. All models use the same generated split.
Unsupported labels are excluded using the recognizers' length requirements,
not fixed image IDs. See [OCR Preparation](../ocr/README.md).

The table above describes the old completed experiment, not mandatory totals
for a new seed. Active validation checks coverage, split disjointness and
model/image/GT alignment instead of insisting on historical row counts. New
splits must not be reported using the old result tables.

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

The historical test has been repeatedly observed. Report it as a historical
benchmark, not as a previously unseen blind holdout. The source manifest and
dataset images are excluded from Git until redistribution terms are reviewed.
