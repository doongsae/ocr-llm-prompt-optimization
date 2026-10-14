# OCR Preparation

From the repository root, after completing [Setup](../README.md#setup):

```bash
python -m pipeline prepare
```

The command reads image IDs and GT from LMDB, creates the seed-based split,
runs all six recognizers on the selected images, and writes the manifest.
Each model runs in a separate sequential process to release GPU memory.
Sampling and label eligibility are specified in [Data Protocol](../docs/DATA_PROTOCOL.md).

## Model Assets

| Models | Source and cache |
| --- | --- |
| CRNN, ViTSTR, TRBA, ABINet, PARSeq | [PARSeq](https://github.com/baudm/parseq) Real9 releases; torch.hub cache |
| SVTRv2-B | [OpenOCR](https://github.com/Topdu/OpenOCR) U14M-Filter weights; `.cache/ocr/` |

SVTRv2 uses OpenOCR commit `45bfc0cc791c055739872e7255fe032cac1abe9c`, its
official CTC inference YAML, and the [official SVTRv2-B checkpoint](https://github.com/Topdu/OpenOCR/blob/45bfc0cc791c055739872e7255fe032cac1abe9c/configs/rec/svtrv2/readme.md).
The source archive and checkpoint downloads are verified by SHA-256.
Assets are downloaded only when absent; valid cached assets are reused.

## Outputs and Reuse

`data_manifest` determines the manifest path and its parent output directory:

| Output | Contents |
| --- | --- |
| `selection.csv` | Split, dataset, LMDB image ID and GT |
| `split.json` | Seed, source eligibility and split counts |
| `predictions/<model>.csv` | Selected-image OCR predictions |
| `predictions/<model>.json` | Output hashes, checkpoint provenance and runtime |
| `preparation.json` | Combined preparation summary |
| `frozen_rows.csv.gz` | Table consumed by prompt optimization |

An existing manifest skips preparation. Otherwise, the saved split and valid
OCR exports are reused, and only missing models run. Mismatched rows or hashes
stop execution. To generate a new split, change `seed` and `data_manifest`.

## Optional Overrides

| Option | Use |
| --- | --- |
| `--step split`, `ocr`, `freeze` | Run one preparation stage instead of all stages |
| `--device cpu` | Run OCR without CUDA |
| `--batch-size` | Change the OCR batch size |
| `--ocr-python` | Use a separate Python environment with OCR dependencies |
| `--parseq-root`, `--openocr-root` | Use existing upstream source checkouts |
| `--svtr-config`, `--svtr-checkpoint` | Use manually managed SVTRv2 assets |

`python -m pipeline prepare --help` lists sampling overrides.
`OCR_DATA_MANIFEST` overrides `data_manifest` for scripted runs.
