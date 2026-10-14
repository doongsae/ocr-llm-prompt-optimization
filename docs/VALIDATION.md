# Validation

Validated on 2026-10-14 with Python 3.10. No paid API calls were made during
repository validation.

| Check | Result |
| --- | --- |
| Unit suite with the historical manifest | 89 tests passed |
| Historical data and prompt contracts | Matched for 35,274 rows and 24 scopes |
| Mocked native optimizer workflow | Matching requests, selected programs, validation outputs and usage |
| OCR integration | Six checkpoints, 15 datasets, 117 images, 702 prediction rows |
| Automatic versus manual model assets | All six prediction CSVs and the split CSV were byte-identical |
| Cache reuse | Completed OCR results and assets reused without inference or download |
| Prepared-data preflight | 24 scopes and 90 Test cells passed |
| Historical error analysis | Aggregate CSVs matched the reference copies |

The OCR integration ran on CPU using 36 Train, 36 Validation and 45 Test images
per model. The automatic-asset run took 36.31 backend seconds, excluding worker
startup. This is an integration check, not a full benchmark result.

The mocked optimizer comparison used three synthetic examples and bounded
native optimizers. Request counts matched: initial prompt 3, GEPA 21,
Bootstrap 23. Timestamps and execution time are excluded from equivalence.

## Run Checks

```bash
python -m unittest discover -s pipeline/tests -v
python tools/check_publication.py --staged
```

The full-data unit test is skipped when the private manifest is unavailable.
For comparison against the original experiment, supply its directory and
manifest:

```bash
python tools/verify_snapshot.py --original <ORIGINAL_DIRECTORY> --manifest <MANIFEST>
python tools/verify_workflow.py --original <ORIGINAL_DIRECTORY>
```

These checks cover input/output fields, normalization, Train/Validation order,
sampling, rewards, formatted messages, optimizer handoff and candidate selection.
They do not establish identical future API generations, blind-test
generalization, or absence of OCR-pretraining overlap. See
[Data Protocol](DATA_PROTOCOL.md) for evaluation limitations.
