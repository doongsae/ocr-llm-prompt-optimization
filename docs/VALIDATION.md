# Validation

Code checks: 2026-10-15. GPU and live API checks: 2026-10-14. Python 3.10.

| Check | Result |
| --- | --- |
| Unit suite with the historical manifest | 95 tests passed |
| Historical data and prompt contracts | Matched for 35,274 rows and 24 scopes |
| Mocked native optimizer workflow | Matching requests, selected programs, validation outputs and usage |
| CPU and CUDA OCR integration | Six checkpoints, 15 datasets, 117 images per model; all 702 prediction rows matched |
| SVTRv2 batch coverage | Batch sizes 1, 2, 4, 8 and 128; exact multiples and partial batches covered |
| Live API smoke | 8 correction calls and 2 reflection calls completed successfully |
| Automatic versus manual model assets | All six prediction CSVs and the split CSV were byte-identical |
| Cache reuse | Completed OCR results and assets reused without inference or download |
| Prepared-data preflight | 24 scopes and 90 Test cells passed |
| Historical error analysis | Aggregate CSVs matched the reference copies |
| Error analysis on new counts | Accepted new outcomes and zero-error inputs; rejected inconsistent inputs under Python `-O` |

The OCR integration used 36 Train, 36 Validation and 45 Test images per model.
The CUDA run used an NVIDIA A100 80GB at batch size 1 and took 29.05 backend
seconds, excluding worker startup. Its predictions matched the CPU run for all
six models. A separate SVTRv2 CPU regression at batch size 4 also matched all
117 reference rows. These are integration checks, not full benchmark results.

The live API smoke used real Train OCR strings with the production adapter and
prompts: 8 corrections, followed by one failure diagnosis and instruction
revision. All 10 requests used `gpt-6-luna` with `reasoning_effort="none"` and
retries disabled. Usage was 2,204 input tokens, 348 output tokens and 0 reasoning
tokens over 18.32 seconds. The generated instruction was not applied; this does
not constitute a full optimizer run or a performance evaluation.

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
