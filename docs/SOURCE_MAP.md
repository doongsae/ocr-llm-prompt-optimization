# Research Source Map

All source paths below are relative to the original `OCR_LLM` workspace. The
original files remain unchanged. No historical run metadata has been rewritten
to claim that the portable copy produced the old experiment results.

| Portable location | Original source | Change |
| --- | --- | --- |
| `pipeline/config.py` | `261004_luna_em_gepa_bootstrap_espo_feedback/config.py` and original shared data constants | Repository-relative paths; centralize model/group/stage definitions |
| `pipeline/core.py` | `ocr_dspy_gepa.py` and original `data_contract.py` factory | Preserve signature, corrector, normalization, seeds, and factory |
| `pipeline/data.py` | `260731_gepa_bootstrap_scope_validated_recovery/common.py` | Package imports; remove unused server/output settings, group lookup, and scope-hash helpers |
| `pipeline/contracts.py` | `261004_luna_em_gepa_bootstrap_espo_feedback/data_contract.py` | Keep validation only, remove re-export facade |
| `pipeline/cli.py`, `training.py`, `evaluation.py`, `artifacts.py` | Functions and optimizer bodies from `261004_luna_em_gepa_bootstrap_espo_feedback/run_experiment.py` | Group related functions; package imports/current fingerprints; remove cross-fingerprint resume compatibility |
| `pipeline/storage.py` | Shared `common.py` hash/JSON functions and runner timestamp function | Centralize storage without changing behavior |
| `pipeline/json_adapter.py`, `prompts/` | Same names under `261004_luna_em_gepa_bootstrap_espo_feedback/` | Byte-for-byte copies |
| `pipeline/metrics.py`, `openai_lm.py`, `reflection.py` | Same names under `261004_luna_em_gepa_bootstrap_espo_feedback/` | Preserve logic; relative imports and relocated API-key guard |
| `pipeline/tests/test_experiment.py` | `261004_luna_em_gepa_bootstrap_espo_feedback/tests/test_experiment.py` | Package imports and owner-specific mocks; private-data skip and temporary manifest fixture |
| `analysis/analyze_errors.py` | `261014_luna_test_error_distribution/analyze.py` | Source/output paths become CLI arguments; protect source directory |
| `ocr/prepare.py` | `prepare_dspy_data.py`, T11 test sampling and existing manifest assembly | GT-first seed split; no fixed IDs or General correctness quota; selected-output merge |
| `ocr/eval_parseq_selected.py` | `eval_parseq_models.py` | Same preprocessing/model API; infer only generated selection |
| `ocr/eval_svtrv2.py` | `260612_t11_general_tables/run_svtrv2_aligned_eval.py` | Keep RatioSampler/LMDB-ID alignment; configurable paths; official inference model without the unused training branch |
| `docs/REFLECTION_SOURCES.md` | `261004_luna_em_gepa_bootstrap_espo_feedback/SOURCES.md` | Byte-for-byte copy |
| `reference_results/main_table.csv` | `261004_luna_em_gepa_bootstrap_espo_feedback/results/main_table.csv` | Aggregated historical copy |
| `reference_results/by_*.csv` | `261014_luna_test_error_distribution/by_*.csv` | Aggregated historical copies |

The repository documentation, publication utilities, and
`pipeline/tests/test_repository.py` and `test_structure.py` are new packaging work, not a new OCR
method. `docs/source_inventory.json` maps each of the 933 locally archived
source files to its original relative path and SHA-256. The archive is ignored
by Git by default and requires separate review before publication.
