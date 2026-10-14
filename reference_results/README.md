# Historical Aggregated Results

The tables are copied from the completed 2026-10-04 Luna EM-only experiment and
the 2026-10-14 read-only error-distribution analysis. They contain aggregated
counts and percentages only; images, GT strings, predictions, prompts containing
train examples, credentials, caches, and transport logs are not included.

`main_table.csv` has 270 rows: 90 cells for each of base, GEPA, and Bootstrap.
`by_dataset.csv`, `by_group.csv`, `by_model.csv`, and `by_cell.csv` classify the
final Bootstrap outputs only. The baseline is raw OCR, not a previous LLM.

Fix, unchanged-wrong, and changed-wrong percentages divide by raw errors.
Harm percentages divide by raw-correct predictions. Error-share percentages
divide by all 4,335 raw errors. The unchanged/changed distinction is literal
string equality, with separate normalized-equality columns provided.

These are historical benchmark results, not a fresh run of this reorganized
snapshot. The `analysis/analyze_errors.py` script deliberately validates that
historical experiment's exact row and outcome counts.
