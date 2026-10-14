# Historical Aggregated Results

The tables come from the completed 2026-10-04 Luna EM-only experiment and
the 2026-10-14 error-distribution analysis. They contain aggregated counts and
percentages; the baseline is raw OCR.

`main_table.csv` has 270 rows: 90 cells for each of base, GEPA, and Bootstrap.
`by_dataset.csv`, `by_group.csv`, `by_model.csv`, and `by_cell.csv` classify the
final Bootstrap outputs only.

Fix, unchanged-wrong, and changed-wrong percentages divide by raw errors.
Harm percentages divide by raw-correct predictions. Error-share percentages
divide by all 4,335 raw errors. The unchanged/changed distinction is literal
string equality, with separate normalized-equality columns provided.

These results use the [historical split](../docs/DATA_PROTOCOL.md#historical-reference-split).
`analysis/analyze_errors.py` validates that experiment's row and outcome counts.
