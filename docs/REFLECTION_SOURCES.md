# Reflection Sources

## ESPO Templates

Source: [ESPO, arXiv:2609.04197v1](https://arxiv.org/html/2609.04197v1),
Appendix C.1-C.2.

- `pipeline/prompts/diagnose.txt` implements C.1's structured-diagnosis
  requirements: pattern name, root cause, error IDs and suggested instruction
  fix. C.1 does not provide a complete literal clustering prompt.
- `pipeline/prompts/revise.txt` starts with the C.2 diagnostic-revision
  strategy verbatim. The remaining text specifies OCR input/output, evidence
  placeholders and JSON response fields.
- Each proposal uses two reflection calls: diagnosis, then revision.
  Evidence comes only from failing rows in the scope's GEPA Train minibatch.

This is an ESPO-inspired proposal hook, not the full ESPO optimizer.
GEPA controls minibatch sampling, candidate acceptance, Pareto exploration
and validation selection. Full-Train diagnosis, four-strategy search and
stability resampling are outside this implementation.

## Optimizers

Sources: [GEPA paper](https://arxiv.org/pdf/2507.19457),
[GEPA code](https://github.com/gepa-ai/gepa), and installed DSPy 3.1.3 /
GEPA 0.0.26.

GEPA uses the `instruction_proposer` extension point. Bootstrap admits a
generated trace only when normalized EM is true; labeled Train demos are also
eligible. Both optimizers select by mean Validation EM. GEPA's metric-call
budget excludes reflection calls.

## API Transport

Sources: [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [Structured Outputs](https://developers.openai.com/cookbook/examples/structured_outputs_intro).

Correction and reflection use `gpt-6-luna` with `reasoning_effort="none"`.
Correction uses a strict JSON schema containing only `corrected_text`.
The transport validates returned JSON before DSPy decoding.

`OCRJSONAdapter` serializes OCR inputs as JSON objects. Output-format
instructions stay in the system message and API schema, preserving literal
OCR strings, including empty values and marker-like text.
