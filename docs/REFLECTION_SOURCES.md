# Source Map

## ESPO

Primary source: [ESPO, arXiv:2609.04197v1](https://arxiv.org/html/2609.04197v1).

- Appendix C.1 describes a structured diagnosis with pattern names, root causes,
  error indices, and suggested instruction fixes. It does not publish a complete
  literal clustering prompt. `prompts/diagnose.txt` implements these requirements,
  with OCR input/output and normalization definitions added.
- Appendix C.2 publishes the diagnostic-revision strategy. The first sentence of
  `prompts/revise.txt` is that strategy verbatim. The remaining text specifies the
  OCR contract, evidence placeholders, and machine-readable output.
- GEPA retains its original minibatch sampling, acceptance, Pareto exploration,
  and validation selection. Diagnosis uses only the supplied failing train rows.
  Pattern count is bounded by that small batch, rather than requiring 3-7 patterns
  when fewer than three failures exist.
- Only diagnosis and diagnostic revision are used here. This is **not full ESPO**:
  no full-train diagnosis, four-strategy population, or stability resampling.
- Two reflection calls per proposal are logged. This changes proposal cost;
  equal GEPA metric-call budgets do not imply equal API-call/token budgets.

No handcrafted OCR error taxonomy, numeric-preservation rule, fragment rule,
wrong-copy praise, or invented reasoning trace is included.

## GEPA / DSPy

Primary sources:

- [GEPA paper](https://arxiv.org/pdf/2507.19457), Section 3 and reflection appendix.
- [Official GEPA code](https://github.com/gepa-ai/gepa).
- Installed DSPy 3.1.3 and GEPA 0.0.26 source, especially
  `dspy/teleprompt/gepa/gepa_utils.py`, `bootstrap.py`, and `random_search.py`.

We use the existing `instruction_proposer` extension point rather than modifying
the installed optimizers. Generated Bootstrap traces require explicit Boolean
normalized EM correctness. Native labeled train demos remain allowed. Candidate
selection is native mean validation EM, not the discarded harm-balanced ranking.

## OpenAI API

- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).
- [Deployment checklist](https://developers.openai.com/api/docs/guides/deployment-checklist).
- [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
- [Official Structured Outputs example](https://developers.openai.com/cookbook/examples/structured_outputs_intro).

Both roles use the exact requested model `gpt-6-luna`, official SDK Chat
Completions, `max_completion_tokens`, and explicit `reasoning_effort=none`.
This preserves the previous non-thinking workload and its temperatures/token
caps. No vLLM template arguments, model substitution, or reasoning fallback.

Correction uses `response_format.type=json_schema`, `strict=true`, all fields
required and `additionalProperties=false`. DSPy's native JSONAdapter formats
the corresponding field contract and demos. The SDK transport enforces the
schema irrespective of DSPy's model-capability registry and validates the
returned JSON with the standard parser before DSPy's decoding. Output control
markers are never stripped from OCR strings or repaired by special-case rules.

The installed JSONAdapter uses marker-style inputs even though its outputs are
JSON. The local `OCRJSONAdapter` overrides only input serialization and the
input-structure example to encode field values as JSON objects. User messages
have no trailing format instructions; system output structure and the API
schema carry them. This is a transport-level correction, not an ESPO prompt
strategy or a new OCR-specific correction rule.
