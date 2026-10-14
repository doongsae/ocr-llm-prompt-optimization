"""ESPO-sourced diagnosis/revision templates inside GEPA's proposal hook.

This is not the full ESPO optimizer: GEPA still controls minibatches and selection.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from .config import HERE
from .storage import write_json
from .metrics import facts


class ProposalFormatError(ValueError):
    """An unusable LM proposal, not a broken data or execution contract."""


def render(name, **values):
    return (HERE / "prompts" / name).read_text().format(**values)


def validate_diagnosis(value, error_ids):
    if not isinstance(value, dict) or set(value) != {"patterns"}:
        raise ValueError("diagnosis must contain only a patterns array")
    patterns = value["patterns"]
    if not isinstance(patterns, list) or not 1 <= len(patterns) <= min(7, len(error_ids)):
        raise ValueError("invalid pattern count")
    covered = set()
    for pattern in patterns:
        if not isinstance(pattern, dict) or set(pattern) != {"name", "root_cause", "error_ids", "suggested_fix"}:
            raise ValueError("unexpected diagnosis fields")
        for field in ("name", "root_cause", "suggested_fix"):
            if not isinstance(pattern[field], str) or not pattern[field].strip():
                raise ValueError(f"empty diagnosis field: {field}")
        ids = pattern["error_ids"]
        if (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= error_ids):
            raise ValueError("diagnosis cited nonexistent or duplicate error IDs")
        covered.update(ids)
    if covered != error_ids:
        raise ValueError("diagnosis did not cover all supplied failures")
    return value


class EspoTemplateProposer:
    def __init__(self, lm, train, audit_dir):
        self.lm = lm
        self.allowed = {(str(ex.ocr_prediction), str(ex.gt)) for ex in train}
        self.audit_dir = Path(audit_dir)
        self.calls = 0
        self.failures = 0
        self.rejected_proposals = 0
        self.attempts = 0
        self.audit = {}

    def ask_json(self, prompt, phase):
        outputs = self.lm(prompt=prompt, response_format={"type": "json_object"})
        self.audit[f"{phase}_prompt"] = prompt
        self.audit[f"{phase}_raw_outputs"] = outputs
        if len(outputs) != 1 or not isinstance(outputs[0], str):
            raise ProposalFormatError("expected one reflection text response")
        try:
            return json.loads(outputs[0])
        except json.JSONDecodeError as exc:
            raise ProposalFormatError("reflection response was not valid JSON") from exc

    def __call__(self, candidate, reflective_dataset, components_to_update):
        if self.failures:
            raise RuntimeError("reflection already failed; refusing further proposals")
        self.attempts += 1
        self.audit = {"attempt": self.attempts, "evidence_split": "train",
                      "parent_instruction": candidate.get("correct")}
        try:
            return self.propose(candidate, reflective_dataset, components_to_update)
        except ProposalFormatError as exc:
            self.rejected_proposals += 1
            write_json(self.audit_dir / f"rejected_proposal_{self.attempts:04d}.json",
                       {**self.audit, "status": "rejected", "error_type": type(exc).__name__,
                        "reason": str(exc), "completed_proposals": self.calls})
            # Native GEPA rejects this mutation and advances its bounded search budget.
            raise
        except Exception as exc:
            self.failures += 1
            write_json(self.audit_dir / f"failed_proposal_{self.failures:04d}.json",
                       {**self.audit, "status": "failed", "error_type": type(exc).__name__,
                        "completed_proposals": self.calls, "evidence_split": "train"})
            raise

    def propose(self, candidate, reflective_dataset, components_to_update):
        if components_to_update != ["correct"]:
            raise ValueError("unexpected GEPA component")
        records = []
        for item in reflective_dataset["correct"]:
            if item["Inputs"] != {"ocr_prediction": item["Inputs"].get("ocr_prediction")}:
                raise ValueError("unexpected reflection input fields")
            record = json.loads(item["Feedback"])
            if (record["ocr_prediction"], record["ground_truth"]) not in self.allowed:
                raise ValueError("reflection evidence is not from this scope's train")
            if item["Inputs"]["ocr_prediction"] != record["ocr_prediction"]:
                raise ValueError("feedback input disagrees with execution trace")
            output = item["Generated Outputs"].get("corrected_text")
            if output != record["corrected_text"]:
                raise ValueError("feedback output disagrees with execution trace")
            expected = facts(SimpleNamespace(gt=record["ground_truth"], ocr_prediction=record["ocr_prediction"]),
                             SimpleNamespace(corrected_text=output))
            if record != expected:
                raise ValueError("reflection facts disagree with deterministic evaluation")
            if record["em"] == 0:
                records.append({"error_id": f"E{len(records)}", **record})
        if not records:
            return {"correct": candidate["correct"]}
        self.audit["errors"] = records
        instruction = candidate["correct"]
        diagnosis_prompt = render("diagnose.txt", instruction=instruction,
                                  errors=json.dumps(records, ensure_ascii=True))
        value = self.ask_json(diagnosis_prompt, "diagnosis")
        try:
            diagnosis = validate_diagnosis(value, {r["error_id"] for r in records})
        except ValueError as exc:
            raise ProposalFormatError(str(exc)) from exc
        revision_prompt = render("revise.txt", instruction=instruction,
                                 diagnosis=json.dumps(diagnosis, ensure_ascii=True))
        revision = self.ask_json(revision_prompt, "revision")
        if (not isinstance(revision, dict) or set(revision) != {"new_instruction"}
                or not isinstance(revision["new_instruction"], str) or not revision["new_instruction"].strip()):
            raise ProposalFormatError("expected a nonempty new_instruction string")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / f"proposal_{self.calls:04d}.json"
        if path.exists():
            raise FileExistsError(path)
        write_json(path, {"source": "ESPO Appendix C.1 requirements + C.2 diagnostic revision",
                          "full_espo_optimizer": False, "evidence_split": "train",
                          "parent_instruction": instruction, "errors": records,
                          "diagnosis_prompt": diagnosis_prompt, "diagnosis": diagnosis,
                          "revision_prompt": revision_prompt, **revision})
        self.calls += 1
        return {"correct": revision["new_instruction"].strip()}
