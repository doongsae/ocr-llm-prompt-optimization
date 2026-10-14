"""GEPA/Bootstrap optimization and the scope training lifecycle."""

import random
import time

import dspy
import pandas as pd
from dspy.teleprompt import GEPA

from . import config
from .artifacts import fingerprint, job_dir, verify_job
from .contracts import verify_program
from .core import fresh_program
from .data import build_error_biased_trainset_safe, load_scope_examples
from .evaluation import evaluate
from .metrics import bootstrap_metric, gepa_metric
from .openai_lm import reset_usage, setup_lms, usage_summary
from .reflection import EspoTemplateProposer
from .storage import now, sha256_path, write_json


def optimize_instruction(program, train, val, reflection, output, seed):
    proposer = EspoTemplateProposer(reflection, train, output / "reflection")
    optimizer = GEPA(metric=gepa_metric, auto=config.SETTINGS["gepa_auto"],
                     reflection_minibatch_size=min(config.SETTINGS["reflection_minibatch_size"], len(train)),
                     reflection_lm=reflection, instruction_proposer=proposer,
                     gepa_kwargs={"stop_callbacks": lambda state: proposer.failures > 0},
                     skip_perfect_score=True, track_stats=True, seed=seed,
                     num_threads=config.SETTINGS["num_threads"], log_dir=str(output / "gepa_logs"))
    program = optimizer.compile(program, trainset=train, valset=val)
    if proposer.failures:
        raise RuntimeError("GEPA reflection failed; refusing to mark this job completed")
    if proposer.rejected_proposals and not proposer.calls:
        raise RuntimeError("GEPA produced no valid reflection proposals; not a completed optimization")
    result = program.detailed_results
    candidates = []
    for index, candidate in enumerate(result.candidates):
        verify_program(candidate, train)
        path = output / "candidates" / f"{index:03d}.json"
        path.parent.mkdir(exist_ok=True)
        candidate.save(str(path))
        candidates.append({"instruction": candidate.correct.signature.instructions,
                           "program": str(path.relative_to(output)),
                           "program_sha256": sha256_path(path)})
    write_json(output / "selection.json", {
        "selection": config.SETTINGS["selection"], "best_index": result.best_idx,
        "validation_scores": result.val_aggregate_scores,
        "validation_subscores": result.val_subscores,
        "instruction_candidates": candidates,
        "proposal_count": proposer.calls, "rejected_proposal_count": proposer.rejected_proposals,
        "max_metric_calls": optimizer.max_metric_calls})
    return program


def optimize_demonstrations(program, train, val, output):
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=bootstrap_metric, max_bootstrapped_demos=config.SETTINGS["bootstrap_max_demos"],
        max_labeled_demos=config.SETTINGS["bootstrap_max_demos"],
        num_candidate_programs=config.SETTINGS["bootstrap_candidates"],
        max_rounds=config.SETTINGS["bootstrap_max_rounds"], num_threads=config.SETTINGS["num_threads"], max_errors=1)
    program = optimizer.compile(program, trainset=train, valset=val)
    candidates = []
    for index, item in enumerate(program.candidate_programs):
        candidate = item["program"]
        verify_program(candidate, train)
        path = output / "candidates" / f"{index:03d}.json"
        path.parent.mkdir(exist_ok=True)
        candidate.save(str(path))
        candidates.append({"seed": item["seed"], "validation_em_percent": item["score"],
                           "subscores": item["subscores"], "program_sha256": sha256_path(path),
                           "program": str(path.relative_to(output))})
    write_json(output / "selection.json", {"selection": config.SETTINGS["selection"], "candidates": candidates})
    return program


def train_scope(model, group, stage):
    output = job_dir(model, group, stage)
    if output.exists():
        return verify_job(model, group, stage)
    output.mkdir(parents=True)
    started = time.monotonic()
    reset_usage()
    state = {"status": "running", "model": model, "group": group, "stage": stage,
             "started_at_utc": now(), **fingerprint()}
    write_json(output / "status.json", state)
    try:
        train, val = load_scope_examples("group", model, group)
        ratio = config.SETTINGS["bootstrap_error_ratio"] if stage == "bootstrap" else config.SETTINGS["gepa_error_ratio"]
        seed = config.SETTINGS["bootstrap_seed"] if stage == "bootstrap" else config.SETTINGS["gepa_seed"]
        biased, n_errors, n_correct, fallback = build_error_biased_trainset_safe(train, ratio, seed)
        random.seed(seed)
        _, reflection = setup_lms(config.SETTINGS)
        program = fresh_program()
        state.update(train_rows=len(train), biased_train_rows=len(biased), train_errors=n_errors,
                     correct_train_kept=n_correct, sampling_fallback=fallback, validation_rows=len(val))
        print(f"[start] {model}/{group}/{stage} train={len(biased)} val={len(val)}", flush=True)
        if stage == "gepa":
            program = optimize_instruction(program, biased, val, reflection, output, seed)
        elif stage == "bootstrap":
            base = verify_job(model, group, "gepa")
            program.load(str(job_dir(model, group, "gepa") / "program.json"))
            program = fresh_program(program.correct.signature.instructions)
            state["gepa_program_sha256"] = base["program_sha256"]
            program = optimize_demonstrations(program, biased, val, output)
        state["optimization_runtime_seconds"] = time.monotonic() - started
        state["contract"] = verify_program(program, biased)
        program.save(str(output / "program.json"))
        (output / "learned_prompt.txt").write_text(program.correct.signature.instructions + "\n")
        state["program_sha256"] = sha256_path(output / "program.json")
        report, rows = evaluate(program, val)
        pd.DataFrame(rows).to_csv(output / "validation_predictions.csv", index=False)
        write_json(output / "validation.json", report)
        state.update(status="completed", validation=report, usage=usage_summary(),
                     finished_at_utc=now(), runtime_seconds=time.monotonic() - started)
        write_json(output / "metadata.json", state)
        print(f"[done] {model}/{group}/{stage} EM={report['candidate_accuracy']:.6f} "
              f"fixes={report['fixes']} harms={report['harms']} seconds={state['runtime_seconds']:.2f}", flush=True)
        return state
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, usage=usage_summary(),
                     finished_at_utc=now(), runtime_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output / "status.json", state)

