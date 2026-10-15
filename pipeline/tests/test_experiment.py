"""No network or real API credentials are used by these tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline import config

import dspy
from dspy.teleprompt import GEPA
from openai.types.chat import ChatCompletion

from pipeline import artifacts, data, evaluation, openai_lm, storage, training
from pipeline.config import MODEL, SETTINGS
from pipeline.config import DATA_MANIFEST
from pipeline.contracts import audit_data, verify_program
from pipeline.core import fresh_program
from pipeline.metrics import bootstrap_metric, exact_metric, facts, gepa_metric, trace_is_exact
from pipeline.openai_lm import CORRECTION_RESPONSE_FORMAT, LunaLM, reset_usage, usage_summary
from pipeline.json_adapter import OCRJSONAdapter
from pipeline.reflection import EspoTemplateProposer, ProposalFormatError, render, validate_diagnosis


def example(ocr="SPED", gt="SPEED", idx="1"):
    return dspy.Example(ocr_prediction=ocr, corrected_text=gt, gt=gt, model="trba",
                        dataset="SVT", image_idx=idx).with_inputs("ocr_prediction")


def prediction(text):
    return dspy.Prediction(corrected_text=text)


def completion(text, finish="stop"):
    return ChatCompletion(id="offline-fixture", created=0, model=MODEL, object="chat.completion",
                          choices=[{"index": 0, "finish_reason": finish,
                                    "message": {"role": "assistant", "content": text}}],
                          usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20})


def diagnosis(ids=("E0",)):
    return {"patterns": [{"name": "Observed pattern", "root_cause": "Observed cause",
                          "error_ids": list(ids), "suggested_fix": "Evidence-based revision"}]}


class MetricTests(unittest.TestCase):
    def test_four_behaviors_are_em_only(self):
        cases = [("OPEN", "OPEN", "OPEN", 1), ("OPEN", "OPEN", "OPENS", 0),
                 ("SPED", "SPEED", "SPEED", 1), ("SPED", "SPEED", "SPED", 0),
                 ("SPEXX", "SPEED", "SPEEX", 0)]
        for raw, gt, output, expected in cases:
            with self.subTest(raw=raw, output=output):
                ex = example(raw, gt)
                pred = prediction(output)
                self.assertEqual(exact_metric(ex, pred), expected)
                self.assertIs(trace_is_exact(ex, pred), bool(expected))
                self.assertIs(bootstrap_metric(ex, pred, trace=[]), bool(expected))
                self.assertEqual(bootstrap_metric(ex, pred), float(expected))
                feedback = gepa_metric(ex, pred, pred_name="correct")
                self.assertEqual(feedback.score, expected)
                self.assertEqual(json.loads(feedback.feedback), facts(ex, pred))

    def test_normalization_ignores_case_and_non_alphanumeric(self):
        self.assertEqual(exact_metric(example("sp-ed", "SPEED!"), prediction("Speed.")), 1)

    def test_missing_empty_and_explanatory_outputs_are_not_successes(self):
        for pred in (dspy.Prediction(), prediction(None), prediction(""),
                     prediction("SPEED\nExplanation"), prediction("<think>SPEED</think>")):
            self.assertEqual(exact_metric(example(), pred), 0)
            self.assertIs(trace_is_exact(example(), pred), False)

    def test_feedback_contains_facts_only(self):
        value = gepa_metric(example("255", "256"), prediction("255"), pred_name="correct")
        record = json.loads(value.feedback)
        self.assertEqual(record["em"], 0)
        self.assertEqual(record["edit_distance_after"], 1)
        self.assertNotIn("NEVER", value.feedback)
        self.assertNotIn("wisely", value.feedback)


class TransportTests(unittest.TestCase):
    def setUp(self):
        reset_usage()

    def test_sdk_request_and_dspy_parsing(self):
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": "OPEN"}')
        lm = LunaLM("student", 32, 0.0)
        with patch("pipeline.openai_lm.sdk_client", return_value=client), dspy.context(lm=lm, adapter=dspy.JSONAdapter(use_native_function_calling=False)):
            pred = fresh_program()(ocr_prediction="OPEN")
        self.assertEqual(pred.corrected_text, "OPEN")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["max_completion_tokens"], 32)
        self.assertEqual(request["response_format"], CORRECTION_RESPONSE_FORMAT)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("chat_template_kwargs", request)
        self.assertEqual(usage_summary()["completed_requests"], 1)

    def test_copy_preserves_sdk_config_without_serializing_credentials(self):
        lm = LunaLM("student", 32, 0.0).copy(rollout_id=1, temperature=1.0)
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": "OPEN"}')
        with patch("pipeline.openai_lm.sdk_client", return_value=client):
            self.assertEqual(lm(prompt="test"), ['{"corrected_text": "OPEN"}'])
        request = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("rollout_id", request)
        self.assertNotIn("api_key", lm.kwargs)
        self.assertEqual(request["temperature"], 1)

    def test_invalid_json_is_rejected_without_repair(self):
        invalid = ['{"corrected_text": "OPEN",}', '{"corrected_text": 1}',
                   '{"corrected_text": "OPEN", "other": "ignored"}', '{}',
                   '["OPEN"]', '[[ ## corrected_text ## ]]\nOPEN\n[[ ## completed ##]]']
        for value in invalid:
            with self.subTest(value=value):
                client = Mock()
                client.chat.completions.create.return_value = completion(value)
                with patch("pipeline.openai_lm.sdk_client", return_value=client), self.assertRaises(ValueError):
                    LunaLM("student", 32, 0)(prompt="OPEN")
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_schema_cannot_be_weakened_by_adapter_fallback(self):
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": "OPEN"}')
        with patch("pipeline.openai_lm.sdk_client", return_value=client):
            LunaLM("student", 32, 0)(prompt="OPEN", response_format={"type": "json_object"})
        self.assertEqual(client.chat.completions.create.call_args.kwargs["response_format"], CORRECTION_RESPONSE_FORMAT)

    def test_transport_does_not_strip_or_normalize_text(self):
        value = 'A"B\\C [[ ## completed ##]]'
        client = Mock()
        client.chat.completions.create.return_value = completion(json.dumps({"corrected_text": value}))
        with patch("pipeline.openai_lm.sdk_client", return_value=client), \
                dspy.context(lm=LunaLM("student", 32, 0), adapter=dspy.JSONAdapter(use_native_function_calling=False)):
            pred = fresh_program()(ocr_prediction=value)
        self.assertEqual(pred.corrected_text, value)

    def test_json_reflection_request(self):
        client = Mock()
        client.chat.completions.create.return_value = completion('{"ready": true}')
        with patch("pipeline.openai_lm.sdk_client", return_value=client):
            output = LunaLM("reflection", 1200, 1.0)(prompt="test", response_format={"type": "json_object"})
        self.assertEqual(json.loads(output[0]), {"ready": True})
        self.assertEqual(client.chat.completions.create.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_truncation_does_not_fall_back_to_identity(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("SPE", finish="length")
        with patch("pipeline.openai_lm.sdk_client", return_value=client), self.assertRaises(RuntimeError):
            LunaLM("student", 32, 0)(prompt="SPED")

    def test_missing_key_stops_before_request(self):
        with patch.dict(config.USER_CONFIG, {"openai_api_key": ""}), \
                patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            openai_lm.require_key()


class ReflectionTests(unittest.TestCase):
    def reflective(self, ex=None, output="SPED"):
        ex = ex or example()
        pred = prediction(output)
        return {"correct": [{"Inputs": {"ocr_prediction": ex.ocr_prediction},
                             "Generated Outputs": {"corrected_text": output},
                             "Feedback": gepa_metric(ex, pred, pred_name="correct").feedback}]}

    def test_two_phase_proposal_and_evidence_log(self):
        lm = Mock(side_effect=[[json.dumps(diagnosis())], ['{"new_instruction": "Revised OCR instruction"}']])
        with tempfile.TemporaryDirectory() as directory:
            proposer = EspoTemplateProposer(lm, [example()], directory)
            output = proposer({"correct": "Initial instruction"}, self.reflective(), ["correct"])
            record = json.loads((Path(directory) / "proposal_0000.json").read_text())
        self.assertEqual(output, {"correct": "Revised OCR instruction"})
        self.assertEqual(lm.call_count, 2)
        self.assertEqual(record["evidence_split"], "train")
        self.assertFalse(record["full_espo_optimizer"])
        self.assertIn("SPED", record["diagnosis_prompt"])
        self.assertIn("SPEED", record["diagnosis_prompt"])

    def test_out_of_scope_evidence_is_rejected_before_lm_call(self):
        lm = Mock()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            EspoTemplateProposer(lm, [example()], directory)(
                {"correct": "seed"}, self.reflective(example("OTHER", "SECRET"), "OTHER"), ["correct"])
        lm.assert_not_called()

    def test_tampered_facts_are_rejected(self):
        data = self.reflective()
        record = json.loads(data["correct"][0]["Feedback"])
        record["em"] = 1
        data["correct"][0]["Feedback"] = json.dumps(record)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            EspoTemplateProposer(Mock(), [example()], directory)({"correct": "seed"}, data, ["correct"])

    def test_empty_error_pool_is_not_a_hallucinated_diagnosis(self):
        lm = Mock()
        ex = example("OPEN", "OPEN")
        with tempfile.TemporaryDirectory() as directory:
            output = EspoTemplateProposer(lm, [ex], directory)(
                {"correct": "seed"}, self.reflective(ex, "OPEN"), ["correct"])
        self.assertEqual(output, {"correct": "seed"})
        lm.assert_not_called()

    def test_diagnosis_rejects_unknown_ids_and_missing_coverage(self):
        with self.assertRaises(ValueError):
            validate_diagnosis(diagnosis(("E9",)), {"E0"})
        with self.assertRaises(ValueError):
            validate_diagnosis(diagnosis(), {"E0", "E1"})

    def test_paper_strategy_is_present_and_ocr_rules_are_absent(self):
        prompt = render("revise.txt", instruction="seed", diagnosis="{}")
        self.assertTrue(prompt.startswith("Revise the instruction to address each error pattern."))
        self.assertNotIn("NEVER modify", prompt)
        self.assertNotIn("leave it unchanged", prompt)

    def test_invalid_proposal_is_audited_and_next_proposal_can_succeed(self):
        lm = Mock(side_effect=[["not JSON"], [json.dumps(diagnosis())], ['{"new_instruction": "Revised"}']])
        with tempfile.TemporaryDirectory() as directory:
            proposer = EspoTemplateProposer(lm, [example()], directory)
            with self.assertRaises(ProposalFormatError):
                proposer({"correct": "seed"}, self.reflective(), ["correct"])
            self.assertEqual(proposer.failures, 0)
            self.assertEqual(proposer.rejected_proposals, 1)
            audit = json.loads((Path(directory) / "rejected_proposal_0001.json").read_text())
            self.assertEqual(audit["diagnosis_raw_outputs"], ["not JSON"])
            output = proposer({"correct": "seed"}, self.reflective(), ["correct"])
            self.assertEqual(output, {"correct": "Revised"})
        self.assertEqual(lm.call_count, 3)

    def test_invalid_pattern_count_is_not_a_fatal_pipeline_failure(self):
        malformed = {"patterns": diagnosis()["patterns"] * 2}
        lm = Mock(return_value=[json.dumps(malformed)])
        with tempfile.TemporaryDirectory() as directory:
            proposer = EspoTemplateProposer(lm, [example()], directory)
            with self.assertRaisesRegex(ProposalFormatError, "invalid pattern count"):
                proposer({"correct": "seed"}, self.reflective(), ["correct"])
            self.assertEqual(proposer.failures, 0)
            self.assertEqual(proposer.rejected_proposals, 1)
            audit = json.loads((Path(directory) / "rejected_proposal_0001.json").read_text())
            self.assertEqual(json.loads(audit["diagnosis_raw_outputs"][0]), malformed)

    def test_transport_failure_remains_fatal(self):
        lm = Mock(side_effect=RuntimeError("API unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            proposer = EspoTemplateProposer(lm, [example()], directory)
            with self.assertRaises(RuntimeError):
                proposer({"correct": "seed"}, self.reflective(), ["correct"])
            self.assertEqual(proposer.failures, 1)
            self.assertEqual(proposer.rejected_proposals, 0)
            with self.assertRaises(RuntimeError):
                proposer({"correct": "seed"}, self.reflective(), ["correct"])
        self.assertEqual(lm.call_count, 1)


class ContractTests(unittest.TestCase):
    def test_runtime_input_is_json_without_trailing_instructions(self):
        adapter = OCRJSONAdapter()
        for text in ("", "OPEN", 'A"B\\C', "[[ ## corrected_text ## ]]", "A\nB"):
            with self.subTest(text=text):
                messages = adapter.format(fresh_program().correct.signature, [example()], {"ocr_prediction": text})
                user_messages = [m["content"] for m in messages if m["role"] == "user"]
                self.assertEqual(json.loads(user_messages[-1]), {"ocr_prediction": text})
                self.assertEqual(json.loads(user_messages[0]), {"ocr_prediction": "SPED"})

    def test_empty_response_is_valid_but_does_not_gain_wrong_copy_reward(self):
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": ""}')
        with patch("pipeline.openai_lm.sdk_client", return_value=client), \
                dspy.context(lm=LunaLM("student", 32, 0), adapter=OCRJSONAdapter()):
            pred = fresh_program()(ocr_prediction="")
        self.assertEqual(pred.corrected_text, "")
        self.assertTrue(facts(example("", "ACTUAL"), pred)["output_valid"])
        self.assertEqual(exact_metric(example("", "ACTUAL"), pred), 0)

    @unittest.skipUnless(DATA_MANIFEST.is_file(), "Private frozen data manifest not supplied")
    def test_entire_frozen_split_is_disjoint_and_complete(self):
        report = audit_data()
        self.assertEqual(report["training_scopes"], 24)
        self.assertEqual(report["test_cells"], 90)
        self.assertEqual(report["rows"], len(data.load_data_manifest()))

    def test_demos_have_exact_train_answers_and_metadata_is_not_formatted(self):
        program = fresh_program()
        program.correct.demos = [example()]
        self.assertTrue(verify_program(program, [example()])["exact_train_demos"])
        program.correct.demos = [dspy.Example(ocr_prediction="SPED", corrected_text="SPED")]
        with self.assertRaises(ValueError):
            verify_program(program, [example()])

    def test_serialization_and_instruction_only_handoff(self):
        program = fresh_program("A learned instruction")
        program.correct.demos = [example()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            program.save(str(path))
            loaded = fresh_program()
            loaded.load(str(path))
        self.assertEqual(loaded.correct.signature.instructions, "A learned instruction")
        self.assertEqual(len(loaded.correct.demos), 1)
        handoff = fresh_program(loaded.correct.signature.instructions)
        self.assertEqual(handoff.correct.demos, [])

    def test_failed_inference_does_not_count_as_an_identity_prediction(self):
        program = Mock()
        program.batch.return_value = [None]
        with self.assertRaises(RuntimeError):
            evaluation.evaluate(program, [example()])

    def test_cell_accounting(self):
        rows = [{"model": "trba", "dataset": "SVT", "baseline_correct": before,
                 "candidate_correct": after} for before, after in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        report = evaluation.summarize_rows(rows)
        self.assertEqual((report["fixes"], report["harms"], report["net_correct"]), (1, 1, 0))
        self.assertEqual((report["wins"], report["ties"], report["losses"]), (0, 1, 0))

    def test_resume_preserves_metadata_and_rejects_stale_fingerprints_or_programs(self):
        target = {"experiment_sha256": "new"}
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "HERE", Path(directory)), \
                patch.object(artifacts, "fingerprint", return_value=target):
            output = artifacts.job_dir("trba", "common", "base")
            output.mkdir(parents=True)
            fresh_program().save(str(output / "program.json"))
            metadata = {"status": "completed", "model": "trba", "group": "common", "stage": "base",
                        "experiment_sha256": "new", "program_sha256": storage.sha256_path(output / "program.json")}
            storage.write_json(output / "metadata.json", metadata)
            original = (output / "metadata.json").read_bytes()
            self.assertEqual(artifacts.verify_job("trba", "common", "base"), metadata)
            self.assertEqual((output / "metadata.json").read_bytes(), original)
            stale = {**metadata, "experiment_sha256": "old"}
            storage.write_json(output / "metadata.json", stale)
            stale_bytes = (output / "metadata.json").read_bytes()
            with self.assertRaises(ValueError):
                artifacts.verify_job("trba", "common", "base")
            self.assertEqual((output / "metadata.json").read_bytes(), stale_bytes)
            storage.write_json(output / "metadata.json", metadata)
            bad = {"experiment_sha256": "changed-transport"}
            with patch.object(artifacts, "fingerprint", return_value=bad), self.assertRaises(ValueError):
                artifacts.verify_job("trba", "common", "base")
            (output / "program.json").write_text((output / "program.json").read_text() + " ")
            with self.assertRaises(ValueError):
                artifacts.verify_job("trba", "common", "base")


class NativeOptimizerTests(unittest.TestCase):
    def test_gepa_stops_when_reflection_fails(self):
        ex = example()
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": "SPED"}')
        with tempfile.TemporaryDirectory() as directory, patch("pipeline.openai_lm.sdk_client", return_value=client), \
                dspy.context(lm=LunaLM("student", 32, 0), adapter=dspy.JSONAdapter(use_native_function_calling=False)):
            proposer = EspoTemplateProposer(Mock(side_effect=RuntimeError("API unavailable")), [ex], directory)
            optimized = GEPA(metric=gepa_metric, max_metric_calls=10, reflection_minibatch_size=1,
                             instruction_proposer=proposer, reflection_lm=LunaLM("reflection", 1200, 1),
                             gepa_kwargs={"stop_callbacks": lambda state: proposer.failures > 0},
                             num_threads=1, track_stats=True).compile(fresh_program(), trainset=[ex], valset=[ex])
        self.assertEqual(proposer.failures, 1)
        self.assertEqual(proposer.calls, 0)
        self.assertEqual(len(optimized.detailed_results.candidates), 1)
        self.assertLess(client.chat.completions.create.call_count, 10)

    def test_native_gepa_continues_after_invalid_pattern_count(self):
        ex = example()
        client = Mock()
        def respond(**request):
            content = str(request["messages"])
            return completion(json.dumps({"corrected_text": "SPEED" if "REVISED" in content else "SPED"}))
        client.chat.completions.create.side_effect = respond
        lm = Mock(side_effect=[[json.dumps({"patterns": diagnosis()["patterns"] * 2})],
                               [json.dumps(diagnosis())], ['{"new_instruction": "REVISED"}']])
        with tempfile.TemporaryDirectory() as directory, patch("pipeline.openai_lm.sdk_client", return_value=client), \
                dspy.context(lm=LunaLM("student", 32, 0), adapter=OCRJSONAdapter()):
            proposer = EspoTemplateProposer(lm, [ex], directory)
            optimized = GEPA(metric=gepa_metric, max_metric_calls=12, reflection_minibatch_size=1,
                             instruction_proposer=proposer, reflection_lm=LunaLM("reflection", 1200, 1),
                             gepa_kwargs={"stop_callbacks": lambda state: proposer.failures > 0},
                             num_threads=1, track_stats=True).compile(fresh_program(), trainset=[ex], valset=[ex])
        self.assertEqual(proposer.rejected_proposals, 1)
        self.assertEqual(proposer.failures, 0)
        self.assertEqual(proposer.calls, 1)
        self.assertEqual(optimized.detailed_results.val_aggregate_scores[-1], 1)

    def test_bootstrap_never_accepts_an_incorrect_generated_trace(self):
        ex = example()
        client = Mock()
        client.chat.completions.create.return_value = completion('{"corrected_text": "SPED"}')
        with patch("pipeline.openai_lm.sdk_client", return_value=client), dspy.context(lm=LunaLM("student", 32, 0), adapter=dspy.JSONAdapter(use_native_function_calling=False)):
            program = dspy.BootstrapFewShot(metric=bootstrap_metric, max_bootstrapped_demos=1,
                                           max_labeled_demos=0, max_rounds=1, max_errors=1).compile(
                                               fresh_program(), trainset=[ex])
        self.assertEqual(program.correct.demos, [])

    def test_runner_stages_and_candidate_serialization_offline(self):
        ex = example()
        client = Mock()

        def respond(**request):
            content = "\n".join(str(m["content"]) for m in request["messages"])
            if "Observed errors (JSON):" in content:
                return completion(json.dumps(diagnosis()))
            if "Structured error diagnosis (JSON):" in content:
                return completion('{"new_instruction": "REVISED: Correct the OCR input."}')
            output = "SPEED" if "REVISED" in content else "SPED"
            return completion(json.dumps({"corrected_text": output}))

        def small_gepa(**kwargs):
            kwargs.pop("auto")
            kwargs["max_metric_calls"] = 10
            return GEPA(**kwargs)

        settings = {**SETTINGS, "num_threads": 1, "reflection_minibatch_size": 1,
                    "bootstrap_candidates": 1, "bootstrap_max_demos": 1}
        client.chat.completions.create.side_effect = respond
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "offline_manifest.csv"
            manifest.write_text("split,model,dataset\n", encoding="utf-8")
            with patch.object(config, "HERE", Path(directory)), patch.object(config, "SETTINGS", settings), \
                    patch.object(config, "DATA_MANIFEST", manifest), \
                    patch.object(training, "load_scope_examples", return_value=([ex], [ex])), \
                    patch.object(training, "GEPA", side_effect=small_gepa), \
                    patch("pipeline.openai_lm.sdk_client", return_value=client):
                for stage in config.STAGES:
                    result = training.train_scope("trba", "common", stage)
                    self.assertEqual(result["status"], "completed")
                    self.assertGreaterEqual(result["runtime_seconds"], 0)
                    self.assertEqual(artifacts.verify_job("trba", "common", stage)["stage"], stage)
                for stage in ("gepa", "bootstrap"):
                    selection = json.loads((artifacts.job_dir("trba", "common", stage) / "selection.json").read_text())
                    candidates = selection.get("instruction_candidates", selection.get("candidates"))
                    self.assertTrue(candidates)
                    for candidate in candidates:
                        self.assertTrue((artifacts.job_dir("trba", "common", stage) / candidate["program"]).is_file())
                path = artifacts.job_dir("trba", "common", "bootstrap") / "program.json"
                path.write_text(path.read_text() + " ")
                with self.assertRaises(ValueError):
                    artifacts.verify_job("trba", "common", "bootstrap")

    def test_test_inference_requires_complete_freeze(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "HERE", Path(directory)):
            with self.assertRaises(RuntimeError):
                evaluation.run_test()
            self.assertFalse((Path(directory) / "results").exists())

    def test_gepa_custom_proposer_then_native_bootstrap_offline(self):
        ex = example()
        client = Mock()

        def respond(**request):
            content = "\n".join(str(m["content"]) for m in request["messages"])
            if "Observed errors (JSON):" in content:
                errors = json.loads(content.split("Observed errors (JSON):\n", 1)[1].split("\n\nReturn", 1)[0])
                return completion(json.dumps(diagnosis(tuple(r["error_id"] for r in errors))))
            if "Structured error diagnosis (JSON):" in content:
                return completion(json.dumps({"new_instruction": "REVISED: Correct the OCR input. Output only corrected_text."}))
            corrected = "SPEED" if "REVISED" in content else "SPED"
            return completion(json.dumps({"corrected_text": corrected}))

        client.chat.completions.create.side_effect = respond
        student = LunaLM("student", 32, 0)
        reflection = LunaLM("reflection", 1200, 1)
        with tempfile.TemporaryDirectory() as directory, patch("pipeline.openai_lm.sdk_client", return_value=client), \
                dspy.context(lm=student, adapter=dspy.JSONAdapter(use_native_function_calling=False)):
            proposer = EspoTemplateProposer(reflection, [ex], Path(directory) / "reflection")
            optimized = GEPA(metric=gepa_metric, max_metric_calls=10, reflection_minibatch_size=1,
                             reflection_lm=reflection, instruction_proposer=proposer,
                             num_threads=1, track_stats=True, seed=42).compile(
                                 fresh_program(), trainset=[ex], valset=[ex])
            self.assertGreater(proposer.calls, 0)
            self.assertEqual(optimized.detailed_results.val_aggregate_scores[-1], 1)
            self.assertEqual(exact_metric(ex, optimized(ocr_prediction="SPED")), 1)
            handoff = fresh_program(optimized.correct.signature.instructions)
            boot = dspy.BootstrapFewShotWithRandomSearch(
                metric=bootstrap_metric, max_bootstrapped_demos=1, max_labeled_demos=1,
                num_candidate_programs=1, max_rounds=1, num_threads=1, max_errors=1).compile(
                    handoff, trainset=[ex], valset=[ex])
            verify_program(boot, [ex])
            for candidate in boot.candidate_programs:
                verify_program(candidate["program"], [ex])
            self.assertEqual(exact_metric(ex, boot(ocr_prediction="SPED")), 1)


if __name__ == "__main__":
    unittest.main()
