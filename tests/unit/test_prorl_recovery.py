from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.infra.farmshare import (
    SlurmArraySpec,
    probe_commands,
    render_slurm_array,
    shard_indices,
)
from polaris.prorl_recovery.diversity import diversity_diagnostics
from polaris.prorl_recovery.logprob import (
    TrajectoryForScoring,
    score_trajectories_batched,
)
from polaris.prorl_recovery.phase3 import (
    assign_bucket,
    assign_high_base_logprob,
    derive_phase3_input_set,
    materialize_phase3_trajectories,
)
from polaris.prorl_recovery.protocol import (
    KARAN_DU_REPLICATION_GATE,
    memory_transplant_claim_passes,
    recoverable_fraction,
)
from polaris.prorl_recovery.orchestration import (
    aggregate_phase0_gate,
    aggregate_rws_exact_phase0_gate,
    aggregate_phase1,
    audit_recovery_cells,
    cell_command,
    derive_prorl_only_problem_ids,
    phase0_cells,
    phase0_rws_exact_cells,
    phase1_cells,
    phase2_cells,
    rws_exact_cell_command,
    token_cap_after_smoke,
    token_cap_for_track,
    write_archive,
)
from polaris.registry import resolve_model
from polaris.runners.condition_runner import get_track_config


def test_farmshare_slurm_array_renders_deterministic_shards():
    script = render_slurm_array(
        SlurmArraySpec(job_name="rf", command="python run.py --x 1", num_shards=4)
    )

    assert "#SBATCH --array=0-3%4" in script
    assert 'export POLARIS_SHARD_ID="${SLURM_ARRAY_TASK_ID}"' in script
    assert 'export POLARIS_NUM_SHARDS="4"' in script
    assert 'export HF_HOME="/scratch/users/$USER/.cache/huggingface"' in script
    assert 'export PATH="/scratch/users/$USER/polaris/envs/polaris/bin:$PATH"' in script
    assert "micromamba run -p" not in script
    assert shard_indices(10, 2, 4) == [2, 6]


def test_farmshare_slurm_array_can_render_phase_matrices():
    script = render_slurm_array(
        SlurmArraySpec(
            job_name="phase1",
            command="python scripts/run_condition.py --track math500",
            num_shards=4,
            array_tasks=16,
            max_concurrent=4,
        )
    )

    assert "#SBATCH --array=0-15%4" in script
    assert 'POLARIS_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"' in script


def test_farmshare_probe_includes_gpu_qos_and_nvidia_smi():
    commands = probe_commands(include_gpu=True)

    joined = " && ".join(commands)
    assert "sacctmgr show qos" in joined
    assert "--qos=gpu" in joined
    assert "nvidia-smi -L" in joined


def test_reasoning_gym_graph_color_size_track_generates_fixed_vertex_count():
    pytest.importorskip("reasoning_gym")
    cfg = get_track_config("reasoning_gym_graph_color_n18")

    problem = cfg.dataset_loader(0, 1)[0]
    payload = json.loads(problem.answer)

    assert cfg.benchmark == "Reasoning Gym graph_color n=18"
    assert payload["task"] == "graph_color"
    assert payload["entry"]["metadata"]["num_vertices"] == 18


def test_prorl_model_registry_pins_revisions():
    base = resolve_model("deepseek-r1-distill-qwen-1.5b")
    prorl = resolve_model("nemotron-prorl-v2")
    brorl = resolve_model("nemotron-brorl")

    assert base.revision == "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    assert base.revision_commit == "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    assert prorl.revision == "main"
    assert prorl.revision_commit == "c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0"
    assert prorl.artifact_etags["model-00001-of-00002.safetensors"].startswith(
        "f477d140"
    )
    assert brorl.revision == "brorl"


def test_prorl_recovery_phase_cells_and_commands_are_locked():
    phase1 = phase1_cells(
        root="/scratch/users/$USER/polaris/runs/prorl_recovery",
        problem_count=20,
        tracks=("math500",),
        num_shards=4,
    )
    phase2 = phase2_cells(
        root="/scratch/users/$USER/polaris/runs/prorl_recovery",
        problem_count=30,
        tracks=("reasoning_gym_boxnet",),
        num_shards=4,
    )
    phase0 = phase0_cells(root="/scratch/users/$USER/polaris/runs/prorl_recovery")

    assert len(phase1) == 16
    assert {cell.samples_per_problem for cell in phase1} == {128}
    assert all(f"shard-{cell.shard_id}.sqlite" in cell.cache_path for cell in phase1)
    phase1_pilot = phase1_cells(
        root="/scratch/users/$USER/polaris/runs/prorl_recovery",
        problem_count=20,
        tracks=("math500",),
        num_shards=4,
        samples_per_problem=16,
    )
    assert {cell.samples_per_problem for cell in phase1_pilot} == {16}
    assert len(phase0) == 4
    assert phase0[0].archive_kind == "rws_math_direct"
    assert {cell.backend for cell in phase0} == {"hf"}
    assert "--backend" in cell_command(phase0[0])
    assert "hf" in cell_command(phase0[0])
    phase1_rg = phase1_cells(
        root="/scratch/users/$USER/polaris/runs/prorl_recovery",
        problem_count=20,
        tracks=("reasoning_gym_boxnet",),
        num_shards=4,
        samples_per_problem=16,
    )
    assert {cell.archive_kind for cell in phase1_rg} == {"reasoning_gym_boxnet_direct"}
    assert "reasoning_gym_boxnet_direct.json" in " ".join(cell_command(phase1_rg[0]))
    assert any(cell.rung == "rung2_bon_t12_k1024" for cell in phase2)
    assert all(f"shard-{cell.shard_id}.sqlite" in cell.cache_path for cell in phase2)
    assert {
        cell.archive_kind
        for cell in phase2
        if cell.rung and cell.rung.startswith(("rung0", "rung1", "rung2", "rung3", "rung4"))
    } == {"reasoning_gym_boxnet_direct"}
    assert {
        cell.archive_kind
        for cell in phase2
        if cell.rung and cell.rung.startswith(("rung5", "rung6", "rung7"))
    } == {"reasoning_gym_seed_archive"}
    rung7 = next(cell for cell in phase2 if cell.rung == "rung7_full_memory")
    cmd = cell_command(rung7)
    assert "--samples-per-problem" in cmd
    assert "128" in cmd
    assert "--memory-mode" in cmd
    assert "--vllm-dtype" in cmd
    assert "bfloat16" in cmd
    assert "$POLARIS_REPO_DIR/runs/generated/prorl_recovery_archives/reasoning_gym_seed_archive.json" in cmd


def test_exact_rws_phase0_cells_match_upstream_shard_seed_matrix():
    cells = phase0_rws_exact_cells(root="/scratch/users/$USER/polaris/runs/prorl_recovery")

    assert len(cells) == 40
    assert {cell.num_shards for cell in cells} == {5}
    assert {cell.num_seeds for cell in cells} == {8}
    assert [(cell.batch_idx, cell.seed) for cell in cells[:10]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 0),
        (1, 1),
    ]
    assert cells[-1].batch_idx == 4
    assert cells[-1].seed == 7
    assert cells[0].expected_csv.endswith(
        "qwen_math/qwen_math_math_base_power_samp_results_10_0.25_0_0.csv"
    )


def test_exact_rws_phase0_command_runs_upstream_script_not_polaris_wrapper():
    cell = phase0_rws_exact_cells(root="/runs/prorl")[0]
    cmd = rws_exact_cell_command(cell, repo_dir="/repo")
    joined = " ".join(cmd)

    assert "upstream/reasoning-with-sampling/llm_experiments" in joined
    assert "power_samp_math.py" in joined
    assert sys.executable in joined
    assert "HF_HUB_OFFLINE=1" in joined
    assert "&& python power_samp_math.py" not in joined
    assert "--batch_idx 0" in joined
    assert "--seed 0" in joined
    assert "scripts/run_condition.py" not in joined


def test_phase2_planner_filters_to_prorl_only_problem_ids():
    rows = [
        {
            "checkpoint": "deepseek-r1-distill-qwen-1.5b",
            "task_family": "reasoning_gym_graph_color",
            "problem_id": "p0",
            "sample_idx": 0,
            "verified": False,
        },
        {
            "checkpoint": "nemotron-prorl-v2",
            "task_family": "reasoning_gym_graph_color",
            "problem_id": "p0",
            "sample_idx": 0,
            "verified": True,
        },
        {
            "checkpoint": "deepseek-r1-distill-qwen-1.5b",
            "task_family": "reasoning_gym_graph_color",
            "problem_id": "p1",
            "sample_idx": 0,
            "verified": True,
        },
        {
            "checkpoint": "nemotron-brorl",
            "task_family": "reasoning_gym_graph_color",
            "problem_id": "p1",
            "sample_idx": 0,
            "verified": True,
        },
    ]

    selected = derive_prorl_only_problem_ids(rows)
    assert selected == {"reasoning_gym_graph_color": ("p0",)}

    cells = phase2_cells(
        root="/runs/prorl",
        problem_count=10,
        tracks=("reasoning_gym_graph_color",),
        num_shards=2,
        selected_problem_ids_by_track=selected,
    )
    assert all(cell.selected_problem_ids == ("p0",) for cell in cells)
    assert "--problem-ids" in cell_command(cells[0])
    assert "p0" in cell_command(cells[0])


def test_token_caps_are_track_level_and_double_once_after_smoke():
    assert token_cap_for_track("phase0", "math500") == 3072
    assert token_cap_for_track("phase1", "math500") == 4096
    assert token_cap_for_track("phase1", "gpqa_diamond") == 2048
    assert token_cap_for_track("phase2", "reasoning_gym_boxnet") == 8192
    assert token_cap_after_smoke("math500", cap_hit_rate=0.05) == 4096
    assert token_cap_after_smoke("math500", cap_hit_rate=0.051) == 8192
    assert token_cap_after_smoke("math500", cap_hit_rate=0.9, already_doubled=True) == 4096


def test_cloudrift_preflight_and_estimator_fail_closed(tmp_path):
    from polaris.infra.cloudrift import (
        cloudrift_environment,
        estimate_cloudrift_cost,
        recommended_gpu_order,
    )
    from polaris.infra.preflight import (
        PaidRunPreflight,
        PreflightError,
        validate_paid_run_preflight,
    )

    spec = PaidRunPreflight(
        run_kind="cloudrift",
        artifact_dir=tmp_path / "run",
        cache_path=tmp_path / "trajectories.sqlite",
        split=(0, 2),
        seed=17,
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        backend="vllm",
        estimated_dollar_cost=0.25,
        cost_cap_dollars=0.50,
        user_authorized=False,
    )
    with pytest.raises(PreflightError, match="user_authorized"):
        validate_paid_run_preflight(spec)

    report = validate_paid_run_preflight(
        PaidRunPreflight(**{**spec.__dict__, "user_authorized": True})
    )
    estimate = estimate_cloudrift_cost("rtx4090", wall_clock_seconds=3600)

    assert report["run_kind"] == "cloudrift"
    assert estimate.dollars == pytest.approx(0.25)
    assert estimate_cloudrift_cost(
        "rtx4090", wall_clock_seconds=3600, hourly_rate=0.27
    ).rate_source == "explicit_ui_rate"
    assert cloudrift_environment()["HF_HOME"] == "/workspace/.cache/huggingface"
    assert recommended_gpu_order() == ("rtx4090", "v100_sxm3")


def test_write_archive_outputs_direct_and_seed_archive(tmp_path):
    direct = tmp_path / "direct.json"
    seed = tmp_path / "seed_archive.json"
    rg_direct = tmp_path / "reasoning_gym_direct.json"
    rg_seed = tmp_path / "reasoning_gym_seed_archive.json"
    rg_family = tmp_path / "reasoning_gym_family_direct.json"
    rg_graph = tmp_path / "reasoning_gym_graph_direct.json"
    rg_boxnet = tmp_path / "reasoning_gym_boxnet_direct.json"

    write_archive(direct, kind="direct")
    write_archive(seed, kind="seed_archive")
    write_archive(rg_direct, kind="reasoning_gym_direct")
    write_archive(rg_seed, kind="reasoning_gym_seed_archive")
    write_archive(rg_family, kind="reasoning_gym_family_direct")
    write_archive(rg_graph, kind="reasoning_gym_graph_color_direct")
    write_archive(rg_boxnet, kind="reasoning_gym_boxnet_direct")

    assert json.loads(direct.read_text())[0]["id"] == "direct"
    assert len(json.loads(seed.read_text())["entries"]) >= 4
    rg_entry = json.loads(rg_direct.read_text())[0]
    assert "<answer>" not in (rg_entry["prefix"] + rg_entry["suffix"])
    assert "<answer>...</answer>" not in rg_entry["prefix"]
    assert "<answer>...</answer>" not in rg_entry["suffix"]
    assert re.search(r"<answer>\s*.*?\s*</answer>", rg_entry["prefix"] + rg_entry["suffix"], re.S) is None
    assert "\\boxed{}" not in rg_entry["suffix"]
    rg_seed_payload = json.loads(rg_seed.read_text())
    assert len(rg_seed_payload["entries"]) >= 4
    assert all("<answer>" not in (entry["prefix"] + entry["suffix"]) for entry in rg_seed_payload["entries"])
    assert all("<answer>...</answer>" not in entry["prefix"] for entry in rg_seed_payload["entries"])
    assert all("<answer>...</answer>" not in entry["suffix"] for entry in rg_seed_payload["entries"])
    assert all("\\boxed{}" not in entry["suffix"] for entry in rg_seed_payload["entries"])
    assert "kinship" in json.loads(rg_family.read_text())[0]["prefix"]
    assert "JSON object" in json.loads(rg_graph.read_text())[0]["prefix"]
    assert "JSON list" in json.loads(rg_boxnet.read_text())[0]["prefix"]


def test_reasoning_gym_verifier_extracts_answer_tag(monkeypatch):
    import types

    seen = {}

    class FakeDataset:
        def score_answer(self, answer, entry):
            seen["answer"] = answer
            return 1.0 if answer == '{"move": "left"}' else 0.0

    fake_reasoning_gym = types.ModuleType("reasoning_gym")
    fake_reasoning_gym.create_dataset = lambda task, size, seed: FakeDataset()
    fake_utils = types.ModuleType("reasoning_gym.utils")
    fake_utils.extract_answer = lambda completion: '{"move": "left"}'
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    monkeypatch.setitem(sys.modules, "reasoning_gym.utils", fake_utils)

    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    result = score_reasoning_gym(
        'reasoning text <answer>{"move": "left"}</answer>',
        json.dumps({"task": "boxnet", "entry": {}, "answer": ""}),
    )

    assert result["passed"] is True
    assert seen["answer"] == '{"move": "left"}'


def test_reasoning_gym_verifier_uses_last_non_placeholder_answer_tag(monkeypatch):
    import types

    seen = {}

    class FakeDataset:
        def score_answer(self, answer, entry):
            seen["answer"] = answer
            return 1.0 if answer == "daughter-in-law" else 0.0

    fake_reasoning_gym = types.ModuleType("reasoning_gym")
    fake_reasoning_gym.create_dataset = lambda task, size, seed: FakeDataset()
    fake_utils = types.ModuleType("reasoning_gym.utils")
    fake_utils.extract_answer = lambda completion: "..."
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    monkeypatch.setitem(sys.modules, "reasoning_gym.utils", fake_utils)

    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    result = score_reasoning_gym(
        "Prompt says <answer>...</answer>. Model says <answer>daughter-in-law</answer>.",
        json.dumps({"task": "family_relationships", "entry": {}, "answer": ""}),
    )

    assert result["passed"] is True
    assert seen["answer"] == "daughter-in-law"


def test_reasoning_gym_verifier_extracts_answer_marker(monkeypatch):
    import types

    seen = {}

    class FakeDataset:
        def score_answer(self, answer, entry):
            seen["answer"] = answer
            return 1.0 if answer == "grandfather" else 0.0

    fake_reasoning_gym = types.ModuleType("reasoning_gym")
    fake_reasoning_gym.create_dataset = lambda task, size, seed: FakeDataset()
    fake_utils = types.ModuleType("reasoning_gym.utils")
    fake_utils.extract_answer = lambda completion: completion
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    monkeypatch.setitem(sys.modules, "reasoning_gym.utils", fake_utils)

    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    result = score_reasoning_gym(
        "Prompt text.\n</think>\n\nAnswer: Grandfather\nextra reasoning ignored",
        json.dumps({"task": "family_relationships", "entry": {}, "answer": ""}),
    )

    assert result["passed"] is True
    assert seen["answer"] == "grandfather"


def test_reasoning_gym_verifier_extracts_json_after_answer_marker_prose(monkeypatch):
    import types

    seen = {}

    class FakeDataset:
        def score_answer(self, answer, entry):
            seen["answer"] = answer
            return 1.0 if answer == '{"0": 1, "1": 2}' else 0.0

    fake_reasoning_gym = types.ModuleType("reasoning_gym")
    fake_reasoning_gym.create_dataset = lambda task, size, seed: FakeDataset()
    fake_utils = types.ModuleType("reasoning_gym.utils")
    fake_utils.extract_answer = lambda completion: completion
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    monkeypatch.setitem(sys.modules, "reasoning_gym.utils", fake_utils)

    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    result = score_reasoning_gym(
        'Prompt.\n</think>\n\nAnswer:\nThe required JSON mapping is:\n{"0": 1, "1": 2}\nThis works.',
        json.dumps({"task": "graph_color", "entry": {}, "answer": ""}),
    )

    assert result["passed"] is True
    assert seen["answer"] == '{"0": 1, "1": 2}'


def test_reasoning_gym_verifier_fails_closed_on_bad_json_shape(monkeypatch):
    import types

    class FakeDataset:
        def score_answer(self, answer, entry):
            raise AttributeError("bad answer shape")

    fake_reasoning_gym = types.ModuleType("reasoning_gym")
    fake_reasoning_gym.create_dataset = lambda task, size, seed: FakeDataset()
    fake_utils = types.ModuleType("reasoning_gym.utils")
    fake_utils.extract_answer = lambda completion: "[1, 2, 3]"
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    monkeypatch.setitem(sys.modules, "reasoning_gym.utils", fake_utils)

    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    result = score_reasoning_gym(
        "bad <answer>[1, 2, 3]</answer>",
        json.dumps({"task": "boxnet", "entry": {}, "answer": ""}),
    )

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["scorer_error"].startswith("AttributeError")


def test_reasoning_gym_game_of_life_halting_uses_exact_boolean():
    from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym

    reference = json.dumps(
        {
            "task": "game_of_life_halting",
            "entry": {"answer": "False"},
            "answer": "False",
        }
    )

    assert score_reasoning_gym("answer: False", reference)["passed"] is True
    assert score_reasoning_gym("answer: True", reference)["passed"] is False
    assert score_reasoning_gym("answer: deliberately wrong", reference)["passed"] is False


def test_hf_generator_passes_revision_to_transformers(monkeypatch):
    seen = {"tokenizer": None, "model": None}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class FakeModel:
        def to(self, device):
            seen["model_to"] = device
            return self

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        seen["tokenizer"] = (args, kwargs)
        return FakeTokenizer()

    def fake_model_from_pretrained(*args, **kwargs):
        seen["model"] = (args, kwargs)
        return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
    )
    fake_rws = types.ModuleType("polaris.vendored.rws.power_samp_utils")
    fake_rws.AutoregressiveSampler = lambda model, tokenizer, device: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "polaris.vendored.rws.power_samp_utils", fake_rws)

    from polaris.infra.serving.hf import RWSGenerator

    gen = RWSGenerator(
        model_id="fake-model",
        revision="v2",
        seed=1,
        torch_dtype="float32",
        score_segments_mode="cached_decode",
        device_map_auto=False,
    )

    assert gen.revision == "v2"
    assert seen["tokenizer"][1]["revision"] == "v2"
    assert seen["model"][1]["revision"] == "v2"
    assert str(seen["model"][1]["torch_dtype"]) == "torch.float32"
    assert seen["model"][1]["device_map"] is None
    assert seen["model_to"] == "cuda"
    assert gen.runtime_metadata()["torch_dtype"] == "float32"
    assert gen.runtime_metadata()["score_segments_mode"] == "cached_decode"
    assert gen.runtime_metadata()["device_map_auto"] is False


def test_vllm_generator_passes_revision_to_loader(monkeypatch):
    seen = {"tokenizer": None, "llm": None}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text):
            return [1]

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    class FakeLLM:
        def __init__(self, **kwargs):
            seen["llm"] = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        seen["tokenizer"] = (args, kwargs)
        return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    from polaris.infra.serving.vllm import VLLMGenerator

    gen = VLLMGenerator(model_id="fake-model", revision="brorl")

    assert gen.revision == "brorl"
    assert seen["tokenizer"][1]["revision"] == "brorl"
    assert seen["llm"]["revision"] == "brorl"
    assert seen["llm"]["tokenizer_revision"] == "brorl"


def test_run_condition_preflight_records_model_revision(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_condition.py",
            "--track",
            "math500",
            "--model-key",
            "nemotron-prorl-v2",
            "--condition",
            "greedy",
            "--archive",
            str(tmp_path / "missing.json"),
            "--split",
            "0",
            "1",
            "--polaris-source-hash",
            "dev",
            "--preregistration-anchor",
            "TODO.md#prorl-recovery",
            "--out",
            str(tmp_path / "run"),
            "--preflight-only",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model_revision"] == "main"
    assert payload["model_revision_commit"] == "c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0"
    assert payload["model_artifact_etags"]["config.json"].startswith("4b1c5c5")


def test_run_condition_preflight_accepts_budget_and_shard_args(tmp_path):
    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps(
            [
                {
                    "id": "direct",
                    "prefix": "Q: ",
                    "suffix": " A:",
                    "descriptor_hint": "direct",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_condition.py",
            "--track",
            "math500",
            "--model-key",
            "deepseek-r1-distill-qwen-1.5b",
            "--condition",
            "bon_temp1",
            "--archive",
            str(archive),
            "--split",
            "0",
            "3",
            "--shard-id",
            "1",
            "--num-shards",
            "4",
            "--samples-per-problem",
            "16",
            "--sampling-temperature",
            "1.2",
            "--max-new-tokens",
            "64",
            "--polaris-source-hash",
            "dev",
            "--preregistration-anchor",
            "TODO.md#prorl-recovery",
            "--out",
            str(tmp_path / "run"),
            "--preflight-only",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr


def test_phase1_aggregation_writes_parquet_and_passk(tmp_path):
    run_dir = tmp_path / "phase1" / "math500" / "base" / "shard-0"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "model",
                "benchmark": "MATH500",
                "config": {"model_key": "base", "track": "math500"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "preflight.json").write_text(
        json.dumps({"backend": "vllm"}),
        encoding="utf-8",
    )
    rows = [
        {
            "problem_id": "p0",
            "sample_index": 0,
            "generation": "bad",
            "generation_token_count": 1,
            "verifier_result": {"passed": False, "score": 0.25},
        },
        {
            "problem_id": "p0",
            "sample_index": 15,
            "generation": "ok",
            "generation_token_count": 2,
            "verifier_result": {"passed": True, "score": 0.75},
        },
        {
            "problem_id": "p1",
            "sample_index": 0,
            "generation": "ok",
            "generation_token_count": 3,
            "verifier_result": {"passed": True, "score": 1.0},
        },
    ]
    with (run_dir / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = aggregate_phase1([run_dir / "candidates.jsonl"], tmp_path / "phase1_results.parquet")

    assert summary["rows"] == 3
    assert (tmp_path / "phase1_results.parquet").exists()
    import pandas as pd

    df = pd.read_parquet(tmp_path / "phase1_results.parquet")
    assert set(df["generation_backend"]) == {"vllm"}
    assert list(df["verifier_score"]) == [0.25, 0.75, 1.0]
    pass16 = next(item for item in summary["pass_at"] if item["k"] == 16)
    assert pass16["accuracy"] == 1.0
    score16 = next(item for item in summary["score_at"] if item["k"] == 16)
    assert score16["best_score_mean"] == 0.875
    assert score16["first_score_mean"] == 0.625
    assert score16["all_score_mean"] == pytest.approx((0.25 + 0.75 + 1.0) / 3)


def test_phase0_gate_aggregation_uses_registered_threshold(tmp_path):
    run_dir = tmp_path / "phase0" / "karan_du_replication" / "shard-0"
    run_dir.mkdir(parents=True)
    with (run_dir / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for idx in range(500):
            f.write(
                json.dumps(
                    {
                        "problem_id": f"p{idx}",
                        "sample_index": 0,
                        "verifier_result": {"passed": idx < 374},
                    }
                )
                + "\n"
            )

    report = aggregate_phase0_gate(
        [run_dir / "candidates.jsonl"],
        tmp_path / "phase0_gate.json",
        expected_problems=500,
    )

    assert report["accuracy"] == pytest.approx(0.748)
    assert report["passed"] is True
    assert report["complete"] is True
    assert report["gate"]["target_accuracy"] == 0.748
    assert json.loads((tmp_path / "phase0_gate.json").read_text())["passed"] is True


def test_exact_rws_phase0_aggregation_uses_mean_over_complete_seeds(tmp_path):
    import pandas as pd

    result_dir = tmp_path / "phase0" / "rws_exact" / "results" / "qwen_math"
    result_dir.mkdir(parents=True)
    for seed, correct in ((0, 374), (1, 364)):
        for batch_idx in range(5):
            rows = []
            for local_idx in range(100):
                global_idx = batch_idx * 100 + local_idx
                ok = global_idx < correct
                rows.append(
                    {
                        "question": f"q{global_idx}",
                        "correct_answer": str(global_idx),
                        "naive_completion": "",
                        "naive_answer": "",
                        "std_completion": "",
                        "std_answer": "",
                        "mcmc_completion": f"\\boxed{{{global_idx if ok else -1}}}",
                        "mcmc_answer": str(global_idx if ok else -1),
                    }
                )
            pd.DataFrame(rows).to_csv(
                result_dir
                / f"qwen_math_math_base_power_samp_results_10_0.25_{batch_idx}_{seed}.csv",
                index=False,
            )

    report = aggregate_rws_exact_phase0_gate(
        tmp_path,
        tmp_path / "rws_exact_gate.json",
        expected_seeds=2,
    )

    assert report["complete"] is True
    assert report["accuracy"] == pytest.approx((0.748 + 0.728) / 2)
    assert report["passed"] is True
    assert len(report["per_seed"]) == 2


def test_recovery_cell_audit_checks_required_files_and_row_counts(tmp_path):
    cells = phase1_cells(
        root=str(tmp_path),
        problem_count=3,
        tracks=("math500",),
        num_shards=4,
        samples_per_problem=2,
    )
    cell = cells[1]
    out = Path(cell.out_dir)
    out.mkdir(parents=True)
    for name in [
        "manifest.json",
        "archive.json",
        "metrics.json",
        "costs.json",
        "rollouts.json",
        "preflight.json",
        "environment.json",
        "run_plan_cell.json",
        "audit.md",
    ]:
        (out / name).write_text("{}", encoding="utf-8")
    for name, rows in {
        "candidates.jsonl": 2,
        "scores.jsonl": 2,
        "selected.jsonl": 1,
    }.items():
        (out / name).write_text("\n".join("{}" for _ in range(rows)) + "\n", encoding="utf-8")

    report = audit_recovery_cells([cell])

    assert report["passed"] is True
    assert report["totals"]["expected_candidates"] == 2
    assert report["totals"]["candidate_rows"] == 2

    (out / "scores.jsonl").write_text("{}\n", encoding="utf-8")
    failed = audit_recovery_cells([cell])
    assert failed["passed"] is False
    assert "scores row count" in failed["failures"][0]["reason"]


def test_phase3_input_derivation_is_deterministic_and_filters_base_rung7_successes():
    phase1 = [
        {"track": "math500", "problem_id": "b", "model_key": "nemotron-brorl", "passed": True},
        {"track": "math500", "problem_id": "a", "model_key": "nemotron-prorl-v2", "passed": True},
        {"track": "gpqa_diamond", "problem_id": "c", "model_key": "nemotron-prorl-v2", "passed": False},
    ]
    rung7 = [{"track": "math500", "problem_id": "b", "passed": True}]

    rows = derive_phase3_input_set(phase1_rows=phase1, rung7_rows=rung7)

    assert rows == [
        {
            "task_family": "math500",
            "problem_id": "a",
            "checkpoint": "nemotron-prorl-v2",
            "phase1_source_id": None,
            "base_rung7_failed": True,
        }
    ]


def test_phase3_trajectory_materialization_uses_successful_raw_candidate():
    phase3_rows = [
        {
            "task_family": "math500",
            "problem_id": "a",
            "checkpoint": "nemotron-prorl-v2",
            "base_rung7_failed": True,
        }
    ]
    candidate_rows = [
        {
            "task_family": "math500",
            "checkpoint": "nemotron-prorl-v2",
            "problem_id": "a",
            "candidate_id": "wrong",
            "sample_index": 0,
            "prompt_text": "Q",
            "generation": "bad",
            "verifier_result": {"passed": False},
        },
        {
            "task_family": "math500",
            "checkpoint": "nemotron-prorl-v2",
            "problem_id": "a",
            "candidate_id": "right",
            "sample_index": 3,
            "prompt_text": "Q",
            "generation": "good",
            "verifier_result": {"passed": True, "score": 1.0},
        },
    ]

    rows = materialize_phase3_trajectories(
        phase3_rows=phase3_rows,
        candidate_rows=candidate_rows,
    )

    assert rows == [
        {
            "row_id": "right",
            "task_family": "math500",
            "problem_id": "a",
            "checkpoint": "nemotron-prorl-v2",
            "prompt_text": "Q",
            "response_text": "good",
            "source_candidate_id": "right",
            "sample_index": 3,
        }
    ]


def test_rf_and_bucket_rules_are_preregistered():
    assert recoverable_fraction(
        base_accuracy=0.40, frozen_inference_accuracy=0.55, trained_accuracy=0.70
    ) == pytest.approx(0.5)
    assert KARAN_DU_REPLICATION_GATE.accepts(0.75)
    assert not KARAN_DU_REPLICATION_GATE.accepts(0.70)
    assert memory_transplant_claim_passes(
        transplant_pass_at_16=0.55, control_pass_at_16=0.44
    )
    assert assign_high_base_logprob(
        base_generated_lp_means={"math500": [-5.0, -4.0, -3.0, -2.0]},
        trace_task_family="math500",
        trace_lp_base_mean=-2.5,
    )
    assert assign_bucket({"high_base_logprob": True}) == "search_limited"
    assert assign_bucket(
        {
            "high_base_logprob": False,
            "prompt_variant_solves": True,
            "memory_transplant_passes": True,
        }
    ) == "prompt_conditional"
    assert assign_bucket(
        {
            "high_base_logprob": False,
            "prompt_variant_solves": False,
            "memory_transplant_passes": True,
        }
    ) == "memory_conditional"
    assert assign_bucket({}) == "weight_only"


def test_logprob_scorer_batches_trajectories_with_scorebatch_contract():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) % 10 for c in text]

    class FakeSampler:
        tokenizer = FakeTokenizer()

        def __init__(self):
            self.calls = []

        def score_segments(self, prefix_ids_batch, target_segments_batch, *, temperature):
            from polaris.infra.serving import ScoreBatch

            self.calls.append((prefix_ids_batch, target_segments_batch, temperature))
            lp_tokens = [[-1.0 for _ in target] for target in target_segments_batch]
            return ScoreBatch(
                lp_norm=[sum(tokens) for tokens in lp_tokens],
                lp_unnorm=[sum(tokens) for tokens in lp_tokens],
                lp_norm_tokens=lp_tokens,
                lp_unnorm_tokens=lp_tokens,
            )

    sampler = FakeSampler()
    rows = score_trajectories_batched(
        sampler=sampler,
        trajectories=[
            TrajectoryForScoring("r1", "math500", "p1", "Q", "abc"),
            TrajectoryForScoring("r2", "math500", "p2", "Q", "de"),
        ],
        batch_size=2,
    )

    assert len(sampler.calls) == 1
    assert sampler.calls[0][2] == 1.0
    assert rows[0]["lp_base_sum"] == -3.0
    assert rows[0]["lp_base_mean"] == -1.0
    assert rows[0]["token_count"] == 3
    assert rows[0]["token_logprobs"] == [-1.0, -1.0, -1.0]


def test_diversity_diagnostics_records_entropy_and_hash_counts():
    diag = diversity_diagnostics(
        [
            {"extracted_answer": "A", "descriptor": "x", "generation": "trace one", "generation_token_count": 3},
            {"extracted_answer": "B", "descriptor": "x", "generation": "trace two", "generation_token_count": 5},
        ]
    )

    assert diag["unique_answer_count"] == 2
    assert diag["trace_hash_count"] == 2
    assert diag["response_length"]["mean"] == 4.0
