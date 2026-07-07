from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.core.fork_search import ForkSearchConfig, run_entropy_gated_fork_search
from polaris.core.inference import Candidate
from polaris.core.memory import distill_reasoning_gym_strategy
from polaris.core.repair import RepairConfig, run_verifier_guided_repair
from polaris.evals.datasets.math500 import Problem
from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym
from polaris.proicl.archive import build_cross_task_curriculum_archive
from polaris.proicl.protocol import ArchiveScope
from polaris.proicl.analysis import compute_proicl_decomposition
from polaris.proicl.analysis import write_proicl_decomposition_by_track
from polaris.proicl.artifact_audit import write_proicl_artifact_audit
from polaris.proicl.naming import (
    make_proicl_run_identity,
    panel_slug,
    standard_run_root,
    write_run_index,
)
from polaris.proicl.launcher import (
    build_signal_cells,
    cell_complete,
    required_artifacts,
    run_cells,
    run_condition_command,
)
from polaris.proicl import launcher as launcher_mod
from polaris.proicl.run_graph import (
    PROICL_PRIMARY_CONDITIONS,
    build_proicl_run_graph,
)
from polaris.runners.condition_runner import run_condition
from scripts.proicl.run_proicl_signal import (
    DEFAULT_SIGNAL_TRACKS,
    _archive_is_live,
    _gepa_archive_command,
    _partition_cells,
    configure_flow_cache_env,
)


def test_proicl_run_graph_exposes_complete_ladder_plus_prorl_reference():
    plan = build_proicl_run_graph(
        root="/runs/proicl",
        tracks=("reasoning_gym_boxnet",),
        problem_count=100,
        rollout_budget=64,
        num_shards=2,
    )

    assert PROICL_PRIMARY_CONDITIONS == (
        "base_greedy",
        "bon_temp1",
        "mcmc_only",
        "mixed_alpha_mcmc",
        "fork_search",
        "gepa_only",
        "gepa_mcmc",
        "gepa_mcmc_repair",
        "gepa_mcmc_fork_repair",
        "gepa_mcmc_fork_repair_memory",
        "gepa_mcmc_memory",
        "prorl_v2_greedy",
    )
    assert {cell["proicl_condition"] for cell in plan["cells"]} == set(
        PROICL_PRIMARY_CONDITIONS
    )

    base = next(c for c in plan["cells"] if c["proicl_condition"] == "base_greedy")
    mcmc = next(c for c in plan["cells"] if c["proicl_condition"] == "mcmc_only")
    mixed = next(c for c in plan["cells"] if c["proicl_condition"] == "mixed_alpha_mcmc")
    fork = next(c for c in plan["cells"] if c["proicl_condition"] == "fork_search")
    gepa = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_only")
    composed = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc")
    repair = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc_repair")
    full = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc_fork_repair_memory")
    memory = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc_memory")
    prorl = next(c for c in plan["cells"] if c["proicl_condition"] == "prorl_v2_greedy")

    assert base["model_key"] == "deepseek-r1-distill-qwen-1.5b"
    assert base["runtime_condition"] == "greedy"
    assert base["condition"] == "greedy"
    assert base["archive_kind"] == "direct"
    assert base["rollout_budget"] == 1

    assert mcmc["model_key"] == "deepseek-r1-distill-qwen-1.5b"
    assert mcmc["runtime_condition"] == "single_prompt_power"
    assert mcmc["condition"] == "single_prompt_power"
    assert mcmc["archive_kind"] == "direct"
    assert mcmc["uses_power_sampling"] is True
    assert mcmc["uses_gepa_archive"] is False

    assert mixed["runtime_condition"] == "mixed_alpha_mcmc"
    assert mixed["alpha_policy"] == "mixed_alpha_4_1"
    assert fork["runtime_condition"] == "fork_search"
    assert fork["uses_fork_search"] is True
    assert fork["fork_search_mode"] == "entropy_gated"

    assert gepa["runtime_condition"] == "gepa_only"
    assert gepa["archive_kind"] == "gepa"
    assert gepa["archive_scope"] == "within_family"
    assert gepa["uses_power_sampling"] is False
    assert gepa["uses_gepa_archive"] is True

    assert composed["runtime_condition"] == "proicl_gepa_mcmc"
    assert composed["archive_kind"] == "gepa"
    assert composed["rollout_budget"] == 64
    assert composed["uses_memory"] is False
    assert composed["alpha_policy"] == "mixed_alpha_4_1"

    assert repair["uses_repair"] is True
    assert repair["repair_mode"] == "verifier_guided"
    assert full["uses_memory"] is True
    assert full["uses_repair"] is True
    assert full["uses_fork_search"] is True
    assert full["memory_protocol"] == "frozen_dev"

    assert memory["runtime_condition"] == "proicl_gepa_mcmc_memory"
    assert memory["condition"] == "proicl_gepa_mcmc_memory"
    assert memory["uses_memory"] is True
    assert memory["memory_mode"] == "distilled_strategies"
    assert memory["memory_protocol"] == "online"
    assert memory["alpha_policy"] == "mixed_alpha_4_1"
    assert memory["memory_store_path"].endswith(
        "reasoning_gym_boxnet-gepa_mcmc_memory-shard-0.sqlite"
    )

    assert prorl["model_key"] == "nemotron-prorl-v2"
    assert prorl["runtime_condition"] == "greedy"
    assert prorl["condition"] == "greedy"
    assert prorl["slow_weight_reference"] is True


def test_proicl_signal_default_tracks_follow_reasoning_gym_difficulty_order():
    assert DEFAULT_SIGNAL_TRACKS == (
        "reasoning_gym_family_relationships",
        "reasoning_gym_graph_color_n5",
        "reasoning_gym_graph_color_n8",
        "reasoning_gym_graph_color_n10",
        "reasoning_gym_graph_color_n13",
        "reasoning_gym_graph_color_n15",
        "reasoning_gym_graph_color_n18",
        "reasoning_gym_graph_color_n20",
        "reasoning_gym_boxnet",
    )


def test_proicl_run_identity_is_parseable_and_safe(tmp_path):
    tracks = (
        "reasoning_gym_family_relationships",
        "reasoning_gym_graph_color_n10",
        "reasoning_gym_boxnet",
    )
    identity = make_proicl_run_identity(
        run_stage="small_real_slice",
        tracks=tracks,
        archive_scope="within_family",
        backend="hf",
        timestamp="20260522T101112Z",
        tag="gate-2",
    )

    assert panel_slug(tracks) == "sustained-rg-3t"
    assert identity.run_id == (
        "proicl_small-real-slice_sustained-rg-3t_within-family_hf_"
        "gate-2_20260522T101112Z"
    )
    assert standard_run_root(tmp_path / "runs", identity) == (
        tmp_path / "runs" / identity.run_id
    )


def test_proicl_run_index_records_trace_layout(tmp_path):
    identity = make_proicl_run_identity(
        run_stage="smoke",
        tracks=("reasoning_gym_boxnet",),
        archive_scope="transductive_support",
        backend="hf",
        timestamp="20260522T101112Z",
    )
    payload = write_run_index(
        tmp_path / "run_index.json",
        identity=identity,
        tracks=("reasoning_gym_boxnet",),
        conditions=("base_greedy", "gepa_mcmc_fork_repair_memory"),
        split=(0, 2),
        rollout_budget=2,
        archive_size=2,
        max_metric_calls=4,
        max_new_tokens=256,
        mcmc_steps=2,
        mcmc_block_num=4,
        num_shards=1,
        memory_num_shards=1,
        reflection_provider="local-hf",
        reflection_model_id="Qwen/Qwen2.5-7B-Instruct",
        run_kind="local",
        cost_cap_dollars=None,
        notes="unit",
    )

    on_disk = json.loads((tmp_path / "run_index.json").read_text(encoding="utf-8"))
    assert on_disk == payload
    assert payload["identity"]["run_id"] == identity.run_id
    assert payload["layout"]["analysis"] == "full/analysis/"
    assert payload["mcmc"] == {"steps": 2, "block_num": 4}
    assert payload["reflection"]["provider"] == "local-hf"


def test_proicl_artifact_audit_reports_missing_trace_files(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "run",
        tracks=("reasoning_gym_boxnet",),
        split=(0, 1),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
        conditions=("gepa_mcmc_fork_repair_memory",),
    )
    plan = {"cells": [cell.to_jsonable() for cell in cells]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = write_proicl_artifact_audit(plan_path, tmp_path / "audit")

    assert report["passed"] is False
    assert report["total_cells"] == 1
    assert report["missing"][0]["condition"] == "gepa_mcmc_fork_repair_memory"
    assert "repair_traces.jsonl" in report["missing"][0]["missing"]
    assert "fork_traces.jsonl" in report["missing"][0]["missing"]
    assert "memory.sqlite" in report["missing"][0]["missing"]
    assert (tmp_path / "audit" / "artifact_audit.json").exists()
    assert (tmp_path / "audit" / "artifact_audit.md").exists()


def test_proicl_signal_launcher_materializes_executable_cells(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 40),
        rollout_budget=8,
        num_shards=2,
        memory_num_shards=1,
    )

    by_condition = {}
    for cell in cells:
        by_condition.setdefault(cell.proicl_condition, []).append(cell)

    assert set(by_condition) == {
        "base_greedy",
        "bon_temp1",
        "mcmc_only",
        "mixed_alpha_mcmc",
        "fork_search",
        "gepa_only",
        "gepa_mcmc",
        "gepa_mcmc_repair",
        "gepa_mcmc_fork_repair",
        "gepa_mcmc_fork_repair_memory",
        "gepa_mcmc_memory",
        "prorl_v2_greedy",
    }
    assert len(by_condition["mcmc_only"]) == 2
    assert len(by_condition["gepa_mcmc_memory"]) == 1

    memory = by_condition["gepa_mcmc_memory"][0]
    assert memory.condition == "proicl_gepa_mcmc_memory"
    assert memory.memory_store_path is not None
    assert "--memory-mode" in memory.extra_args
    assert "memory.sqlite" in required_artifacts(memory)
    assert "archive_build_manifest.json" in required_artifacts(memory)
    assert "archive.json" in required_artifacts(memory)
    assert "rollouts.json" in required_artifacts(memory)

    assert cell_complete(memory) is False

    full = by_condition["gepa_mcmc_fork_repair_memory"][0]
    assert full.uses_repair is True
    assert full.uses_fork_search is True
    assert full.memory_protocol == "frozen_dev"
    assert "--enable-repair" in full.extra_args
    assert "--enable-fork-search" in full.extra_args
    assert "--online-memory" not in full.extra_args
    assert "repair_traces.jsonl" in required_artifacts(full)
    assert "fork_traces.jsonl" in required_artifacts(full)


def test_proicl_signal_launcher_exposes_sps_aliases(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_graph_color_n12",),
        split=(20, 70),
        rollout_budget=8,
        num_shards=1,
        memory_num_shards=1,
        archive_scope=ArchiveScope.CROSS_FAMILY_CURRICULUM,
        archive_train_tracks=("reasoning_gym_family_relationships",),
        archive_heldout_tracks=("reasoning_gym_graph_color_n12",),
        conditions=("base_greedy", "sps_only", "gepa_sps_fixed", "prorl_v2_greedy"),
    )

    by_condition = {cell.proicl_condition: cell for cell in cells}
    sps = by_condition["sps_only"]
    gepa = by_condition["gepa_sps_fixed"]

    assert sps.runtime_condition == "single_prompt_power"
    assert sps.archive_build_id == "none"
    assert sps.uses_gepa_archive is False
    assert gepa.runtime_condition == "full_archive_fixed"
    assert gepa.archive_build_id.startswith("proicl_cross_family_curriculum")
    assert gepa.uses_gepa_archive is True
    assert gepa.archive_train_tracks == ("reasoning_gym_family_relationships",)
    assert gepa.archive_heldout_tracks == ("reasoning_gym_graph_color_n12",)


def test_proicl_signal_cell_partitioning_is_disjoint_and_complete(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet", "reasoning_gym_graph_color"),
        split=(20, 40),
        rollout_budget=8,
        num_shards=2,
        memory_num_shards=1,
    )

    partitions = [
        _partition_cells(cells, stride=4, offset=offset)
        for offset in range(4)
    ]
    flattened = [cell for partition in partitions for cell in partition]

    assert len(flattened) == len(cells)
    assert {id(cell) for cell in flattened} == {id(cell) for cell in cells}
    assert all(partitions)


def test_proicl_signal_launcher_rejects_parallel_memory_shards(tmp_path):
    with pytest.raises(ValueError, match="memory_num_shards=1"):
        build_signal_cells(
            root=tmp_path / "proicl",
            tracks=("reasoning_gym_boxnet",),
            split=(20, 40),
            rollout_budget=8,
            num_shards=2,
            memory_num_shards=2,
        )


def test_proicl_launcher_reuses_the_gpu_that_finished(tmp_path, monkeypatch):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 24),
        rollout_budget=2,
        num_shards=3,
        memory_num_shards=1,
    )[:3]
    launches: list[str] = []
    durations = [3, 1, 1]
    completed: set[str] = set()
    stale = Path(cells[0].artifact_dir) / "candidates.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale partial row\n", encoding="utf-8")

    class FakeProc:
        def __init__(self, ticks: int, artifact_dir: str):
            self.ticks = ticks
            self.artifact_dir = artifact_dir

        def poll(self):
            self.ticks -= 1
            if self.ticks > 0:
                return None
            completed.add(self.artifact_dir)
            return 0

    def fake_popen(cmd, *, cwd, env, stdout, stderr):
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        return FakeProc(durations.pop(0), cmd[1])

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        launcher_mod,
        "run_condition_command",
        lambda **kwargs: ["fake", kwargs["cell"].artifact_dir],
    )
    monkeypatch.setattr(
        launcher_mod,
        "cell_complete",
        lambda cell: cell.artifact_dir in completed,
    )

    run_cells(
        repo_root=tmp_path,
        cells=cells,
        gpus=["0", "1"],
        events_path=tmp_path / "events.jsonl",
        backend="hf",
        local_files_only=False,
        cost_cap_dollars=1.0,
        estimated_dollar_cost=0.1,
        estimated_wall_clock_seconds=10,
        run_kind="local",
        run_stage="smoke",
        max_new_tokens=8,
    )

    assert launches == ["0", "1", "1"]
    assert not stale.exists()


def test_proicl_launcher_failure_reports_stderr_tail(tmp_path, monkeypatch):
    cell = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 21),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
    )[0]

    class FakeProc:
        def poll(self):
            return 1

    def fake_popen(cmd, *, cwd, env, stdout, stderr):
        stderr.write("useful failure context\n")
        stderr.flush()
        return FakeProc()

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        launcher_mod,
        "run_condition_command",
        lambda **kwargs: ["fake", kwargs["cell"].artifact_dir],
    )
    monkeypatch.setattr(launcher_mod, "cell_complete", lambda cell: False)

    with pytest.raises(RuntimeError, match="useful failure context"):
        run_cells(
            repo_root=tmp_path,
            cells=[cell],
            gpus=["0"],
            events_path=tmp_path / "events.jsonl",
            backend="hf",
            local_files_only=False,
            cost_cap_dollars=1.0,
            estimated_dollar_cost=0.1,
            estimated_wall_clock_seconds=10,
            run_kind="local",
            run_stage="smoke",
            max_new_tokens=8,
        )


def test_proicl_launcher_resumes_matching_partial_cell(tmp_path, monkeypatch):
    cell = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 21),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
    )[0]
    out = Path(cell.artifact_dir)
    out.mkdir(parents=True)
    (out / "proicl_launch_cell.json").write_text(
        json.dumps(cell.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial = out / "selected.jsonl"
    partial.write_text('{"problem_id": "partial"}\n', encoding="utf-8")
    completed: set[str] = set()

    class FakeProc:
        def poll(self):
            completed.add(cell.artifact_dir)
            return 0

    def fake_popen(cmd, *, cwd, env, stdout, stderr):
        return FakeProc()

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        launcher_mod,
        "run_condition_command",
        lambda **kwargs: ["fake", kwargs["cell"].artifact_dir],
    )
    monkeypatch.setattr(
        launcher_mod,
        "cell_complete",
        lambda launch_cell: launch_cell.artifact_dir in completed,
    )

    run_cells(
        repo_root=tmp_path,
        cells=[cell],
        gpus=["0"],
        events_path=tmp_path / "events.jsonl",
        backend="hf",
        local_files_only=False,
        cost_cap_dollars=1.0,
        estimated_dollar_cost=0.1,
        estimated_wall_clock_seconds=10,
        run_kind="local",
        run_stage="smoke",
        max_new_tokens=8,
    )

    assert partial.exists()


def test_proicl_launcher_active_cell_snapshot_reports_checkpoint_progress(tmp_path):
    cell = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 22),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
    )[0]
    out = Path(cell.artifact_dir)
    out.mkdir(parents=True)
    (out / "selected.jsonl").write_text('{"problem_id": "p1"}\n', encoding="utf-8")
    (out / "checkpoint.json").write_text(
        json.dumps(
            {
                "completed_problems": 1,
                "expected_problems": 2,
                "complete": False,
            }
        ),
        encoding="utf-8",
    )
    (out / "stderr.log").write_text("still running\n", encoding="utf-8")
    active = launcher_mod._ActiveCell(
        proc=SimpleNamespace(pid=1234),
        cell=cell,
        gpu="2",
        started_at=10.0,
    )

    snapshot = launcher_mod._active_cell_snapshot(active, now=75.5)
    text = launcher_mod._format_active_cell_snapshot(snapshot)

    assert snapshot["pid"] == 1234
    assert snapshot["gpu"] == "2"
    assert snapshot["elapsed_seconds"] == 65.5
    assert snapshot["completed_problems"] == 1
    assert snapshot["expected_problems"] == 2
    assert snapshot["selected_rows"] == 1
    assert "checkpoint=1/2" in text
    assert "stderr_bytes=" in text


def test_proicl_condition_command_propagates_vllm_scoring_mode(tmp_path):
    cell = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 21),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
    )[1]

    cmd = run_condition_command(
        repo_root=Path("/repo"),
        cell=cell,
        backend="vllm",
        local_files_only=True,
        polaris_source_hash="hash",
        cost_cap_dollars=1.0,
        estimated_dollar_cost=0.1,
        estimated_wall_clock_seconds=10.0,
        run_kind="local",
        run_stage="smoke",
        max_new_tokens=64,
        mcmc_steps=2,
        mcmc_block_num=4,
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.7,
        vllm_max_model_len=2048,
        vllm_scoring_mode="native_segment",
        vllm_parity_artifact="runs/calibration/calibration_summary.json",
    )

    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "vllm"
    assert cmd[cmd.index("--vllm-scoring-mode") + 1] == "native_segment"
    assert cmd[cmd.index("--vllm-dtype") + 1] == "float32"
    assert cmd[cmd.index("--vllm-model-impl") + 1] == "transformers"
    assert cmd[cmd.index("--vllm-gpu-memory-utilization") + 1] == "0.7"
    assert cmd[cmd.index("--vllm-max-model-len") + 1] == "2048"
    assert cmd[cmd.index("--mcmc-steps") + 1] == "2"
    assert cmd[cmd.index("--mcmc-block-num") + 1] == "4"
    assert (
        cmd[cmd.index("--vllm-parity-artifact") + 1]
        == "runs/calibration/calibration_summary.json"
    )
    assert "--local-files-only" in cmd


def test_proicl_signal_flow_cache_overrides_workspace_defaults(tmp_path):
    env = {
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    }

    configure_flow_cache_env(
        env,
        run_kind="flow",
        mnt_local=tmp_path,
        cache_root=tmp_path / "proicl-cache",
    )

    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["HF_HUB_CACHE"].startswith(str(tmp_path))
    assert env["HUGGINGFACE_HUB_CACHE"] == env["HF_HUB_CACHE"]
    assert env["PIP_CACHE_DIR"].startswith(str(tmp_path))


def test_proicl_gepa_archive_command_can_pin_single_gpu(tmp_path):
    args = SimpleNamespace(
        env_file=tmp_path / ".env",
        local_files_only=False,
        backend="hf",
        reflection_provider="local-hf",
        reflection_model_id="Qwen/Qwen2.5-7B-Instruct",
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.85,
        vllm_max_model_len=None,
        vllm_scoring_mode="forced_decode_v0",
        vllm_parity_artifact=None,
    )

    cmd = _gepa_archive_command(
        args=args,
        out=tmp_path / "archive",
        tracks=["reasoning_gym_boxnet", "reasoning_gym_graph_color"],
        dev_split=(0, 6),
        archive_size=8,
        max_metric_calls=64,
        sampler_max_new_tokens=512,
        reflection_max_new_tokens=512,
        cuda_visible_devices="0",
    )

    assert "--cuda-visible-devices" in cmd
    assert cmd[cmd.index("--cuda-visible-devices") + 1] == "0"
    assert "--live-gepa" in cmd
    assert "--reflection-provider" in cmd
    assert cmd[cmd.index("--reflection-provider") + 1] == "local-hf"
    assert cmd[cmd.index("--reflection-model-id") + 1] == "Qwen/Qwen2.5-7B-Instruct"


def test_proicl_signal_archive_liveness_accepts_local_hf_reflection(tmp_path):
    out = tmp_path / "archive"
    out.mkdir()
    (out / "archive_build_manifest.json").write_text(
        json.dumps(
                {
                    "archive_scope": "within_family",
                    "heldout_tracks": ["reasoning_gym_boxnet", "reasoning_gym_graph_color"],
                    "tracks": ["reasoning_gym_boxnet", "reasoning_gym_graph_color"],
                    "dev_split": [0, 6],
                "gepa": {"dry_run": False, "max_metric_calls": 64},
                "reflection": {
                    "provider": "local_hf",
                    "config": {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
                },
            }
        ),
        encoding="utf-8",
    )
    (out / "archive.json").write_text(
        json.dumps({"entries": [{"id": str(idx)} for idx in range(8)]}),
        encoding="utf-8",
    )

    assert _archive_is_live(
        out,
        archive_size=8,
        tracks=["reasoning_gym_boxnet", "reasoning_gym_graph_color"],
        dev_split=(0, 6),
        max_metric_calls=64,
        reflection_provider="local-hf",
        reflection_model_id="Qwen/Qwen2.5-7B-Instruct",
    )


def test_proicl_gepa_archive_command_propagates_vllm_sampler_backend(tmp_path):
    args = SimpleNamespace(
        env_file=tmp_path / ".env",
        local_files_only=False,
        backend="vllm",
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.65,
        vllm_max_model_len=4096,
        vllm_scoring_mode="forced_decode_v0",
        vllm_parity_artifact=tmp_path / "calibration_summary.json",
    )

    cmd = _gepa_archive_command(
        args=args,
        out=tmp_path / "archive",
        tracks=["reasoning_gym_boxnet"],
        dev_split=(0, 6),
        archive_size=8,
        max_metric_calls=64,
        sampler_max_new_tokens=512,
        reflection_max_new_tokens=512,
        cuda_visible_devices="0",
    )

    assert cmd[cmd.index("--sampler-backend") + 1] == "vllm"
    assert cmd[cmd.index("--vllm-dtype") + 1] == "float32"
    assert cmd[cmd.index("--vllm-gpu-memory-utilization") + 1] == "0.65"
    assert cmd[cmd.index("--vllm-max-model-len") + 1] == "4096"
    assert cmd[cmd.index("--vllm-parity-artifact") + 1] == str(
        tmp_path / "calibration_summary.json"
    )


def test_proicl_decomposition_reports_rf_and_slow_residual():
    report = compute_proicl_decomposition(
        {
            "base": 0.10,
            "mcmc_only": 0.25,
            "mixed_alpha_mcmc": 0.30,
            "fork_search": 0.31,
            "gepa_only": 0.34,
            "gepa_mcmc": 0.42,
            "gepa_mcmc_repair": 0.46,
            "gepa_mcmc_fork_repair": 0.48,
            "gepa_mcmc_fork_repair_memory": 0.50,
            "gepa_mcmc_memory": 0.50,
            "prorl_v2_greedy": 0.70,
        }
    )

    assert report["A_base"] == 0.10
    assert report["A_mcmc"] == 0.25
    assert report["A_mixed"] == 0.30
    assert report["A_fork"] == 0.31
    assert report["A_gepa"] == 0.34
    assert report["A_gepa_mcmc"] == 0.42
    assert report["A_repair"] == 0.46
    assert report["A_fork_repair"] == 0.48
    assert report["A_full_memory"] == 0.50
    assert report["A_memory"] == 0.50
    assert report["A_prorl_v2"] == 0.70
    assert report["rf_valid"] is True
    assert report["RF_mcmc"] == pytest.approx(0.25)
    assert report["RF_gepa"] == pytest.approx(0.4)
    assert report["RF_gepa_mcmc"] == pytest.approx(0.5333333333)
    assert report["RF_memory"] == pytest.approx(0.6666666667)
    assert report["slow_weight_residual"] == pytest.approx(0.20)
    assert report["repair_gain"] == pytest.approx(0.04)
    assert report["fork_repair_gain"] == pytest.approx(0.02)
    assert report["memory_gain"] == pytest.approx(0.02)


def test_proicl_decomposition_accepts_sps_aliases():
    report = compute_proicl_decomposition(
        {
            "base": 0.10,
            "sps_only": 0.30,
            "gepa_sps_fixed": 0.45,
            "prorl_v2_greedy": 0.60,
        }
    )

    assert report["A_sps"] == 0.30
    assert report["A_mcmc"] == 0.30
    assert report["A_gepa_sps_fixed"] == 0.45
    assert report["A_gepa_mcmc"] == 0.45
    assert report["RF_sps"] == pytest.approx(0.40)
    assert report["RF_gepa_sps_fixed"] == pytest.approx(0.70)
    assert report["confound_delta_gepa_archive_sps_vs_base_sps"] == pytest.approx(0.15)
    assert report["confound_check_base_sps_vs_gepa_archive_sps"] == {
        "base_sps_accuracy": 0.30,
        "gepa_archive_sps_accuracy": 0.45,
        "delta": pytest.approx(0.15),
        "status": "gepa_archive_sps_above_base_sps",
    }


def test_proicl_decomposition_tolerates_nonpositive_rf_denominator():
    report = compute_proicl_decomposition(
        {
            "base": 0.50,
            "mcmc_only": 0.55,
            "gepa_only": 0.55,
            "gepa_mcmc": 0.60,
            "gepa_mcmc_memory": 0.60,
            "prorl_v2_greedy": 0.50,
        }
    )

    assert report["rf_valid"] is False
    assert report["rf_denominator"] == pytest.approx(0.0)
    assert report["RF_mcmc"] is None
    assert "analysis_warning" in report


def test_proicl_aggregate_reads_shard_metrics(tmp_path):
    values = {
        "base_greedy": [0.10, 0.20],
        "mcmc_only": [0.25, 0.35],
        "gepa_only": [0.34, 0.44],
        "gepa_mcmc": [0.42, 0.52],
        "gepa_mcmc_memory": [0.50, 0.60],
        "prorl_v2_greedy": [0.70, 0.80],
    }
    root = tmp_path / "proicl"
    for condition, accuracies in values.items():
        for shard_id, accuracy in enumerate(accuracies):
            path = root / "runs" / "reasoning_gym_boxnet" / condition / f"shard-{shard_id}"
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(
                json.dumps({"accuracy": accuracy, "n_problems": 5}),
                encoding="utf-8",
            )

    report = write_proicl_decomposition_by_track(
        root=root,
        tracks=("reasoning_gym_boxnet",),
        out_dir=tmp_path / "analysis",
    )

    decomp = report["reasoning_gym_boxnet"]["decomposition"]
    assert decomp["A_base"] == pytest.approx(0.15)
    assert decomp["A_prorl_v2"] == pytest.approx(0.75)
    assert (tmp_path / "analysis" / "proicl_decomposition.json").exists()
    md = (tmp_path / "analysis" / "proicl_decomposition.md")
    assert md.exists()
    text = md.read_text()
    assert "discovery_gain" in text
    assert "composition_gain" in text
    assert "confound_delta_gepa_archive_sps_vs_base_sps" in text


def test_cross_task_curriculum_archive_writes_k16_gepa_artifacts(tmp_path):
    out = tmp_path / "archive"
    payload = build_cross_task_curriculum_archive(
        out_dir=out,
        tracks=("reasoning_gym_boxnet", "reasoning_gym_graph_color"),
        dev_split=(0, 2),
        archive_size=16,
        dry_run=True,
        max_metric_calls=32,
        reflection_provider="none",
    )

    archive = json.loads((out / "archive.json").read_text())
    manifest = json.loads((out / "archive_build_manifest.json").read_text())

    assert payload["archive_build_id"].startswith("proicl_within_family")
    assert len(archive["entries"]) == 16
    assert manifest["archive_scope"] == "within_family"
    assert manifest["train_tracks"] == [
        "reasoning_gym_boxnet",
        "reasoning_gym_graph_color",
    ]
    assert manifest["gepa"]["dry_run"] is True
    assert manifest["gepa"]["max_metric_calls"] == 32
    assert manifest["leakage_policy"]["no_prorl_or_brorl_traces"] is True


def test_cross_family_archive_rejects_heldout_track_leakage(tmp_path):
    with pytest.raises(ValueError, match="leaks held-out target tracks"):
        build_cross_task_curriculum_archive(
            out_dir=tmp_path / "archive",
            tracks=("reasoning_gym_boxnet", "reasoning_gym_graph_color_n10"),
            archive_scope=ArchiveScope.CROSS_FAMILY_CURRICULUM,
            heldout_tracks=("reasoning_gym_boxnet",),
            dev_split=(0, 2),
            archive_size=2,
            dry_run=True,
            max_metric_calls=4,
            reflection_provider="none",
        )


def test_reasoning_gym_verifier_emits_structured_graph_feedback(monkeypatch):
    class FakeDataset:
        def score_answer(self, *, answer, entry):
            return 0.0

    fake_reasoning_gym = SimpleNamespace(create_dataset=lambda task, size, seed: FakeDataset())
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)

    result = score_reasoning_gym(
        "bad final answer",
        json.dumps(
            {
                "task": "graph_color",
                "entry": {"question": "color graph", "answer": {}},
                "answer": {},
            }
        ),
    )

    assert result["passed"] is False
    assert result["format_valid"] is False
    assert result["failure_type"] == "graph_color_invalid_json"
    assert "adjacent nodes" in result["repair_hint"]


def _candidate(*, generation: str, verifier_result: dict) -> Candidate:
    return Candidate(
        prompt_id="direct",
        sample_index=0,
        alpha=1.0,
        prompt_text="Solve.",
        generation=generation,
        response_contains_prompt=False,
        prompt_token_count=1,
        generation_token_count=1,
        wall_clock_seconds=0.0,
        estimated_dollar_cost=0.0,
        acceptance_ratio=None,
        verifier_result=verifier_result,
    )


def test_repair_controller_records_repaired_candidate():
    class RepairSampler:
        def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
            return SimpleNamespace(
                generation="fixed",
                response_contains_prompt=False,
                prompt_token_count=len(prompt_text.split()),
                generation_token_count=1,
                wall_clock_seconds=0.0,
                estimated_dollar_cost=0.0,
                acceptance_ratio=None,
                token_ids=[],
                logprobs_norm=[],
                logprobs_unnorm=[],
            )

    parent = _candidate(
        generation="wrong",
        verifier_result={
            "score": 0.0,
            "passed": False,
            "failure_type": "answer_mismatch",
            "repair_hint": "say fixed",
        },
    )
    repairs = run_verifier_guided_repair(
        parent=parent,
        sampler=RepairSampler(),
        scorer=lambda response, reference: {"score": 1.0, "passed": "fixed" in response},
        reference="fixed",
        max_new_tokens=8,
        config=RepairConfig(max_attempts=2),
    )

    assert len(repairs) == 1
    assert repairs[0].repair_attempt == 1
    assert repairs[0].parent_candidate_id == "direct:0"
    assert repairs[0].verifier_result["passed"] is True


def test_entropy_gated_fork_search_uses_sampler_distribution():
    class ForkSampler:
        def next_token_distribution(self, prompt_text, *, top_k):
            return [(" A", 0.5), (" B", 0.3), (" C", 0.2)]

        def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
            return SimpleNamespace(
                generation=" answer",
                response_contains_prompt=False,
                prompt_token_count=len(prompt_text.split()),
                generation_token_count=1,
                wall_clock_seconds=0.0,
                estimated_dollar_cost=0.0,
                acceptance_ratio=None,
                token_ids=[],
                logprobs_norm=[],
                logprobs_unnorm=[],
            )

    candidates = run_entropy_gated_fork_search(
        prompt_id="direct",
        sample_index_start=4,
        prompt_text="Solve.",
        sampler=ForkSampler(),
        scorer=lambda response, reference: {"score": 1.0, "passed": True},
        reference="answer",
        max_new_tokens=8,
        config=ForkSearchConfig(top_k=3, max_depth=1, entropy_threshold=0.1),
    )

    assert [c.sample_index for c in candidates] == [4, 5, 6]
    assert all(c.search_strategy == "entropy_gated_fork_search" for c in candidates)
    assert candidates[0].fork_token == " A"
    assert candidates[0].fork_entropy is not None


def test_reasoning_gym_memory_distillers_are_task_specific():
    assert "adjacency constraints" in distill_reasoning_gym_strategy("trace", "graph_color")
    assert "family graph" in distill_reasoning_gym_strategy("trace", "family_relationships")
    assert "grid state" in distill_reasoning_gym_strategy("trace", "boxnet")


def test_live_cross_task_gepa_uses_vendored_gepa_without_external_package(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "gepa", None)

    rows = [
        Problem(
            problem_id="rg-dev-0",
            prompt="Return the exact word pass.",
            answer="pass",
            source="unit",
        )
    ]
    monkeypatch.setattr(
        "polaris.proicl.archive._load_rows",
        lambda *, tracks, split, dry_run: (rows, "optimizer_dev"),
    )

    class Sampler:
        def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
            text = "pass" if "Use the exact requested word" in prompt_text else "fail"
            return SimpleNamespace(
                generation=text,
                response_contains_prompt=False,
                prompt_token_count=len(prompt_text.split()),
                generation_token_count=1,
                wall_clock_seconds=0.0,
                estimated_dollar_cost=0.0,
            )

    class ReflectionLM:
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        def __call__(self, prompt):
            self.total_tokens_in += 1
            self.total_tokens_out += 1
            return "```Use the exact requested word and place it inside <answer> tags.```"

    def scorer(response: str, reference: str) -> dict:
        passed = response.strip().endswith("pass")
        return {"score": 1.0 if passed else 0.0, "passed": passed}

    payload = build_cross_task_curriculum_archive(
        out_dir=tmp_path / "archive",
        tracks=("reasoning_gym_boxnet",),
        dev_split=(0, 1),
        archive_size=2,
        dry_run=False,
        max_metric_calls=4,
        reflection_provider="local_hf",
        sampler=Sampler(),
        scorer=scorer,
        reflection_lm=ReflectionLM(),
    )

    manifest = json.loads(
        (tmp_path / "archive" / "archive_build_manifest.json").read_text()
    )
    assert payload["gepa"]["dry_run"] is False
    assert manifest["reflection"]["provider"] == "local_hf"
    assert manifest["reflection"]["usage"]["total_tokens_out"] >= 1
    archive = json.loads((tmp_path / "archive" / "archive.json").read_text())
    assert len(archive["entries"]) == 2


class _FakeSampler:
    def generate_greedy(self, prompt_text, max_new_tokens):
        return self._gen(prompt_text)

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        return self._gen(prompt_text)

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        return self._gen(prompt_text, acceptance_ratio=1.0)

    def _gen(self, prompt_text, acceptance_ratio=None):
        return SimpleNamespace(
            generation=" solved",
            response_contains_prompt=False,
            prompt_token_count=len(prompt_text.split()),
            generation_token_count=1,
            wall_clock_seconds=0.0,
            estimated_dollar_cost=0.0,
            acceptance_ratio=acceptance_ratio,
            token_ids=[],
            logprobs_norm=[],
            logprobs_unnorm=[],
        )


def test_proicl_gepa_mcmc_memory_writes_memory_ledgers(tmp_path):
    archive = FrozenArchive(
        entries=(
            PromptEntry(
                id="direct",
                prefix="Q: ",
                suffix=" A:",
                descriptor_hint="direct",
            ),
        )
    )
    problem = Problem(
        problem_id="toy-0",
        prompt="2+2?",
        answer="4",
        source="unit",
    )

    metrics = run_condition(
        track="math500",
        split=(0, 1),
        model_key="deepseek-r1-distill-qwen-1.5b",
        out_dir=tmp_path / "run",
        condition="proicl_gepa_mcmc_memory",
        archive=archive,
        cell_fitness={"direct": 1.0},
        sampler=_FakeSampler(),
        seed=17,
        archive_hash="archive",
        polaris_source_hash="test",
        vendored_commits={},
        preregistration_anchor="TODO.PROICL.md",
        model_id="fake",
        model_revision="main",
        model_revision_commit="fake",
        problems=[problem],
        budget_override=1,
        max_new_tokens=8,
        archive_build_id="proicl-cross-task-gepa-test",
        memory_build_id="proicl-memory-test",
        serving_backend_metadata={
            "backend": "vllm",
            "vllm_scoring_mode": "native_segment",
            "vllm_version": "unit",
            "vllm_commit": "abc123",
            "dtype": "float32",
            "model_impl": "transformers",
            "tokenizer_revision": "main",
            "parity_artifact_path": "runs/calibration/calibration_summary.json",
        },
    )

    assert metrics["condition"] == "proicl_gepa_mcmc_memory"
    assert metrics["alpha_policy_id"] == "mixed_alpha_4_1"
    assert metrics["backend"] == "vllm"
    assert metrics["vllm_scoring_mode"] == "native_segment"
    assert (tmp_path / "run" / "memory.sqlite").exists()
    assert (tmp_path / "run" / "memory_events.jsonl").exists()
    assert (tmp_path / "run" / "archive_build_manifest.json").exists()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["config"]["serving_backend"]["vllm_version"] == "unit"
    assert (
        manifest["config"]["vllm_parity_artifact"]
        == "runs/calibration/calibration_summary.json"
    )
