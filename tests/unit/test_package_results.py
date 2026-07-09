from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_packager():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "package_results.py"
    spec = importlib.util.spec_from_file_location("package_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_package_results_writes_small_bundle(tmp_path, monkeypatch):
    packager = _load_packager()
    run_root = tmp_path / "run"
    full = run_root / "full"
    shard = (
        full
        / "runs"
        / "reasoning_gym_graph_color_n12"
        / "gepa_sps_fixed"
        / "shard-0"
    )
    _write_json(
        shard / "metrics.json",
        {
            "accuracy": 1.0,
            "mean_selected_score": 1.0,
            "n_problems": 1,
            "n_candidates": 2,
            "B_per_problem": 2,
            "power_sampler": "sps",
            "alpha_policy_id": "fixed_alpha_4",
            "backend": "vllm",
            "vllm_scoring_mode": "forced_decode_v0",
        },
    )
    _write_jsonl(
        shard / "selected.jsonl",
        [
            {
                "problem_id": "reasoning_gym:graph_color:seed=0:index=23",
                "selected_passed": True,
                "selected_score": 1.0,
                "prompt_id": "gepa-0",
                "sample_index": 0,
                "alpha": 4.0,
            }
        ],
    )
    _write_jsonl(
        shard / "scores.jsonl",
        [{"problem_id": "reasoning_gym:graph_color:seed=0:index=23", "score": 1.0}],
    )
    _write_json(
        shard / "checkpoint.json",
        {
            "complete": True,
            "completed_problem_ids": ["reasoning_gym:graph_color:seed=0:index=23"],
            "expected_problem_ids": ["reasoning_gym:graph_color:seed=0:index=23"],
        },
    )
    (shard / "audit.md").write_text("# audit\n", encoding="utf-8")
    _write_json(run_root / "run_index.json", {"power_sampler": "sps"})
    (full / "events.jsonl").write_text("", encoding="utf-8")
    _write_json(
        full / "proicl_signal_plan.json",
        [
            {
                "track": "reasoning_gym_graph_color_n12",
                "proicl_condition": "gepa_sps_fixed",
                "shard_id": 0,
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["package_results.py", "--run-root", str(run_root)],
    )
    packager.main()

    bundle = run_root / "results_bundle"
    assert (bundle / "summary" / "metrics.csv").exists()
    assert (bundle / "summary" / "per_problem_selected.csv").exists()
    assert (bundle / "summary" / "per_problem_agreement.csv").exists()
    assert (bundle / "summary" / "cell_status.csv").exists()
    assert (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "selected.jsonl").exists()
    assert (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "checkpoint.json").exists()
    assert not (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "candidates.jsonl").exists()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completion_status"] == "complete"
    assert manifest["is_final"] is True
    assert manifest["planned_cells"] == 1
    assert manifest["completed_cells"] == 1
    assert (run_root / "results_bundle.tar.gz").exists()


def test_package_results_requires_allow_partial_for_incomplete_plan(tmp_path, monkeypatch):
    packager = _load_packager()
    run_root = tmp_path / "run"
    full = run_root / "full"
    complete_shard = full / "runs" / "reasoning_gym_boxnet" / "base_greedy" / "shard-0"
    partial_shard = full / "runs" / "reasoning_gym_boxnet" / "gepa_sps_fixed" / "shard-0"
    _write_json(
        complete_shard / "metrics.json",
        {"accuracy": 0.0, "n_problems": 1, "n_candidates": 1},
    )
    _write_jsonl(
        complete_shard / "selected.jsonl",
        [{"problem_id": "boxnet-20", "selected_passed": False, "selected_score": 0.0}],
    )
    _write_jsonl(
        partial_shard / "selected.jsonl",
        [{"problem_id": "boxnet-20", "selected_passed": False, "selected_score": 0.0}],
    )
    _write_json(
        partial_shard / "checkpoint.json",
        {
            "complete": False,
            "completed_problem_ids": ["boxnet-20"],
            "expected_problem_ids": ["boxnet-20", "boxnet-21"],
        },
    )
    _write_json(
        full / "proicl_signal_plan.json",
        [
            {"track": "reasoning_gym_boxnet", "proicl_condition": "base_greedy", "shard_id": 0},
            {"track": "reasoning_gym_boxnet", "proicl_condition": "gepa_sps_fixed", "shard_id": 0},
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["package_results.py", "--run-root", str(run_root)],
    )
    with pytest.raises(SystemExit, match="run is incomplete"):
        packager.main()
    assert not (run_root / "results_bundle.tar.gz").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["package_results.py", "--run-root", str(run_root), "--allow-partial"],
    )
    packager.main()

    bundle = run_root / "partial_results_bundle"
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completion_status"] == "partial"
    assert manifest["is_final"] is False
    assert manifest["planned_cells"] == 2
    assert manifest["completed_cells"] == 1
    assert manifest["partial_cells"] == 1
    assert (bundle / "summary" / "cell_status.csv").exists()
    assert (bundle / "runs" / "reasoning_gym_boxnet" / "gepa_sps_fixed" / "shard-0" / "checkpoint.json").exists()
    assert (run_root / "partial_results_bundle.tar.gz").exists()
