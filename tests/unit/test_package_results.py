from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_packager():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "proicl" / "package_results.py"
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
    problem_id = "reasoning_gym:graph_color:seed=0:index=23"
    for condition, passed, prompt_id in [
        ("sps_only", False, "direct"),
        ("gepa_sps_fixed", True, "gepa-0"),
    ]:
        shard = (
            full
            / "runs"
            / "reasoning_gym_graph_color_n12"
            / condition
            / "shard-0"
        )
        _write_json(
            shard / "metrics.json",
            {
                "accuracy": 1.0 if passed else 0.0,
                "mean_selected_score": 1.0 if passed else 0.0,
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
                    "problem_id": problem_id,
                    "selected_passed": passed,
                    "selected_score": 1.0 if passed else 0.0,
                    "prompt_id": prompt_id,
                    "sample_index": 0,
                    "alpha": 4.0,
                }
            ],
        )
        _write_jsonl(
            shard / "scores.jsonl",
            [{"problem_id": problem_id, "score": 1.0 if passed else 0.0}],
        )
        (shard / "audit.md").write_text("# audit\n", encoding="utf-8")
    _write_json(run_root / "run_index.json", {"power_sampler": "sps"})
    (full / "events.jsonl").write_text("", encoding="utf-8")

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
    assert (bundle / "summary" / "confound_checks.csv").exists()
    metrics_rows = list(csv.DictReader((bundle / "summary" / "metrics.csv").open()))
    assert {row["condition"] for row in metrics_rows} == {"sps_only", "gepa_sps_fixed"}
    assert all(row["correct"] in {"0", "1"} for row in metrics_rows)
    assert all(row["ci95_low"] for row in metrics_rows)
    confound_rows = list(csv.DictReader((bundle / "summary" / "confound_checks.csv").open()))
    assert len(confound_rows) == 1
    confound = confound_rows[0]
    assert confound["track"] == "reasoning_gym_graph_color_n12"
    assert confound["common_n"] == "1"
    assert confound["sps_correct"] == "0"
    assert confound["gepa_sps_correct"] == "1"
    assert confound["gepa_only_passed"] == "1"
    assert confound["confound_status"] == "gepa_archive_sps_above_base_sps"
    assert float(confound["sps_pass_rate"]) == 0.0
    assert float(confound["gepa_sps_pass_rate"]) == 1.0
    assert float(confound["gepa_minus_sps"]) == 1.0
    assert 0.0 <= float(confound["sps_ci95_low"]) <= float(confound["sps_ci95_high"]) <= 1.0
    assert 0.0 <= float(confound["gepa_sps_ci95_low"]) <= float(confound["gepa_sps_ci95_high"]) <= 1.0
    assert (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "selected.jsonl").exists()
    assert not (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "candidates.jsonl").exists()
    assert (run_root / "results_bundle.tar.gz").exists()
