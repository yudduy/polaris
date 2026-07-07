"""Package a ProICL experiment run into a small, shareable result bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also include candidates.jsonl. This can be large.",
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _full_root(run_root: Path) -> Path:
    return run_root / "full" if (run_root / "full").exists() else run_root


def _condition_key(path: Path) -> tuple[str, str, str]:
    shard = path.parent.name
    condition = path.parent.parent.name
    track = path.parent.parent.parent.name
    return track, condition, shard


def _wilson_interval(correct: int, n: int, *, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = correct / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _selected_correct_count(path: Path) -> tuple[int, int] | None:
    rows = _jsonl(path)
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get("selected_passed"))), len(rows)


def _metrics_correct_count(metrics: dict[str, Any]) -> tuple[int, int]:
    n = int(metrics.get("n_problems") or 0)
    accuracy = float(metrics.get("accuracy") or 0.0)
    return int(round(accuracy * n)), n


def _collect_metrics(full_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((full_root / "runs").glob("*/*/shard-*/metrics.json")):
        track, condition, shard = _condition_key(metrics_path)
        metrics = _json(metrics_path)
        correct, n = _selected_correct_count(metrics_path.parent / "selected.jsonl") or _metrics_correct_count(metrics)
        ci_low, ci_high = _wilson_interval(correct, n)
        rows.append(
            {
                "track": track,
                "condition": condition,
                "shard": shard,
                "accuracy": metrics.get("accuracy"),
                "correct": correct,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_selected_score": metrics.get("mean_selected_score"),
                "n_problems": metrics.get("n_problems"),
                "n_candidates": metrics.get("n_candidates"),
                "B_per_problem": metrics.get("B_per_problem"),
                "power_sampler": metrics.get("power_sampler"),
                "sps_top_k": metrics.get("sps_top_k"),
                "sps_candidate_pool_size": metrics.get("sps_candidate_pool_size"),
                "sps_rollouts_per_candidate": metrics.get("sps_rollouts_per_candidate"),
                "sps_rollout_horizon": metrics.get("sps_rollout_horizon"),
                "alpha_policy_id": metrics.get("alpha_policy_id"),
                "backend": metrics.get("backend"),
                "vllm_scoring_mode": metrics.get("vllm_scoring_mode"),
            }
        )
    return rows


def _collect_selected(full_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected_path in sorted((full_root / "runs").glob("*/*/shard-*/selected.jsonl")):
        track, condition, shard = _condition_key(selected_path)
        metrics = {}
        metrics_path = selected_path.parent / "metrics.json"
        if metrics_path.exists():
            metrics = _json(metrics_path)
        for row in _jsonl(selected_path):
            rows.append(
                {
                    "track": track,
                    "condition": condition,
                    "shard": shard,
                    "problem_id": row.get("problem_id"),
                    "selected_passed": row.get("selected_passed"),
                    "selected_score": row.get("selected_score"),
                    "prompt_id": row.get("prompt_id"),
                    "sample_index": row.get("sample_index"),
                    "alpha": row.get("alpha"),
                    "power_sampler": metrics.get("power_sampler"),
                    "alpha_policy_id": metrics.get("alpha_policy_id"),
                }
            )
    return rows


def _agreement_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    conditions: set[str] = set()
    for row in selected_rows:
        key = (str(row["track"]), str(row["problem_id"]))
        condition = str(row["condition"])
        conditions.add(condition)
        payload = grouped.setdefault(
            key,
            {"track": row["track"], "problem_id": row["problem_id"]},
        )
        payload[f"{condition}_passed"] = row.get("selected_passed")
        payload[f"{condition}_score"] = row.get("selected_score")

    preferred = [
        "base_greedy",
        "sps_only",
        "mcmc_only",
        "gepa_sps_fixed",
        "gepa_mcmc",
        "prorl_v2_greedy",
    ]
    ordered = [c for c in preferred if c in conditions] + sorted(conditions - set(preferred))
    fields = ["track", "problem_id"]
    for condition in ordered:
        fields.extend([f"{condition}_passed", f"{condition}_score"])
    return [
        {field: row.get(field) for field in fields}
        for row in sorted(grouped.values(), key=lambda r: (str(r["track"]), str(r["problem_id"])))
    ]


def _confound_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in selected_rows:
        condition = str(row.get("condition"))
        if condition not in {"sps_only", "gepa_sps_fixed"}:
            continue
        track = str(row.get("track"))
        problem_id = str(row.get("problem_id"))
        grouped.setdefault(track, {}).setdefault(problem_id, {})[condition] = row

    rows: list[dict[str, Any]] = []
    for track, problems in sorted(grouped.items()):
        paired = [payload for payload in problems.values() if "sps_only" in payload and "gepa_sps_fixed" in payload]
        common_n = len(paired)
        sps_correct = sum(1 for payload in paired if bool(payload["sps_only"].get("selected_passed")))
        gepa_correct = sum(1 for payload in paired if bool(payload["gepa_sps_fixed"].get("selected_passed")))
        both = sum(
            1
            for payload in paired
            if bool(payload["sps_only"].get("selected_passed"))
            and bool(payload["gepa_sps_fixed"].get("selected_passed"))
        )
        sps_only_passed = sum(
            1
            for payload in paired
            if bool(payload["sps_only"].get("selected_passed"))
            and not bool(payload["gepa_sps_fixed"].get("selected_passed"))
        )
        gepa_only_passed = sum(
            1
            for payload in paired
            if not bool(payload["sps_only"].get("selected_passed"))
            and bool(payload["gepa_sps_fixed"].get("selected_passed"))
        )
        neither = common_n - both - sps_only_passed - gepa_only_passed
        sps_rate = sps_correct / common_n if common_n else None
        gepa_rate = gepa_correct / common_n if common_n else None
        sps_low, sps_high = _wilson_interval(sps_correct, common_n)
        gepa_low, gepa_high = _wilson_interval(gepa_correct, common_n)
        delta = None if sps_rate is None or gepa_rate is None else gepa_rate - sps_rate
        if common_n == 0:
            status = "missing_paired_base_sps_and_gepa_archive_sps"
        elif delta is not None and delta > 0.0:
            status = "gepa_archive_sps_above_base_sps"
        else:
            status = "no_gepa_archive_sps_gain_over_base_sps"
        rows.append(
            {
                "track": track,
                "common_n": common_n,
                "sps_correct": sps_correct,
                "gepa_sps_correct": gepa_correct,
                "both_passed": both,
                "sps_only_passed": sps_only_passed,
                "gepa_only_passed": gepa_only_passed,
                "neither_passed": neither,
                "sps_pass_rate": sps_rate,
                "gepa_sps_pass_rate": gepa_rate,
                "sps_ci95_low": sps_low,
                "sps_ci95_high": sps_high,
                "gepa_sps_ci95_low": gepa_low,
                "gepa_sps_ci95_high": gepa_high,
                "gepa_minus_sps": delta,
                "confound_status": status,
            }
        )
    return rows


def _copy_cell_artifacts(full_root: Path, bundle_root: Path, *, include_candidates: bool) -> None:
    rels = [
        "metrics.json",
        "selected.jsonl",
        "scores.jsonl",
        "preflight.json",
        "run_plan_cell.json",
        "manifest.json",
        "audit.md",
        "stdout.json",
        "stderr.log",
    ]
    if include_candidates:
        rels.append("candidates.jsonl")
    for shard_dir in sorted((full_root / "runs").glob("*/*/shard-*")):
        if not shard_dir.is_dir():
            continue
        rel = shard_dir.relative_to(full_root)
        for name in rels:
            _copy_if_exists(shard_dir / name, bundle_root / rel / name)


def _copy_top_level(full_root: Path, run_root: Path, bundle_root: Path) -> None:
    for rel in [
        "launch_config.json",
        "resource_probe.json",
        "runtime_profile.json",
        "backend_preflight/backend_preflight.json",
        "wandb.json",
        "run_index.json",
        "events.jsonl",
        "proicl_signal_plan.json",
        "proicl_signal_plan.worker-0-of-1.json",
    ]:
        _copy_if_exists(run_root / rel, bundle_root / rel)
        _copy_if_exists(full_root / rel, bundle_root / "full" / rel)
    if (full_root / "analysis").exists():
        shutil.copytree(full_root / "analysis", bundle_root / "analysis", dirs_exist_ok=True)
    if (run_root / "probes").exists():
        shutil.copytree(run_root / "probes", bundle_root / "probes", dirs_exist_ok=True)
    for manifest in sorted((full_root / "archives").glob("*/archive_build_manifest.json")):
        dst = bundle_root / "archives" / manifest.parent.name / "archive_build_manifest.json"
        _copy_if_exists(manifest, dst)


def _make_tarball(bundle_root: Path) -> Path:
    tar_path = bundle_root.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_root, arcname=bundle_root.name)
    return tar_path


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    full_root = _full_root(run_root)
    if not (full_root / "runs").exists():
        raise SystemExit(f"could not find run artifacts under {full_root / 'runs'}")

    out = (args.out or (run_root / "results_bundle")).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    metrics_rows = _collect_metrics(full_root)
    selected_rows = _collect_selected(full_root)
    agreement_rows = _agreement_rows(selected_rows)
    confound_rows = _confound_rows(selected_rows)

    _write_csv(
        out / "summary" / "metrics.csv",
        metrics_rows,
        [
            "track",
            "condition",
            "shard",
            "accuracy",
            "correct",
            "ci95_low",
            "ci95_high",
            "mean_selected_score",
            "n_problems",
            "n_candidates",
            "B_per_problem",
            "power_sampler",
            "sps_top_k",
            "sps_candidate_pool_size",
            "sps_rollouts_per_candidate",
            "sps_rollout_horizon",
            "alpha_policy_id",
            "backend",
            "vllm_scoring_mode",
        ],
    )
    _write_csv(
        out / "summary" / "per_problem_selected.csv",
        selected_rows,
        [
            "track",
            "condition",
            "shard",
            "problem_id",
            "selected_passed",
            "selected_score",
            "prompt_id",
            "sample_index",
            "alpha",
            "power_sampler",
            "alpha_policy_id",
        ],
    )
    if agreement_rows:
        fields = list(agreement_rows[0].keys())
    else:
        fields = ["track", "problem_id"]
    _write_csv(out / "summary" / "per_problem_agreement.csv", agreement_rows, fields)
    _write_csv(
        out / "summary" / "confound_checks.csv",
        confound_rows,
        [
            "track",
            "common_n",
            "sps_correct",
            "gepa_sps_correct",
            "both_passed",
            "sps_only_passed",
            "gepa_only_passed",
            "neither_passed",
            "sps_pass_rate",
            "gepa_sps_pass_rate",
            "sps_ci95_low",
            "sps_ci95_high",
            "gepa_sps_ci95_low",
            "gepa_sps_ci95_high",
            "gepa_minus_sps",
            "confound_status",
        ],
    )

    _copy_top_level(full_root, run_root, out)
    _copy_cell_artifacts(full_root, out, include_candidates=args.include_candidates)
    manifest = {
        "schema": "proicl_results_bundle.v1",
        "run_root": str(run_root),
        "full_root": str(full_root),
        "metrics_rows": len(metrics_rows),
        "selected_rows": len(selected_rows),
        "agreement_rows": len(agreement_rows),
        "confound_rows": len(confound_rows),
        "included_candidates": bool(args.include_candidates),
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tar_path = _make_tarball(out)
    print(json.dumps({"bundle_dir": str(out), "tarball": str(tar_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
