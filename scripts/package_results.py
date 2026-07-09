"""Package a ProICL experiment run into a small, shareable result bundle."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Package an interrupted/incomplete run for debugging. Without this, "
            "standard ProICL runs must have every planned cell complete."
        ),
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


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def _planned_cells(full_root: Path) -> list[dict[str, Any]]:
    for rel in (
        "proicl_signal_plan.json",
        "proicl_signal_plan.worker-0-of-1.json",
    ):
        path = full_root / rel
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
    return []


def _plan_cell_key(cell: dict[str, Any]) -> tuple[str, str, str]:
    track = str(cell.get("track") or "")
    condition = str(cell.get("proicl_condition") or cell.get("condition") or "")
    shard = f"shard-{int(cell.get('shard_id', 0))}"
    return track, condition, shard


def _checkpoint_expected(checkpoint: dict[str, Any]) -> int | None:
    expected = checkpoint.get("expected_problems")
    if expected is not None:
        return int(expected)
    expected_ids = checkpoint.get("expected_problem_ids")
    if isinstance(expected_ids, list):
        return len(expected_ids)
    return None


def _checkpoint_completed(checkpoint: dict[str, Any]) -> int | None:
    completed = checkpoint.get("completed_problems")
    if completed is not None:
        return int(completed)
    completed_ids = checkpoint.get("completed_problem_ids")
    if isinstance(completed_ids, list):
        return len(completed_ids)
    return None


def _cell_status_rows(full_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    planned = _planned_cells(full_root)
    if planned:
        keys = [_plan_cell_key(cell) for cell in planned]
    else:
        keys = [
            _condition_key(path)
            for path in sorted((full_root / "runs").glob("*/*/shard-*"))
            if path.is_dir()
        ]

    for track, condition, shard in sorted(keys):
        shard_dir = full_root / "runs" / track / condition / shard
        metrics_path = shard_dir / "metrics.json"
        selected_path = shard_dir / "selected.jsonl"
        checkpoint_path = shard_dir / "checkpoint.json"
        metrics: dict[str, Any] = {}
        checkpoint: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = _json(metrics_path)
        if checkpoint_path.exists():
            checkpoint = _json(checkpoint_path)

        selected_rows = _jsonl_count(selected_path)
        metrics_n_problems = metrics.get("n_problems")
        expected_problems = (
            _checkpoint_expected(checkpoint)
            if checkpoint
            else int(metrics_n_problems)
            if metrics_n_problems is not None
            else None
        )
        completed_problems = (
            _checkpoint_completed(checkpoint)
            if checkpoint
            else selected_rows
            if selected_path.exists()
            else None
        )
        metrics_complete = (
            metrics_path.exists()
            and metrics_n_problems is not None
            and int(metrics_n_problems) > 0
            and selected_rows == int(metrics_n_problems)
        )
        if metrics_complete:
            reason = "complete"
        elif selected_rows > 0:
            reason = "partial_cell"
        elif shard_dir.exists():
            reason = "started_no_selected_rows"
        else:
            reason = "not_started"

        rows.append(
            {
                "track": track,
                "condition": condition,
                "shard": shard,
                "planned": bool(planned),
                "complete": metrics_complete,
                "reason": reason,
                "metrics_exists": metrics_path.exists(),
                "checkpoint_exists": checkpoint_path.exists(),
                "checkpoint_complete": checkpoint.get("complete"),
                "selected_rows": selected_rows,
                "completed_problems": completed_problems,
                "expected_problems": expected_problems,
                "n_problems": metrics_n_problems,
                "n_candidates": metrics.get("n_candidates"),
                "accuracy": metrics.get("accuracy"),
                "mean_selected_score": metrics.get("mean_selected_score"),
            }
        )
    return rows


def _completion_summary(status_rows: list[dict[str, Any]]) -> dict[str, Any]:
    planned_known = any(bool(row.get("planned")) for row in status_rows)
    completed = [row for row in status_rows if row.get("complete")]
    incomplete = [row for row in status_rows if not row.get("complete")]
    if not status_rows:
        status = "unknown_no_cells"
    elif not planned_known:
        status = "unknown_no_plan"
    elif not incomplete:
        status = "complete"
    else:
        status = "partial"
    return {
        "completion_status": status,
        "is_final": status == "complete",
        "planned_cells": len(status_rows) if planned_known else None,
        "completed_cells": len(completed),
        "incomplete_cells": len(incomplete) if planned_known else None,
        "partial_cells": sum(
            1
            for row in incomplete
            if int(row.get("selected_rows") or 0) > 0
        ),
        "pending_cells": sum(
            1
            for row in incomplete
            if int(row.get("selected_rows") or 0) == 0
        ),
        "incomplete_cell_details": [
            {
                "track": row["track"],
                "condition": row["condition"],
                "shard": row["shard"],
                "reason": row["reason"],
                "selected_rows": row["selected_rows"],
                "completed_problems": row["completed_problems"],
                "expected_problems": row["expected_problems"],
                "metrics_exists": row["metrics_exists"],
                "checkpoint_exists": row["checkpoint_exists"],
            }
            for row in incomplete
        ]
        if planned_known
        else [],
    }


def _require_complete(summary: dict[str, Any], *, allow_partial: bool) -> None:
    if allow_partial or summary.get("completion_status") in {"complete", "unknown_no_plan"}:
        return
    details = summary.get("incomplete_cell_details") or []
    preview = "; ".join(
        f"{row['track']}/{row['condition']}/{row['shard']} "
        f"{row['reason']} {row['completed_problems']}/{row['expected_problems']}"
        for row in details[:5]
    )
    suffix = f" First incomplete cells: {preview}." if preview else ""
    raise SystemExit(
        "run is incomplete; refusing to create final results_bundle.tar.gz "
        f"({summary.get('completed_cells')}/{summary.get('planned_cells')} cells complete)."
        f"{suffix} Re-run with --allow-partial to create a diagnostic partial bundle."
    )


def _collect_metrics(full_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((full_root / "runs").glob("*/*/shard-*/metrics.json")):
        track, condition, shard = _condition_key(metrics_path)
        metrics = _json(metrics_path)
        rows.append(
            {
                "track": track,
                "condition": condition,
                "shard": shard,
                "accuracy": metrics.get("accuracy"),
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


def _copy_cell_artifacts(full_root: Path, bundle_root: Path, *, include_candidates: bool) -> None:
    rels = [
        "metrics.json",
        "selected.jsonl",
        "scores.jsonl",
        "preflight.json",
        "run_plan_cell.json",
        "manifest.json",
        "checkpoint.json",
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
    for manifest in sorted((full_root / "archives").glob("*/archive_build_manifest.json")):
        dst = bundle_root / "archives" / manifest.parent.name / "archive_build_manifest.json"
        _copy_if_exists(manifest, dst)
    repo_root = Path(__file__).resolve().parents[1]
    for rel in [
        "configs/archive_build.yaml",
        "configs/eval.yaml",
    ]:
        _copy_if_exists(repo_root / rel, bundle_root / rel)


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

    status_rows = _cell_status_rows(full_root)
    completion = _completion_summary(status_rows)
    _require_complete(completion, allow_partial=args.allow_partial)

    default_bundle_name = (
        "partial_results_bundle"
        if args.allow_partial and completion["completion_status"] != "complete"
        else "results_bundle"
    )
    out = (args.out or (run_root / default_bundle_name)).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    metrics_rows = _collect_metrics(full_root)
    selected_rows = _collect_selected(full_root)
    agreement_rows = _agreement_rows(selected_rows)

    _write_csv(
        out / "summary" / "metrics.csv",
        metrics_rows,
        [
            "track",
            "condition",
            "shard",
            "accuracy",
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
        out / "summary" / "cell_status.csv",
        status_rows,
        [
            "track",
            "condition",
            "shard",
            "planned",
            "complete",
            "reason",
            "metrics_exists",
            "checkpoint_exists",
            "checkpoint_complete",
            "selected_rows",
            "completed_problems",
            "expected_problems",
            "n_problems",
            "n_candidates",
            "accuracy",
            "mean_selected_score",
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
        "included_candidates": bool(args.include_candidates),
        **completion,
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tar_path = _make_tarball(out)
    print(json.dumps({"bundle_dir": str(out), "tarball": str(tar_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
