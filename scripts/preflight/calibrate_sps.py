"""Validate SPS sharpening on a MATH500 sampled-baseline slice.

The slow block-MCMC approximation comparison is optional. The experiment gate
compares SPS against a matched temperature-1 sampled baseline rather than
greedy@1. It gates on matched selected pass@B and records per-candidate sample
accuracy as a diagnostic, because SPS should show up against sampled baselines
at a matched sampling budget.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-key", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--split", type=int, nargs=2, default=(0, 100))
    parser.add_argument("--baseline-condition", choices=["bon_temp1"], default="bon_temp1")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--samples-per-problem", type=int, default=8)
    parser.add_argument("--sps-alpha", type=float, default=4.0)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--sps-block-size", type=int, default=192)
    parser.add_argument("--sps-top-k", type=int, default=8)
    parser.add_argument("--sps-candidate-pool-size", type=int, default=8)
    parser.add_argument("--sps-rollouts-per-candidate", type=int, default=8)
    parser.add_argument("--sps-rollout-horizon", type=int, default=128)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--vllm-parity-artifact", type=Path, required=True)
    parser.add_argument("--vllm-dtype", default="float32")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=6144)
    parser.add_argument("--vllm-scoring-mode", default="forced_decode_v0")
    parser.add_argument("--run-kind", default="local")
    parser.add_argument("--cost-cap-dollars", type=float, default=0.0)
    parser.add_argument("--estimated-dollar-cost-per-cell", type=float, default=0.0)
    parser.add_argument("--estimated-wall-clock-seconds-per-cell", type=float, default=7200)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-sharpening-gain", type=float, default=0.01)
    parser.add_argument("--max-capped-failed-rate", type=float, default=0.01)
    parser.add_argument("--skip-mcmc-approximation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)


def _write_math_archive(path: Path) -> None:
    payload = [
        {
            "id": "direct",
            "prefix": (
                "Solve this math problem. Keep the solution concise and avoid "
                "long exploratory reasoning.\n\nProblem:\n"
            ),
            "suffix": (
                "\n\nUse at most 12 short lines. End immediately after the final "
                "answer, and put that answer inside \\boxed{}."
            ),
            "descriptor_hint": "concise_math_direct",
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cell(
    *,
    args: argparse.Namespace,
    archive: Path,
    out: Path,
    condition: str,
    power_sampler: str,
    block_num: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_condition.py",
        "--track",
        "math500",
        "--model-key",
        args.model_key,
        "--condition",
        condition,
        "--archive",
        str(archive),
        "--split",
        str(args.split[0]),
        str(args.split[1]),
        "--seed",
        "17",
        "--proicl-source-hash",
        "sps-calibration",
        "--preregistration-anchor",
        "TODO.md#sps-transition-amendment",
        "--out",
        str(out),
        "--backend",
        args.backend,
        "--samples-per-problem",
        str(args.samples_per_problem),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--mcmc-steps",
        str(args.mcmc_steps),
        "--mcmc-block-num",
        str(block_num),
        "--power-sampler",
        power_sampler,
        "--sps-top-k",
        str(args.sps_top_k),
        "--sps-candidate-pool-size",
        str(args.sps_candidate_pool_size),
        "--sps-rollouts-per-candidate",
        str(args.sps_rollouts_per_candidate),
        "--sps-rollout-horizon",
        str(args.sps_rollout_horizon),
        "--vllm-dtype",
        args.vllm_dtype,
        "--vllm-model-impl",
        args.vllm_model_impl,
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-scoring-mode",
        args.vllm_scoring_mode,
        "--vllm-parity-artifact",
        str(args.vllm_parity_artifact),
        "--trajectory-cache",
        str(out / "trajectory_cache.sqlite"),
        "--run-stage",
        "small_real_slice",
        "--run-kind",
        args.run_kind,
        "--estimated-dollar-cost",
        str(args.estimated_dollar_cost_per_cell),
        "--estimated-wall-clock-seconds",
        str(args.estimated_wall_clock_seconds_per_cell),
        "--cost-cap-dollars",
        str(args.cost_cap_dollars),
        "--user-authorized-paid-run",
    ]
    if condition == "single_prompt_power":
        cmd.extend(["--fixed-alpha", str(args.sps_alpha)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    _run(cmd)


def _read_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _correct(metrics: dict) -> int:
    return int(round(float(metrics["accuracy"]) * int(metrics["n_problems"])))


def _row_passed(row: dict) -> bool:
    verifier = row.get("verifier_result")
    if isinstance(verifier, dict):
        if verifier.get("passed") is True:
            return True
        try:
            return float(verifier.get("score") or 0.0) >= 1.0
        except (TypeError, ValueError):
            return False
    if row.get("selected_passed") is True:
        return True
    try:
        return float(row.get("selected_score") or row.get("score") or 0.0) >= 1.0
    except (TypeError, ValueError):
        return False


def _jsonl_stats(path: Path, *, max_new_tokens: int) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correct = sum(1 for row in rows if _row_passed(row))
    capped = [
        row
        for row in rows
        if int(row.get("generation_token_count") or 0) >= max_new_tokens
    ]
    capped_failed = [row for row in capped if not _row_passed(row)]
    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "capped_rows": len(capped),
        "capped_failed_rows": len(capped_failed),
        "max_generation_tokens": max(
            (int(row.get("generation_token_count") or 0) for row in rows),
            default=0,
        ),
    }


def _condition_stats(path: Path, *, max_new_tokens: int) -> dict:
    return {
        "selected": _jsonl_stats(path / "selected.jsonl", max_new_tokens=max_new_tokens),
        "candidates": _jsonl_stats(path / "candidates.jsonl", max_new_tokens=max_new_tokens),
    }


def _capped_failed_rate(stats: dict) -> float:
    n = int(stats["candidates"]["n"])
    if n == 0:
        return 0.0
    return int(stats["candidates"]["capped_failed_rows"]) / n


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "archive" / "rws_math_direct.json"
    _write_math_archive(archive)
    block_num = max(1, math.ceil(args.max_new_tokens / args.sps_block_size))

    baseline_out = args.out / args.baseline_condition
    mcmc_out = args.out / "mcmc"
    sps_out = args.out / "sps"
    _run_cell(
        args=args,
        archive=archive,
        out=baseline_out,
        condition=args.baseline_condition,
        power_sampler="mcmc",
        block_num=block_num,
    )
    _run_cell(
        args=args,
        archive=archive,
        out=sps_out,
        condition="single_prompt_power",
        power_sampler="sps",
        block_num=block_num,
    )

    baseline = _read_metrics(baseline_out)
    baseline_stats = _condition_stats(baseline_out, max_new_tokens=args.max_new_tokens)
    if args.skip_mcmc_approximation:
        mcmc = None
        mcmc_stats = None
    else:
        _run_cell(
            args=args,
            archive=archive,
            out=mcmc_out,
            condition="single_prompt_power",
            power_sampler="mcmc",
            block_num=block_num,
        )
        mcmc = _read_metrics(mcmc_out)
        mcmc_stats = _condition_stats(mcmc_out, max_new_tokens=args.max_new_tokens)
    sps = _read_metrics(sps_out)
    sps_stats = _condition_stats(sps_out, max_new_tokens=args.max_new_tokens)
    if int(baseline["n_problems"]) != int(sps["n_problems"]):
        raise SystemExit(
            "SPS calibration baseline/SPS problem counts differ: "
            f"{baseline['n_problems']} vs {sps['n_problems']}"
        )
    if int(baseline_stats["candidates"]["n"]) != int(sps_stats["candidates"]["n"]):
        raise SystemExit(
            "SPS calibration baseline/SPS candidate counts differ: "
            f"{baseline_stats['candidates']['n']} vs {sps_stats['candidates']['n']}"
        )
    candidate_diff = (
        None
        if mcmc_stats is None
        else abs(
            float(mcmc_stats["candidates"]["accuracy"])
            - float(sps_stats["candidates"]["accuracy"])
        )
    )
    selected_sharpening_gain = float(sps["accuracy"]) - float(baseline["accuracy"])
    candidate_sharpening_gain = (
        float(sps_stats["candidates"]["accuracy"])
        - float(baseline_stats["candidates"]["accuracy"])
    )
    selected_diff = (
        None
        if mcmc is None
        else abs(float(mcmc["accuracy"]) - float(sps["accuracy"]))
    )
    approximation_passed = (
        True if selected_diff is None else selected_diff <= args.tolerance
    )
    sharpening_passed = selected_sharpening_gain >= args.min_sharpening_gain
    baseline_capped_failed_rate = (
        baseline_stats["selected"]["capped_failed_rows"]
        / max(1, baseline_stats["selected"]["n"])
    )
    sps_capped_failed_rate = (
        sps_stats["selected"]["capped_failed_rows"]
        / max(1, sps_stats["selected"]["n"])
    )
    mcmc_capped_failed_rate = (
        None
        if mcmc_stats is None
        else (
            mcmc_stats["selected"]["capped_failed_rows"]
            / max(1, mcmc_stats["selected"]["n"])
        )
    )
    truncation_passed = (
        baseline_capped_failed_rate <= args.max_capped_failed_rate
        and sps_capped_failed_rate <= args.max_capped_failed_rate
        and (
            mcmc_capped_failed_rate is None
            or mcmc_capped_failed_rate <= args.max_capped_failed_rate
        )
    )
    n_problems = int(sps["n_problems"])
    summary = {
        "calibration_kind": "sps_vs_matched_sampled_baseline_math500",
        "passed": approximation_passed and sharpening_passed and truncation_passed,
        "approximation_passed": approximation_passed,
        "sharpening_passed": sharpening_passed,
        "truncation_passed": truncation_passed,
        "mcmc_approximation_skipped": args.skip_mcmc_approximation,
        "gate_metric": "selected_pass_at_b",
        "baseline_condition": args.baseline_condition,
        "samples_per_problem": args.samples_per_problem,
        "sps_alpha": args.sps_alpha,
        "tolerance": args.tolerance,
        "accuracy_abs_diff": selected_diff,
        "candidate_accuracy_abs_diff": candidate_diff,
        "min_sharpening_gain": args.min_sharpening_gain,
        "max_capped_failed_rate": args.max_capped_failed_rate,
        "baseline_capped_failed_rate": baseline_capped_failed_rate,
        "mcmc_capped_failed_rate": mcmc_capped_failed_rate,
        "sps_capped_failed_rate": sps_capped_failed_rate,
        "discrete_gain_resolution": 1.0 / n_problems,
        "sps_sharpening_gain": selected_sharpening_gain,
        "candidate_sharpening_gain": candidate_sharpening_gain,
        "baseline_accuracy": baseline["accuracy"],
        "baseline_correct": _correct(baseline),
        "mcmc_accuracy": None if mcmc is None else mcmc["accuracy"],
        "sps_accuracy": sps["accuracy"],
        "sps_correct": _correct(sps),
        "baseline_candidate_accuracy": baseline_stats["candidates"]["accuracy"],
        "baseline_candidate_correct": baseline_stats["candidates"]["correct"],
        "mcmc_candidate_accuracy": (
            None if mcmc_stats is None else mcmc_stats["candidates"]["accuracy"]
        ),
        "sps_candidate_accuracy": sps_stats["candidates"]["accuracy"],
        "sps_candidate_correct": sps_stats["candidates"]["correct"],
        "baseline_stats": baseline_stats,
        "mcmc_stats": mcmc_stats,
        "sps_stats": sps_stats,
        "n_problems": n_problems,
        "max_new_tokens": args.max_new_tokens,
        "sps_block_size": args.sps_block_size,
        "block_num": block_num,
        "sps_top_k": args.sps_top_k,
        "sps_candidate_pool_size": args.sps_candidate_pool_size,
        "sps_rollouts_per_candidate": args.sps_rollouts_per_candidate,
        "sps_rollout_horizon": args.sps_rollout_horizon,
        "artifacts": {
            "baseline": str(baseline_out),
            "mcmc": None if mcmc is None else str(mcmc_out),
            "sps": str(sps_out),
            "archive": str(archive),
        },
    }
    (args.out / "sps_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("SPS calibration failed")


if __name__ == "__main__":
    main()
