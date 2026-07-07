"""Run one ProICL condition for any registered track.

`--preflight-only` validates launch gates and exits before loading model
backends or datasets. Paid/full runs still require explicit authorization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        required=True,
        choices=[
            "math500",
            "humaneval_plus",
            "gpqa_diamond",
            "reasoning_gym_boxnet",
            "reasoning_gym_graph_color",
            "reasoning_gym_graph_color_n5",
            "reasoning_gym_graph_color_n8",
            "reasoning_gym_graph_color_n10",
            "reasoning_gym_graph_color_n12",
            "reasoning_gym_graph_color_n13",
            "reasoning_gym_graph_color_n14",
            "reasoning_gym_graph_color_n15",
            "reasoning_gym_graph_color_n16",
            "reasoning_gym_graph_color_n18",
            "reasoning_gym_graph_color_n20",
            "reasoning_gym_family_relationships",
            "reasoning_gym_acre",
            "reasoning_gym_game_of_life_halting",
            "reasoning_gym_maze",
            "reasoning_gym_palindrome_generation",
            "reasoning_gym_palindrome",
            "reasoning_gym_letter_counting",
        ],
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=int, nargs=2, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--proicl-source-hash",
        "--polaris-source-hash",
        dest="polaris_source_hash",
        metavar="PROICL_SOURCE_HASH",
        required=True,
    )
    parser.add_argument("--preregistration-anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "vllm", "sglang"], default="hf")
    parser.add_argument("--sglang-base-url", default="http://localhost:30000")
    parser.add_argument("--samples-per-problem", type=int, default=None)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--mcmc-steps", type=int, default=None)
    parser.add_argument("--mcmc-block-num", type=int, default=None)
    parser.add_argument(
        "--power-block-num",
        type=int,
        default=None,
        help="Public alias for the number of power-sampling blocks.",
    )
    parser.add_argument(
        "--sps-block-num",
        type=int,
        default=None,
        help="SPS-facing alias for --power-block-num.",
    )
    parser.add_argument("--sps-top-k", type=int, default=8)
    parser.add_argument("--sps-candidate-pool-size", type=int, default=8)
    parser.add_argument("--sps-rollouts-per-candidate", type=int, default=8)
    parser.add_argument("--sps-rollout-horizon", type=int, default=None)
    parser.add_argument(
        "--power-sampler",
        choices=["mcmc", "sps"],
        default="mcmc",
        help="Power-sampling implementation for alpha>1 candidates.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Override power-sampling alpha for calibration/sweeps.",
    )
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--trajectory-cache", type=Path, default=None)
    parser.add_argument("--memory-store", type=Path, default=None)
    parser.add_argument(
        "--memory-mode",
        choices=["off", "raw_traces", "distilled_strategies", "cross_task"],
        default="off",
    )
    parser.add_argument("--admit-memory", action="store_true")
    parser.add_argument("--online-memory", action="store_true")
    parser.add_argument("--enable-repair", action="store_true")
    parser.add_argument("--enable-fork-search", action="store_true")
    parser.add_argument("--archive-build-id", default=None)
    parser.add_argument("--memory-build-id", default=None)
    parser.add_argument(
        "--run-stage",
        choices=["smoke", "small_real_slice", "final"],
        default="smoke",
    )
    parser.add_argument(
        "--run-kind",
        choices=[
            "modal",
            "mithril",
            "flow",
            "phase",
            "farmshare",
            "cloudrift",
            "bulk",
            "local",
        ],
        default="local",
    )
    parser.add_argument("--problem-ids", nargs="+", default=None)
    parser.add_argument("--run-plan", type=Path, default=None)
    parser.add_argument("--estimated-dollar-cost", type=float, default=None)
    parser.add_argument("--estimated-wall-clock-seconds", type=float, default=None)
    parser.add_argument("--cost-cap-dollars", type=float, default=None)
    parser.add_argument("--user-authorized-paid-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--vllm-dtype", default="float32")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument(
        "--no-vllm-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching for GPUs/backends where prefix-prefill is unstable.",
    )
    parser.add_argument(
        "--vllm-scoring-mode",
        choices=["forced_decode_v0", "native_segment"],
        default="forced_decode_v0",
    )
    parser.add_argument("--vllm-parity-artifact", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rws-commit", default="")
    parser.add_argument("--evalplus-commit", default="")
    parser.add_argument("--gepa-commit", default="")
    parser.add_argument("--dc-commit", default="")
    return parser.parse_args()


def _resolve_power_block_num(args: argparse.Namespace) -> int | None:
    supplied = [
        value
        for value in (args.mcmc_block_num, args.power_block_num, args.sps_block_num)
        if value is not None
    ]
    if not supplied:
        return None
    if len(set(supplied)) != 1:
        raise SystemExit(
            "--mcmc-block-num, --power-block-num, and --sps-block-num are aliases; "
            "provide at most one value or use the same value for each."
        )
    return supplied[0]


def _protocol_sync_passed() -> bool:
    guard = REPO_ROOT / "scripts" / "preflight" / "check_protocol_sync.sh"
    if not guard.exists():
        return True
    result = subprocess.run(
        ["bash", str(guard)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _load_run_plan_cell(path: Path | None, args: argparse.Namespace) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        if (
            cell.get("track") == args.track
            and cell.get("model_key") == args.model_key
            and cell.get("condition") == args.condition
            and list(cell.get("split", [])) == list(args.split)
            and int(cell.get("seed", args.seed)) == args.seed
            and cell.get("run_stage") == args.run_stage
        ):
            return cell
    raise SystemExit("run plan does not contain the requested cell")


def _condition_requires_vllm_calibration(condition: str) -> bool:
    """Only MCMC/power-sampling cells need HF-vLLM scorer equivalence.

    Greedy and low-temperature BoN cells may still use vLLM for generation, but
    they do not call the teacher-forced scorer that the calibration artifact
    validates.
    """
    return condition in {
        "single_prompt_power",
        "full_archive_fixed",
        "full_archive_mixed",
        "full_archive_decaying",
        "polaris_full_verified_memory",
        "proicl_gepa_mcmc",
        "proicl_gepa_mcmc_memory",
        "mixed_alpha_mcmc",
        "proicl_gepa_mcmc_repair",
        "proicl_gepa_mcmc_fork_repair",
        "proicl_gepa_mcmc_fork_repair_memory",
    }


def main() -> None:
    args = _parse_args()
    power_block_num = _resolve_power_block_num(args)

    from polaris.infra.preflight import (
        PaidRunPreflight,
        PreflightError,
        validate_paid_run_preflight,
    )
    from polaris.registry import (
        resolve_model,
        validate_condition_for_track,
        validate_model_for_track,
    )

    validate_model_for_track(args.model_key, args.track)
    validate_condition_for_track(args.condition, args.track)
    if args.power_sampler == "sps" and args.backend not in {"vllm", "sglang"}:
        raise SystemExit("power_sampler=sps currently requires --backend vllm or sglang")
    model = resolve_model(args.model_key)
    model_revision = args.model_revision or model.revision
    protocol_sync = _protocol_sync_passed()
    vllm_calibration = None
    vllm_calibration_required = (
        args.backend == "vllm"
        and _condition_requires_vllm_calibration(args.condition)
    )
    vllm_parity_artifact_for_condition = (
        args.vllm_parity_artifact if vllm_calibration_required else None
    )
    if vllm_calibration_required:
        from polaris.infra.vllm_calibration import (
            CalibrationArtifactError,
            validate_vllm_calibration_artifact,
        )

        try:
            vllm_calibration = validate_vllm_calibration_artifact(
                vllm_parity_artifact_for_condition,
                expected_model_id=model.hf_id,
            )
        except CalibrationArtifactError as exc:
            raise SystemExit(f"vLLM calibration gate failed: {exc}") from exc

    preflight_report = {
        "required": args.run_stage != "smoke",
        "passed": args.run_stage == "smoke",
        "protocol_sync_passed": protocol_sync,
        "model_revision": model_revision,
        "model_revision_commit": model.revision_commit,
        "model_artifact_etags": model.artifact_etags or {},
        "backend": args.backend,
        "power_sampler": args.power_sampler,
        "vllm_scoring_mode": args.vllm_scoring_mode if args.backend == "vllm" else None,
        "vllm_calibration_required": vllm_calibration_required
        if args.backend == "vllm"
        else None,
        "vllm_parity_artifact": str(vllm_parity_artifact_for_condition)
        if vllm_parity_artifact_for_condition is not None
        else None,
        "vllm_calibration_passed": vllm_calibration is not None
        if vllm_calibration_required
        else None,
    }
    if args.run_stage != "smoke" or args.user_authorized_paid_run:
        try:
            preflight_report = validate_paid_run_preflight(
                PaidRunPreflight(
                    run_kind=args.run_kind,
                    artifact_dir=args.out,
                    cache_path=args.trajectory_cache,
                    split=(args.split[0], args.split[1]),
                    seed=args.seed,
                    model_id=model.hf_id,
                    backend=args.backend,
                    estimated_dollar_cost=args.estimated_dollar_cost,
                    estimated_wall_clock_seconds=args.estimated_wall_clock_seconds,
                    cost_cap_dollars=args.cost_cap_dollars,
                    user_authorized=args.user_authorized_paid_run,
                )
            )
            preflight_report["model_revision"] = model_revision
            preflight_report["model_revision_commit"] = model.revision_commit
            preflight_report["model_artifact_etags"] = model.artifact_etags or {}
            preflight_report["protocol_sync_passed"] = protocol_sync
            preflight_report["power_sampler"] = args.power_sampler
            preflight_report["vllm_scoring_mode"] = (
                args.vllm_scoring_mode if args.backend == "vllm" else None
            )
            preflight_report["vllm_calibration_required"] = (
                vllm_calibration_required if args.backend == "vllm" else None
            )
            preflight_report["vllm_parity_artifact"] = (
                str(vllm_parity_artifact_for_condition)
                if vllm_parity_artifact_for_condition is not None
                else None
            )
            preflight_report["vllm_calibration_passed"] = (
                vllm_calibration is not None if vllm_calibration_required else None
            )
        except PreflightError as exc:
            raise SystemExit(f"paid-run preflight failed: {exc}") from exc

    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return

    from polaris.baselines import PUBLISHED_COMPARATORS
    from polaris.config import MCMC_BLOCK_NUM, MCMC_STEPS
    from polaris.core.archive import FrozenArchive
    from polaris.io.manifest import compute_archive_hash
    from polaris.io.trajectory_cache import TrajectoryCache
    from polaris.runners.condition_runner import run_condition

    archive_payload = json.loads(args.archive.read_text(encoding="utf-8"))
    if isinstance(archive_payload, dict):
        entries = archive_payload.get("entries", archive_payload)
        cell_fitness = archive_payload.get("cell_fitness", {})
    else:
        entries = archive_payload
        cell_fitness = {}
    archive = FrozenArchive.from_entries(entries)
    archive_hash = compute_archive_hash(archive)

    if args.condition in PUBLISHED_COMPARATORS:
        sampler = None
    elif args.backend == "hf":
        from polaris.infra.serving.hf import RWSGenerator

        sampler = RWSGenerator(
            model_id=model.hf_id,
            revision=model_revision,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    elif args.backend == "vllm":
        from polaris.infra.serving.vllm import VLLMGenerator

        sampler = VLLMGenerator(
            model_id=model.hf_id,
            revision=model_revision,
            seed=args.seed,
            dtype=args.vllm_dtype,
            model_impl=args.vllm_model_impl,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            local_files_only=args.local_files_only,
            scoring_mode=args.vllm_scoring_mode,
            enable_prefix_caching=not args.no_vllm_prefix_caching,
            parity_artifact_path=str(vllm_parity_artifact_for_condition)
            if vllm_parity_artifact_for_condition is not None
            else None,
        )
    else:
        from polaris.infra.serving.sglang import SGLangGenerator

        sampler = SGLangGenerator(
            model_id=model.hf_id,
            base_url=args.sglang_base_url,
            seed=args.seed,
        )
    if sampler is None:
        serving_backend_metadata = {
            "backend": args.backend,
            "scoring_mode": "published_comparator",
        }
    else:
        metadata_fn = getattr(sampler, "runtime_metadata", None)
        serving_backend_metadata = (
            metadata_fn()
            if callable(metadata_fn)
            else {"backend": args.backend, "scoring_mode": "unknown"}
        )

    trajectory_cache = (
        TrajectoryCache(args.trajectory_cache) if args.trajectory_cache is not None else None
    )
    problems = None
    if (
        args.shard_id is not None
        or args.num_shards is not None
        or args.problem_ids is not None
    ):
        from polaris.runners.condition_runner import load_track_slice

        all_problems = load_track_slice(args.track, args.split[0], args.split[1])
        if args.problem_ids is not None:
            requested_ids = set(args.problem_ids)
            by_id = {str(problem.problem_id): problem for problem in all_problems}
            missing = sorted(requested_ids - set(by_id))
            if missing:
                raise SystemExit(
                    "--problem-ids contains ids outside selected split: "
                    + ", ".join(missing)
                )
            all_problems = [problem for problem in all_problems if str(problem.problem_id) in requested_ids]
        problems = all_problems
    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None:
            raise SystemExit("--shard-id and --num-shards must be provided together")
        from polaris.infra.farmshare import shard_indices

        selected = shard_indices(len(problems or []), args.shard_id, args.num_shards)
        problems = [(problems or [])[idx] for idx in selected]
    try:
        metrics = run_condition(
            track=args.track,
            split=(args.split[0], args.split[1]),
            model_key=args.model_key,
            out_dir=args.out,
            condition=args.condition,
            archive=archive,
            cell_fitness=cell_fitness,
            sampler=sampler,
            seed=args.seed,
            archive_hash=archive_hash,
            polaris_source_hash=args.polaris_source_hash,
            vendored_commits={
                "rws": args.rws_commit,
                "evalplus": args.evalplus_commit,
                "gepa": args.gepa_commit,
                "dc": args.dc_commit,
            },
            preregistration_anchor=args.preregistration_anchor,
            model_id=model.hf_id,
            model_revision=model_revision,
            model_revision_commit=model.revision_commit,
            model_artifact_etags=model.artifact_etags or {},
            trajectory_cache=trajectory_cache,
            memory_store_path=args.memory_store,
            memory_mode=args.memory_mode,
            admit_memory=args.admit_memory,
            online_memory=args.online_memory,
            enable_repair=args.enable_repair,
            enable_fork_search=args.enable_fork_search,
            archive_build_id=args.archive_build_id,
            memory_build_id=args.memory_build_id,
            run_stage=args.run_stage,
            run_plan_cell=_load_run_plan_cell(args.run_plan, args),
            preflight_report=preflight_report,
            dataset_lock_id=None,
            problems=problems,
            budget_override=args.samples_per_problem,
            low_temp_temperature=args.sampling_temperature,
            max_new_tokens=args.max_new_tokens
            if args.max_new_tokens is not None
            else None,
            mcmc_steps=args.mcmc_steps
            if args.mcmc_steps is not None
            else MCMC_STEPS,
            mcmc_block_num=power_block_num
            if power_block_num is not None
            else MCMC_BLOCK_NUM,
            power_sampler=args.power_sampler,
            fixed_alpha=args.fixed_alpha,
            sps_top_k=args.sps_top_k,
            sps_candidate_pool_size=args.sps_candidate_pool_size,
            sps_rollouts_per_candidate=args.sps_rollouts_per_candidate,
            sps_rollout_horizon=args.sps_rollout_horizon,
            serving_backend_metadata=serving_backend_metadata,
        )
    finally:
        if trajectory_cache is not None:
            trajectory_cache.close()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
