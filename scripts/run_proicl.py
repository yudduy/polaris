"""Plan, build, and analyze the ProICL fast-weight recovery audit."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--root", default="runs/proicl")
    plan.add_argument("--tracks", nargs="+", default=None)
    plan.add_argument("--problem-count", type=int, default=100)
    plan.add_argument("--rollout-budget", type=int, default=128)
    plan.add_argument("--num-shards", type=int, default=4)
    plan.add_argument("--seed", type=int, default=17)
    plan.add_argument(
        "--archive-scope",
        choices=["transductive_support", "within_family", "cross_family_curriculum"],
        default="within_family",
    )
    plan.add_argument("--archive-train-tracks", nargs="+", default=None)
    plan.add_argument("--archive-heldout-tracks", nargs="+", default=None)
    plan.add_argument("--conditions", nargs="+", default=None)
    plan.add_argument("--out", type=Path, required=True)

    direct = sub.add_parser("write-direct-archives")
    direct.add_argument("--root", default="runs/proicl")
    direct.add_argument("--tracks", nargs="+", default=None)

    archive = sub.add_parser("build-archive")
    archive.add_argument("--out", type=Path, required=True)
    archive.add_argument(
        "--tracks",
        nargs="+",
        default=[
            "reasoning_gym_family_relationships",
            "reasoning_gym_graph_color_n5",
            "reasoning_gym_graph_color_n8",
            "reasoning_gym_graph_color_n10",
            "reasoning_gym_graph_color_n13",
            "reasoning_gym_graph_color_n15",
            "reasoning_gym_graph_color_n18",
            "reasoning_gym_graph_color_n20",
            "reasoning_gym_boxnet",
        ],
    )
    archive.add_argument(
        "--archive-scope",
        choices=["transductive_support", "within_family", "cross_family_curriculum"],
        default="within_family",
    )
    archive.add_argument("--heldout-tracks", nargs="+", default=None)
    archive.add_argument("--dev-split", type=int, nargs=2, default=(0, 100))
    archive.add_argument("--archive-size", type=int, default=16)
    archive.add_argument("--max-metric-calls", type=int, default=1000)
    archive.add_argument(
        "--reflection-provider",
        choices=["none", "xai", "local-hf"],
        default="none",
    )
    archive.add_argument("--live-gepa", action="store_true")
    archive.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    archive.add_argument("--model-key", default="deepseek-r1-distill-qwen-1.5b")
    archive.add_argument("--sampler-max-new-tokens", type=int, default=512)
    archive.add_argument("--sampler-backend", choices=["hf", "vllm"], default="hf")
    archive.add_argument("--vllm-dtype", default="float32")
    archive.add_argument("--vllm-model-impl", default="transformers")
    archive.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    archive.add_argument("--vllm-max-model-len", type=int, default=None)
    archive.add_argument("--no-vllm-prefix-caching", action="store_true")
    archive.add_argument(
        "--vllm-scoring-mode",
        choices=["forced_decode_v0", "native_segment"],
        default="forced_decode_v0",
    )
    archive.add_argument("--vllm-parity-artifact", default=None)
    archive.add_argument("--local-files-only", action="store_true")
    archive.add_argument("--cuda-visible-devices", default=None)
    archive.add_argument("--reflection-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    archive.add_argument("--reflection-revision", default=None)
    archive.add_argument("--reflection-temperature", type=float, default=0.7)
    archive.add_argument("--reflection-max-new-tokens", type=int, default=1024)
    archive.add_argument("--seed", type=int, default=17)
    archive.add_argument("--dry-run", action="store_true")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--accuracies-json", type=Path, required=True)
    analyze.add_argument("--out-dir", type=Path, required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--tracks", nargs="+", required=True)
    aggregate.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _cmd_plan(args: argparse.Namespace) -> None:
    from polaris.proicl.run_graph import (
        PROICL_PRIMARY_TRACKS,
        build_proicl_run_graph,
        write_proicl_run_graph,
    )

    tracks = tuple(args.tracks) if args.tracks else PROICL_PRIMARY_TRACKS
    graph = build_proicl_run_graph(
        root=args.root,
        tracks=tracks,
        problem_count=args.problem_count,
        rollout_budget=args.rollout_budget,
        num_shards=args.num_shards,
        seed=args.seed,
        archive_scope=args.archive_scope,
        archive_train_tracks=args.archive_train_tracks,
        archive_heldout_tracks=args.archive_heldout_tracks,
        conditions=tuple(args.conditions) if args.conditions else None,
    )
    write_proicl_run_graph(args.out, graph)
    print(json.dumps({"out": str(args.out), "cells": len(graph["cells"])}, sort_keys=True))


def _cmd_write_direct_archives(args: argparse.Namespace) -> None:
    from polaris.proicl.run_graph import PROICL_PRIMARY_TRACKS
    from polaris.prorl_recovery.orchestration import (
        direct_archive_kind_for_track,
        write_archive,
    )

    tracks = tuple(args.tracks) if args.tracks else PROICL_PRIMARY_TRACKS
    written = []
    for track in tracks:
        path = Path(args.root) / "archives" / track / "direct.json"
        write_archive(path, kind=direct_archive_kind_for_track(track))
        written.append(str(path))
    print(json.dumps({"archives": written}, sort_keys=True))


def _install_gepa_archive_stop_handlers(out_dir: Path) -> None:
    stop_file = out_dir / "gepa_run" / "gepa.stop"

    def _handle(signum: int, _frame: object) -> None:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        stop_file.write_text(
            f"signal {signum} requested graceful GEPA stop at {ts}\n",
            encoding="utf-8",
        )
        print(f"[ProICL] requested GEPA checkpoint stop: {stop_file}", file=sys.stderr, flush=True)

    for sig_name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle)


def _cmd_build_archive(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.live_gepa:
        raise ValueError("live GEPA archive construction requires --live-gepa")
    if not args.dry_run and args.reflection_provider == "none":
        raise ValueError("live GEPA requires --reflection-provider xai or local-hf")
    if not args.dry_run:
        _install_gepa_archive_stop_handlers(args.out)

    sampler = None
    scorer = None
    reflection_lm = None
    reflection_config = None
    provider = "local_hf" if args.reflection_provider == "local-hf" else args.reflection_provider
    if not args.dry_run:
        if args.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        from polaris.config import MODEL_REGISTRY
        from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym
        from polaris.gepa_reflection import (
            LocalHFReflectionConfig,
            XAIReflectionConfig,
            load_env_file,
            make_local_hf_reflection_lm,
            make_xai_reflection_lm,
        )

        model_spec = MODEL_REGISTRY[args.model_key]
        if args.sampler_backend == "vllm":
            from polaris.infra.serving.vllm import VLLMGenerator

            sampler = VLLMGenerator(
                model_id=model_spec["hf_id"],
                revision=model_spec.get("revision"),
                seed=args.seed,
                dtype=args.vllm_dtype,
                model_impl=args.vllm_model_impl,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                local_files_only=args.local_files_only,
                scoring_mode=args.vllm_scoring_mode,
                enable_prefix_caching=not args.no_vllm_prefix_caching,
                parity_artifact_path=args.vllm_parity_artifact,
            )
        else:
            from polaris.infra.serving.hf import RWSGenerator

            sampler = RWSGenerator(
                model_id=model_spec["hf_id"],
                revision=model_spec.get("revision"),
                seed=args.seed,
                local_files_only=args.local_files_only,
            )
        scorer = score_reasoning_gym
        if args.reflection_provider == "xai":
            load_env_file(args.env_file)
            reflection_config = XAIReflectionConfig.from_env(require_key=True)
            reflection_config = XAIReflectionConfig(
                api_key=reflection_config.api_key,
                base_url=reflection_config.base_url,
                model=reflection_config.model,
                litellm_model=reflection_config.litellm_model,
                input_price_per_million=reflection_config.input_price_per_million,
                output_price_per_million=reflection_config.output_price_per_million,
                initial_cost_cap_dollars=reflection_config.initial_cost_cap_dollars,
                hard_cost_cap_dollars=reflection_config.hard_cost_cap_dollars,
                temperature=args.reflection_temperature,
                max_tokens=args.reflection_max_new_tokens,
            )
            reflection_lm = make_xai_reflection_lm(reflection_config)
        elif args.reflection_provider == "local-hf":
            reflection_config = LocalHFReflectionConfig(
                model_id=args.reflection_model_id,
                revision=args.reflection_revision,
                temperature=args.reflection_temperature,
                max_new_tokens=args.reflection_max_new_tokens,
                local_files_only=args.local_files_only,
            )
            reflection_lm = make_local_hf_reflection_lm(reflection_config)

    from polaris.proicl.archive import build_cross_task_curriculum_archive

    manifest = build_cross_task_curriculum_archive(
        out_dir=args.out,
        tracks=tuple(args.tracks),
        archive_scope=args.archive_scope,
        heldout_tracks=tuple(args.heldout_tracks or args.tracks),
        dev_split=(args.dev_split[0], args.dev_split[1]),
        archive_size=args.archive_size,
        dry_run=args.dry_run,
        max_metric_calls=args.max_metric_calls,
        reflection_provider=provider,
        sampler=sampler,
        scorer=scorer,
        reflection_lm=reflection_lm,
        reflection_config=reflection_config,
        sampler_max_new_tokens=args.sampler_max_new_tokens,
        seed=args.seed,
    )
    print(json.dumps({"out": str(args.out), "archive_build_id": manifest["archive_build_id"]}, sort_keys=True))


def _cmd_analyze(args: argparse.Namespace) -> None:
    from polaris.proicl.analysis import write_proicl_decomposition

    accuracies = json.loads(args.accuracies_json.read_text(encoding="utf-8"))
    report = write_proicl_decomposition(accuracies=accuracies, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


def _cmd_aggregate(args: argparse.Namespace) -> None:
    from polaris.proicl.analysis import write_proicl_decomposition_by_track

    report = write_proicl_decomposition_by_track(
        root=args.root,
        tracks=tuple(args.tracks),
        out_dir=args.out_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    dispatch = {
        "plan": _cmd_plan,
        "write-direct-archives": _cmd_write_direct_archives,
        "build-archive": _cmd_build_archive,
        "analyze": _cmd_analyze,
        "aggregate": _cmd_aggregate,
    }
    dispatch[args.action](args)


if __name__ == "__main__":
    main()
