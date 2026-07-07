"""Fail-fast vLLM/SPS runtime preflight for release experiment launchers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class RuntimeCandidate:
    dtype: str
    attention_backend: str | None
    prefix_caching: bool
    sps_vllm_batch_size: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--track", default="reasoning_gym_boxnet")
    parser.add_argument("--split", type=int, nargs=2, required=True)
    parser.add_argument("--model-key", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--vllm-parity-artifact", type=Path, required=True)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-attention-backend", default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-prefix-caching", choices=["0", "1"], default="1")
    parser.add_argument("--sps-vllm-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sps-block-num", type=int, default=6)
    parser.add_argument("--sps-top-k", type=int, default=8)
    parser.add_argument("--sps-candidate-pool-size", type=int, default=8)
    parser.add_argument("--sps-rollouts-per-candidate", type=int, default=8)
    parser.add_argument("--sps-rollout-horizon", type=int, default=128)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--gpu-profile", default="generic")
    parser.add_argument("--gpu-names", default="unknown")
    parser.add_argument("--gpu-min-memory-mib", type=int, default=0)
    parser.add_argument("--run-kind", default="local")
    parser.add_argument("--estimated-wall-clock-seconds", type=float, default=7200)
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _request_key(args: argparse.Namespace) -> dict[str, Any]:
    from polaris.proicl.launcher import source_hash

    payload = {
        "schema": "proicl_backend_preflight_request.v1",
        "source_hash": source_hash(REPO_ROOT),
        "track": args.track,
        "split": [int(args.split[0]), int(args.split[1])],
        "model_key": args.model_key,
        "gpu_profile": args.gpu_profile,
        "gpu_names": args.gpu_names,
        "gpu_min_memory_mib": int(args.gpu_min_memory_mib),
        "vllm_version": _version("vllm"),
        "torch_version": _version("torch"),
        "runtime_request": {
            "dtype": args.vllm_dtype,
            "model_impl": args.vllm_model_impl,
            "attention_backend": args.vllm_attention_backend,
            "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "max_model_len": int(args.vllm_max_model_len),
            "prefix_caching": args.vllm_prefix_caching == "1",
            "sps_vllm_batch_size": int(args.sps_vllm_batch_size),
        },
        "sps": {
            "max_new_tokens": int(args.max_new_tokens),
            "block_num": int(args.sps_block_num),
            "top_k": int(args.sps_top_k),
            "candidate_pool_size": int(args.sps_candidate_pool_size),
            "rollouts_per_candidate": int(args.sps_rollouts_per_candidate),
            "rollout_horizon": int(args.sps_rollout_horizon),
        },
        "calibration_artifact_sha256": _sha256(args.vllm_parity_artifact),
    }
    return payload


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_failure(returncode: int, stderr_tail: str) -> str:
    text = stderr_tail.lower()
    if "paid-run preflight failed" in text:
        return "paid_run_preflight"
    if "vllm calibration gate failed" in text:
        return "vllm_calibration_gate"
    if "outofresources" in text and "shared memory" in text:
        return "vllm_attention_shared_memory"
    if "token id" in text and "out of vocabulary" in text:
        return "invalid_token_id"
    if "bfloat16 is only supported" in text and "compute capability" in text:
        return "bf16_unsupported"
    if "cuda out of memory" in text or "torch.cuda.outofmemoryerror" in text:
        return "cuda_oom"
    if returncode == -9 or "oom_kill" in text or "out of memory" in text:
        return "host_oom_or_sigkill"
    if "flashattention" in text or "flash attention" in text:
        return "flash_attention_unavailable"
    if "no module named" in text and "vllm" in text:
        return "vllm_not_installed"
    return "unknown_runtime_failure"


def _suggestion(kind: str) -> str:
    return {
        "vllm_attention_shared_memory": (
            "Use bfloat16/FlashAttention and lower SPS_VLLM_BATCH_SIZE; "
            "avoid float32/xFormers for SPS on 48GB GPUs."
        ),
        "invalid_token_id": "Pull the latest token-mask fix and rerun this preflight.",
        "bf16_unsupported": "Use VLLM_DTYPE=float16 on pre-Ampere GPUs such as TITAN RTX, T4, or V100.",
        "cuda_oom": "Lower VLLM_GPU_MEMORY_UTILIZATION or SPS_VLLM_BATCH_SIZE.",
        "host_oom_or_sigkill": "Lower MAX_PARALLEL_CELLS and request more host RAM.",
        "flash_attention_unavailable": (
            "Install a vLLM stack with FlashAttention support or use an H100/A100 image that has it."
        ),
        "vllm_not_installed": "Install requirements.txt inside the experiment environment.",
        "paid_run_preflight": "The backend preflight wrapper is missing required run metadata.",
        "vllm_calibration_gate": (
            "Use a passing calibration_summary.json. Production runtime may be BF16, "
            "but the parity calibration should remain the known passing float32 artifact."
        ),
    }.get(kind, "Inspect stderr.log; the runtime failed before the full experiment started.")


def _candidate_list(args: argparse.Namespace) -> list[RuntimeCandidate]:
    first = RuntimeCandidate(
        dtype=args.vllm_dtype,
        attention_backend=args.vllm_attention_backend or None,
        prefix_caching=args.vllm_prefix_caching == "1",
        sps_vllm_batch_size=max(1, int(args.sps_vllm_batch_size)),
    )
    candidates = [first]
    if args.allow_fallbacks:
        fallback_dtypes = [first.dtype]
        if first.dtype == "bfloat16":
            fallback_dtypes.append("float16")
        for dtype in fallback_dtypes:
            candidates.append(
                RuntimeCandidate(
                    dtype=dtype,
                    attention_backend=first.attention_backend,
                    prefix_caching=first.prefix_caching,
                    sps_vllm_batch_size=first.sps_vllm_batch_size,
                )
            )
            if first.prefix_caching:
                candidates.append(
                    RuntimeCandidate(
                        dtype=dtype,
                        attention_backend=first.attention_backend,
                        prefix_caching=False,
                        sps_vllm_batch_size=first.sps_vllm_batch_size,
                    )
                )
            size = first.sps_vllm_batch_size
            while size > 1:
                size = max(1, size // 2)
                candidates.append(
                    RuntimeCandidate(
                        dtype=dtype,
                        attention_backend=first.attention_backend,
                        prefix_caching=False,
                        sps_vllm_batch_size=size,
                    )
                )
                if size == 1:
                    break
    seen: set[tuple[str, str | None, bool, int]] = set()
    unique = []
    for candidate in candidates:
        key = (
            candidate.dtype,
            candidate.attention_backend,
            candidate.prefix_caching,
            candidate.sps_vllm_batch_size,
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _write_archive(path: Path, track: str) -> None:
    from polaris.prorl_recovery.orchestration import (
        direct_archive_kind_for_track,
        write_archive,
    )

    write_archive(path, kind=direct_archive_kind_for_track(track))


def _run_attempt(
    *,
    args: argparse.Namespace,
    candidate: RuntimeCandidate,
    attempt_dir: Path,
) -> dict[str, Any]:
    archive = attempt_dir / "archives" / args.track / "direct.json"
    _write_archive(archive, args.track)
    run_dir = attempt_dir / "run"
    cache_dir = attempt_dir / "cache"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if candidate.attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = candidate.attention_backend
    else:
        env.pop("VLLM_ATTENTION_BACKEND", None)
    env["PROICL_SPS_VLLM_BATCH_SIZE"] = str(candidate.sps_vllm_batch_size)
    env["SPS_VLLM_BATCH_SIZE"] = str(candidate.sps_vllm_batch_size)

    split_end = min(int(args.split[1]), int(args.split[0]) + 1)
    cmd = [
        sys.executable,
        "scripts/run_condition.py",
        "--track",
        args.track,
        "--model-key",
        args.model_key,
        "--condition",
        "single_prompt_power",
        "--archive",
        str(archive),
        "--split",
        str(args.split[0]),
        str(split_end),
        "--seed",
        "17",
        "--proicl-source-hash",
        "backend-preflight",
        "--preregistration-anchor",
        "TODO.PROICL.md#backend-preflight",
        "--out",
        str(run_dir),
        "--trajectory-cache",
        str(cache_dir),
        "--backend",
        "vllm",
        "--samples-per-problem",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--sps-block-num",
        str(args.sps_block_num),
        "--power-sampler",
        "sps",
        "--sps-top-k",
        str(args.sps_top_k),
        "--sps-candidate-pool-size",
        str(args.sps_candidate_pool_size),
        "--sps-rollouts-per-candidate",
        str(args.sps_rollouts_per_candidate),
        "--sps-rollout-horizon",
        str(args.sps_rollout_horizon),
        "--vllm-dtype",
        candidate.dtype,
        "--vllm-model-impl",
        args.vllm_model_impl,
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-parity-artifact",
        str(args.vllm_parity_artifact),
        "--run-stage",
        "smoke",
        "--run-kind",
        args.run_kind,
        "--estimated-dollar-cost",
        "0",
        "--estimated-wall-clock-seconds",
        str(args.estimated_wall_clock_seconds),
        "--cost-cap-dollars",
        "0",
        "--user-authorized-paid-run",
    ]
    if not candidate.prefix_caching:
        cmd.append("--no-vllm-prefix-caching")
    if args.local_files_only:
        cmd.append("--local-files-only")

    stdout = attempt_dir / "stdout.json"
    stderr = attempt_dir / "stderr.log"
    with stdout.open("w", encoding="utf-8") as out_f, stderr.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=out_f,
            stderr=err_f,
            check=False,
        )
    stderr_tail = _tail(stderr)
    failure_kind = None if proc.returncode == 0 else _classify_failure(proc.returncode, stderr_tail)
    return {
        "candidate": asdict(candidate),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0 and (run_dir / "metrics.json").exists(),
        "artifact_dir": str(attempt_dir),
        "stderr_tail": stderr_tail,
        "failure_kind": failure_kind,
        "suggestion": _suggestion(failure_kind) if failure_kind else None,
        "command": cmd,
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "backend_preflight.json"
    request_key = _request_key(args)
    previous = _json(summary_path)
    if previous and previous.get("passed") is True and previous.get("request_key") == request_key:
        runtime = previous["selected_runtime"]
        _write_json(args.runtime_profile, runtime)
        print(json.dumps({"passed": True, "cached": True, "runtime": runtime}, sort_keys=True))
        return

    attempts = []
    selected: dict[str, Any] | None = None
    for index, candidate in enumerate(_candidate_list(args)):
        attempt = _run_attempt(
            args=args,
            candidate=candidate,
            attempt_dir=args.out_dir / f"attempt-{index}",
        )
        attempts.append(attempt)
        if attempt["passed"]:
            selected = {
                "schema": "proicl_runtime_profile.v1",
                "vllm_dtype": candidate.dtype,
                "vllm_model_impl": args.vllm_model_impl,
                "vllm_attention_backend": candidate.attention_backend,
                "vllm_prefix_caching": candidate.prefix_caching,
                "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
                "vllm_max_model_len": int(args.vllm_max_model_len),
                "sps_vllm_batch_size": int(candidate.sps_vllm_batch_size),
                "backend_preflight": str(summary_path),
            }
            break

    summary = {
        "schema": "proicl_backend_preflight.v1",
        "passed": selected is not None,
        "request_key": request_key,
        "selected_runtime": selected,
        "attempts": attempts,
    }
    if selected is None and attempts:
        summary["failure_kind"] = attempts[-1].get("failure_kind")
        summary["suggestion"] = attempts[-1].get("suggestion")
    _write_json(summary_path, summary)
    if selected is not None:
        _write_json(args.runtime_profile, selected)
        print(json.dumps({"passed": True, "cached": False, "runtime": selected}, sort_keys=True))
        return
    print(
        json.dumps(
            {
                "passed": False,
                "summary": str(summary_path),
                "failure_kind": summary.get("failure_kind"),
                "attempts": [
                    {
                        "candidate": attempt.get("candidate"),
                        "returncode": attempt.get("returncode"),
                        "failure_kind": attempt.get("failure_kind"),
                        "suggestion": attempt.get("suggestion"),
                    }
                    for attempt in attempts
                ],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(
        "Backend preflight failed before GEPA/full eval. "
        f"See {summary_path}. {summary.get('suggestion', '')}"
    )


if __name__ == "__main__":
    main()
