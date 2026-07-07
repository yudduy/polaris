from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from polaris.prorl_recovery.protocol import BASE_MODEL_KEY, PRORL_V2_MODEL_KEY
from polaris.proicl.analysis import write_proicl_decomposition_by_track
from polaris.proicl.protocol import (
    ArchiveScope,
    ForkSearchMode,
    MemoryProtocol,
    PAPER_ALIGNED_SUSTAINED_TRACKS,
    ProICLConditionSpec,
    RepairMode,
    archive_scope_id,
    validate_archive_scope_membership,
)


REASONING_GYM_TRACKS: tuple[str, ...] = PAPER_ALIGNED_SUSTAINED_TRACKS

DEFAULT_SIGNAL_CONDITIONS: tuple[str, ...] = (
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

HELDOUT_EXPERIMENT_CONDITIONS: tuple[str, ...] = (
    "base_greedy",
    "sps_only",
    "gepa_sps_fixed",
    "prorl_v2_greedy",
)

COMMON_ARTIFACTS: tuple[str, ...] = (
    "metrics.json",
    "candidates.jsonl",
    "scores.jsonl",
    "selected.jsonl",
    "costs.json",
    "archive.json",
    "rollouts.json",
    "preflight.json",
    "environment.json",
    "manifest.json",
    "run_plan_cell.json",
    "audit.md",
)


@dataclass(frozen=True)
class LaunchCell:
    proicl_condition: str
    runtime_condition: str
    condition: str
    track: str
    model_key: str
    split: tuple[int, int]
    shard_id: int
    num_shards: int
    rollout_budget: int
    archive_path: str
    archive_build_id: str
    memory_build_id: str
    cache_path: str
    artifact_dir: str
    archive_scope: str = ArchiveScope.CROSS_FAMILY_CURRICULUM.value
    archive_scope_id: str = "none"
    archive_train_tracks: tuple[str, ...] = field(default_factory=tuple)
    archive_heldout_tracks: tuple[str, ...] = field(default_factory=tuple)
    uses_gepa_archive: bool = False
    uses_memory: bool = False
    uses_repair: bool = False
    uses_fork_search: bool = False
    memory_protocol: str = "off"
    repair_mode: str = "off"
    fork_search_mode: str = "off"
    memory_store_path: str | None = None
    seed: int = 17
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def to_jsonable(self) -> dict[str, Any]:
        payload = {
            "proicl_condition": self.proicl_condition,
            "runtime_condition": self.runtime_condition,
            "condition": self.condition,
            "track": self.track,
            "model_key": self.model_key,
            "split": list(self.split),
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "rollout_budget": self.rollout_budget,
            "archive_path": self.archive_path,
            "archive_build_id": self.archive_build_id,
            "archive_scope": self.archive_scope,
            "archive_scope_id": self.archive_scope_id,
            "archive_train_tracks": list(self.archive_train_tracks),
            "archive_heldout_tracks": list(self.archive_heldout_tracks),
            "memory_build_id": self.memory_build_id,
            "cache_path": self.cache_path,
            "artifact_dir": self.artifact_dir,
            "uses_gepa_archive": self.uses_gepa_archive,
            "uses_memory": self.uses_memory,
            "uses_repair": self.uses_repair,
            "uses_fork_search": self.uses_fork_search,
            "memory_protocol": self.memory_protocol,
            "repair_mode": self.repair_mode,
            "fork_search_mode": self.fork_search_mode,
            "memory_store_path": self.memory_store_path,
            "seed": self.seed,
            "extra_args": list(self.extra_args),
        }
        return payload


@dataclass(frozen=True)
class _ActiveCell:
    proc: subprocess.Popen
    cell: LaunchCell
    gpu: str
    started_at: float


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}
    row.update(payload)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


class _PlainProgress:
    def __init__(self, *, total: int, initial: int, desc: str) -> None:
        self.total = total
        self.n = initial
        self.desc = desc
        if total:
            print(f"{desc}: {self.n}/{self.total} cells", flush=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        print(f"{self.desc}: {self.n}/{self.total} cells", flush=True)

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        if value:
            print(f"{self.desc}: {value}", flush=True)

    def write(self, value: str) -> None:
        print(value, flush=True)

    def close(self) -> None:
        return


def _make_progress(*, total: int, initial: int, desc: str):
    if os.environ.get("PROICL_DISABLE_TQDM", "1") == "1" or not sys.stderr.isatty():
        return _PlainProgress(total=total, initial=initial, desc=desc)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _PlainProgress(total=total, initial=initial, desc=desc)
    return tqdm(total=total, initial=initial, desc=desc, unit="cell", dynamic_ncols=True)


def _progress_write(progress: Any, message: str) -> None:
    writer = getattr(progress, "write", None)
    if callable(writer):
        writer(message)
    else:
        print(message, flush=True)


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _jsonl_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _cell_checkpoint_status(out: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "selected_rows": _jsonl_row_count(out / "selected.jsonl"),
        "completed_problems": None,
        "expected_problems": None,
        "checkpoint_complete": None,
    }
    checkpoint_path = out / "checkpoint.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return status
    status["completed_problems"] = checkpoint.get("completed_problems")
    status["expected_problems"] = checkpoint.get("expected_problems")
    status["checkpoint_complete"] = checkpoint.get("complete")
    return status


def _active_cell_snapshot(active: _ActiveCell, *, now: float) -> dict[str, Any]:
    out = Path(active.cell.artifact_dir)
    stderr_path = out / "stderr.log"
    try:
        stderr_bytes = stderr_path.stat().st_size
    except OSError:
        stderr_bytes = 0
    status = {
        "pid": getattr(active.proc, "pid", None),
        "gpu": active.gpu,
        "track": active.cell.track,
        "condition": active.cell.proicl_condition,
        "shard": active.cell.shard_id,
        "num_shards": active.cell.num_shards,
        "artifact_dir": active.cell.artifact_dir,
        "stderr_log": str(stderr_path),
        "stderr_bytes": stderr_bytes,
        "elapsed_seconds": round(max(0.0, now - active.started_at), 1),
    }
    status.update(_cell_checkpoint_status(out))
    return status


def _format_active_cell_snapshot(status: dict[str, Any]) -> str:
    completed = status.get("completed_problems")
    expected = status.get("expected_problems")
    if completed is not None and expected is not None:
        progress = f"checkpoint={completed}/{expected}"
    else:
        progress = f"selected_rows={status.get('selected_rows', 0)}"
    return (
        "[ProICL] active "
        f"gpu={status['gpu']} pid={status.get('pid')} "
        f"elapsed={status['elapsed_seconds']}s "
        f"track={status['track']} condition={status['condition']} "
        f"shard={status['shard']}/{status['num_shards']} "
        f"{progress} stderr_bytes={status['stderr_bytes']} "
        f"log={status['stderr_log']}"
    )


def _cell_heartbeat_interval_seconds() -> float:
    raw = os.environ.get("PROICL_CELL_HEARTBEAT_SECONDS", "60")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


def source_hash(repo_root: Path) -> str:
    h = hashlib.sha256()
    roots = ["src", "scripts", "docs", "tests", "configs", "TODO.PROICL.md", "pyproject.toml"]
    for rel in roots:
        path = repo_root / rel
        if not path.exists():
            continue
        if path.is_file():
            files = [path]
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            h.update(str(file.relative_to(repo_root)).encode("utf-8"))
            h.update(b"\0")
            h.update(file.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def vendored_commit(repo_root: Path, rel: str) -> str:
    env_key = {
        "upstream/reasoning-with-sampling": "PROICL_RWS_COMMIT",
        "upstream/gepa": "PROICL_GEPA_COMMIT",
        "upstream/evalplus": "PROICL_EVALPLUS_COMMIT",
        "upstream/dynamic-cheatsheet": "PROICL_DC_COMMIT",
    }.get(rel)
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    result = subprocess.run(
        ["git", "-C", str(repo_root / rel), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def build_signal_cells(
    *,
    root: Path,
    tracks: Iterable[str],
    split: tuple[int, int],
    rollout_budget: int,
    num_shards: int,
    memory_num_shards: int,
    seed: int = 17,
    archive_scope: ArchiveScope | str = ArchiveScope.WITHIN_FAMILY,
    archive_train_tracks: Iterable[str] | None = None,
    archive_heldout_tracks: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> list[LaunchCell]:
    if memory_num_shards != 1:
        raise ValueError(
            "ProICL memory cells must use memory_num_shards=1 so a track has "
            "one serialized verifier-gated curriculum store."
        )
    track_tuple = tuple(tracks)
    archive_scope = ArchiveScope(archive_scope)
    train_tracks = tuple(archive_train_tracks or track_tuple)
    heldout_tracks = tuple(archive_heldout_tracks or track_tuple)
    validate_archive_scope_membership(
        archive_scope=archive_scope,
        train_tracks=train_tracks,
        heldout_tracks=heldout_tracks,
    )
    scope_id = archive_scope_id(archive_scope, train_tracks)
    specs = (
        ProICLConditionSpec("base_greedy", "greedy", BASE_MODEL_KEY, "direct", "greedy", "greedy"),
        ProICLConditionSpec("bon_temp1", "bon_temp1", BASE_MODEL_KEY, "direct", "matched", "bon_temp1"),
        ProICLConditionSpec("sps_only", "single_prompt_power", BASE_MODEL_KEY, "direct", "matched", "fixed_alpha_4", uses_power_sampling=True),
        ProICLConditionSpec("mcmc_only", "single_prompt_power", BASE_MODEL_KEY, "direct", "matched", "fixed_alpha_4", uses_power_sampling=True),
        ProICLConditionSpec("mixed_alpha_mcmc", "mixed_alpha_mcmc", BASE_MODEL_KEY, "direct", "matched", "mixed_alpha_4_1", uses_power_sampling=True),
        ProICLConditionSpec("fork_search", "fork_search", BASE_MODEL_KEY, "direct", "matched", "bon_temp1", uses_fork_search=True, fork_search_mode=ForkSearchMode.ENTROPY_GATED),
        ProICLConditionSpec("gepa_only", "gepa_only", BASE_MODEL_KEY, "gepa", "matched", "bon_temp1", uses_gepa_archive=True),
        ProICLConditionSpec("gepa_sps_fixed", "full_archive_fixed", BASE_MODEL_KEY, "gepa", "matched", "fixed_alpha_4", uses_power_sampling=True, uses_gepa_archive=True),
        ProICLConditionSpec("gepa_mcmc", "proicl_gepa_mcmc", BASE_MODEL_KEY, "gepa", "matched", "mixed_alpha_4_1", uses_power_sampling=True, uses_gepa_archive=True),
        ProICLConditionSpec("gepa_mcmc_repair", "proicl_gepa_mcmc_repair", BASE_MODEL_KEY, "gepa", "matched", "mixed_alpha_4_1", uses_power_sampling=True, uses_gepa_archive=True, uses_repair=True, repair_mode=RepairMode.VERIFIER_GUIDED),
        ProICLConditionSpec("gepa_mcmc_fork_repair", "proicl_gepa_mcmc_fork_repair", BASE_MODEL_KEY, "gepa", "matched", "mixed_alpha_4_1", uses_power_sampling=True, uses_gepa_archive=True, uses_repair=True, uses_fork_search=True, repair_mode=RepairMode.VERIFIER_GUIDED, fork_search_mode=ForkSearchMode.ENTROPY_GATED),
        ProICLConditionSpec("gepa_mcmc_fork_repair_memory", "proicl_gepa_mcmc_fork_repair_memory", BASE_MODEL_KEY, "gepa", "matched", "mixed_alpha_4_1", uses_power_sampling=True, uses_gepa_archive=True, uses_memory=True, uses_repair=True, uses_fork_search=True, memory_mode="distilled_strategies", memory_protocol=MemoryProtocol.FROZEN_DEV, repair_mode=RepairMode.VERIFIER_GUIDED, fork_search_mode=ForkSearchMode.ENTROPY_GATED),
        ProICLConditionSpec("gepa_mcmc_memory", "proicl_gepa_mcmc_memory", BASE_MODEL_KEY, "gepa", "matched", "mixed_alpha_4_1", uses_power_sampling=True, uses_gepa_archive=True, uses_memory=True, memory_mode="distilled_strategies", memory_protocol=MemoryProtocol.ONLINE, preliminary=True),
        ProICLConditionSpec("prorl_v2_greedy", "greedy", PRORL_V2_MODEL_KEY, "direct", "greedy", "greedy", slow_weight_reference=True),
    )
    spec_by_key = {spec.key: spec for spec in specs}
    condition_tuple = tuple(conditions or DEFAULT_SIGNAL_CONDITIONS)
    unknown_conditions = sorted(set(condition_tuple) - set(spec_by_key))
    if unknown_conditions:
        raise ValueError("unknown ProICL signal conditions: " + ", ".join(unknown_conditions))
    cells: list[LaunchCell] = []
    for track in track_tuple:
        direct_archive = root / "archives" / track / "direct.json"
        gepa_archive = root / "archives" / scope_id / "archive.json"
        for proicl_condition in condition_tuple:
            spec = spec_by_key[proicl_condition]
            budget = 1 if spec.budget_role == "greedy" else rollout_budget
            shards = memory_num_shards if spec.uses_memory else num_shards
            for shard_id in range(shards):
                archive_path = gepa_archive if spec.archive_kind == "gepa" else direct_archive
                memory_path = (
                    root / "memory" / track / f"{spec.key}.sqlite"
                    if spec.uses_memory
                    else None
                )
                extra_parts: list[str] = []
                if spec.uses_memory:
                    extra_parts.extend(["--memory-mode", spec.memory_mode])
                if spec.memory_protocol == MemoryProtocol.ONLINE:
                    extra_parts.extend(["--admit-memory", "--online-memory"])
                if spec.uses_repair:
                    extra_parts.append("--enable-repair")
                if spec.uses_fork_search:
                    extra_parts.append("--enable-fork-search")
                cells.append(
                    LaunchCell(
                        proicl_condition=spec.key,
                        runtime_condition=spec.runtime_condition,
                        condition=spec.runtime_condition,
                        track=track,
                        model_key=spec.model_key,
                        split=split,
                        shard_id=shard_id,
                        num_shards=shards,
                        rollout_budget=budget,
                        archive_path=str(archive_path),
                        archive_build_id=scope_id if spec.uses_gepa_archive else "none",
                        archive_scope=archive_scope.value,
                        archive_scope_id=scope_id,
                        archive_train_tracks=train_tracks if spec.uses_gepa_archive else (),
                        archive_heldout_tracks=heldout_tracks if spec.uses_gepa_archive else (),
                        memory_build_id=(
                            f"proicl-memory-{spec.memory_protocol.value}-{track}"
                            if spec.uses_memory
                            else "none"
                        ),
                        cache_path=str(
                            root
                            / "trajectory_cache"
                            / f"{track}-{spec.key}-shard-{shard_id}.sqlite"
                        ),
                        artifact_dir=str(
                            root / "runs" / track / spec.key / f"shard-{shard_id}"
                        ),
                        uses_gepa_archive=spec.uses_gepa_archive,
                        uses_memory=spec.uses_memory,
                        uses_repair=spec.uses_repair,
                        uses_fork_search=spec.uses_fork_search,
                        memory_protocol=spec.memory_protocol.value,
                        repair_mode=spec.repair_mode.value,
                        fork_search_mode=spec.fork_search_mode.value,
                        memory_store_path=str(memory_path) if memory_path else None,
                        seed=seed,
                        extra_args=tuple(extra_parts),
                    )
                )
    return cells


def required_artifacts(cell: LaunchCell) -> tuple[str, ...]:
    extra: list[str] = []
    if cell.uses_gepa_archive or cell.uses_memory:
        extra.append("archive_build_manifest.json")
    if cell.uses_repair:
        extra.append("repair_traces.jsonl")
    if cell.uses_fork_search:
        extra.append("fork_traces.jsonl")
    if cell.uses_memory:
        extra.extend(["memory.sqlite", "memory_events.jsonl"])
    return COMMON_ARTIFACTS + tuple(extra)


def cell_complete(cell: LaunchCell) -> bool:
    out = Path(cell.artifact_dir)
    if not out.exists():
        return False
    for rel in required_artifacts(cell):
        path = out / rel
        if not path.exists() or path.stat().st_size == 0:
            return False
    try:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(metrics.get("n_problems", 0)) > 0


def _launch_cell_matches(out: Path, cell: LaunchCell) -> bool:
    path = out / "proicl_launch_cell.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == cell.to_jsonable()
    except Exception:
        return False


def _prepare_cell_artifact_dir(cell: LaunchCell) -> str:
    out = Path(cell.artifact_dir)
    if out.exists():
        if not _launch_cell_matches(out, cell):
            shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "proicl_launch_cell.json", cell.to_jsonable())
            return "fresh"
        write_json(out / "proicl_launch_cell.json", cell.to_jsonable())
        return "resume"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "proicl_launch_cell.json", cell.to_jsonable())
    return "fresh"


def run_condition_command(
    *,
    repo_root: Path,
    cell: LaunchCell,
    backend: str,
    local_files_only: bool,
    polaris_source_hash: str,
    cost_cap_dollars: float,
    estimated_dollar_cost: float,
    estimated_wall_clock_seconds: float,
    run_kind: str,
    run_stage: str,
    max_new_tokens: int,
    power_sampler: str = "mcmc",
    mcmc_steps: int | None = None,
    mcmc_block_num: int | None = None,
    fixed_alpha: float | None = None,
    sps_top_k: int = 8,
    sps_candidate_pool_size: int = 8,
    sps_rollouts_per_candidate: int = 8,
    sps_rollout_horizon: int | None = None,
    vllm_dtype: str = "float32",
    vllm_model_impl: str = "transformers",
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_max_model_len: int | None = None,
    vllm_scoring_mode: str = "forced_decode_v0",
    vllm_parity_artifact: str | None = None,
    vllm_enable_prefix_caching: bool = True,
) -> list[str]:
    frozen_conditions = {
        "bon_temp1",
        "sps_only",
        "mcmc_only",
        "mixed_alpha_mcmc",
        "fork_search",
        "gepa_only",
        "gepa_sps_fixed",
        "gepa_mcmc",
        "gepa_mcmc_repair",
        "gepa_mcmc_fork_repair",
        "gepa_mcmc_fork_repair_memory",
        "gepa_mcmc_memory",
    }
    if cell.proicl_condition in frozen_conditions and cell.model_key != BASE_MODEL_KEY:
        raise ValueError(
            "ProICL frozen-base conditions must use the base model, not "
            f"{cell.model_key!r}"
        )
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_condition.py"),
        "--track",
        cell.track,
        "--model-key",
        cell.model_key,
        "--condition",
        cell.condition,
        "--archive",
        cell.archive_path,
        "--split",
        str(cell.split[0]),
        str(cell.split[1]),
        "--shard-id",
        str(cell.shard_id),
        "--num-shards",
        str(cell.num_shards),
        "--seed",
        str(cell.seed),
        "--backend",
        backend,
        "--samples-per-problem",
        str(cell.rollout_budget),
        "--sampling-temperature",
        "1.0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--power-sampler",
        power_sampler,
        "--proicl-source-hash",
        polaris_source_hash,
        "--preregistration-anchor",
        "TODO.PROICL.md#proicl-fast-weight-recovery-audit",
        "--out",
        cell.artifact_dir,
        "--trajectory-cache",
        cell.cache_path,
        "--archive-build-id",
        cell.archive_build_id,
        "--memory-build-id",
        cell.memory_build_id,
        "--run-stage",
        run_stage,
        "--run-kind",
        run_kind,
        "--estimated-dollar-cost",
        str(estimated_dollar_cost),
        "--estimated-wall-clock-seconds",
        str(estimated_wall_clock_seconds),
        "--cost-cap-dollars",
        str(cost_cap_dollars),
        "--user-authorized-paid-run",
        "--rws-commit",
        vendored_commit(repo_root, "upstream/reasoning-with-sampling"),
        "--gepa-commit",
        vendored_commit(repo_root, "upstream/gepa"),
        "--evalplus-commit",
        vendored_commit(repo_root, "upstream/evalplus"),
        "--dc-commit",
        vendored_commit(repo_root, "upstream/dynamic-cheatsheet"),
    ]
    if mcmc_steps is not None:
        cmd.extend(["--mcmc-steps", str(mcmc_steps)])
    if mcmc_block_num is not None:
        cmd.extend(["--mcmc-block-num", str(mcmc_block_num)])
    if fixed_alpha is not None and cell.uses_power_sampling:
        cmd.extend(["--fixed-alpha", str(fixed_alpha)])
    cmd.extend(
        [
            "--sps-top-k",
            str(sps_top_k),
            "--sps-candidate-pool-size",
            str(sps_candidate_pool_size),
            "--sps-rollouts-per-candidate",
            str(sps_rollouts_per_candidate),
        ]
    )
    if sps_rollout_horizon is not None:
        cmd.extend(["--sps-rollout-horizon", str(sps_rollout_horizon)])
    if local_files_only:
        cmd.append("--local-files-only")
    if backend == "vllm":
        cmd.extend(
            [
                "--vllm-dtype",
                vllm_dtype,
                "--vllm-model-impl",
                vllm_model_impl,
                "--vllm-gpu-memory-utilization",
                str(vllm_gpu_memory_utilization),
                "--vllm-scoring-mode",
                vllm_scoring_mode,
            ]
        )
        if vllm_max_model_len is not None:
            cmd.extend(["--vllm-max-model-len", str(vllm_max_model_len)])
        if vllm_parity_artifact is not None:
            cmd.extend(["--vllm-parity-artifact", vllm_parity_artifact])
        if not vllm_enable_prefix_caching:
            cmd.append("--no-vllm-prefix-caching")
    if cell.memory_store_path is not None:
        cmd.extend(["--memory-store", cell.memory_store_path])
    cmd.extend(cell.extra_args)
    return cmd


def run_cells(
    *,
    repo_root: Path,
    cells: list[LaunchCell],
    gpus: list[str],
    events_path: Path,
    backend: str,
    local_files_only: bool,
    cost_cap_dollars: float,
    estimated_dollar_cost: float,
    estimated_wall_clock_seconds: float,
    run_kind: str,
    run_stage: str,
    max_new_tokens: int,
    power_sampler: str = "mcmc",
    mcmc_steps: int | None = None,
    mcmc_block_num: int | None = None,
    fixed_alpha: float | None = None,
    sps_top_k: int = 8,
    sps_candidate_pool_size: int = 8,
    sps_rollouts_per_candidate: int = 8,
    sps_rollout_horizon: int | None = None,
    vllm_dtype: str = "float32",
    vllm_model_impl: str = "transformers",
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_max_model_len: int | None = None,
    vllm_scoring_mode: str = "forced_decode_v0",
    vllm_parity_artifact: str | None = None,
    vllm_enable_prefix_caching: bool = True,
    stop_on_failure: bool = True,
) -> None:
    if not gpus:
        raise ValueError("at least one GPU slot is required")
    polaris_hash = source_hash(repo_root)
    queue = [cell for cell in cells if not cell_complete(cell)]
    skipped = len(cells) - len(queue)
    append_event(events_path, "queue_start", cells=len(cells), pending=len(queue), skipped=skipped)
    active: list[_ActiveCell] = []
    available_gpus = list(gpus)
    failures: list[dict[str, Any]] = []
    progress = _make_progress(total=len(cells), initial=skipped, desc="ProICL cells")
    heartbeat_interval = _cell_heartbeat_interval_seconds()
    next_heartbeat = (
        time.monotonic() + heartbeat_interval
        if heartbeat_interval > 0
        else float("inf")
    )
    if skipped:
        _progress_write(progress, f"[ProICL] skipped {skipped} completed cells")
    try:
        while queue or active:
            while queue and available_gpus:
                cell = queue.pop(0)
                gpu = available_gpus.pop(0)
                out = Path(cell.artifact_dir)
                launch_mode = _prepare_cell_artifact_dir(cell)
                cmd = run_condition_command(
                    repo_root=repo_root,
                    cell=cell,
                    backend=backend,
                    local_files_only=local_files_only,
                    polaris_source_hash=polaris_hash,
                    cost_cap_dollars=cost_cap_dollars,
                    estimated_dollar_cost=estimated_dollar_cost,
                    estimated_wall_clock_seconds=estimated_wall_clock_seconds,
                    run_kind=run_kind,
                    run_stage=run_stage,
                    max_new_tokens=max_new_tokens,
                    power_sampler=power_sampler,
                    mcmc_steps=mcmc_steps,
                    mcmc_block_num=mcmc_block_num,
                    fixed_alpha=fixed_alpha,
                    sps_top_k=sps_top_k,
                    sps_candidate_pool_size=sps_candidate_pool_size,
                    sps_rollouts_per_candidate=sps_rollouts_per_candidate,
                    sps_rollout_horizon=sps_rollout_horizon,
                    vllm_dtype=vllm_dtype,
                    vllm_model_impl=vllm_model_impl,
                    vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                    vllm_max_model_len=vllm_max_model_len,
                    vllm_scoring_mode=vllm_scoring_mode,
                    vllm_parity_artifact=vllm_parity_artifact,
                    vllm_enable_prefix_caching=vllm_enable_prefix_caching,
                )
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                progress.set_postfix_str(
                    f"gpu={gpu} {launch_mode} "
                    f"{cell.track}/{cell.proicl_condition}/shard-{cell.shard_id}",
                    refresh=True,
                )
                _progress_write(
                    progress,
                    f"[ProICL] {launch_mode} "
                    f"gpu={gpu} track={cell.track} condition={cell.proicl_condition} "
                    f"shard={cell.shard_id}/{cell.num_shards} log={out / 'stderr.log'}",
                )
                stdout = (out / "stdout.json").open("a", encoding="utf-8")
                stderr = (out / "stderr.log").open("a", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=stdout, stderr=stderr)
                stdout.close()
                stderr.close()
                append_event(
                    events_path,
                    "cell_start",
                    gpu=gpu,
                    pid=getattr(proc, "pid", None),
                    track=cell.track,
                    condition=cell.proicl_condition,
                    shard=cell.shard_id,
                    mode=launch_mode,
                    artifact_dir=cell.artifact_dir,
                    stderr_log=str(out / "stderr.log"),
                )
                active.append(
                    _ActiveCell(
                        proc=proc,
                        cell=cell,
                        gpu=gpu,
                        started_at=time.monotonic(),
                    )
                )
            time.sleep(2)
            now = time.monotonic()
            still_active: list[_ActiveCell] = []
            for active_cell in active:
                rc = active_cell.proc.poll()
                if rc is None:
                    still_active.append(active_cell)
                    continue
                cell = active_cell.cell
                gpu = active_cell.gpu
                if rc == 0 and cell_complete(cell):
                    available_gpus.append(gpu)
                    progress.update(1)
                    progress.set_postfix_str(
                        f"gpu={gpu} done {cell.track}/{cell.proicl_condition}/shard-{cell.shard_id}",
                        refresh=True,
                    )
                    append_event(
                        events_path,
                        "cell_done",
                        gpu=gpu,
                        track=cell.track,
                        condition=cell.proicl_condition,
                        shard=cell.shard_id,
                    )
                else:
                    available_gpus.append(gpu)
                    progress.update(1)
                    stderr_tail = _tail_text(Path(cell.artifact_dir) / "stderr.log")
                    failure = {
                        "track": cell.track,
                        "condition": cell.proicl_condition,
                        "shard": cell.shard_id,
                        "returncode": rc,
                        "artifact_dir": cell.artifact_dir,
                        "stderr_tail": stderr_tail,
                    }
                    failures.append(failure)
                    append_event(events_path, "cell_failed", **failure)
                    _progress_write(progress, f"[ProICL] failed {failure}")
                    if stop_on_failure:
                        for live in still_active:
                            live.proc.terminate()
                        raise RuntimeError(f"ProICL cell failed: {failure}")
            active = still_active
            if active and now >= next_heartbeat:
                for active_cell in active:
                    snapshot = _active_cell_snapshot(active_cell, now=now)
                    append_event(events_path, "cell_heartbeat", **snapshot)
                    _progress_write(progress, _format_active_cell_snapshot(snapshot))
                next_heartbeat = now + heartbeat_interval
    finally:
        progress.close()
    append_event(events_path, "queue_done", failures=len(failures))
    if failures:
        raise RuntimeError(f"ProICL queue had failures: {failures}")


def aggregate(root: Path, tracks: Iterable[str], out_dir: Path) -> dict[str, Any]:
    return write_proicl_decomposition_by_track(
        root=root,
        tracks=tuple(tracks),
        out_dir=out_dir,
    )
