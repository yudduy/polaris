from __future__ import annotations

import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from polaris.config import MODEL_REGISTRY
from polaris.core.archive import FrozenArchive, MATH500_ARCHIVE_V1, PromptEntry
from polaris.infra.farmshare import shard_indices
from polaris.prorl_recovery.protocol import (
    BASE_MODEL_KEY,
    BRO_RL_MODEL_KEY,
    KARAN_DU_REPLICATION_GATE,
    PRORL_V1_MODEL_KEY,
    PRORL_V2_MODEL_KEY,
)


DEFAULT_MAIN_TRACKS: tuple[str, ...] = (
    "math500",
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
)
PRORL_CHECKPOINTS: tuple[str, ...] = (
    BASE_MODEL_KEY,
    PRORL_V1_MODEL_KEY,
    PRORL_V2_MODEL_KEY,
    BRO_RL_MODEL_KEY,
)
BON_K: tuple[int, ...] = (4, 16, 64, 256, 1024)
PHASE0_TOKEN_CAP = 3072
TRACK_TOKEN_CAPS: dict[str, int] = {
    "math500": 4096,
    "gpqa_diamond": 2048,
    "reasoning_gym_boxnet": 8192,
    "reasoning_gym_graph_color": 8192,
    "reasoning_gym_graph_color_n5": 8192,
    "reasoning_gym_graph_color_n8": 8192,
    "reasoning_gym_graph_color_n10": 8192,
    "reasoning_gym_graph_color_n12": 8192,
    "reasoning_gym_graph_color_n13": 8192,
    "reasoning_gym_graph_color_n14": 8192,
    "reasoning_gym_graph_color_n15": 8192,
    "reasoning_gym_graph_color_n16": 8192,
    "reasoning_gym_graph_color_n18": 8192,
    "reasoning_gym_graph_color_n20": 8192,
    "reasoning_gym_family_relationships": 8192,
}
GENERATED_PRORL_ARCHIVE_REL = "runs/generated/prorl_recovery_archives"


@dataclass(frozen=True)
class RecoveryCell:
    phase: str
    track: str
    model_key: str
    condition: str
    split: tuple[int, int]
    shard_id: int
    num_shards: int
    samples_per_problem: int
    sampling_temperature: float
    max_new_tokens: int
    max_model_len: int
    archive_kind: str
    out_dir: str
    cache_path: str
    rung: str | None = None
    seed: int = 17
    backend: str = "vllm"
    selected_problem_ids: tuple[str, ...] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = list(self.split)
        if self.selected_problem_ids is not None:
            payload["selected_problem_ids"] = list(self.selected_problem_ids)
        return payload


@dataclass(frozen=True)
class RWSExactCell:
    root: str
    cell_index: int
    batch_idx: int
    seed: int
    num_shards: int = 5
    num_seeds: int = 8
    model: str = "qwen_math"
    temperature: float = 0.25
    mcmc_steps: int = 10

    @property
    def save_str(self) -> str:
        return f"{self.root}/phase0/rws_exact/results"

    @property
    def out_dir(self) -> str:
        return f"{self.root}/phase0/rws_exact/cells/batch-{self.batch_idx}/seed-{self.seed}"

    @property
    def expected_csv(self) -> str:
        return (
            f"{self.save_str}/{self.model}/"
            f"{self.model}_math_base_power_samp_results_"
            f"{self.mcmc_steps}_{self.temperature}_{self.batch_idx}_{self.seed}.csv"
        )

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["save_str"] = self.save_str
        payload["out_dir"] = self.out_dir
        payload["expected_csv"] = self.expected_csv
        return payload


def token_cap_for_track(phase: str, track: str) -> int:
    if phase == "phase0":
        return PHASE0_TOKEN_CAP
    try:
        return TRACK_TOKEN_CAPS[track]
    except KeyError as exc:
        raise ValueError(f"unknown token-cap track: {track!r}") from exc


def token_cap_after_smoke(
    track: str,
    *,
    cap_hit_rate: float,
    already_doubled: bool = False,
) -> int:
    cap = token_cap_for_track("phase1", track)
    if cap_hit_rate > 0.05 and not already_doubled:
        return cap * 2
    return cap


def checkpoint_max_new_tokens(model_key: str) -> int:
    if model_key == BRO_RL_MODEL_KEY:
        return 16384
    if model_key == PRORL_V2_MODEL_KEY:
        return 16384
    return 8192


def checkpoint_max_model_len(model_key: str) -> int:
    return 16384 if checkpoint_max_new_tokens(model_key) > 8192 else 8192


def gpqa_available(env: dict[str, str] | None = None) -> bool:
    import os

    env = env or os.environ
    return bool(
        env.get("GPQA_DIAMOND_PATH")
        or env.get("HF_TOKEN")
        or env.get("HUGGINGFACE_HUB_TOKEN")
    )


def selected_tracks(*, include_gpqa: bool = False) -> tuple[str, ...]:
    tracks = list(DEFAULT_MAIN_TRACKS)
    if include_gpqa:
        tracks.insert(1, "gpqa_diamond")
    return tuple(tracks)


def _row_checkpoint(row: dict[str, Any]) -> str:
    return str(row.get("checkpoint") or row.get("model_key") or row.get("model"))


def _row_track(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("track") or row.get("benchmark"))


def _row_passed(row: dict[str, Any]) -> bool:
    if "verified" in row:
        return bool(row["verified"])
    if "passed" in row:
        return bool(row["passed"])
    if "selected_passed" in row:
        return bool(row["selected_passed"])
    verifier = row.get("verifier_result")
    if isinstance(verifier, dict):
        return bool(verifier.get("passed", False))
    return False


def derive_prorl_only_problem_ids(
    phase1_rows: Iterable[dict[str, Any]],
    *,
    base_key: str = BASE_MODEL_KEY,
    solver_keys: tuple[str, ...] = (PRORL_V2_MODEL_KEY, BRO_RL_MODEL_KEY),
) -> dict[str, tuple[str, ...]]:
    """Return problems solved by a trained checkpoint and not solved by base."""

    base_solved: dict[str, set[str]] = {}
    solver_solved: dict[str, set[str]] = {}
    for row in phase1_rows:
        if not _row_passed(row):
            continue
        track = _row_track(row)
        problem_id = str(row.get("problem_id"))
        checkpoint = _row_checkpoint(row)
        if checkpoint == base_key:
            base_solved.setdefault(track, set()).add(problem_id)
        elif checkpoint in solver_keys:
            solver_solved.setdefault(track, set()).add(problem_id)

    selected: dict[str, tuple[str, ...]] = {}
    for track, solved in solver_solved.items():
        problem_ids = tuple(sorted(solved - base_solved.get(track, set())))
        if problem_ids:
            selected[track] = problem_ids
    return selected


def direct_archive() -> FrozenArchive:
    return FrozenArchive(entries=(MATH500_ARCHIVE_V1.entries[0],))


def rws_math_direct_archive() -> FrozenArchive:
    from polaris.vendored.rws.constants import COT, PROMPT

    return FrozenArchive(
        entries=(
            PromptEntry(
                id="direct",
                prefix=PROMPT,
                suffix=COT,
                descriptor_hint="rws_cot_direct",
            ),
        )
    )


def reasoning_gym_direct_archive(task: str = "generic") -> FrozenArchive:
    if task == "family_relationships":
        task_guidance = (
            "Answer the kinship question with only the relationship word. No reasoning."
        )
    elif task == "graph_color":
        task_guidance = (
            "Solve the graph coloring task. The final tagged value must be a raw "
            "JSON object mapping every vertex string to one allowed color."
        )
    elif task == "boxnet":
        task_guidance = (
            "Solve the box-moving planning task. The final tagged value must be a "
            "raw JSON list of action dictionaries and contain no prose."
        )
    else:
        task_guidance = (
            "Solve the verifier-scored task and return exactly the value requested "
            "by the task."
        )
    return FrozenArchive(
        entries=(
            PromptEntry(
                id="direct",
                prefix=(
                    f"{task_guidance}\n\nTask:\n"
                ),
                suffix=(
                    "\n\nReturn only the final value. Do not include a placeholder, "
                    "example answer, or reasoning in the final response."
                ),
                descriptor_hint="reasoning_gym_direct_answer_tag",
            ),
        )
    )


def reasoning_gym_seed_archive() -> FrozenArchive:
    base_suffix = (
        "\n\nReturn only the final value. Do not include a placeholder, example "
        "answer, or reasoning in the final response."
    )
    entries = (
        PromptEntry(
            id="direct",
            prefix=(
                "Solve the following task. Reason if needed, then put only the final "
                "answer in the final response. For JSON tasks, the final response must "
                "be valid JSON and contain no prose.\n\nTask:\n"
            ),
            suffix=base_suffix,
            descriptor_hint="reasoning_gym_direct_answer_tag",
        ),
        PromptEntry(
            id="planner",
            prefix=(
                "You are solving a verifier-scored reasoning task. Build the solution "
                "carefully, check it against the task rules, and put the final answer "
                "in XML tags named answer. JSON answers must be raw valid JSON inside "
                "the tags.\n\nTask:\n"
            ),
            suffix=base_suffix,
            descriptor_hint="reasoning_gym_rule_checked",
        ),
        PromptEntry(
            id="format_strict",
            prefix=(
                "Solve the task. Keep all reasoning outside the final answer. The text "
                "inside the final answer-tag block must contain exactly the value that "
                "the programmatic grader should parse.\n\nTask:\n"
            ),
            suffix=base_suffix,
            descriptor_hint="reasoning_gym_format_strict",
        ),
        PromptEntry(
            id="minimal_final",
            prefix=(
                "Find the correct final response for this task. If the final response is "
                "JSON, return syntactically valid JSON only inside the answer tag; if it "
                "is a word or phrase, return only that word or phrase.\n\nTask:\n"
            ),
            suffix=base_suffix,
            descriptor_hint="reasoning_gym_minimal_final",
        ),
    )
    return FrozenArchive(entries=entries)


def direct_archive_kind_for_track(track: str) -> str:
    if track == "reasoning_gym_family_relationships":
        return "reasoning_gym_family_direct"
    if track.startswith("reasoning_gym_graph_color"):
        return "reasoning_gym_graph_color_direct"
    if track == "reasoning_gym_boxnet":
        return "reasoning_gym_boxnet_direct"
    return "reasoning_gym_direct" if track.startswith("reasoning_gym_") else "direct"


def seed_archive_kind_for_track(track: str) -> str:
    return "reasoning_gym_seed_archive" if track.startswith("reasoning_gym_") else "seed_archive"


def recovery_archive_path(kind: str, *, repo_dir: str = "$POLARIS_REPO_DIR") -> str:
    return f"{repo_dir}/{GENERATED_PRORL_ARCHIVE_REL}/{kind}.json"


def ensure_recovery_archive(kind: str, *, repo_dir: str | Path) -> Path:
    raw_repo_dir = os.path.expandvars(str(repo_dir))
    if "$" in raw_repo_dir:
        raise ValueError(f"repo_dir must be concrete to generate archives: {repo_dir!r}")
    path = Path(raw_repo_dir) / GENERATED_PRORL_ARCHIVE_REL / f"{kind}.json"
    write_archive(path, kind=kind)
    return path


def write_archive(path: Path, *, kind: str) -> None:
    if kind == "direct":
        archive = direct_archive()
        payload: Any = archive.to_jsonable()
    elif kind == "reasoning_gym_direct":
        archive = reasoning_gym_direct_archive()
        payload = archive.to_jsonable()
    elif kind == "reasoning_gym_family_direct":
        archive = reasoning_gym_direct_archive("family_relationships")
        payload = archive.to_jsonable()
    elif kind == "reasoning_gym_graph_color_direct":
        archive = reasoning_gym_direct_archive("graph_color")
        payload = archive.to_jsonable()
    elif kind == "reasoning_gym_boxnet_direct":
        archive = reasoning_gym_direct_archive("boxnet")
        payload = archive.to_jsonable()
    elif kind == "reasoning_gym_seed_archive":
        archive = reasoning_gym_seed_archive()
        payload = {
            "entries": archive.to_jsonable(),
            "cell_fitness": {
                entry.descriptor_hint: 1.0 for entry in archive.entries
            },
            "archive_build_id": "reasoning-gym-seed-archive",
            "frozen": True,
        }
    elif kind == "rws_math_direct":
        archive = rws_math_direct_archive()
        payload = archive.to_jsonable()
    elif kind == "seed_archive":
        archive = MATH500_ARCHIVE_V1
        payload = {
            "entries": archive.to_jsonable(),
            "cell_fitness": {
                entry.descriptor_hint: 1.0 for entry in archive.entries
            },
            "archive_build_id": "seed-archive",
            "frozen": True,
        }
    else:
        raise ValueError(f"unknown archive kind: {kind!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def phase1_cells(
    *,
    root: str,
    problem_count: int,
    tracks: Iterable[str],
    num_shards: int = 4,
    seed: int = 17,
    samples_per_problem: int = 128,
) -> list[RecoveryCell]:
    if samples_per_problem <= 0:
        raise ValueError("samples_per_problem must be positive")
    cells: list[RecoveryCell] = []
    for track in tracks:
        for model_key in PRORL_CHECKPOINTS:
            max_new_tokens = token_cap_for_track("phase1", track)
            for shard_id in range(num_shards):
                rel = f"phase1/{track}/{model_key}/shard-{shard_id}"
                cells.append(
                    RecoveryCell(
                        phase="phase1",
                        track=track,
                        model_key=model_key,
                        condition="bon_temp1",
                        split=(0, problem_count),
                        shard_id=shard_id,
                        num_shards=num_shards,
                        samples_per_problem=samples_per_problem,
                        sampling_temperature=1.0,
                        max_new_tokens=max_new_tokens,
                        max_model_len=max(8192, max_new_tokens),
                        archive_kind=direct_archive_kind_for_track(track),
                        out_dir=f"{root}/{rel}",
                        cache_path=(
                            f"{root}/trajectory_cache/phase1-{track}-{model_key}"
                            f"-shard-{shard_id}.sqlite"
                        ),
                        seed=seed,
                    )
                )
    return cells


def phase2_cells(
    *,
    root: str,
    problem_count: int,
    tracks: Iterable[str],
    num_shards: int = 4,
    seed: int = 17,
    selected_problem_ids_by_track: dict[str, tuple[str, ...]] | None = None,
) -> list[RecoveryCell]:
    cells: list[RecoveryCell] = []
    for track in tracks:
        direct_kind = direct_archive_kind_for_track(track)
        seed_kind = seed_archive_kind_for_track(track)
        if selected_problem_ids_by_track is None:
            selected_problem_ids = None
        else:
            selected_problem_ids = selected_problem_ids_by_track.get(track, ())
            if not selected_problem_ids:
                continue
        max_new_tokens = token_cap_for_track("phase2", track)
        shard_ids = range(num_shards)
        if selected_problem_ids is not None:
            shard_ids = [
                shard_id
                for shard_id in range(num_shards)
                if shard_indices(len(selected_problem_ids), shard_id, num_shards)
            ]
        for shard_id in shard_ids:
            base = {
                "phase": "phase2",
                "track": track,
                "model_key": BASE_MODEL_KEY,
                "split": (0, problem_count),
                "shard_id": shard_id,
                "num_shards": num_shards,
                "max_new_tokens": max_new_tokens,
                "max_model_len": max(8192, max_new_tokens),
                "seed": seed,
                "selected_problem_ids": selected_problem_ids,
            }
            cells.append(
                RecoveryCell(
                    **base,
                    rung="rung0_greedy",
                    condition="greedy",
                    samples_per_problem=1,
                    sampling_temperature=0.0,
                    archive_kind=direct_kind,
                    out_dir=f"{root}/phase2/{track}/rung0_greedy/shard-{shard_id}",
                    cache_path=f"{root}/trajectory_cache/phase2-{track}-rung0-shard-{shard_id}.sqlite",
                )
            )
            for k in BON_K:
                for temp, rung in ((1.0, "rung1_bon_t1"), (1.2, "rung2_bon_t12")):
                    cells.append(
                        RecoveryCell(
                            **base,
                            rung=f"{rung}_k{k}",
                            condition="bon_temp1",
                            samples_per_problem=k,
                            sampling_temperature=temp,
                            archive_kind=direct_kind,
                            out_dir=f"{root}/phase2/{track}/{rung}_k{k}/shard-{shard_id}",
                            cache_path=(
                                f"{root}/trajectory_cache/phase2-{track}-{rung}-k{k}"
                                f"-shard-{shard_id}.sqlite"
                            ),
                        )
                    )
            cells.extend(
                [
                    RecoveryCell(
                        **base,
                        rung="rung3_rws_mcmc",
                        condition="single_prompt_power",
                        samples_per_problem=64,
                        sampling_temperature=1.0,
                        archive_kind=direct_kind,
                        out_dir=f"{root}/phase2/{track}/rung3_rws_mcmc/shard-{shard_id}",
                        cache_path=f"{root}/trajectory_cache/phase2-{track}-rung3-shard-{shard_id}.sqlite",
                    ),
                    RecoveryCell(
                        **base,
                        rung="rung4_mixed_alpha",
                        condition="full_archive_mixed",
                        samples_per_problem=64,
                        sampling_temperature=1.0,
                        archive_kind=direct_kind,
                        out_dir=f"{root}/phase2/{track}/rung4_mixed_alpha/shard-{shard_id}",
                        cache_path=f"{root}/trajectory_cache/phase2-{track}-rung4-shard-{shard_id}.sqlite",
                    ),
                    RecoveryCell(
                        **base,
                        rung="rung5_gepa_archive",
                        condition="gepa_only",
                        samples_per_problem=16,
                        sampling_temperature=1.0,
                        archive_kind=seed_kind,
                        out_dir=f"{root}/phase2/{track}/rung5_gepa_archive/shard-{shard_id}",
                        cache_path=f"{root}/trajectory_cache/phase2-{track}-rung5-shard-{shard_id}.sqlite",
                    ),
                    RecoveryCell(
                        **base,
                        rung="rung6_archive_mixed",
                        condition="full_archive_mixed",
                        samples_per_problem=128,
                        sampling_temperature=1.0,
                        archive_kind=seed_kind,
                        out_dir=f"{root}/phase2/{track}/rung6_archive_mixed/shard-{shard_id}",
                        cache_path=f"{root}/trajectory_cache/phase2-{track}-rung6-shard-{shard_id}.sqlite",
                    ),
                    RecoveryCell(
                        **base,
                        rung="rung7_full_memory",
                        condition="polaris_full_verified_memory",
                        samples_per_problem=128,
                        sampling_temperature=1.0,
                        archive_kind=seed_kind,
                        out_dir=f"{root}/phase2/{track}/rung7_full_memory/shard-{shard_id}",
                        cache_path=f"{root}/trajectory_cache/phase2-{track}-rung7-shard-{shard_id}.sqlite",
                    ),
                ]
            )
    return cells


def phase0_cells(*, root: str, num_shards: int = 4, seed: int = 17) -> list[RecoveryCell]:
    return [
        RecoveryCell(
            phase="phase0",
            rung="karan_du_replication",
            track="math500",
            model_key="qwen2.5-math-7b",
            condition="single_prompt_power",
            split=(0, 500),
            shard_id=shard_id,
            num_shards=num_shards,
            samples_per_problem=1,
            sampling_temperature=1.0,
            max_new_tokens=token_cap_for_track("phase0", "math500"),
            max_model_len=8192,
            archive_kind="rws_math_direct",
            out_dir=f"{root}/phase0/karan_du_replication/shard-{shard_id}",
            cache_path=f"{root}/trajectory_cache/phase0-karan-du-shard-{shard_id}.sqlite",
            seed=seed,
            backend="hf",
        )
        for shard_id in range(num_shards)
    ]


def phase0_rws_exact_cells(
    *, root: str, num_shards: int = 5, num_seeds: int = 8
) -> list[RWSExactCell]:
    """Render the upstream RWS Phase 0 matrix without POLARIS wrapper changes."""

    cells: list[RWSExactCell] = []
    for batch_idx in range(num_shards):
        for seed in range(num_seeds):
            cells.append(
                RWSExactCell(
                    root=root,
                    cell_index=len(cells),
                    batch_idx=batch_idx,
                    seed=seed,
                    num_shards=num_shards,
                    num_seeds=num_seeds,
                )
            )
    return cells


def rws_exact_cell_command(
    cell: RWSExactCell, *, repo_dir: str = "$POLARIS_REPO_DIR"
) -> list[str]:
    """Return the shell command that runs upstream RWS `power_samp_math.py`."""

    llm_dir = f"{repo_dir}/upstream/reasoning-with-sampling/llm_experiments"
    cmd = (
        f"export HF_HUB_OFFLINE=1 && "
        f"cd {shlex.quote(llm_dir)} && "
        f"{shlex.quote(sys.executable)} power_samp_math.py "
        f"--save_str {shlex.quote(cell.save_str)} "
        f"--model {shlex.quote(cell.model)} "
        f"--temperature {cell.temperature} "
        f"--mcmc_steps {cell.mcmc_steps} "
        f"--batch_idx {cell.batch_idx} "
        f"--seed {cell.seed}"
    )
    return ["bash", "-lc", cmd]


def _read_hf_snapshot(model_id: str) -> str | None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None
    ref = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
    if not ref.exists():
        return None
    return ref.read_text(encoding="utf-8").strip() or None


def _rws_exact_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "hf_home": os.environ.get("HF_HOME"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
    }
    try:
        import numpy
        import torch
        import transformers

        payload.update(
            {
                "numpy_version": numpy.__version__,
                "torch_version": torch.__version__,
                "torch_cuda": getattr(torch.version, "cuda", None),
                "transformers_version": transformers.__version__,
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic only.
        payload["version_probe_error"] = repr(exc)
    return payload


def run_rws_exact_cell(cell: RWSExactCell, *, repo_dir: str) -> dict[str, Any]:
    """Run one upstream-RWS MATH500 cell and write forensic metadata."""

    out_dir = Path(cell.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell.json").write_text(
        json.dumps(cell.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "environment.json").write_text(
        json.dumps(_rws_exact_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cmd = rws_exact_cell_command(cell, repo_dir=repo_dir)
    started = platform.node()
    subprocess.run(cmd, check=True)
    snapshot = _read_hf_snapshot("Qwen/Qwen2.5-Math-7B")
    manifest = {
        "cell": cell.to_jsonable(),
        "host": started,
        "upstream_repo": "upstream/reasoning-with-sampling",
        "upstream_commit": _git_rev_parse(Path(repo_dir) / "upstream" / "reasoning-with-sampling"),
        "model_id": "Qwen/Qwen2.5-Math-7B",
        "model_revision": snapshot,
        "expected_csv_exists": Path(cell.expected_csv).exists(),
        "expected_csv": cell.expected_csv,
        "command": cmd,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _git_rev_parse(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def cell_command(
    cell: RecoveryCell,
    *,
    repo_dir: str = "$POLARIS_REPO_DIR",
    run_kind: str = "cloudrift",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    estimated_wall_clock_seconds: float | None = None,
    user_authorized_paid_run: bool = False,
) -> list[str]:
    model_revision = MODEL_REGISTRY[cell.model_key].get("revision")
    archive_path = recovery_archive_path(cell.archive_kind, repo_dir=repo_dir)
    cmd = [
        "python",
        "scripts/run_condition.py",
        "--track",
        cell.track,
        "--model-key",
        cell.model_key,
        "--condition",
        cell.condition,
        "--archive",
        archive_path,
        "--split",
        str(cell.split[0]),
        str(cell.split[1]),
        "--seed",
        str(cell.seed),
        "--polaris-source-hash",
        "filesystem-cloudrift-prorl-recovery",
        "--preregistration-anchor",
        "TODO.PROICL.md#proicl-fast-weight-recovery-audit",
        "--out",
        cell.out_dir,
        "--backend",
        cell.backend,
        "--samples-per-problem",
        str(cell.samples_per_problem),
        "--sampling-temperature",
        str(cell.sampling_temperature),
        "--max-new-tokens",
        str(cell.max_new_tokens),
        "--shard-id",
        str(cell.shard_id),
        "--num-shards",
        str(cell.num_shards),
        "--trajectory-cache",
        cell.cache_path,
        "--run-kind",
        run_kind,
        "--run-stage",
        "small_real_slice",
        "--vllm-dtype",
        "bfloat16",
        "--vllm-gpu-memory-utilization",
        "0.90",
        "--vllm-max-model-len",
        str(cell.max_model_len),
    ]
    if model_revision is not None:
        cmd.extend(["--model-revision", str(model_revision)])
    if estimated_dollar_cost is not None:
        cmd.extend(["--estimated-dollar-cost", str(estimated_dollar_cost)])
    if cost_cap_dollars is not None:
        cmd.extend(["--cost-cap-dollars", str(cost_cap_dollars)])
    if estimated_wall_clock_seconds is not None:
        cmd.extend(["--estimated-wall-clock-seconds", str(estimated_wall_clock_seconds)])
    if user_authorized_paid_run:
        cmd.append("--user-authorized-paid-run")
    if cell.selected_problem_ids is not None:
        cmd.append("--problem-ids")
        cmd.extend(cell.selected_problem_ids)
    if cell.condition == "polaris_full_verified_memory":
        cmd.extend(
            [
                "--memory-store",
                f"{cell.out_dir}/memory.sqlite",
                "--memory-mode",
                "distilled_strategies",
                "--admit-memory",
                "--online-memory",
                "--archive-build-id",
                f"{cell.track}-seed-archive",
                "--memory-build-id",
                f"{cell.track}-base-verified-memory",
            ]
        )
    return cmd


def _parse_rws_exact_filename(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    prefix = "qwen_math_math_base_power_samp_results_10_0.25_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    try:
        batch_s, seed_s = rest.split("_")
        return int(batch_s), int(seed_s)
    except ValueError:
        return None


def _safe_rws_grade(ans: Any, correct_ans: Any) -> int:
    from polaris.vendored.rws.grader_utils.math_grader import grade_answer

    try:
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0


def aggregate_rws_exact_phase0_gate(
    root: Path,
    out_path: Path,
    *,
    expected_problems: int = 500,
    expected_shards: int = 5,
    expected_seeds: int = 8,
) -> dict[str, Any]:
    """Aggregate exact upstream RWS CSV outputs into the Karan-Du gate."""

    import pandas as pd

    result_dir = root / "phase0" / "rws_exact" / "results" / "qwen_math"
    files = sorted(result_dir.glob("qwen_math_math_base_power_samp_results_10_0.25_*_*.csv"))
    by_seed: dict[int, list[Path]] = {}
    for path in files:
        parsed = _parse_rws_exact_filename(path)
        if parsed is None:
            continue
        _, seed = parsed
        by_seed.setdefault(seed, []).append(path)

    per_seed: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        seed_files = sorted(by_seed[seed])
        total = 0
        correct = 0
        batches = []
        for path in seed_files:
            parsed = _parse_rws_exact_filename(path)
            if parsed is None:
                continue
            batch_idx, _ = parsed
            batches.append(batch_idx)
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                total += 1
                correct += _safe_rws_grade(row.get("mcmc_answer", ""), row.get("correct_answer", ""))
        per_seed.append(
            {
                "seed": seed,
                "files": [str(path) for path in seed_files],
                "batches": sorted(batches),
                "complete": total == expected_problems
                and sorted(batches) == list(range(expected_shards)),
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    complete_seeds = [row for row in per_seed if row["complete"]]
    accuracy = (
        sum(float(row["accuracy"]) for row in complete_seeds) / len(complete_seeds)
        if complete_seeds
        else 0.0
    )
    complete = len(complete_seeds) == expected_seeds
    gate_passed = complete and KARAN_DU_REPLICATION_GATE.accepts(accuracy)
    report = {
        "gate": {
            **KARAN_DU_REPLICATION_GATE.to_jsonable(),
            "aggregation": "mean_complete_seed_accuracy_from_upstream_rws_csv",
            "expected_shards": expected_shards,
            "expected_seeds": expected_seeds,
        },
        "result_dir": str(result_dir),
        "files": len(files),
        "complete": complete,
        "accuracy": accuracy,
        "passed": gate_passed,
        "per_seed": per_seed,
        "out": str(out_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def aggregate_phase1(candidate_files: Iterable[Path], out_path: Path) -> dict[str, Any]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for path in candidate_files:
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preflight_path = path.parent / "preflight.json"
        preflight = (
            json.loads(preflight_path.read_text(encoding="utf-8"))
            if preflight_path.exists()
            else {}
        )
        config = manifest.get("config", {})
        checkpoint = config.get("model_key") or manifest.get("model_id")
        task_family = config.get("track") or manifest.get("benchmark")
        generation_backend = preflight.get("backend") or config.get("backend")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                verifier = raw.get("verifier_result", {})
                token_ids = raw.get("token_ids") or []
                logprobs = raw.get("logprobs_norm") or []
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "task_family": task_family,
                        "problem_id": raw["problem_id"],
                        "sample_idx": int(raw["sample_index"]),
                        "response": raw.get("generation", ""),
                        "verified": bool(verifier.get("passed", False)),
                        "verifier_score": float(verifier.get("score", 0.0)),
                        "logprob_sum": float(sum(logprobs)) if logprobs else math.nan,
                        "response_length": int(raw.get("generation_token_count", 0)),
                        "token_ids": token_ids,
                        "generation_backend": generation_backend,
                        "source_file": str(path),
                    }
                )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    pass_rows: list[dict[str, Any]] = []
    if not df.empty:
        for keys, group in df.groupby(["checkpoint", "task_family", "problem_id"]):
            checkpoint, task_family, problem_id = keys
            for k in (1, 16, 128):
                pass_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "task_family": task_family,
                        "problem_id": problem_id,
                        "k": k,
                        "passed": bool(group[group["sample_idx"] < k]["verified"].any()),
                    }
                )
    pass_df = pd.DataFrame(pass_rows)
    summary: dict[str, Any] = {"rows": len(df), "out": str(out_path), "pass_at": []}
    if not pass_df.empty:
        for keys, group in pass_df.groupby(["checkpoint", "task_family", "k"]):
            checkpoint, task_family, k = keys
            summary["pass_at"].append(
                {
                    "checkpoint": checkpoint,
                    "task_family": task_family,
                    "k": int(k),
                    "accuracy": float(group["passed"].mean()),
                    "n_problems": int(len(group)),
                }
            )
    summary["score_at"] = []
    if not df.empty:
        for keys, group in df.groupby(["checkpoint", "task_family"]):
            checkpoint, task_family = keys
            for k in (1, 16, 128):
                first_scores: list[float] = []
                best_scores: list[float] = []
                all_scores: list[float] = []
                for _, problem_group in group.groupby("problem_id"):
                    window = problem_group[problem_group["sample_idx"] < k].sort_values(
                        "sample_idx"
                    )
                    if window.empty:
                        continue
                    scores = [float(x) for x in window["verifier_score"]]
                    first_scores.append(scores[0])
                    best_scores.append(max(scores))
                    all_scores.extend(scores)
                summary["score_at"].append(
                    {
                        "checkpoint": checkpoint,
                        "task_family": task_family,
                        "k": int(k),
                        "first_score_mean": (
                            float(sum(first_scores) / len(first_scores))
                            if first_scores
                            else math.nan
                        ),
                        "best_score_mean": (
                            float(sum(best_scores) / len(best_scores))
                            if best_scores
                            else math.nan
                        ),
                        "all_score_mean": (
                            float(sum(all_scores) / len(all_scores))
                            if all_scores
                            else math.nan
                        ),
                        "n_problems": int(len(first_scores)),
                        "n_samples": int(len(all_scores)),
                    }
                )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def aggregate_phase0_gate(
    candidate_files: Iterable[Path],
    out_path: Path,
    *,
    expected_problems: int = 500,
) -> dict[str, Any]:
    """Aggregate the Karan-Du replication gate into a hard pass/fail report."""

    by_problem: dict[str, bool] = {}
    rows = 0
    for path in candidate_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                problem_id = str(raw["problem_id"])
                passed = bool(raw.get("verifier_result", {}).get("passed", False))
                by_problem[problem_id] = by_problem.get(problem_id, False) or passed
                rows += 1

    n_problems = len(by_problem)
    accuracy = (
        sum(1 for passed in by_problem.values() if passed) / n_problems
        if n_problems
        else 0.0
    )
    complete = n_problems == expected_problems and rows == expected_problems
    gate_passed = complete and KARAN_DU_REPLICATION_GATE.accepts(accuracy)
    report = {
        "gate": KARAN_DU_REPLICATION_GATE.to_jsonable(),
        "rows": rows,
        "n_problems": n_problems,
        "expected_problems": expected_problems,
        "complete": complete,
        "accuracy": accuracy,
        "passed": gate_passed,
        "out": str(out_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


REQUIRED_CELL_ARTIFACTS: tuple[str, ...] = (
    "manifest.json",
    "archive.json",
    "candidates.jsonl",
    "scores.jsonl",
    "selected.jsonl",
    "metrics.json",
    "costs.json",
    "rollouts.json",
    "preflight.json",
    "environment.json",
    "run_plan_cell.json",
    "audit.md",
)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def audit_recovery_cells(cells: Iterable[RecoveryCell]) -> dict[str, Any]:
    """Audit required artifacts and row counts for a rendered recovery plan."""

    failures: list[dict[str, Any]] = []
    totals = {
        "cells": 0,
        "expected_problems": 0,
        "selected_rows": 0,
        "expected_candidates": 0,
        "candidate_rows": 0,
        "score_rows": 0,
    }
    for cell in cells:
        totals["cells"] += 1
        out = Path(cell.out_dir)
        missing = [name for name in REQUIRED_CELL_ARTIFACTS if not (out / name).exists()]
        if missing:
            failures.append(
                {
                    "cell": cell.to_jsonable(),
                    "reason": "missing required artifacts",
                    "missing": missing,
                }
            )
            continue
        problem_count = (
            len(cell.selected_problem_ids)
            if cell.selected_problem_ids is not None
            else cell.split[1] - cell.split[0]
        )
        expected_problems = len(
            shard_indices(problem_count, cell.shard_id, cell.num_shards)
        )
        expected_candidates = expected_problems * cell.samples_per_problem
        candidate_rows = _line_count(out / "candidates.jsonl")
        score_rows = _line_count(out / "scores.jsonl")
        selected_rows = _line_count(out / "selected.jsonl")
        totals["expected_problems"] += expected_problems
        totals["selected_rows"] += selected_rows
        totals["expected_candidates"] += expected_candidates
        totals["candidate_rows"] += candidate_rows
        totals["score_rows"] += score_rows
        if selected_rows != expected_problems:
            failures.append(
                {
                    "cell": cell.to_jsonable(),
                    "reason": (
                        f"selected row count {selected_rows} != expected "
                        f"{expected_problems}"
                    ),
                }
            )
        if candidate_rows != expected_candidates:
            failures.append(
                {
                    "cell": cell.to_jsonable(),
                    "reason": (
                        f"candidate row count {candidate_rows} != expected "
                        f"{expected_candidates}"
                    ),
                }
            )
        if score_rows != expected_candidates:
            failures.append(
                {
                    "cell": cell.to_jsonable(),
                    "reason": (
                        f"scores row count {score_rows} != expected "
                        f"{expected_candidates}"
                    ),
                }
            )
        if cell.condition == "polaris_full_verified_memory":
            memory_missing = [
                name
                for name in ("memory.sqlite", "memory_events.jsonl")
                if not (out / name).exists()
            ]
            if memory_missing:
                failures.append(
                    {
                        "cell": cell.to_jsonable(),
                        "reason": "missing memory artifacts",
                        "missing": memory_missing,
                    }
                )
    return {
        "passed": not failures,
        "totals": totals,
        "failures": failures,
    }
