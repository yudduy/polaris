from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.evals.datasets.math500 import Problem
from polaris.gepa_reflection import reflection_manifest
from polaris.io.rollouts import RolloutLedger
from polaris.proicl.protocol import (
    ArchiveScope,
    archive_scope_id,
    validate_archive_scope_membership,
)


PROICL_REASONING_GYM_TRACKS: tuple[str, ...] = (
    "reasoning_gym_family_relationships",
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
    "reasoning_gym_boxnet",
    "reasoning_gym_acre",
    "reasoning_gym_game_of_life_halting",
    "reasoning_gym_maze",
    "reasoning_gym_palindrome_generation",
    "reasoning_gym_letter_counting",
)

_DEFAULT_SUFFIX = (
    "\n\nReturn only the final value requested by the task. Do not include a "
    "placeholder, example answer, or extra prose in the final response."
)

_SEED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("direct", "Solve the verifier-scored task. Reason briefly, then answer.\n\nTask:\n"),
    ("rules", "Identify the task rules, apply them step by step, then answer.\n\nTask:\n"),
    ("state", "Track the current state explicitly before choosing the final response.\n\nTask:\n"),
    ("constraint", "List the constraints that must be satisfied, check them, then answer.\n\nTask:\n"),
    ("planner", "Make a compact plan, execute it, verify the result, then answer.\n\nTask:\n"),
    ("inverse", "Work backward from the required final structure when helpful.\n\nTask:\n"),
    ("search", "Search over plausible candidates and reject those violating the rules.\n\nTask:\n"),
    ("format", "Solve the task while keeping the final answer parseable by the grader.\n\nTask:\n"),
    ("graph", "Represent entities and relations as a small graph before answering.\n\nTask:\n"),
    ("table", "Use a small table of intermediate facts when the task has multiple objects.\n\nTask:\n"),
    ("simulate", "Simulate each operation carefully and check the terminal state.\n\nTask:\n"),
    ("minimal", "Focus on the minimal valid output required by the verifier.\n\nTask:\n"),
    ("audit", "After deriving an answer, audit it against every stated condition.\n\nTask:\n"),
    ("decompose", "Decompose the problem into primitives, solve each primitive, then combine.\n\nTask:\n"),
    ("counterexample", "Try to find a contradiction to the candidate answer before finalizing.\n\nTask:\n"),
    ("transfer", "Use analogous solved task patterns only when they preserve the task rules.\n\nTask:\n"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, default=default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _fallback_rows(tracks: tuple[str, ...], split: tuple[int, int]) -> list[Problem]:
    rows: list[Problem] = []
    for track in tracks:
        for idx in range(split[0], split[1]):
            rows.append(
                Problem(
                    problem_id=f"{track}:dry-{idx}",
                    prompt=f"Dry-run {track} task {idx}.",
                    answer="",
                    source="dry_run",
                )
            )
    return rows


def _load_rows(
    *,
    tracks: tuple[str, ...],
    split: tuple[int, int],
    dry_run: bool,
) -> tuple[list[Any], str]:
    from polaris.runners.condition_runner import load_track_optimizer_dev_slice

    rows: list[Any] = []
    try:
        for track in tracks:
            rows.extend(load_track_optimizer_dev_slice(track, split[0], split[1]))
        return rows, "optimizer_dev"
    except Exception:
        if not dry_run:
            raise
        return _fallback_rows(tracks, split), "dry_run_fallback"


def _seed_archive(archive_size: int) -> FrozenArchive:
    if archive_size <= 0:
        raise ValueError("archive_size must be positive")
    entries = []
    for idx in range(archive_size):
        name, prefix = _SEED_PREFIXES[idx % len(_SEED_PREFIXES)]
        entries.append(
            PromptEntry(
                id=f"proicl_{idx:02d}_{name}",
                prefix=prefix,
                suffix=_DEFAULT_SUFFIX,
                descriptor_hint=f"proicl_cross_task_{name}",
            )
        )
    return FrozenArchive(entries=tuple(entries))


def _archive_from_gepa_candidates(candidates: list[dict[str, str]], archive_size: int) -> FrozenArchive:
    entries: list[PromptEntry] = []
    seen: set[str] = set()
    for idx, candidate in enumerate(candidates):
        text = str(candidate.get("instruction") or next(iter(candidate.values()), "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        entries.append(
            PromptEntry(
                id=f"proicl_gepa_{len(entries):02d}",
                prefix=text.rstrip() + "\n\nTask:\n",
                suffix=_DEFAULT_SUFFIX,
                descriptor_hint=f"proicl_gepa_candidate_{idx}",
            )
        )
        if len(entries) >= archive_size:
            break
    if len(entries) < archive_size:
        entries.extend(_seed_archive(archive_size - len(entries)).entries)
    return FrozenArchive(entries=tuple(entries[:archive_size]))


def _run_live_gepa(
    *,
    out_dir: Path,
    rows: list[Any],
    archive_size: int,
    sampler: Any,
    scorer: Any,
    reflection_lm: Any,
    max_metric_calls: int,
    sampler_max_new_tokens: int,
    seed: int,
) -> tuple[FrozenArchive, dict[str, Any]]:
    from polaris.gepa_integration import PolarisGEPAAdapter
    from polaris.vendored.gepa import optimize

    adapter = PolarisGEPAAdapter(
        sampler=sampler,
        scorer=scorer,
        max_new_tokens=sampler_max_new_tokens,
        run_dir=out_dir / "gepa_adapter",
    )
    result = optimize(
        seed_candidate={
            "instruction": (
                "Solve the verifier-scored reasoning task. Reason through the "
                "task rules and return exactly one final answer-tag block in the "
                "requested format.\n\n"
            )
        },
        trainset=rows,
        valset=rows,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        run_dir=str(out_dir / "gepa_run"),
        seed=seed,
        display_progress_bar=False,
    )
    _write_json(out_dir / "gepa_result.json", result.to_dict())
    return _archive_from_gepa_candidates(result.candidates, archive_size), {
        "dry_run": False,
        "max_metric_calls": max_metric_calls,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": len(result.candidates),
    }


def build_cross_task_curriculum_archive(
    *,
    out_dir: Path,
    tracks: tuple[str, ...] = PROICL_REASONING_GYM_TRACKS,
    archive_scope: ArchiveScope | str = ArchiveScope.WITHIN_FAMILY,
    heldout_tracks: tuple[str, ...] = (),
    dev_split: tuple[int, int] = (0, 100),
    archive_size: int = 16,
    dry_run: bool = True,
    max_metric_calls: int = 1000,
    reflection_provider: str = "none",
    sampler: Any | None = None,
    scorer: Any | None = None,
    reflection_lm: Any | None = None,
    reflection_config: Any | None = None,
    sampler_max_new_tokens: int = 512,
    seed: int = 17,
) -> dict[str, Any]:
    """Build a cross-task Reasoning Gym prompt archive for ProICL.

    Dry-run mode writes the same artifact contract with deterministic seed
    prompts. Live GEPA requires a sampler, scorer, and reflection LM so paid
    calls cannot happen accidentally from this helper.
    """

    archive_scope = ArchiveScope(archive_scope)
    heldout_tracks = tuple(heldout_tracks or tracks)
    if not tracks:
        raise ValueError("at least one track is required")
    validate_archive_scope_membership(
        archive_scope=archive_scope,
        train_tracks=tracks,
        heldout_tracks=heldout_tracks,
    )
    if archive_size <= 0:
        raise ValueError("archive_size must be positive")
    if max_metric_calls <= 0:
        raise ValueError("max_metric_calls must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, dev_source = _load_rows(tracks=tracks, split=dev_split, dry_run=dry_run)
    if dry_run:
        archive = _seed_archive(archive_size)
        gepa_payload = {
            "dry_run": True,
            "max_metric_calls": max_metric_calls,
            "total_metric_calls": 0,
            "num_candidates": archive_size,
        }
    else:
        if sampler is None or scorer is None or reflection_lm is None:
            raise ValueError("live GEPA requires sampler, scorer, and reflection_lm")
        archive, gepa_payload = _run_live_gepa(
            out_dir=out_dir,
            rows=rows,
            archive_size=archive_size,
            sampler=sampler,
            scorer=scorer,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            sampler_max_new_tokens=sampler_max_new_tokens,
            seed=seed,
        )

    scope_id = archive_scope_id(archive_scope, tracks)
    archive_build_id = f"{scope_id}-k{archive_size}-seed-{seed}-{dev_split[0]}-{dev_split[1]}"
    archive_payload = {
        "entries": archive.to_jsonable(),
        "cell_fitness": {
            entry.descriptor_hint: 1.0 for entry in archive.entries
        },
        "archive_build_id": archive_build_id,
        "selection_rule": f"gepa_pareto_{archive_scope.value}",
        "frozen": True,
    }
    _write_json(out_dir / "archive.json", archive_payload)
    _write_jsonl(
        out_dir / "optimizer_feedback_rows.jsonl",
        [
            {
                "problem_id": getattr(row, "problem_id", f"row-{idx}"),
                "source": getattr(row, "source", "unknown"),
            }
            for idx, row in enumerate(rows)
        ],
    )
    ledger = RolloutLedger(archive_construction=int(gepa_payload["total_metric_calls"]))
    ledger.write(out_dir / "rollouts.json")

    reflection_config_payload = reflection_config
    reflection_usage = {}
    if reflection_lm is not None:
        reflection_usage = {
            "total_cost": float(getattr(reflection_lm, "total_cost", 0.0) or 0.0),
            "total_tokens_in": int(getattr(reflection_lm, "total_tokens_in", 0) or 0),
            "total_tokens_out": int(getattr(reflection_lm, "total_tokens_out", 0) or 0),
        }
    manifest = {
        "archive_build_id": archive_build_id,
        "archive_scope": archive_scope.value,
        "archive_scope_id": scope_id,
        "tracks": list(tracks),
        "train_tracks": list(tracks),
        "heldout_tracks": list(heldout_tracks),
        "dev_split": list(dev_split),
        "dev_source": dev_source,
        "archive_size": archive_size,
        "seed": seed,
        "gepa": gepa_payload,
        "reflection": reflection_manifest(
            provider=reflection_provider,
            config=reflection_config_payload,
            status="dry_run" if dry_run else "complete",
            usage=reflection_usage,
        ),
        "leakage_policy": {
            "optimizer_feedback_split_only": True,
            "no_prorl_or_brorl_traces": True,
            "heldout_eval_not_used_for_prompt_evolution": True,
            "cross_family_target_leakage_blocked": (
                archive_scope == ArchiveScope.CROSS_FAMILY_CURRICULUM
            ),
        },
        "artifact_paths": {
            "archive": str(out_dir / "archive.json"),
            "rollouts": str(out_dir / "rollouts.json"),
            "optimizer_feedback_rows": str(out_dir / "optimizer_feedback_rows.jsonl"),
        },
    }
    _write_json(out_dir / "archive_build_manifest.json", manifest)
    return manifest
