from __future__ import annotations

import datetime as _dt
import os
import platform
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from polaris.config import MAX_NEW_TOKENS, MCMC_BLOCK_NUM, MCMC_STEPS, MODEL_ID
from polaris.core.archive import FrozenArchive
from polaris.core.inference import Candidate, polaris_inference
from polaris.core.fork_search import ForkSearchConfig, run_entropy_gated_fork_search
from polaris.core.memory import (
    MAX_RETRIEVED_MEMORY_ENTRIES,
    MAX_RETRIEVED_MEMORY_TOKENS,
    MemoryStore,
)
from polaris.core.mixed_alpha import (
    DECAYING_ALPHA_4_TO_1,
    FIXED_ALPHA_4,
    MIXED_ALPHA_4_1,
    AlphaSchedule,
)
from polaris.core.persistent_memory import PersistentMemoryLedger
from polaris.core.repair import RepairConfig, run_verifier_guided_repair
from polaris.evals.datasets.math500 import MATH500_TEST_SLICE, Problem
from polaris.evals.verifiers.math import VERIFIER_ID, score_math
from polaris.io.artifacts import append_jsonl, write_json, write_jsonl
from polaris.io.manifest import write_run_manifest
from polaris.io.rollouts import RolloutLedger
from polaris.io.trajectory_cache import TrajectoryCache, TrajectoryKey, TrajectoryRecord

CONDITIONS: tuple[str, ...] = (
    "greedy",
    "bon_temp1",
    "bon_temp1_archive",
    "single_prompt_power",
    "single_best_prompt",
    "full_archive_fixed",
    "full_archive_mixed",
    "full_archive_decaying",
    "polaris_full_verified_memory",
    "proicl_gepa_mcmc",
    "proicl_gepa_mcmc_memory",
    "mixed_alpha_mcmc",
    "fork_search",
    "proicl_gepa_mcmc_repair",
    "proicl_gepa_mcmc_fork_repair",
    "proicl_gepa_mcmc_fork_repair_memory",
    "dynamic_cheatsheet",
    "ace",
    "gepa_only",
)


class _SamplerLike(Protocol):
    def generate_greedy(self, prompt_text: str, max_new_tokens: int) -> Any: ...
    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int = ...,
        block_num: int = ...,
    ) -> Any: ...
    def generate_low_temp(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int
    ) -> Any: ...
    def generate_sps_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        block_num: int = ...,
        top_k: int = ...,
        candidate_pool_size: int = ...,
        rollouts_per_candidate: int = ...,
        rollout_horizon: int | None = ...,
        seed_base: int | None = ...,
        seed_offsets: list[int] | None = ...,
    ) -> list[Any]: ...


_ALPHA_1_ONLY = AlphaSchedule(policy_id="bon_temp1", alphas=(1.0,))


@dataclass(frozen=True)
class _CandidateJob:
    prompt_id: str
    sample_index: int
    alpha: float
    prompt_text: str
    stable_seed_offset: int


BestSelector = Callable[[list[Candidate]], tuple[Candidate, dict[str, Any]]]


def _select_archive_subset(
    archive: FrozenArchive, cell_fitness: dict[str, float], condition: str
) -> FrozenArchive:
    if condition in (
        "greedy",
        "bon_temp1",
        "single_prompt_power",
        "mixed_alpha_mcmc",
        "fork_search",
        "dynamic_cheatsheet",
        "ace",
    ):
        direct = next(e for e in archive.entries if e.id == "direct")
        return FrozenArchive(entries=(direct,))
    if condition == "single_best_prompt":
        if not cell_fitness:
            raise ValueError(
                "single_best_prompt requires cell_fitness from MAP-Elites archive.json"
            )
        best_descriptor = max(cell_fitness.items(), key=lambda kv: kv[1])[0]
        best = next(e for e in archive.entries if e.descriptor_hint == best_descriptor)
        return FrozenArchive(entries=(best,))
    if condition in (
        "full_archive_fixed",
        "full_archive_mixed",
        "full_archive_decaying",
        "polaris_full_verified_memory",
        "proicl_gepa_mcmc",
        "proicl_gepa_mcmc_memory",
        "proicl_gepa_mcmc_repair",
        "proicl_gepa_mcmc_fork_repair",
        "proicl_gepa_mcmc_fork_repair_memory",
        "bon_temp1_archive",
        "gepa_only",
    ):
        return archive
    raise ValueError(f"unknown condition: {condition!r}")


def _select_schedule(condition: str) -> AlphaSchedule:
    if condition in (
        "full_archive_mixed",
        "polaris_full_verified_memory",
        "proicl_gepa_mcmc",
        "proicl_gepa_mcmc_memory",
        "mixed_alpha_mcmc",
        "proicl_gepa_mcmc_repair",
        "proicl_gepa_mcmc_fork_repair",
        "proicl_gepa_mcmc_fork_repair_memory",
    ):
        return MIXED_ALPHA_4_1
    if condition == "full_archive_decaying":
        return DECAYING_ALPHA_4_TO_1
    if condition in ("bon_temp1", "bon_temp1_archive", "dynamic_cheatsheet", "ace", "gepa_only", "fork_search"):
        return _ALPHA_1_ONLY
    return FIXED_ALPHA_4


def _budget_for(condition: str) -> int:
    if condition == "full_archive_decaying":
        return 16
    if condition in ("greedy", "dynamic_cheatsheet", "ace"):
        return 1
    return 8


def _greedy_candidate(
    sampler: _SamplerLike,
    archive_subset: FrozenArchive,
    problem: Problem,
    max_new_tokens: int,
    scorer: Callable[[str, str], dict],
) -> Candidate:
    entry = archive_subset.entries[0]
    prompt_text = entry.compose(problem.prompt)
    gen = sampler.generate_greedy(prompt_text, max_new_tokens=max_new_tokens)
    full_response = (
        gen.generation if gen.response_contains_prompt else prompt_text + gen.generation
    )
    verifier_result = scorer(full_response, problem.answer)
    return Candidate(
        prompt_id=entry.id,
        sample_index=0,
        alpha=0.0,  # greedy isn't a power-sampling alpha
        prompt_text=prompt_text,
        generation=gen.generation,
        response_contains_prompt=gen.response_contains_prompt,
        prompt_token_count=gen.prompt_token_count,
        generation_token_count=gen.generation_token_count,
        wall_clock_seconds=gen.wall_clock_seconds,
        estimated_dollar_cost=gen.estimated_dollar_cost,
        acceptance_ratio=None,
        verifier_result=verifier_result,
        token_ids=_list_attr(gen, "token_ids"),
        logprobs_norm=_list_attr(gen, "logprobs_norm"),
        logprobs_unnorm=_list_attr(gen, "logprobs_unnorm"),
    )


def _supports_batch_generation(sampler: _SamplerLike, *, power_sampler: str) -> bool:
    if not callable(getattr(sampler, "generate_low_temp_batch", None)):
        return False
    if power_sampler == "sps":
        return callable(getattr(sampler, "generate_sps_power_batch", None))
    return callable(getattr(sampler, "generate_power_batch", None))


def _list_attr(obj: Any, name: str) -> list:
    value = getattr(obj, name, None)
    return list(value) if value is not None else []


def _candidate_from_cached(
    *,
    job: _CandidateJob,
    cached,
    reference: str,
    scorer: Callable[[str, str], dict],
    trajectory_cache: TrajectoryCache,
) -> Candidate:
    verifier_result = cached.verifier_result
    if verifier_result is None:
        full_response = (
            cached.generation
            if cached.response_contains_prompt
            else job.prompt_text + cached.generation
        )
        verifier_result = scorer(full_response, reference)
        trajectory_cache.mark_verified(cached.key, verifier_result=verifier_result)
    return Candidate(
        prompt_id=job.prompt_id,
        sample_index=job.sample_index,
        alpha=job.alpha,
        prompt_text=job.prompt_text,
        generation=cached.generation,
        response_contains_prompt=cached.response_contains_prompt,
        prompt_token_count=cached.prompt_token_count,
        generation_token_count=cached.generation_token_count,
        wall_clock_seconds=cached.wall_clock_seconds,
        estimated_dollar_cost=cached.dollar_cost,
        acceptance_ratio=cached.acceptance_ratio,
        verifier_result=verifier_result,
        token_ids=list(cached.token_ids),
        logprobs_norm=list(cached.logprobs_norm),
        logprobs_unnorm=list(cached.logprobs_unnorm),
    )


def _candidate_from_generation(
    *,
    job: _CandidateJob,
    gen,
    reference: str,
    scorer: Callable[[str, str], dict],
    trajectory_cache: TrajectoryCache | None,
    model_id: str,
    track: str,
    problem_id: str,
    seed: int,
) -> Candidate:
    full_response = (
        gen.generation if gen.response_contains_prompt else job.prompt_text + gen.generation
    )
    verifier_result = scorer(full_response, reference)
    candidate = Candidate(
        prompt_id=job.prompt_id,
        sample_index=job.sample_index,
        alpha=job.alpha,
        prompt_text=job.prompt_text,
        generation=gen.generation,
        response_contains_prompt=gen.response_contains_prompt,
        prompt_token_count=gen.prompt_token_count,
        generation_token_count=gen.generation_token_count,
        wall_clock_seconds=gen.wall_clock_seconds,
        estimated_dollar_cost=gen.estimated_dollar_cost,
        acceptance_ratio=gen.acceptance_ratio,
        verifier_result=verifier_result,
        token_ids=_list_attr(gen, "token_ids"),
        logprobs_norm=_list_attr(gen, "logprobs_norm"),
        logprobs_unnorm=_list_attr(gen, "logprobs_unnorm"),
    )
    if trajectory_cache is not None:
        trajectory_cache.put(
            TrajectoryRecord(
                key=TrajectoryKey(
                    model_id=model_id,
                    track=track,
                    problem_id=problem_id,
                    prompt_id=job.prompt_id,
                    sample_idx=job.sample_index,
                    alpha=job.alpha,
                    seed=seed,
                ),
                prompt_hash=TrajectoryCache.prompt_hash(job.prompt_text),
                generation=gen.generation,
                response_contains_prompt=gen.response_contains_prompt,
                token_ids=_list_attr(gen, "token_ids"),
                logprobs_norm=_list_attr(gen, "logprobs_norm"),
                logprobs_unnorm=_list_attr(gen, "logprobs_unnorm"),
                acceptance_ratio=gen.acceptance_ratio,
                prompt_token_count=gen.prompt_token_count,
                generation_token_count=gen.generation_token_count,
                wall_clock_seconds=gen.wall_clock_seconds,
                dollar_cost=gen.estimated_dollar_cost,
                verifier_result=verifier_result,
            )
        )
    return candidate


def _best_candidate(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("batched inference produced no candidates")
    best_idx = 0
    best_score = candidates[0].verifier_result.get("score", 0.0)
    for idx in range(1, len(candidates)):
        score = candidates[idx].verifier_result.get("score", 0.0)
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]


def _default_best_selector(candidates: list[Candidate]) -> tuple[Candidate, dict[str, Any]]:
    best = _best_candidate(candidates)
    return best, {
        "selector_id": "argmax_verifier_score_iteration_tiebreak-v1",
        "oracle_used": True,
    }


def _cache_key_payload(
    *,
    model_id: str,
    track: str,
    problem_id: str,
    candidate: Candidate,
    seed: int,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "track": track,
        "problem_id": problem_id,
        "prompt_id": candidate.prompt_id,
        "sample_idx": candidate.sample_index,
        "alpha": candidate.alpha,
        "seed": seed,
    }


def _candidate_row(
    *,
    candidate: Candidate,
    problem: Problem,
    model_id: str,
    track: str,
    seed: int,
) -> dict[str, Any]:
    row = asdict(candidate)
    row["problem_id"] = problem.problem_id
    row["source"] = getattr(problem, "source", None)
    row["candidate_id"] = (
        f"{problem.problem_id}:{candidate.search_trace_id}"
        if candidate.search_trace_id
        else (
            f"{problem.problem_id}:{candidate.prompt_id}:"
            f"{candidate.sample_index}:alpha={candidate.alpha:g}"
        )
    )
    row["cache_key"] = _cache_key_payload(
        model_id=model_id,
        track=track,
        problem_id=problem.problem_id,
        candidate=candidate,
        seed=seed,
    )
    row["prompt_hash"] = TrajectoryCache.prompt_hash(candidate.prompt_text)
    row["token_counts"] = {
        "prompt": candidate.prompt_token_count,
        "generation": candidate.generation_token_count,
    }
    row["dollar_estimate"] = candidate.estimated_dollar_cost
    return row


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env_present": {
            "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
            "HUGGINGFACE_HUB_TOKEN": bool(os.environ.get("HUGGINGFACE_HUB_TOKEN")),
            "GPQA_DIAMOND_PATH": bool(os.environ.get("GPQA_DIAMOND_PATH")),
            "HUMANEVAL_OVERRIDE_PATH": bool(os.environ.get("HUMANEVAL_OVERRIDE_PATH")),
            "CUDA_VISIBLE_DEVICES": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        },
    }


def _memory_enabled(
    *,
    condition: str,
    memory_store: MemoryStore | None,
    memory_store_path: Path | None,
    memory_mode: str,
) -> bool:
    return (
        condition == "polaris_full_verified_memory"
        or condition == "proicl_gepa_mcmc_memory"
        or condition == "proicl_gepa_mcmc_fork_repair_memory"
        or memory_store is not None
        or memory_store_path is not None
        or memory_mode != "off"
    )


def _archive_with_memory(archive: FrozenArchive) -> FrozenArchive:
    return FrozenArchive(
        entries=archive.entries,
        max_retrieved_memory_entries=MAX_RETRIEVED_MEMORY_ENTRIES,
        max_retrieved_memory_tokens=MAX_RETRIEVED_MEMORY_TOKENS,
    )


def _memory_store_from_ledger(ledger: PersistentMemoryLedger) -> MemoryStore:
    return MemoryStore(entries=ledger.entries())


def _write_memory_events(path: Path, ledger: PersistentMemoryLedger) -> None:
    with path.open("w", encoding="utf-8") as f:
        for event in ledger.events():
            f.write(json.dumps(event, sort_keys=True) + "\n")


def _copy_memory_sqlite(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    had_invalid_line = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    had_invalid_line = True
    if had_invalid_line:
        write_jsonl(path, rows)
    return rows


def _problem_id_from_row(row: dict[str, Any]) -> str | None:
    value = row.get("problem_id")
    return str(value) if value is not None else None


def _normalize_selected_rows(
    path: Path,
    *,
    expected_problem_ids: set[str],
) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    changed = False
    for row in rows:
        problem_id = _problem_id_from_row(row)
        if problem_id is None or problem_id not in expected_problem_ids:
            changed = True
            continue
        if problem_id in seen:
            changed = True
            continue
        seen.add(problem_id)
        kept.append(row)
    if changed:
        write_jsonl(path, kept)
    elif not path.exists():
        path.touch()
    return kept


def _filter_jsonl_to_completed_problems(
    path: Path,
    *,
    completed_problem_ids: set[str],
    keep_problemless_rows: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    kept: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        problem_id = _problem_id_from_row(row)
        if problem_id is None:
            if keep_problemless_rows:
                kept.append(row)
            else:
                changed = True
            continue
        if problem_id in completed_problem_ids:
            kept.append(row)
        else:
            changed = True
    if changed:
        write_jsonl(path, kept)
    elif not path.exists():
        path.touch()
    return kept


def _selected_score(row: dict[str, Any]) -> float:
    if "selected_score" in row:
        return float(row.get("selected_score") or 0.0)
    verifier = row.get("verifier_result")
    if isinstance(verifier, dict):
        return float(verifier.get("score", 0.0) or 0.0)
    return 0.0


def _selected_passed(row: dict[str, Any]) -> bool:
    if "selected_passed" in row:
        return bool(row.get("selected_passed"))
    verifier = row.get("verifier_result")
    return bool(verifier.get("passed", False)) if isinstance(verifier, dict) else False


def _checkpoint_payload(
    *,
    track: str,
    condition: str,
    run_stage: str,
    expected_problem_ids: Sequence[str],
    completed_problem_ids: set[str],
    n_candidates: int,
) -> dict[str, Any]:
    ordered_completed = [
        problem_id
        for problem_id in expected_problem_ids
        if problem_id in completed_problem_ids
    ]
    return {
        "schema_version": 1,
        "track": track,
        "condition": condition,
        "run_stage": run_stage,
        "updated_at": _dt.datetime.now(_dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "complete": len(ordered_completed) == len(expected_problem_ids),
        "completed_problem_ids": ordered_completed,
        "completed_problems": len(ordered_completed),
        "expected_problems": len(expected_problem_ids),
        "n_candidates": n_candidates,
    }


def _write_checkpoint(
    path: Path,
    *,
    track: str,
    condition: str,
    run_stage: str,
    expected_problem_ids: Sequence[str],
    completed_problem_ids: set[str],
    n_candidates: int,
) -> None:
    write_json(
        path,
        _checkpoint_payload(
            track=track,
            condition=condition,
            run_stage=run_stage,
            expected_problem_ids=expected_problem_ids,
            completed_problem_ids=completed_problem_ids,
            n_candidates=n_candidates,
        ),
    )


def _eligible_memory_ids(store: MemoryStore | None, prompt_id: str) -> list[str]:
    if store is None:
        return []
    return [entry.id for entry in store.entries if entry.archive_prompt_id == prompt_id]


def _run_problem_batched(
    *,
    archive_subset: FrozenArchive,
    schedule: AlphaSchedule,
    sampler: _SamplerLike,
    problem: Problem,
    seed: int,
    model_id: str,
    track: str,
    max_new_tokens: int,
    mcmc_steps: int,
    mcmc_block_num: int,
    power_sampler: str,
    sps_top_k: int,
    sps_candidate_pool_size: int,
    sps_rollouts_per_candidate: int,
    sps_rollout_horizon: int | None,
    scorer: Callable[[str, str], dict],
    total_samples: int,
    trajectory_cache: TrajectoryCache | None,
    low_temp_temperature: float = 1.0,
) -> tuple[Candidate, list[Candidate]]:
    if archive_subset.max_retrieved_memory_entries != 0:
        raise NotImplementedError(
            "v1 inference does not implement memory retrieval; archive must have max_retrieved_memory_entries=0"
        )

    allocation = archive_subset.allocate(total_samples)
    jobs: list[_CandidateJob] = []
    stable_pos = 0
    for entry in archive_subset.entries:
        for sample_index in range(allocation.get(entry.id, 0)):
            jobs.append(
                _CandidateJob(
                    prompt_id=entry.id,
                    sample_index=sample_index,
                    alpha=schedule.alpha(entry.id, sample_index),
                    prompt_text=entry.compose(problem.prompt),
                    stable_seed_offset=stable_pos,
                )
            )
            stable_pos += 1

    candidates_by_pos: list[Candidate | None] = [None for _ in jobs]
    low_temp_jobs: list[tuple[int, _CandidateJob]] = []
    power_jobs_by_temperature: dict[float, list[tuple[int, _CandidateJob]]] = {}

    for pos, job in enumerate(jobs):
        cached = None
        if trajectory_cache is not None:
            cached = trajectory_cache.get(
                model_id=model_id,
                track=track,
                problem_id=problem.problem_id,
                prompt_id=job.prompt_id,
                sample_idx=job.sample_index,
                alpha=job.alpha,
                seed=seed,
            )
        if cached is not None:
            candidates_by_pos[pos] = _candidate_from_cached(
                job=job,
                cached=cached,
                reference=problem.answer,
                scorer=scorer,
                trajectory_cache=trajectory_cache,
            )
            continue
        if job.alpha <= 1.0:
            low_temp_jobs.append((pos, job))
        else:
            power_jobs_by_temperature.setdefault(1.0 / job.alpha, []).append((pos, job))

    if low_temp_jobs:
        gens = sampler.generate_low_temp_batch(
            [job.prompt_text for _, job in low_temp_jobs],
            temperature=low_temp_temperature,
            max_new_tokens=max_new_tokens,
            seed_base=seed,
            seed_offsets=[job.stable_seed_offset for _, job in low_temp_jobs],
        )
        for (pos, job), gen in zip(low_temp_jobs, gens):
            candidates_by_pos[pos] = _candidate_from_generation(
                job=job,
                gen=gen,
                reference=problem.answer,
                scorer=scorer,
                trajectory_cache=trajectory_cache,
                model_id=model_id,
                track=track,
                problem_id=problem.problem_id,
                seed=seed,
            )

    for temperature, power_jobs in sorted(power_jobs_by_temperature.items()):
        if power_sampler == "sps":
            sps_batch = getattr(sampler, "generate_sps_power_batch", None)
            if not callable(sps_batch):
                raise NotImplementedError(
                    "power_sampler='sps' requires sampler.generate_sps_power_batch"
                )
            gens = sps_batch(
                [job.prompt_text for _, job in power_jobs],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                block_num=mcmc_block_num,
                top_k=sps_top_k,
                candidate_pool_size=sps_candidate_pool_size,
                rollouts_per_candidate=sps_rollouts_per_candidate,
                rollout_horizon=sps_rollout_horizon,
                seed_base=seed,
                seed_offsets=[job.stable_seed_offset for _, job in power_jobs],
            )
        elif power_sampler == "mcmc":
            gens = sampler.generate_power_batch(
                [job.prompt_text for _, job in power_jobs],
                temperature=temperature,
                mcmc_steps=mcmc_steps,
                max_new_tokens=max_new_tokens,
                block_num=mcmc_block_num,
                seed_base=seed,
                seed_offsets=[job.stable_seed_offset for _, job in power_jobs],
            )
        else:
            raise ValueError(f"unknown power_sampler: {power_sampler!r}")
        for (pos, job), gen in zip(power_jobs, gens):
            candidates_by_pos[pos] = _candidate_from_generation(
                job=job,
                gen=gen,
                reference=problem.answer,
                scorer=scorer,
                trajectory_cache=trajectory_cache,
                model_id=model_id,
                track=track,
                problem_id=problem.problem_id,
                seed=seed,
            )

    candidates = [candidate for candidate in candidates_by_pos if candidate is not None]
    if len(candidates) != len(candidates_by_pos):
        raise RuntimeError("batched inference failed to materialize every candidate")
    return _best_candidate(candidates), candidates


def run_condition(
    *,
    out_dir: Path,
    condition: str,
    archive: FrozenArchive,
    cell_fitness: dict[str, float],
    sampler: _SamplerLike,
    problems: Sequence[Problem],
    seed: int,
    archive_hash: str,
    polaris_source_hash: str,
    vendored_commits: dict[str, str],
    preregistration_anchor: str,
    benchmark: str = "MATH500",
    split: tuple[int, int] = MATH500_TEST_SLICE,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    model_revision_commit: str | None = None,
    model_artifact_etags: dict[str, str] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    scorer: Callable[[str, str], dict] = score_math,
    track: str = "math500",
    verifier_id: str = VERIFIER_ID,
    selector_id: str = "argmax_verifier_score_iteration_tiebreak-v1",
    best_selector: BestSelector | None = None,
    model_key: str | None = None,
    run_stage: str = "smoke",
    run_plan_cell: dict[str, Any] | None = None,
    preflight_report: dict[str, Any] | None = None,
    dataset_lock_id: str | None = None,
    dataset_source_kind: str | None = None,
    started_at: str | None = None,
    host: str | None = None,
    trajectory_cache: TrajectoryCache | None = None,
    memory_store: MemoryStore | None = None,
    memory_store_path: Path | None = None,
    memory_mode: str = "off",
    admit_memory: bool = False,
    online_memory: bool = False,
    enable_repair: bool = False,
    enable_fork_search: bool = False,
    archive_build_id: str | None = None,
    memory_build_id: str | None = None,
    budget_override: int | None = None,
    low_temp_temperature: float = 1.0,
    serving_backend_metadata: dict[str, Any] | None = None,
    mcmc_steps: int = MCMC_STEPS,
    mcmc_block_num: int = MCMC_BLOCK_NUM,
    power_sampler: str = "mcmc",
    fixed_alpha: float | None = None,
    sps_top_k: int = 8,
    sps_candidate_pool_size: int = 8,
    sps_rollouts_per_candidate: int = 8,
    sps_rollout_horizon: int | None = None,
) -> dict[str, Any]:
    """Run one condition over `problems`; emit full production artifact bundle."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    serving_backend_metadata = dict(serving_backend_metadata or {})
    if budget_override is not None and budget_override <= 0:
        raise ValueError("budget_override must be positive")
    if mcmc_steps <= 0:
        raise ValueError("mcmc_steps must be positive")
    if mcmc_block_num <= 0:
        raise ValueError("mcmc_block_num must be positive")
    if power_sampler not in {"mcmc", "sps"}:
        raise ValueError("power_sampler must be one of {'mcmc', 'sps'}")
    if fixed_alpha is not None and fixed_alpha <= 0.0:
        raise ValueError("fixed_alpha must be positive when set")
    if sps_top_k <= 0:
        raise ValueError("sps_top_k must be positive")
    if sps_candidate_pool_size <= 0:
        raise ValueError("sps_candidate_pool_size must be positive")
    if sps_rollouts_per_candidate < 0:
        raise ValueError("sps_rollouts_per_candidate must be non-negative")
    if sps_rollout_horizon is not None and sps_rollout_horizon <= 0:
        raise ValueError("sps_rollout_horizon must be positive when set")
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_subset = _select_archive_subset(archive, cell_fitness, condition)
    memory_is_enabled = _memory_enabled(
        condition=condition,
        memory_store=memory_store,
        memory_store_path=memory_store_path,
        memory_mode=memory_mode,
    )
    if condition in (
        "polaris_full_verified_memory",
        "proicl_gepa_mcmc_memory",
        "proicl_gepa_mcmc_fork_repair_memory",
    ) and memory_mode == "off":
        memory_mode = "distilled_strategies"
    if memory_is_enabled:
        archive_subset = _archive_with_memory(archive_subset)
    if track == "gpqa_diamond":
        admit_memory = False
    schedule = (
        AlphaSchedule(policy_id=f"fixed_alpha_{fixed_alpha:g}", alphas=(fixed_alpha,))
        if fixed_alpha is not None
        else _select_schedule(condition)
    )
    B = int(budget_override or _budget_for(condition))

    candidates_path = out_dir / "candidates.jsonl"
    scores_path = out_dir / "scores.jsonl"
    selected_path = out_dir / "selected.jsonl"
    expected_problem_ids = [str(problem.problem_id) for problem in problems]
    selected_rows = _normalize_selected_rows(
        selected_path,
        expected_problem_ids=set(expected_problem_ids),
    )
    completed_problem_ids = {
        str(row["problem_id"])
        for row in selected_rows
        if row.get("problem_id") is not None
    }
    candidate_rows = _filter_jsonl_to_completed_problems(
        candidates_path,
        completed_problem_ids=completed_problem_ids,
    )
    _filter_jsonl_to_completed_problems(
        scores_path,
        completed_problem_ids=completed_problem_ids,
    )

    ledger_path = memory_store_path or (out_dir / "memory.sqlite")
    memory_ledger: PersistentMemoryLedger | None = None
    if memory_is_enabled:
        memory_ledger = PersistentMemoryLedger(ledger_path)
        memory_ledger.rollback_incomplete_queries(
            completed_problem_ids,
            expected_query_ids=expected_problem_ids,
        )
        if memory_store is None:
            memory_store = _memory_store_from_ledger(memory_ledger)

    n_correct = sum(1 for row in selected_rows if _selected_passed(row))
    total_selected_score = sum(_selected_score(row) for row in selected_rows)
    total_wall = sum(
        float(row.get("wall_clock_seconds", 0.0) or 0.0)
        for row in candidate_rows
    )
    total_dollars = sum(
        float(row.get("estimated_dollar_cost", row.get("dollar_estimate", 0.0)) or 0.0)
        for row in candidate_rows
    )
    total_input_tokens = sum(
        int(row.get("prompt_token_count", 0) or 0)
        for row in candidate_rows
    )
    total_output_tokens = sum(
        int(row.get("generation_token_count", 0) or 0) for row in candidate_rows
    )
    n_candidates = len(candidate_rows)
    ledger = RolloutLedger()
    if n_candidates:
        ledger.charge_inference(condition, n_candidates)
    repair_enabled = enable_repair or condition in {
        "proicl_gepa_mcmc_repair",
        "proicl_gepa_mcmc_fork_repair",
        "proicl_gepa_mcmc_fork_repair_memory",
    }
    fork_enabled = enable_fork_search or condition in {
        "fork_search",
        "proicl_gepa_mcmc_fork_repair",
        "proicl_gepa_mcmc_fork_repair_memory",
    }
    repair_traces_path = out_dir / "repair_traces.jsonl"
    fork_traces_path = out_dir / "fork_traces.jsonl"
    checkpoint_path = out_dir / "checkpoint.json"
    _write_checkpoint(
        checkpoint_path,
        track=track,
        condition=condition,
        run_stage=run_stage,
        expected_problem_ids=expected_problem_ids,
        completed_problem_ids=completed_problem_ids,
        n_candidates=n_candidates,
    )
    if repair_enabled:
        repair_rows = _filter_jsonl_to_completed_problems(
            repair_traces_path,
            completed_problem_ids=completed_problem_ids,
            keep_problemless_rows=True,
        )
        if not any(row.get("event") == "repair_enabled" for row in repair_rows):
            append_jsonl(
                repair_traces_path,
                {"event": "repair_enabled", "condition": condition},
            )
    if fork_enabled:
        fork_rows = _filter_jsonl_to_completed_problems(
            fork_traces_path,
            completed_problem_ids=completed_problem_ids,
            keep_problemless_rows=True,
        )
        if not any(row.get("event") == "fork_search_enabled" for row in fork_rows):
            append_jsonl(
                fork_traces_path,
                {"event": "fork_search_enabled", "condition": condition},
            )

    for problem in problems:
        problem_id = str(problem.problem_id)
        if problem_id in completed_problem_ids:
            continue
        if condition == "greedy":
            best = _greedy_candidate(
                sampler, archive_subset, problem, max_new_tokens, scorer
            )
            candidates: list[Candidate] = [best]
        else:
            if (
                _supports_batch_generation(sampler, power_sampler=power_sampler)
                and not memory_is_enabled
            ):
                best, candidates = _run_problem_batched(
                    archive_subset=archive_subset,
                    schedule=schedule,
                    sampler=sampler,
                    problem=problem,
                    seed=seed,
                    model_id=model_id,
                    track=track,
                    max_new_tokens=max_new_tokens,
                    mcmc_steps=mcmc_steps,
                    mcmc_block_num=mcmc_block_num,
                    power_sampler=power_sampler,
                    sps_top_k=sps_top_k,
                    sps_candidate_pool_size=sps_candidate_pool_size,
                    sps_rollouts_per_candidate=sps_rollouts_per_candidate,
                    sps_rollout_horizon=sps_rollout_horizon,
                    scorer=scorer,
                    total_samples=B,
                    trajectory_cache=trajectory_cache,
                    low_temp_temperature=low_temp_temperature,
                )
            else:
                known_memory_ids = (
                    {entry.id for entry in memory_store.entries}
                    if memory_store is not None
                    else set()
                )
                best, candidates = polaris_inference(
                    question=problem.prompt,
                    reference=problem.answer,
                    archive=archive_subset,
                    sampler=sampler,
                    alpha_schedule=schedule,
                    total_samples=B,
                    max_new_tokens=max_new_tokens,
                    scorer=scorer,
                    trajectory_cache=trajectory_cache,
                    cache_model_id=model_id,
                    cache_track=track,
                    cache_problem_id=problem.problem_id,
                    cache_seed=seed,
                    memory_store=memory_store,
                    admit_memory=admit_memory
                    or condition
                    in ("polaris_full_verified_memory", "proicl_gepa_mcmc_memory"),
                    memory_independent_check=(
                        lambda trace, reference=problem.answer: bool(
                            scorer(trace, reference).get("passed", False)
                        )
                    ),
                    mcmc_steps=mcmc_steps,
                    mcmc_block_num=mcmc_block_num,
                    power_sampler=power_sampler,
                )
                if memory_ledger is not None and memory_store is not None:
                    for c in candidates:
                        if c.retrieved_memory_ids:
                            memory_ledger.record_retrieval(
                                query_id=problem.problem_id,
                                eligible_ids=_eligible_memory_ids(
                                    memory_store, c.prompt_id
                                ),
                                retrieved_ids=c.retrieved_memory_ids,
                                verifier_metadata={
                                    "verifier_id": verifier_id,
                                    "condition": condition,
                                },
                            )
                            memory_ledger.update_posterior(
                                c.retrieved_memory_ids,
                                verifier_outcome=(
                                    1
                                    if c.verifier_result.get("passed", False)
                                    else 0
                                ),
                                query_id=problem.problem_id,
                            )
                        if c.admitted_memory_id:
                            for entry in memory_store.entries:
                                if (
                                    entry.id == c.admitted_memory_id
                                    and entry.id not in known_memory_ids
                                ):
                                    memory_ledger.admit(
                                        entry,
                                        track=track,
                                        verifier_id=verifier_id,
                                        metadata={
                                            "condition": condition,
                                            "memory_mode": memory_mode,
                                            "model_key": model_key,
                                            "model_id": model_id,
                                            "model_revision": model_revision,
                                        },
                                    )
                                    known_memory_ids.add(entry.id)
                                    break
                        elif not c.verifier_result.get("passed", False):
                            memory_ledger.reject(
                                candidate_trace_id=(
                                    f"{problem.problem_id}:{c.prompt_id}:"
                                    f"{c.sample_index}"
                                ),
                                query_id=problem.problem_id,
                                reason="candidate_failed_verifier",
                                metadata={"condition": condition},
                            )

        if fork_enabled:
            fork_candidates: list[Candidate] = []
            for entry in archive_subset.entries:
                fork_candidates.extend(
                    run_entropy_gated_fork_search(
                        prompt_id=entry.id,
                        sample_index_start=len(candidates) + len(fork_candidates),
                        prompt_text=entry.compose(problem.prompt),
                        sampler=sampler,
                        scorer=scorer,
                        reference=problem.answer,
                        max_new_tokens=max_new_tokens,
                        config=ForkSearchConfig(),
                    )
                )
            for c in fork_candidates:
                append_jsonl(
                    fork_traces_path,
                    {
                        "problem_id": problem.problem_id,
                        "prompt_id": c.prompt_id,
                        "sample_index": c.sample_index,
                        "fork_depth": c.fork_depth,
                        "fork_entropy": c.fork_entropy,
                        "fork_token": c.fork_token,
                        "fork_probability": c.fork_probability,
                        "passed": c.verifier_result.get("passed", False),
                        "score": c.verifier_result.get("score", 0.0),
                    },
                )
            candidates.extend(fork_candidates)

        if repair_enabled:
            repair_candidates: list[Candidate] = []
            for c in list(candidates):
                repair_candidates.extend(
                    run_verifier_guided_repair(
                        parent=c,
                        sampler=sampler,
                        scorer=scorer,
                        reference=problem.answer,
                        max_new_tokens=max_new_tokens,
                        config=RepairConfig(),
                    )
                )
            for c in repair_candidates:
                append_jsonl(
                    repair_traces_path,
                    {
                        "problem_id": problem.problem_id,
                        "parent_candidate_id": c.parent_candidate_id,
                        "repair_attempt": c.repair_attempt,
                        "passed": c.verifier_result.get("passed", False),
                        "score": c.verifier_result.get("score", 0.0),
                        "failure_type": c.verifier_result.get("failure_type"),
                    },
                )
            candidates.extend(repair_candidates)

        selection_metadata: dict[str, Any]
        if best_selector is not None:
            best, selection_metadata = best_selector(candidates)
        else:
            best, selection_metadata = _default_best_selector(candidates)
        selection_metadata.setdefault("selector_id", selector_id)

        for c in candidates:
            row = _candidate_row(
                candidate=c,
                problem=problem,
                model_id=model_id,
                track=track,
                seed=seed,
            )
            append_jsonl(candidates_path, row)
            append_jsonl(
                scores_path,
                {
                    "problem_id": problem.problem_id,
                    "prompt_id": c.prompt_id,
                    "sample_index": c.sample_index,
                    "score": c.verifier_result.get("score", 0.0),
                    "passed": c.verifier_result.get("passed", False),
                    "verifier_id": c.verifier_result.get("verifier_id", verifier_id),
                    "offline_only": c.verifier_result.get("offline_only", False),
                },
            )
            total_wall += c.wall_clock_seconds
            total_dollars += c.estimated_dollar_cost
            total_input_tokens += c.prompt_token_count
            total_output_tokens += c.generation_token_count
            n_candidates += 1
        ledger.charge_inference(condition, len(candidates))

        selected_score = float(best.verifier_result.get("score", 0.0))
        total_selected_score += selected_score
        if best.verifier_result.get("passed"):
            n_correct += 1
        selected_row = _candidate_row(
            candidate=best,
            problem=problem,
            model_id=model_id,
            track=track,
            seed=seed,
        )
        selected_row["selection"] = selection_metadata
        selected_row["selected_score"] = selected_score
        selected_row["selected_passed"] = best.verifier_result.get("passed", False)
        append_jsonl(selected_path, selected_row)
        completed_problem_ids.add(problem_id)
        _write_checkpoint(
            checkpoint_path,
            track=track,
            condition=condition,
            run_stage=run_stage,
            expected_problem_ids=expected_problem_ids,
            completed_problem_ids=completed_problem_ids,
            n_candidates=n_candidates,
        )

    accuracy = n_correct / len(problems) if problems else 0.0
    mean_selected_score = total_selected_score / len(problems) if problems else 0.0
    metrics = {
        "track": track,
        "model_key": model_key,
        "run_stage": run_stage,
        "condition": condition,
        "accuracy": accuracy,
        "mean_selected_score": mean_selected_score,
        "n_problems": len(problems),
        "n_candidates": n_candidates,
        "B_per_problem": B,
        "sampling_temperature": low_temp_temperature,
        "mcmc_steps": mcmc_steps,
        "mcmc_block_num": mcmc_block_num,
        "power_sampler": power_sampler,
        "sps_top_k": sps_top_k,
        "sps_candidate_pool_size": sps_candidate_pool_size,
        "sps_rollouts_per_candidate": sps_rollouts_per_candidate,
        "sps_rollout_horizon": sps_rollout_horizon,
        "alpha_policy_id": schedule.policy_id,
        "repair_enabled": repair_enabled,
        "fork_search_enabled": fork_enabled,
        "selector_id": selector_id,
        "verifier_id": verifier_id,
        "backend": serving_backend_metadata.get("backend"),
        "vllm_scoring_mode": serving_backend_metadata.get("vllm_scoring_mode"),
    }
    costs = {
        "wall_clock_seconds": total_wall,
        "estimated_dollar_cost": total_dollars,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "rollout_total": ledger.total,
    }

    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "costs.json", costs)
    write_json(out_dir / "archive.json", archive_subset.to_jsonable())
    ledger.write(out_dir / "rollouts.json")
    environment_payload = _environment_snapshot()
    environment_payload["serving_backend"] = serving_backend_metadata
    write_json(out_dir / "environment.json", environment_payload)
    preflight_payload = dict(preflight_report or {})
    preflight_payload.setdefault("required", run_stage != "smoke")
    preflight_payload.setdefault("passed", run_stage == "smoke")
    preflight_payload.setdefault("protocol_sync_passed", False)
    preflight_payload.setdefault("model_artifact_etags", dict(model_artifact_etags or {}))
    preflight_payload.setdefault("backend", serving_backend_metadata.get("backend"))
    preflight_payload.setdefault(
        "vllm_scoring_mode", serving_backend_metadata.get("vllm_scoring_mode")
    )
    preflight_payload.setdefault(
        "vllm_parity_artifact", serving_backend_metadata.get("parity_artifact_path")
    )
    write_json(out_dir / "preflight.json", preflight_payload)
    cell_payload = dict(
        run_plan_cell
        or {
            "track": track,
            "model_key": model_key,
            "model_revision": model_revision,
            "model_revision_commit": model_revision_commit,
            "split_id": "custom",
            "split": list(split),
            "condition": condition,
            "archive": archive_hash,
            "archive_build_id": archive_build_id,
            "memory_build_id": memory_build_id,
            "memory_mode": memory_mode if memory_is_enabled else "none",
            "alpha_policy": schedule.policy_id,
            "repair_enabled": repair_enabled,
            "fork_search_enabled": fork_enabled,
            "seed": seed,
            "budget": B,
            "sampling_temperature": low_temp_temperature,
            "mcmc_steps": mcmc_steps,
            "mcmc_block_num": mcmc_block_num,
            "power_sampler": power_sampler,
            "sps_top_k": sps_top_k,
            "sps_candidate_pool_size": sps_candidate_pool_size,
            "sps_rollouts_per_candidate": sps_rollouts_per_candidate,
            "sps_rollout_horizon": sps_rollout_horizon,
            "cache_path": str(trajectory_cache.path)
            if trajectory_cache is not None
            else None,
            "artifact_dir": str(out_dir),
            "cost_cap_dollars": None,
            "run_stage": run_stage,
        }
    )
    if archive_build_id is not None:
        cell_payload.setdefault("archive_build_id", archive_build_id)
    if memory_build_id is not None:
        cell_payload.setdefault("memory_build_id", memory_build_id)
    cell_payload.setdefault("serving_backend", serving_backend_metadata)
    write_json(out_dir / "run_plan_cell.json", cell_payload)
    if archive_build_id is not None or condition in (
        "polaris_full_verified_memory",
        "proicl_gepa_mcmc_memory",
        "proicl_gepa_mcmc_fork_repair_memory",
    ):
        write_json(
            out_dir / "archive_build_manifest.json",
            {
                "archive_build_id": archive_build_id or "inline-archive",
                "memory_build_id": memory_build_id,
                "condition": condition,
                "track": track,
                "memory_mode": memory_mode if memory_is_enabled else "none",
                "frozen": True,
            },
        )
    (out_dir / "audit.md").write_text(
        f"# audit\n\n"
        f"- track: {track}\n"
        f"- condition: {condition}\n"
        f"- run_stage: {run_stage}\n"
        f"- seed: {seed}\n"
        f"- problems: {len(problems)}\n"
        f"- B/problem: {B}\n"
        f"- mcmc_steps: {mcmc_steps}\n"
        f"- mcmc_block_num: {mcmc_block_num}\n"
        f"- sps_top_k: {sps_top_k}\n"
        f"- sps_candidate_pool_size: {sps_candidate_pool_size}\n"
        f"- sps_rollouts_per_candidate: {sps_rollouts_per_candidate}\n"
        f"- sps_rollout_horizon: {sps_rollout_horizon}\n"
        f"- alpha_policy: {schedule.policy_id}\n"
        f"- repair_enabled: {repair_enabled}\n"
        f"- fork_search_enabled: {fork_enabled}\n"
        f"- selector: {selector_id}\n"
        f"- backend: {serving_backend_metadata.get('backend')}\n"
        f"- vllm_scoring_mode: {serving_backend_metadata.get('vllm_scoring_mode')}\n"
        f"- accuracy: {accuracy:.4f}\n"
        f"- mean_selected_score: {mean_selected_score:.4f}\n",
        encoding="utf-8",
    )

    config = {
        "track": track,
        "model_key": model_key,
        "model_revision": model_revision,
        "model_revision_commit": model_revision_commit,
        "model_artifact_etags": dict(model_artifact_etags or {}),
        "run_stage": run_stage,
        "max_new_tokens": max_new_tokens,
        "B_per_problem": B,
        "sampling_temperature": low_temp_temperature,
        "mcmc_steps": mcmc_steps,
        "mcmc_block_num": mcmc_block_num,
        "power_sampler": power_sampler,
        "sps_top_k": sps_top_k,
        "sps_candidate_pool_size": sps_candidate_pool_size,
        "sps_rollouts_per_candidate": sps_rollouts_per_candidate,
        "sps_rollout_horizon": sps_rollout_horizon,
        "schedule_alphas": list(schedule.alphas),
        "repair_enabled": repair_enabled,
        "fork_search_enabled": fork_enabled,
        "selector_id": selector_id,
        "dataset_lock_id": dataset_lock_id,
        "archive_build_id": archive_build_id,
        "memory_build_id": memory_build_id,
        "memory_mode": memory_mode if memory_is_enabled else "none",
        "memory_store_path": str(memory_store_path) if memory_store_path else None,
        "dataset_source_kind": dataset_source_kind
        or (
            "fixture"
            if any(getattr(problem, "source", "") == "smoke" for problem in problems)
            else "real_or_cached"
        ),
        "production_artifacts_schema": 1,
        "trajectory_cache": str(trajectory_cache.path)
        if trajectory_cache is not None
        else None,
        "serving_backend": serving_backend_metadata,
        "vllm_parity_artifact": serving_backend_metadata.get("parity_artifact_path"),
    }
    write_run_manifest(
        path=out_dir / "manifest.json",
        model_id=model_id,
        model_revision=model_revision,
        model_revision_commit=model_revision_commit,
        benchmark=benchmark,
        split=split,
        seeds=[seed],
        condition=condition,
        archive_hash=archive_hash,
        alpha_policy_id=schedule.policy_id,
        config=config,
        polaris_source_hash=polaris_source_hash,
        vendored_commits=vendored_commits,
        verifier_id=verifier_id,
        preregistration_anchor=preregistration_anchor,
        started_at=started_at,
        host=host,
    )
    if memory_ledger is not None:
        _write_memory_events(out_dir / "memory_events.jsonl", memory_ledger)
        memory_ledger.close()
        _copy_memory_sqlite(ledger_path, out_dir / "memory.sqlite")
    return metrics
