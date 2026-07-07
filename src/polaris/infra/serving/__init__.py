"""Serving backends behind a unified Sampler protocol.

`hf.py` is the correctness oracle (slow, exact, used in tests + parity checks).
`vllm.py` is the R5 optimized MCMC candidate after passing the HF
`score_segments` parity gate in bounded Modal smoke tests.
`sglang.py` remains a shared-prefix generation candidate, but its MCMC scoring
path is blocked until it matches the HF oracle.

`core/inference.py` and `infra/mcmc.py` consume the `Sampler` protocol — they
do not import a concrete backend directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ScoreBatch:
    """Per-chain log-prob scores for existing target segments.

    `*_tokens` preserve per-token scores so trajectory cache rows can replay
    downstream ablations without recomputing base-model likelihoods.
    """

    lp_norm: list[float]
    lp_unnorm: list[float]
    lp_norm_tokens: list[list[float]]
    lp_unnorm_tokens: list[list[float]]


@runtime_checkable
class Sampler(Protocol):
    """Backend contract used by inference + MCMC.

    `Generation` is the dataclass defined by each backend; duck-typed here to
    avoid a circular import. Required fields:
        generation: str
        prompt_text: str
        response_contains_prompt: bool
        prompt_token_count: int
        generation_token_count: int
        wall_clock_seconds: float
        estimated_dollar_cost: float
        acceptance_ratio: float | None
    """

    def generate_greedy(self, prompt_text: str, *, max_new_tokens: int) -> Any: ...
    def generate_low_temp(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int
    ) -> Any: ...
    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int = ...,
        block_num: int = ...,
    ) -> Any: ...
    def score_segments(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch: ...


@runtime_checkable
class BatchedSampler(Sampler, Protocol):
    """Optional high-throughput sampler surface used by the MATH500 runner."""

    def generate_low_temp_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Any]: ...

    def generate_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int = ...,
        block_num: int = ...,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Any]: ...

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
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Any]: ...
