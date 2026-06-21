from __future__ import annotations

import json
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from polaris.config import (
    MAX_NEW_TOKENS,
    MCMC_BLOCK_NUM,
    MCMC_STEPS,
    MODEL_ID,
    SEED,
    estimate_cost,
)
from polaris.infra.serving import ScoreBatch
from polaris.infra.serving.sglang_logits import (
    build_forced_segment_score_request,
    build_next_token_score_request,
    extract_output_token_id_logprob,
)


@dataclass
class Generation:
    generation: str
    prompt_text: str
    response_contains_prompt: bool
    prompt_token_count: int
    generation_token_count: int
    wall_clock_seconds: float
    estimated_dollar_cost: float
    acceptance_ratio: float | None = None
    token_ids: list[int] | None = None
    meta_info: dict[str, Any] | None = None


class SGLangGenerator:
    """SGLang OpenAI-compatible client for R5 cheap validation.

    MCMC production use is gated by `score_segments` parity against HF. This
    client intentionally talks to an already-running SGLang server so Modal and
    Mithril launch policy stay outside the inference core.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        base_url: str = "http://localhost:30000",
        seed: int = SEED,
        request_timeout: float = 600.0,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.request_timeout = request_timeout
        self._transport = transport or self._post_json

    def generate_greedy(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self._completion(
            prompt_text=prompt_text,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    def generate_low_temp(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self._completion(
            prompt_text=prompt_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        mcmc_steps: int = MCMC_STEPS,
        block_num: int = MCMC_BLOCK_NUM,
    ) -> Generation:
        raise NotImplementedError(
            "SGLang MCMC generation is blocked until score_segments parity passes "
            "against the HF oracle. Use generate_low_temp/score_segments smokes first."
        )

    def generate_sps_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
        top_k: int = 8,
        candidate_pool_size: int = 8,
        rollouts_per_candidate: int = 8,
        rollout_horizon: int | None = None,
        seed_base: int | None = None,
        seed_offset: int = 0,
    ) -> Generation:
        return self.generate_sps_power_batch(
            [prompt_text],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            rollouts_per_candidate=rollouts_per_candidate,
            rollout_horizon=rollout_horizon,
            seed_base=seed_base,
            seed_offsets=[seed_offset],
        )[0]

    def generate_sps_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
        top_k: int = 8,
        candidate_pool_size: int = 8,
        rollouts_per_candidate: int = 8,
        rollout_horizon: int | None = None,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Generation]:
        if not prompt_texts:
            return []
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if block_num <= 0:
            raise ValueError(f"block_num must be > 0, got {block_num}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if candidate_pool_size <= 0:
            raise ValueError(
                f"candidate_pool_size must be > 0, got {candidate_pool_size}"
            )
        if rollouts_per_candidate < 0:
            raise ValueError(
                f"rollouts_per_candidate must be >= 0, got {rollouts_per_candidate}"
            )
        if seed_offsets is None:
            offsets = list(range(len(prompt_texts)))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != len(prompt_texts):
                raise ValueError("seed_offsets length must match prompt_texts length")

        alpha = 1.0 / float(temperature)
        if alpha <= 1.0:
            return [
                self.generate_low_temp(
                    prompt_text,
                    temperature=1.0,
                    max_new_tokens=max_new_tokens,
                )
                for prompt_text in prompt_texts
            ]

        base_seed = self.seed if seed_base is None else int(seed_base)
        block_size = max(1, (max_new_tokens + block_num - 1) // block_num)
        sampling_params = [
            {
                "max_new_tokens": max_new_tokens,
                "power_sampling": {
                    "alpha": alpha,
                    "block_size": block_size,
                    "candidate_pool_size": max(candidate_pool_size, top_k),
                    "candidate_top_k": top_k,
                    "rollouts_per_candidate": rollouts_per_candidate,
                    "rollout_horizon": rollout_horizon,
                    "jackknife": True,
                    "seed": base_seed + offsets[i] * 1_000_003,
                },
            }
            for i in range(len(prompt_texts))
        ]
        payload = {
            "text": prompt_texts,
            "sampling_params": sampling_params,
            "stream": False,
        }
        started = time.monotonic()
        response = self._transport("/generate", payload)
        elapsed = time.monotonic() - started
        rows = response if isinstance(response, list) else [response]
        if len(rows) != len(prompt_texts):
            raise RuntimeError(
                f"SGLang SPS returned {len(rows)} rows for {len(prompt_texts)} prompts"
            )
        per_row_wall = elapsed / max(1, len(rows))
        return [
            _native_generation(row, prompt, per_row_wall)
            for row, prompt in zip(rows, prompt_texts)
        ]

    def score_segments(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch:
        """Score fixed target segments through SGLang's input-logprob path.

        This is the R5 parity surface. It intentionally performs both normalized
        and base-temperature scoring so the MH ratio can be checked against HF
        before any long generation run is allowed.
        """
        if len(prefix_ids_batch) != len(target_segments_batch):
            raise ValueError("prefix_ids_batch and target_segments_batch length mismatch")

        lp_norm_tokens: list[list[float]] = []
        lp_unnorm_tokens: list[list[float]] = []
        lp_norm: list[float] = []
        lp_unnorm: list[float] = []
        for prefix_ids, target_ids in zip(prefix_ids_batch, target_segments_batch):
            if not target_ids:
                lp_norm_tokens.append([])
                lp_unnorm_tokens.append([])
                lp_norm.append(0.0)
                lp_unnorm.append(0.0)
                continue

            forced_scores = self._try_score_segment_forced(
                prefix_ids, target_ids, temperature=temperature
            )
            if forced_scores is not None:
                norm_vals, unnorm_vals = forced_scores
                lp_norm_tokens.append(norm_vals)
                lp_unnorm_tokens.append(unnorm_vals)
                lp_norm.append(float(sum(norm_vals)))
                lp_unnorm.append(float(sum(unnorm_vals)))
                continue

            context_ids = list(prefix_ids)
            norm_vals: list[float] = []
            base_vals: list[float] = []
            for target_id in target_ids:
                norm_req = build_next_token_score_request(
                    context_ids, target_id, temperature=temperature
                )
                base_req = build_next_token_score_request(
                    context_ids, target_id, temperature=1.0
                )
                norm_resp = self._transport("/generate", norm_req.to_payload())
                base_resp = self._transport("/generate", base_req.to_payload())
                norm_vals.append(extract_output_token_id_logprob(norm_resp, target_id))
                base_vals.append(extract_output_token_id_logprob(base_resp, target_id))
                context_ids.append(target_id)
            unnorm_vals = [(1.0 / temperature) * v for v in base_vals]
            lp_norm_tokens.append(norm_vals)
            lp_unnorm_tokens.append(unnorm_vals)
            lp_norm.append(float(sum(norm_vals)))
            lp_unnorm.append(float(sum(unnorm_vals)))

        return ScoreBatch(
            lp_norm=lp_norm,
            lp_unnorm=lp_unnorm,
            lp_norm_tokens=lp_norm_tokens,
            lp_unnorm_tokens=lp_unnorm_tokens,
        )

    def _try_score_segment_forced(
        self,
        prefix_ids: list[int],
        target_ids: list[int],
        *,
        temperature: float,
    ) -> tuple[list[float], list[float]] | None:
        score_id = uuid.uuid4().hex
        score_path = tempfile.gettempdir() + f"/polaris-sglang-score-{score_id}.jsonl"
        try:
            req = build_forced_segment_score_request(
                prefix_ids,
                target_ids,
                temperature=temperature,
                score_path=score_path,
                score_id=score_id,
            )
        except Exception:
            return None

        self._transport("/generate", req.to_payload())
        try:
            with open(score_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError as exc:
            raise RuntimeError("SGLang forced-token scorer did not write side channel") from exc

        records = [r for r in records if r.get("score_id") == score_id]
        records.sort(key=lambda r: int(r["position"]))
        if len(records) != len(target_ids):
            raise RuntimeError(
                "SGLang forced-token scorer returned "
                f"{len(records)} records for {len(target_ids)} targets"
            )
        seen_targets = [int(r["target_token_id"]) for r in records]
        if seen_targets != [int(x) for x in target_ids]:
            raise RuntimeError(
                f"SGLang forced-token scorer target mismatch: {seen_targets} != {target_ids}"
            )
        return (
            [float(r["lp_norm"]) for r in records],
            [float(r["lp_unnorm"]) for r in records],
        )

    def _completion(
        self,
        *,
        prompt_text: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Generation:
        payload = {
            "model": self.model_id,
            "prompt": prompt_text,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
            "seed": self.seed,
        }
        started = time.monotonic()
        response = self._transport("/v1/completions", payload)
        elapsed = time.monotonic() - started
        text = _completion_text(response)
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost = estimate_cost(prompt_tokens, completion_tokens)
        return Generation(
            generation=text,
            prompt_text=prompt_text,
            response_contains_prompt=False,
            prompt_token_count=prompt_tokens,
            generation_token_count=completion_tokens,
            wall_clock_seconds=elapsed,
            estimated_dollar_cost=cost.dollars,
            acceptance_ratio=None,
            token_ids=_completion_token_ids(response),
        )

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        first = choices[0]
        return str(first.get("text") or first.get("message", {}).get("content") or "")
    return str(response.get("text") or "")


def _completion_token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices") or []
    if choices:
        token_ids = choices[0].get("token_ids")
        if token_ids is not None:
            return [int(x) for x in token_ids]
    token_ids = response.get("token_ids")
    return [int(x) for x in token_ids] if token_ids is not None else []


def _native_generation(
    response: dict[str, Any], prompt_text: str, wall_clock_seconds: float
) -> Generation:
    meta_info = dict(response.get("meta_info") or {})
    prompt_tokens = int(meta_info.get("prompt_tokens", 0))
    completion_tokens = int(meta_info.get("completion_tokens", 0))
    token_ids = response.get("output_ids") or response.get("token_ids") or []
    if not completion_tokens:
        completion_tokens = len(token_ids)
    cost = estimate_cost(prompt_tokens, completion_tokens)
    return Generation(
        generation=str(response.get("text") or ""),
        prompt_text=prompt_text,
        response_contains_prompt=False,
        prompt_token_count=prompt_tokens,
        generation_token_count=completion_tokens,
        wall_clock_seconds=wall_clock_seconds,
        estimated_dollar_cost=cost.dollars,
        acceptance_ratio=None,
        token_ids=[int(x) for x in token_ids],
        meta_info=meta_info,
    )
