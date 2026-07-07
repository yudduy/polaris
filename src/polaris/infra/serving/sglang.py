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
    power_sampling: dict[str, Any] | None = None


class SGLangGenerator:
    """SGLang native `/generate` client for R5 cheap validation.

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
        transport: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.request_timeout = request_timeout
        self._transport = transport or self._post_json

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backend": "sglang",
            "model_id": self.model_id,
            "base_url": self.base_url,
            "generation_endpoint": "/generate",
            "scoring_endpoint": "/generate",
        }

    def generate_greedy(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self.generate_low_temp_batch(
            [prompt_text],
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_low_temp(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self.generate_low_temp_batch(
            [prompt_text],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_low_temp_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
        top_p: float | None = None,
    ) -> list[Generation]:
        if not prompt_texts:
            return []
        offsets = self._seed_offsets(prompt_texts, seed_offsets)
        base_seed = self.seed if seed_base is None else int(seed_base)
        sampling_params = []
        for offset in offsets:
            params: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "sampling_seed": base_seed + int(offset),
            }
            if top_p is not None:
                params["top_p"] = top_p
            sampling_params.append(params)
        return self._generate_native_batch(prompt_texts, sampling_params)

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

    def generate_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        mcmc_steps: int = MCMC_STEPS,
        block_num: int = MCMC_BLOCK_NUM,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Generation]:
        raise NotImplementedError(
            "SGLang MCMC generation is blocked until score_segments parity passes "
            "against the HF oracle. Use generate_sps_power_batch for native SPS."
        )

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
        alpha = 1.0 / float(temperature)
        if alpha <= 1.0:
            return self.generate_low_temp_batch(
                prompt_texts,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                seed_base=seed_base,
                seed_offsets=seed_offsets,
            )
        offsets = self._seed_offsets(prompt_texts, seed_offsets)
        base_seed = self.seed if seed_base is None else int(seed_base)
        sampling_params = []
        for offset in offsets:
            sampling_params.append(
                {
                    "max_new_tokens": max_new_tokens,
                    "power_sampling": {
                        "alpha": alpha,
                        "block_size": block_num,
                        "candidate_pool_size": candidate_pool_size,
                        "candidate_top_k": top_k,
                        "rollouts_per_candidate": rollouts_per_candidate,
                        "rollout_horizon": rollout_horizon,
                        "jackknife": True,
                        "seed": base_seed + int(offset),
                    },
                }
            )
        return self._generate_native_batch(prompt_texts, sampling_params)

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

    def _generate_native_batch(
        self,
        prompt_texts: list[str],
        sampling_params: list[dict[str, Any]],
    ) -> list[Generation]:
        if len(prompt_texts) != len(sampling_params):
            raise ValueError("prompt_texts and sampling_params length mismatch")
        started = time.monotonic()
        response = self._transport(
            "/generate",
            {
                "text": prompt_texts if len(prompt_texts) > 1 else prompt_texts[0],
                "sampling_params": sampling_params
                if len(sampling_params) > 1
                else sampling_params[0],
                "stream": False,
            },
        )
        elapsed = time.monotonic() - started
        rows = response if isinstance(response, list) else [response]
        if len(rows) != len(prompt_texts):
            raise RuntimeError(
                f"SGLang returned {len(rows)} generations for {len(prompt_texts)} prompts"
            )
        per_candidate_wall = elapsed / max(1, len(rows))
        generations: list[Generation] = []
        for prompt_text, row in zip(prompt_texts, rows):
            meta = dict(row.get("meta_info", {}) or {})
            prompt_tokens = int(meta.get("prompt_tokens", 0))
            completion_tokens = int(
                meta.get("completion_tokens", len(row.get("output_ids") or []))
            )
            cost = estimate_cost(prompt_tokens, completion_tokens)
            generations.append(
                Generation(
                    generation=str(row.get("text") or ""),
                    prompt_text=prompt_text,
                    response_contains_prompt=False,
                    prompt_token_count=prompt_tokens,
                    generation_token_count=completion_tokens,
                    wall_clock_seconds=float(meta.get("e2e_latency", per_candidate_wall)),
                    estimated_dollar_cost=cost.dollars,
                    acceptance_ratio=None,
                    token_ids=[int(x) for x in row.get("output_ids") or []],
                    power_sampling=meta.get("power_sampling"),
                )
            )
        return generations

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _seed_offsets(
        self,
        prompt_texts: list[str],
        seed_offsets: list[int] | None,
    ) -> list[int]:
        if seed_offsets is None:
            return list(range(len(prompt_texts)))
        offsets = [int(x) for x in seed_offsets]
        if len(offsets) != len(prompt_texts):
            raise ValueError("seed_offsets length must match prompt_texts length")
        return offsets
