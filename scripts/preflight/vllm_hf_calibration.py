"""Write HF-vLLM scorer calibration artifacts for POLARIS SPS/MCMC.

This is a bounded calibration harness, not a production science run. It loads
HF as the oracle scorer, loads vLLM as the candidate scorer, verifies token IDs
before scoring, and writes the artifact bundle consumed by run manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        default=None,
        help="Optional POLARIS model registry key; supplies model id/revision.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--segment-lens", type=int, nargs="+", default=[1, 2, 8, 32, 128])
    parser.add_argument(
        "--hf-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
        help="HF oracle dtype for calibration. float32 matches the accepted vLLM parity contract.",
    )
    parser.add_argument(
        "--hf-scoring-mode",
        choices=["forward", "cached_decode"],
        default="cached_decode",
        help="HF oracle scoring path for fixed segments.",
    )
    parser.add_argument(
        "--hf-device-map-auto",
        action="store_true",
        help="Use Transformers device_map='auto' for the HF oracle. By default calibration loads directly on CUDA.",
    )
    parser.add_argument(
        "--vllm-scoring-mode",
        choices=["forced_decode_v0", "native_segment"],
        default="forced_decode_v0",
    )
    parser.add_argument("--vllm-dtype", default="float32")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument(
        "--no-vllm-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching for this calibration candidate.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _extend(ids: list[int], length: int, *, fallback: int) -> list[int]:
    if length <= 0:
        return []
    if not ids:
        ids = [fallback]
    out = list(ids)
    while len(out) < length:
        out.extend(ids)
    return out[:length]


def _segments(tokenizer: Any, segment_lens: list[int]) -> tuple[list[int], list[list[int]], dict[str, Any]]:
    prompt = (
        "Solve the problem and put the final answer in boxed form.\n"
        "Problem: Convert the point $(0,3)$ in rectangular coordinates to polar "
        "coordinates. Enter your answer as $(r,\\theta)$ with $r > 0$.\n"
        "Solution:"
    )
    target = " The radius is 3 and the angle is pi/2, so the final answer is boxed."
    prefix_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    fallback = eos_id if eos_id is not None else (target_ids[0] if target_ids else 0)
    repeated = target_ids[0] if target_ids else fallback
    rows: list[list[int]] = []
    cases: list[dict[str, Any]] = []
    for length in segment_lens:
        rows.append(_extend(target_ids, length, fallback=fallback))
        cases.append({"kind": "plain", "length": length})
    rows.append([fallback])
    cases.append({"kind": "eos", "length": 1})
    repeated_len = max(2, min(8, max(segment_lens)))
    rows.append([repeated for _ in range(repeated_len)])
    cases.append({"kind": "repeated_token", "length": repeated_len})
    return prefix_ids, rows, {"prompt": prompt, "target": target, "cases": cases}


def _full_chain_replay(
    *,
    hf,
    vllm,
    prefix_ids: list[int],
    initial: list[int],
    proposals: list[list[int]],
    uniforms: list[float],
    temperature: float,
) -> dict[str, Any]:
    hf_chain = list(initial)
    vllm_chain = list(initial)
    hf_accept_positions: list[int] = []
    vllm_accept_positions: list[int] = []
    rows: list[dict[str, Any]] = []
    for pos, (proposal, u) in enumerate(zip(proposals, uniforms)):
        cur_hf = hf.score_segments([prefix_ids], [hf_chain], temperature=temperature)
        prop_hf = hf.score_segments([prefix_ids], [proposal], temperature=temperature)
        cur_vllm = vllm.score_segments([prefix_ids], [vllm_chain], temperature=temperature)
        prop_vllm = vllm.score_segments([prefix_ids], [proposal], temperature=temperature)
        hf_log_r = (
            prop_hf.lp_unnorm[0]
            + cur_hf.lp_norm[0]
            - cur_hf.lp_unnorm[0]
            - prop_hf.lp_norm[0]
        )
        vllm_log_r = (
            prop_vllm.lp_unnorm[0]
            + cur_vllm.lp_norm[0]
            - cur_vllm.lp_unnorm[0]
            - prop_vllm.lp_norm[0]
        )
        log_u = math.log(float(u))
        hf_accept = log_u < hf_log_r
        vllm_accept = log_u < vllm_log_r
        if hf_accept:
            hf_chain = list(proposal)
            hf_accept_positions.append(pos)
        if vllm_accept:
            vllm_chain = list(proposal)
            vllm_accept_positions.append(pos)
        rows.append(
            {
                "step": pos,
                "u": u,
                "log_u": log_u,
                "hf_log_r": hf_log_r,
                "vllm_log_r": vllm_log_r,
                "hf_accept": hf_accept,
                "vllm_accept": vllm_accept,
                "log_r_abs_diff": abs(hf_log_r - vllm_log_r),
            }
        )
    return {
        "passed": hf_chain == vllm_chain and hf_accept_positions == vllm_accept_positions,
        "final_token_chain_match": hf_chain == vllm_chain,
        "acceptance_count_match": len(hf_accept_positions) == len(vllm_accept_positions),
        "acceptance_positions_match": hf_accept_positions == vllm_accept_positions,
        "hf_accept_positions": hf_accept_positions,
        "vllm_accept_positions": vllm_accept_positions,
        "hf_final_token_ids": hf_chain,
        "vllm_final_token_ids": vllm_chain,
        "steps": rows,
    }


def main() -> None:
    args = _parse_args()

    from polaris.infra.serving.hf import RWSGenerator
    from polaris.infra.serving.vllm import VLLMGenerator
    from polaris.infra.vllm_calibration import (
        mh_replay_row,
        score_parity_rows,
        write_calibration_artifacts,
    )
    if args.model_key:
        from polaris.registry import resolve_model

        model_spec = resolve_model(args.model_key)
        args.model_id = model_spec.hf_id
        if args.model_revision is None:
            args.model_revision = model_spec.revision

    hf = RWSGenerator(
        model_id=args.model_id,
        revision=args.model_revision,
        torch_dtype=args.hf_dtype,
        score_segments_mode=args.hf_scoring_mode,
        device_map_auto=args.hf_device_map_auto,
        local_files_only=args.local_files_only,
    )
    vllm = VLLMGenerator(
        model_id=args.model_id,
        revision=args.model_revision,
        dtype=args.vllm_dtype,
        model_impl=args.vllm_model_impl,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        local_files_only=args.local_files_only,
        scoring_mode=args.vllm_scoring_mode,
        enable_prefix_caching=not args.no_vllm_prefix_caching,
        parity_artifact_path=str(args.out),
    )

    hf_prefix_ids, target_segments, segment_meta = _segments(hf.tokenizer, args.segment_lens)
    vllm_prefix_ids, vllm_segments, _ = _segments(vllm.tokenizer, args.segment_lens)
    tokenizer_parity = {
        "passed": hf_prefix_ids == vllm_prefix_ids and target_segments == vllm_segments,
        "model_id": args.model_id,
        "prefix_len": len(hf_prefix_ids),
        "segment_lens": [len(row) for row in target_segments],
        "model_revision": args.model_revision,
        "hf_tokenizer_revision": args.model_revision,
        "vllm_tokenizer_revision": args.model_revision,
    }
    if not tokenizer_parity["passed"]:
        raise SystemExit("HF/vLLM tokenizer parity failed before scoring")

    prefixes = [hf_prefix_ids for _ in target_segments]
    hf_scores = hf.score_segments(prefixes, target_segments, temperature=args.temperature)
    vllm_scores = vllm.score_segments(prefixes, target_segments, temperature=args.temperature)
    score_rows = score_parity_rows(
        hf=hf_scores,
        vllm=vllm_scores,
        prefix_lens=[len(hf_prefix_ids) for _ in target_segments],
        target_segments=target_segments,
        temperature=args.temperature,
    )

    mh_rows = []
    for idx in range(max(0, len(target_segments) - 1)):
        cur = target_segments[idx]
        prop = target_segments[idx + 1]
        cur_hf = hf.score_segments([hf_prefix_ids], [cur], temperature=args.temperature)
        prop_hf = hf.score_segments([hf_prefix_ids], [prop], temperature=args.temperature)
        cur_vllm = vllm.score_segments([hf_prefix_ids], [cur], temperature=args.temperature)
        prop_vllm = vllm.score_segments([hf_prefix_ids], [prop], temperature=args.temperature)
        mh_rows.append(
            mh_replay_row(
                row_id=f"mh-{idx}",
                cur_hf=cur_hf,
                prop_hf=prop_hf,
                cur_vllm=cur_vllm,
                prop_vllm=prop_vllm,
                row_index=0,
                suffix_len=max(len(cur), len(prop)),
                temperature=args.temperature,
                u=0.173 + 0.071 * (idx % 7),
            )
        )

    chain_len = min(8, len(target_segments[0]))
    initial = _extend(target_segments[0], chain_len, fallback=target_segments[0][0])
    proposals = [
        _extend(row, chain_len, fallback=initial[0])
        for row in target_segments[1 : min(5, len(target_segments))]
    ]
    uniforms = [0.19, 0.41, 0.73, 0.29][: len(proposals)]
    full_chain_replay = _full_chain_replay(
        hf=hf,
        vllm=vllm,
        prefix_ids=hf_prefix_ids,
        initial=initial,
        proposals=proposals,
        uniforms=uniforms,
        temperature=args.temperature,
    )
    full_chain_replay["tokenizer_parity"] = tokenizer_parity
    full_chain_replay["runtime_metadata"] = {
        "backend": "vllm",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "hf_dtype": args.hf_dtype,
        "hf_scoring_mode": args.hf_scoring_mode,
        "hf_device_map_auto": args.hf_device_map_auto,
        "vllm_scoring_mode": args.vllm_scoring_mode,
        "vllm_dtype": args.vllm_dtype,
        "vllm_model_impl": args.vllm_model_impl,
        "vllm_prefix_caching": not args.no_vllm_prefix_caching,
    }
    full_chain_replay["segment_cases"] = segment_meta["cases"]
    full_chain_replay["hf_runtime_metadata"] = hf.runtime_metadata()
    full_chain_replay["vllm_runtime_metadata"] = vllm.runtime_metadata()

    summary = write_calibration_artifacts(
        args.out,
        score_rows=score_rows,
        mh_rows=mh_rows,
        full_chain_replay=full_chain_replay,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
