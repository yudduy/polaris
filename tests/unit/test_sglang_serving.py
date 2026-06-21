from __future__ import annotations

import pytest

from polaris.infra.serving.sglang import SGLangGenerator
from polaris.infra.serving.sglang_logits import (
    build_next_token_score_request,
    build_segment_score_request,
    extract_input_token_logprobs,
    extract_output_token_id_logprob,
)


def test_sglang_generator_uses_injected_transport_for_greedy():
    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return {
            "choices": [{"text": "\\boxed{1}", "token_ids": [10, 11]}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    gen = SGLangGenerator(model_id="m", transport=transport)
    out = gen.generate_greedy("Q", max_new_tokens=8)

    assert calls[0][0] == "/v1/completions"
    assert calls[0][1]["temperature"] == 0.0
    assert calls[0][1]["max_tokens"] == 8
    assert out.generation == "\\boxed{1}"
    assert out.prompt_token_count == 4
    assert out.generation_token_count == 2
    assert out.token_ids == [10, 11]


def test_sglang_score_segments_uses_norm_and_base_requests():
    calls = []
    responses = iter(
        [
            {"meta_info": {"output_token_ids_logprobs": [[[-0.1, 2, None]]]}},
            {"meta_info": {"output_token_ids_logprobs": [[[-1.0, 2, None]]]}},
            {"meta_info": {"output_token_ids_logprobs": [[[-0.2, 3, None]]]}},
            {"meta_info": {"output_token_ids_logprobs": [[[-2.0, 3, None]]]}},
        ]
    )

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return next(responses)

    gen = SGLangGenerator(model_id="m", transport=transport)
    scores = gen.score_segments([[1]], [[2, 3]], temperature=0.25)

    assert [call[0] for call in calls] == ["/generate", "/generate"] * 2
    assert calls[0][1]["input_ids"] == [1]
    assert calls[2][1]["input_ids"] == [1, 2]
    assert calls[0][1]["sampling_params"]["temperature"] == 0.25
    assert calls[1][1]["sampling_params"]["temperature"] == 1.0
    assert calls[0][1]["token_ids_logprob"] == [2]
    assert calls[2][1]["token_ids_logprob"] == [3]
    assert scores.lp_norm == pytest.approx([-0.3])
    assert scores.lp_unnorm == pytest.approx([-12.0])
    assert scores.lp_norm_tokens == [[-0.1, -0.2]]
    assert scores.lp_unnorm_tokens == [[-4.0, -8.0]]


def test_sglang_score_segments_uses_forced_processor_when_available(monkeypatch, tmp_path):
    import polaris.infra.serving.sglang as sglang_mod

    calls = []

    class _Req:
        def to_payload(self):
            return {"forced": True}

    def fake_build(prefix_ids, target_ids, *, temperature, score_path, score_id):
        from pathlib import Path

        Path(score_path).write_text(
            "\n".join(
                [
                    '{"score_id": "%s", "position": 0, "target_token_id": 2, "lp_norm": -0.1, "lp_unnorm": -1.0}'
                    % score_id,
                    '{"score_id": "%s", "position": 1, "target_token_id": 3, "lp_norm": -0.2, "lp_unnorm": -2.0}'
                    % score_id,
                ]
            )
            + "\n"
        )
        return _Req()

    monkeypatch.setattr(sglang_mod, "build_forced_segment_score_request", fake_build)
    monkeypatch.setattr(sglang_mod.tempfile, "gettempdir", lambda: str(tmp_path))

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return {"text": ""}

    gen = SGLangGenerator(model_id="m", transport=transport)
    scores = gen.score_segments([[1]], [[2, 3]], temperature=0.25)

    assert calls == [("/generate", {"forced": True})]
    assert scores.lp_norm == pytest.approx([-0.3])
    assert scores.lp_unnorm == pytest.approx([-3.0])
    assert scores.lp_norm_tokens == [[-0.1, -0.2]]
    assert scores.lp_unnorm_tokens == [[-1.0, -2.0]]


def test_next_token_score_payload_requests_target_token_logprob():
    req = build_next_token_score_request([10, 11], 12, temperature=0.25)
    assert req.input_ids == [10, 11]
    assert req.target_token_id == 12
    assert req.to_payload()["token_ids_logprob"] == [12]
    assert req.to_payload()["sampling_params"]["temperature"] == 0.25


def test_segment_score_payload_starts_at_last_prefix_token():
    req = build_segment_score_request([10, 11, 12], [13, 14], temperature=0.5)
    assert req.input_ids == [10, 11, 12, 13, 14]
    assert req.logprob_start_len == 2
    assert req.target_len == 2


def test_extract_input_token_logprobs_accepts_common_shapes():
    response = {
        "meta_info": {
            "input_token_logprobs": [
                {"logprob": -3.0, "token_id": 1},
                [-2.0, 2],
                -1.0,
            ]
        }
    }
    assert extract_input_token_logprobs(response, 2) == [-2.0, -1.0]


def test_extract_output_token_id_logprob_accepts_common_shapes():
    response = {
        "meta_info": {
            "output_token_ids_logprobs": [
                [[-3.0, 1, None], [-2.0, 2, None]],
            ]
        }
    }
    assert extract_output_token_id_logprob(response, 2) == -2.0


def test_extract_output_token_id_logprob_accepts_val_idx_shape():
    response = {
        "meta_info": {
            "output_token_ids_logprobs_val": [[-3.0, -2.0]],
            "output_token_ids_logprobs_idx": [[1, 2]],
        }
    }
    assert extract_output_token_id_logprob(response, 2) == -2.0


def test_sglang_power_generation_blocked_until_parity():
    gen = SGLangGenerator(model_id="m", transport=lambda endpoint, payload: {})
    with pytest.raises(NotImplementedError, match="parity"):
        gen.generate_power("Q", temperature=0.25, max_new_tokens=8)


def test_sglang_generate_sps_power_batch_uses_native_generate():
    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return [
            {
                "text": "A",
                "output_ids": [1, 2],
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "power_sampling": {"candidate_count": 4},
                },
            },
            {
                "text": "B",
                "output_ids": [3, 4],
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "power_sampling": {"candidate_count": 4},
                },
            },
        ]

    gen = SGLangGenerator(model_id="m", seed=17, transport=transport)
    outs = gen.generate_sps_power_batch(
        ["Q1", "Q2"],
        temperature=0.25,
        max_new_tokens=4,
        block_num=2,
        top_k=2,
        candidate_pool_size=2,
        rollouts_per_candidate=1,
        rollout_horizon=1,
        seed_base=100,
        seed_offsets=[0, 10],
    )

    assert calls[0][0] == "/generate"
    payload = calls[0][1]
    assert payload["text"] == ["Q1", "Q2"]
    assert payload["stream"] is False
    assert [p["max_new_tokens"] for p in payload["sampling_params"]] == [4, 4]
    power = [p["power_sampling"] for p in payload["sampling_params"]]
    assert [p["alpha"] for p in power] == [4.0, 4.0]
    assert [p["block_size"] for p in power] == [2, 2]
    assert [p["candidate_top_k"] for p in power] == [2, 2]
    assert [p["rollouts_per_candidate"] for p in power] == [1, 1]
    assert [p["seed"] for p in power] == [100, 100 + 10 * 1_000_003]
    assert [out.generation for out in outs] == ["A", "B"]
    assert [out.token_ids for out in outs] == [[1, 2], [3, 4]]
    assert outs[0].meta_info["power_sampling"]["candidate_count"] == 4


def test_sglang_generate_sps_power_batch_rejects_bad_seed_offsets():
    gen = SGLangGenerator(model_id="m", transport=lambda endpoint, payload: {})
    with pytest.raises(ValueError, match="seed_offsets"):
        gen.generate_sps_power_batch(
            ["Q1", "Q2"],
            temperature=0.25,
            max_new_tokens=4,
            seed_offsets=[0],
        )
