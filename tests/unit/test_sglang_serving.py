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
            "text": "\\boxed{1}",
            "output_ids": [10, 11],
            "meta_info": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    gen = SGLangGenerator(model_id="m", transport=transport)
    out = gen.generate_greedy("Q", max_new_tokens=8)

    assert calls[0][0] == "/generate"
    assert calls[0][1]["text"] == "Q"
    assert calls[0][1]["stream"] is False
    assert calls[0][1]["sampling_params"]["temperature"] == 0.0
    assert calls[0][1]["sampling_params"]["max_new_tokens"] == 8
    assert out.generation == "\\boxed{1}"
    assert out.prompt_token_count == 4
    assert out.generation_token_count == 2
    assert out.token_ids == [10, 11]


def test_sglang_low_temp_batch_uses_native_generate():
    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return [
            {
                "text": "A",
                "output_ids": [1],
                "meta_info": {"prompt_tokens": 3, "completion_tokens": 1},
            },
            {
                "text": "B",
                "output_ids": [2, 3],
                "meta_info": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ]

    gen = SGLangGenerator(model_id="m", transport=transport)
    outs = gen.generate_low_temp_batch(
        ["Q1", "Q2"],
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=8,
        seed_base=10,
        seed_offsets=[0, 7],
    )

    assert calls == [
        (
            "/generate",
            {
                "text": ["Q1", "Q2"],
                "sampling_params": [
                    {
                        "max_new_tokens": 8,
                        "temperature": 0.6,
                        "sampling_seed": 10,
                        "top_p": 0.95,
                    },
                    {
                        "max_new_tokens": 8,
                        "temperature": 0.6,
                        "sampling_seed": 17,
                        "top_p": 0.95,
                    },
                ],
                "stream": False,
            },
        )
    ]
    assert [out.generation for out in outs] == ["A", "B"]
    assert [out.generation_token_count for out in outs] == [1, 2]


def test_sglang_sps_power_batch_uses_native_generate_power_sampling():
    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, payload))
        return [
            {
                "text": "A",
                "output_ids": [1],
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "power_sampling": {
                        "alpha": 4.0,
                        "block_count": 1,
                        "candidate_count": 8,
                        "rollout_count": 8,
                    },
                },
            },
            {
                "text": "B",
                "output_ids": [2],
                "meta_info": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "power_sampling": {
                        "alpha": 4.0,
                        "block_count": 1,
                        "candidate_count": 8,
                        "rollout_count": 8,
                    },
                },
            },
        ]

    gen = SGLangGenerator(model_id="m", transport=transport)
    outs = gen.generate_sps_power_batch(
        ["Q1", "Q2"],
        temperature=0.25,
        max_new_tokens=256,
        block_num=64,
        top_k=8,
        candidate_pool_size=8,
        rollouts_per_candidate=8,
        rollout_horizon=64,
        seed_base=0,
        seed_offsets=[0, 1],
    )

    params = calls[0][1]["sampling_params"]
    assert calls[0][0] == "/generate"
    assert calls[0][1]["text"] == ["Q1", "Q2"]
    assert params[0]["max_new_tokens"] == 256
    assert "temperature" not in params[0]
    assert "top_p" not in params[0]
    assert params[0]["power_sampling"]["alpha"] == 4.0
    assert params[0]["power_sampling"]["block_size"] == 64
    assert params[0]["power_sampling"]["seed"] == 0
    assert params[1]["power_sampling"]["seed"] == 1
    assert outs[0].power_sampling["candidate_count"] == 8


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
