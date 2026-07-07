from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from polaris.infra.preflight import (
    PaidRunPreflight,
    PreflightError,
    validate_paid_run_preflight,
)


def _valid_preflight(tmp_path: Path) -> PaidRunPreflight:
    return PaidRunPreflight(
        run_kind="modal",
        artifact_dir=tmp_path / "run",
        cache_path=tmp_path / "trajectories.sqlite",
        split=(0, 2),
        seed=17,
        model_id="Qwen/Qwen2.5-Math-7B",
        backend="vllm",
        estimated_dollar_cost=0.25,
        cost_cap_dollars=0.50,
        user_authorized=True,
    )


def test_paid_run_preflight_accepts_complete_bounded_run(tmp_path):
    report = validate_paid_run_preflight(_valid_preflight(tmp_path))

    assert report["passed"] is True
    assert report["run_kind"] == "modal"
    assert report["estimated_dollar_cost"] == 0.25


def test_paid_run_preflight_accepts_complete_farmshare_zero_dollar_run(tmp_path):
    report = validate_paid_run_preflight(
        PaidRunPreflight(
            run_kind="farmshare",
            artifact_dir=tmp_path / "run",
            cache_path=tmp_path / "trajectories.sqlite",
            split=(0, 3),
            seed=17,
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            backend="vllm",
            estimated_dollar_cost=0.0,
            cost_cap_dollars=0.0,
            user_authorized=True,
        )
    )

    assert report["passed"] is True
    assert report["run_kind"] == "farmshare"
    assert report["estimated_dollar_cost"] == 0.0


def test_paid_run_preflight_rejects_missing_required_fields(tmp_path):
    spec = PaidRunPreflight(
        run_kind="modal",
        artifact_dir=tmp_path / "run",
        cache_path=None,
        split=(0, 2),
        seed=None,
        model_id="",
        backend="",
        estimated_dollar_cost=None,
        cost_cap_dollars=None,
        user_authorized=False,
    )

    with pytest.raises(PreflightError) as exc:
        validate_paid_run_preflight(spec)

    message = str(exc.value)
    assert "cache_path is required" in message
    assert "seed is required" in message
    assert "model_id is required" in message
    assert "backend is required" in message
    assert "estimated_dollar_cost is required" in message
    assert "cost_cap_dollars is required" in message
    assert "user_authorized must be true" in message


def test_paid_run_preflight_rejects_over_budget(tmp_path):
    spec = _valid_preflight(tmp_path)
    spec = PaidRunPreflight(
        **{
            **spec.__dict__,
            "estimated_dollar_cost": 0.75,
            "cost_cap_dollars": 0.50,
        }
    )

    with pytest.raises(PreflightError, match="exceeds cost_cap_dollars"):
        validate_paid_run_preflight(spec)


def test_paid_run_preflight_rejects_invalid_split(tmp_path):
    spec = _valid_preflight(tmp_path)
    spec = PaidRunPreflight(**{**spec.__dict__, "split": (2, 2)})

    with pytest.raises(PreflightError, match="split must satisfy"):
        validate_paid_run_preflight(spec)


def test_modal_gpu_entrypoints_require_preflight():
    root = Path(__file__).resolve().parents[2]
    for rel in ("scripts/modal/modal_vllm_app.py",):
        path = root / rel
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            is_modal_function = False
            has_gpu = False
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if getattr(decorator.func, "attr", None) == "function":
                    is_modal_function = True
                    has_gpu = any(keyword.arg == "gpu" for keyword in decorator.keywords)
            if is_modal_function and has_gpu:
                body = ast.get_source_segment(source, node) or ""
                assert "_require_modal_preflight" in body, f"{rel}:{node.name}"


def test_backend_preflight_classifies_vllm_runtime_failures():
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "backend_preflight",
        root / "scripts" / "preflight" / "backend_preflight.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend_preflight"] = module
    spec.loader.exec_module(module)

    assert (
        module._classify_failure(
            1,
            "triton.runtime.errors.OutOfResources: out of resource: shared memory, "
            "Required: 131072, Hardware limit: 65536",
        )
        == "vllm_attention_shared_memory"
    )
    assert (
        module._classify_failure(1, "ValueError: Token id 151703 is out of vocabulary")
        == "invalid_token_id"
    )
    assert module._classify_failure(-9, "Detected 3 oom_kill events") == "host_oom_or_sigkill"
