from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from polaris.infra.farmshare import (
    SlurmArraySpec,
    default_paths,
    env_commands,
    probe_commands,
    render_slurm_array,
    rsync_to_farmshare_command,
    shard_indices,
)
from polaris.infra.preflight import (
    PaidRunPreflight,
    PreflightError,
    validate_paid_run_preflight,
)


def test_farmshare_preflight_requires_zero_cluster_cost(tmp_path):
    spec = PaidRunPreflight(
        run_kind="farmshare",
        artifact_dir=tmp_path / "run",
        cache_path=tmp_path / "cache.sqlite",
        split=(0, 20),
        seed=17,
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        backend="vllm",
        estimated_dollar_cost=0.0,
        cost_cap_dollars=0.0,
        user_authorized=True,
    )

    assert validate_paid_run_preflight(spec)["run_kind"] == "farmshare"

    with pytest.raises(PreflightError, match="farmshare estimated_dollar_cost must be 0.0"):
        validate_paid_run_preflight(
            PaidRunPreflight(**{**spec.__dict__, "estimated_dollar_cost": 0.01})
        )


def test_farmshare_layout_matches_scratch_contract():
    paths = default_paths("duynguy")

    assert paths.env_prefix == "/scratch/users/duynguy/polaris/envs/polaris"
    assert paths.hf_home == "/scratch/users/duynguy/.cache/huggingface"
    assert paths.artifacts_dir == "/scratch/users/duynguy/polaris/runs/prorl_recovery"


def test_farmshare_array_script_uses_four_one_gpu_shards():
    script = render_slurm_array(
        SlurmArraySpec(
            job_name="prorl-smoke",
            num_shards=4,
            command="python scripts/run_condition.py --track math500",
        )
    )

    assert "#SBATCH --array=0-3%4" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --output=/scratch/users/%u/polaris/slurm/%x-%A-%a.out" in script
    assert 'mkdir -p "/scratch/users/$USER/polaris/slurm"' in script
    assert 'POLARIS_NUM_SHARDS="4"' in script
    assert 'export HF_HOME="/scratch/users/$USER/.cache/huggingface"' in script
    assert 'export PATH="/scratch/users/$USER/polaris/envs/polaris/bin:$PATH"' in script
    assert "micromamba run -p" not in script
    assert "python scripts/run_condition.py --track math500" in script


def test_deterministic_shard_assignment():
    assert shard_indices(10, 0, 4) == [0, 4, 8]
    assert shard_indices(10, 3, 4) == [3, 7]
    with pytest.raises(ValueError):
        shard_indices(10, 4, 4)


def test_gpu_probe_sbatch_is_short_one_gpu_job():
    commands = probe_commands(include_gpu=True)

    assert any("--gres=gpu:1" in command for command in commands)
    assert any("--time=00:05:00" in command for command in commands)
    assert any("nvidia-smi -L" in command for command in commands)


def test_farmshare_env_installs_accelerate_for_hf_device_map():
    commands = env_commands()
    joined = " ".join(commands)

    assert "accelerate" in joined
    assert "reasoning-gym==0.1.25" in joined
    assert "import torch, transformers, accelerate, vllm" in joined


def test_project_pins_vllm_compatible_transformers():
    pyproject = tomllib.loads(open("pyproject.toml", "rb").read().decode())
    deps = pyproject["project"]["dependencies"]

    assert any(dep == "transformers==4.51.3" for dep in deps)


def test_farmshare_sync_up_preserves_protocol_progress_file():
    cmd = rsync_to_farmshare_command(
        local_repo=Path("/tmp/polaris"),
        remote_repo="/scratch/users/$USER/polaris/repo",
    )

    joined = " ".join(str(part) for part in cmd)
    assert "--include runs/" in joined
    assert "--include runs/progress.md" in joined
    assert "--exclude runs/**" in joined


def test_farmshare_sync_up_includes_rws_reference_source():
    cmd = rsync_to_farmshare_command(
        local_repo=Path("/tmp/polaris"),
        remote_repo="/scratch/users/$USER/polaris/repo",
    )

    joined = " ".join(str(part) for part in cmd)
    assert "--include upstream/reasoning-with-sampling/llm_experiments/***" in joined
    assert "--include upstream/reasoning-with-sampling/environment.yml" in joined
    assert "--exclude upstream/**" in joined
    assert not any(
        cmd[idx] == "--exclude" and cmd[idx + 1] == "upstream/"
        for idx in range(len(cmd) - 1)
    )
