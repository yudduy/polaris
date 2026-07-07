from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOST = "farmshare"
DEFAULT_REMOTE_ROOT_TEMPLATE = "/scratch/users/{user}/polaris"
DEFAULT_NUM_SHARDS = 4


@dataclass(frozen=True)
class FarmSharePaths:
    remote_root: str
    repo_dir: str
    env_prefix: str
    hf_home: str
    artifacts_dir: str


@dataclass(frozen=True)
class SlurmArraySpec:
    job_name: str
    command: str
    remote_root: str = "/scratch/users/$USER/polaris"
    repo_dir: str = "/scratch/users/$USER/polaris/repo"
    num_shards: int = DEFAULT_NUM_SHARDS
    partition: str = "gpu"
    qos: str = "gpu"
    cpus_per_task: int = 8
    mem: str = "48G"
    time_limit: str = "02:00:00"
    env_name: str = "polaris"
    output_dir: str = "/scratch/users/%u/polaris/slurm"
    gres: str = "gpu:1"
    array_tasks: int | None = None
    max_concurrent: int | None = None


def default_paths(user: str = "$USER") -> FarmSharePaths:
    remote_root = DEFAULT_REMOTE_ROOT_TEMPLATE.format(user=user)
    return FarmSharePaths(
        remote_root=remote_root,
        repo_dir=f"{remote_root}/repo",
        env_prefix=f"{remote_root}/envs/polaris",
        hf_home=f"/scratch/users/{user}/.cache/huggingface",
        artifacts_dir=f"{remote_root}/runs/prorl_recovery",
    )


def shard_indices(total: int, shard_id: int, num_shards: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return [idx for idx in range(total) if idx % num_shards == shard_id]


def render_slurm_array(spec: SlurmArraySpec) -> str:
    if spec.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not spec.command.strip():
        raise ValueError("command is required")
    array_tasks = spec.array_tasks if spec.array_tasks is not None else spec.num_shards
    if array_tasks <= 0:
        raise ValueError("array_tasks must be positive")
    max_concurrent = spec.max_concurrent if spec.max_concurrent is not None else spec.num_shards
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    array = f"0-{array_tasks - 1}%{max_concurrent}"
    shell_output_dir = spec.output_dir.replace("%u", "$USER")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --qos={spec.qos}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
        f"#SBATCH --mem={spec.mem}",
        f"#SBATCH --time={spec.time_limit}",
        f"#SBATCH --gres={spec.gres}",
        f"#SBATCH --array={array}",
        f"#SBATCH --output={spec.output_dir}/%x-%A-%a.out",
        f"#SBATCH --error={spec.output_dir}/%x-%A-%a.err",
        "",
        "set -euo pipefail",
        "",
        f'export POLARIS_REMOTE_ROOT="{spec.remote_root}"',
        f'export POLARIS_REPO_DIR="{spec.repo_dir}"',
        'export POLARIS_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"',
        'export POLARIS_SHARD_ID="${SLURM_ARRAY_TASK_ID}"',
        f'export POLARIS_NUM_SHARDS="{spec.num_shards}"',
        'export HF_HOME="/scratch/users/$USER/.cache/huggingface"',
        'export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"',
        'export TRANSFORMERS_CACHE="${HF_HOME}"',
        'export XDG_CACHE_HOME="${POLARIS_REMOTE_ROOT}/xdg-cache"',
        'export TMPDIR="${POLARIS_REMOTE_ROOT}/tmp"',
        'export VLLM_USE_V1="0"',
        f'export VIRTUAL_ENV="{spec.remote_root}/envs/{spec.env_name}"',
        f'export PATH="{spec.remote_root}/envs/{spec.env_name}/bin:$PATH"',
        'mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"',
        f'mkdir -p "{shell_output_dir}"',
        'cd "$POLARIS_REPO_DIR"',
        spec.command,
        "",
    ]
    return "\n".join(lines)


def probe_commands(include_gpu: bool = True) -> list[str]:
    commands = [
        "hostname",
        "sinfo -o '%P %a %D %G %l'",
        "sacctmgr show qos format=name%16,maxsubmitjobspu,maxjobspu,mintres%20,maxtrespu%25,maxwall",
    ]
    if include_gpu:
        commands.append(
            "srun --partition=gpu --qos=gpu --gres=gpu:1 --time=00:05:00 nvidia-smi -L"
        )
    return commands


def env_commands(remote_root: str = "/scratch/users/$USER/polaris") -> list[str]:
    return [
        f"export MAMBA_PKGS_DIRS={remote_root}/mamba-pkgs",
        f"export PIP_CACHE_DIR={remote_root}/pip-cache",
        "export HF_HOME=/scratch/users/$USER/.cache/huggingface",
        "export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub",
        "export TRANSFORMERS_CACHE=$HF_HOME",
        (
            "mkdir -p "
            f"{remote_root}/repo "
            f"{remote_root}/envs "
            f"{remote_root}/tmp "
            f"{remote_root}/slurm "
            f"{remote_root}/xdg-cache "
            f"{remote_root}/mamba-pkgs "
            f"{remote_root}/pip-cache "
            f"{remote_root}/runs/prorl_recovery "
            "$HUGGINGFACE_HUB_CACHE"
        ),
        f"cd {remote_root}/repo",
        (
            f"if [ ! -x {remote_root}/envs/polaris/bin/python ]; then "
            f"micromamba create -y -p {remote_root}/envs/polaris python=3.11 pip; "
            "fi"
        ),
        (
            f"micromamba run -p {remote_root}/envs/polaris "
            "python -m pip install -e '.[code,dc]' vllm==0.9.2 "
            "reasoning-gym==0.1.25 accelerate pytest pyarrow"
        ),
        (
            f"micromamba run -p {remote_root}/envs/polaris "
            "python - <<'PY'\n"
            "import torch, transformers, accelerate, vllm, datasets, reasoning_gym, pandas, pyarrow\n"
            "print('imports_ok')\n"
            "PY"
        ),
    ]


def rsync_to_farmshare_command(
    *,
    host: str = DEFAULT_HOST,
    local_repo: Path,
    remote_repo: str = "/scratch/users/$USER/polaris/repo",
) -> list[str]:
    return [
        "rsync",
        "-az",
        "--delete",
        "--exclude",
        ".venv-eval/",
        "--include",
        "runs/",
        "--include",
        "runs/progress.md",
        "--exclude",
        "runs/**",
        "--exclude",
        "__pycache__/",
        "--exclude",
        ".pytest_cache/",
        "--exclude",
        ".claude/",
        "--include",
        "upstream/",
        "--include",
        "upstream/reasoning-with-sampling/",
        "--include",
        "upstream/reasoning-with-sampling/README.md",
        "--include",
        "upstream/reasoning-with-sampling/environment.yml",
        "--include",
        "upstream/reasoning-with-sampling/llm_experiments/",
        "--include",
        "upstream/reasoning-with-sampling/llm_experiments/***",
        "--exclude",
        "upstream/**",
        f"{local_repo}/",
        f"{host}:{remote_repo}/",
    ]
