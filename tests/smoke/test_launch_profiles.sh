#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_RUN_ROOT="$(mktemp -d)"
tmpbin=""
trap 'rm -rf "$TEST_RUN_ROOT" "${tmpbin:-}"' EXIT

run_case() {
  local name="$1"
  shift
  local output
  output="$(env -i PATH="$PATH" HOME="${HOME:-}" "$@" 2>&1)"
  printf '%s\n' "$output" | sed "s/^/[$name] /"
  echo "$output"
}

assert_contains() {
  local output="$1"
  local needle="$2"
  if ! grep -Fq -- "$needle" <<<"$output"; then
    echo "Expected output to contain: $needle" >&2
    exit 1
  fi
}

assert_not_contains() {
  local output="$1"
  local needle="$2"
  if grep -Fq -- "$needle" <<<"$output"; then
    echo "Expected output not to contain: $needle" >&2
    exit 1
  fi
}

common_env=(
  DRY_RUN=1
  SKIP_INSTALL=1
  SKIP_CALIBRATION=0
  RUN_ROOT="$TEST_RUN_ROOT"
  GPU_MEMORY_MIB=49140
  GPU_NAMES="NVIDIA L40S"
)

doctor_root="$(mktemp -d)"
out="$(run_case doctor-h100 \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  RUN_ROOT="$doctor_root" \
  HOST_MEMORY_MIB=131072 \
  GPU_MEMORY_MIB=81559 \
  GPU_COMPUTE_CAP=9.0 \
  GPU_NAMES="NVIDIA H100 80GB HBM3" \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment.sh" h100 --doctor)"
assert_contains "$out" "ProICL doctor:"
assert_contains "$out" "constraints=none"
test -f "$doctor_root/cluster_probe.json"
rm -rf "$doctor_root"

status_empty_root="$(mktemp -d)"
out="$(run_case status-empty \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  RUN_ROOT="$status_empty_root" \
  bash "$ROOT/scripts/run_experiment.sh" --status latest)"
assert_contains "$out" "No ProICL runs found"
assert_not_contains "$out" "ProICL launch profile:"
rm -rf "$status_empty_root"

status_root="$(mktemp -d)"
status_run="$status_root/proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_20260529T120000Z"
status_cell="$status_run/full/runs/reasoning_gym_boxnet/sps_only/shard-0"
mkdir -p "$status_cell"
printf 'gepa running\n' > "$status_run/full/logs_gepa.tmp"
printf '%s\n' \
  '{"event":"queue_start","ts":"20260529T120000Z","cells":1,"pending":1,"skipped":0}' \
  '{"archive_dir":"'"$status_run"'/full/archives/proicl_cross_family_curriculum_test","event":"gepa_archive_start","gpu":"0","pid":'"$$"',"stderr_log":"'"$status_run"'/full/logs_gepa.tmp","ts":"20260529T120000Z"}' \
  '{"elapsed_seconds":60,"event":"gepa_archive_heartbeat","gpu":"0","pid":'"$$"',"ts":"20260529T120100Z"}' \
  '{"artifact_dir":"'"$status_cell"'","condition":"sps_only","event":"cell_start","gpu":"0","pid":'"$$"',"shard":0,"stderr_log":"'"$status_cell"'/stderr.log","track":"reasoning_gym_boxnet","ts":"20260529T120001Z"}' \
  > "$status_run/full/events.jsonl"
printf '{"complete":false,"completed_problems":1,"expected_problems":2}\n' > "$status_cell/checkpoint.json"
printf '{"problem_id":"p0"}\n' > "$status_cell/selected.jsonl"
printf 'working\n' > "$status_cell/stderr.log"
out="$(run_case status-active \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  RUN_ROOT="$status_root" \
  bash "$ROOT/scripts/run_experiment.sh" --status latest)"
assert_contains "$out" "cells planned=1 observed=1 complete=0 failed=0 active=1 stale=0"
assert_contains "$out" "gepa_archive=status=active"
assert_contains "$out" "checkpoint=1/2"
rm -rf "$status_root"

out="$(run_case l40-one \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_profile=l40 gpu_count=1"
assert_contains "$out" "gpu_detection_source=SLURM_STEP_GPUS"
assert_contains "$out" "num_shards=1"
assert_contains "$out" "max_parallel_cells=1 smoke_max_parallel_cells=1"
assert_contains "$out" "vllm_dtype=bfloat16"
assert_contains "$out" "vllm_attention_backend=FLASH_ATTN"
assert_contains "$out" "calibration_dtype=float32"
assert_contains "$out" "resume_source=fresh run_timestamp=new"
assert_contains "$out" "progress_interval_seconds=60"
assert_contains "$out" "sps_vllm_batch_size=32"
assert_contains "$out" "--skip-smoke"
assert_not_contains "$out" "--smoke-only"
assert_not_contains "$out" "Running production-shaped SPS/vLLM backend preflight"

out="$(run_case l40-five-tight-host \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0,1,2,3,4 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_profile=l40 gpu_count=5"
assert_contains "$out" "num_shards=5"
assert_contains "$out" "max_parallel_cells=1 smoke_max_parallel_cells=1"

out="$(run_case l40-two-roomy-host \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=163840 \
  SLURM_GPUS_ON_NODE=2 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_profile=l40 gpu_count=2"
assert_contains "$out" "gpu_detection_source=SLURM_GPUS_ON_NODE"
assert_contains "$out" "max_parallel_cells=2 smoke_max_parallel_cells=1"
assert_contains "$out" "overlap_gepa_and_cells=1"

out="$(run_case h100-four \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=262144 \
  GPU_MEMORY_MIB=81559 \
  GPU_COMPUTE_CAP=9.0 \
  GPU_NAMES="NVIDIA H100 80GB HBM3" \
  SLURM_STEP_GPUS=0,1,2,3 \
  bash "$ROOT/scripts/run_experiment.sh" h100)"
assert_contains "$out" "gpu_profile=h100 gpu_count=4"
assert_contains "$out" "num_shards=4"
assert_contains "$out" "max_parallel_cells=4 smoke_max_parallel_cells=1"
assert_contains "$out" "overlap_gepa_and_cells=1"
assert_contains "$out" "sps_chain_batch_size=2"
assert_contains "$out" "sps_vllm_batch_size=64"
assert_contains "$out" "vllm_dtype=bfloat16"

out="$(run_case titan-rtx-pre-ampere \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=131072 \
  GPU_MEMORY_MIB=24220 \
  GPU_COMPUTE_CAP=7.5 \
  GPU_NAMES="NVIDIA TITAN RTX" \
  SLURM_STEP_GPUS=0,1 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_profile=l40 gpu_count=2"
assert_contains "$out" "gpu_min_compute_cap_x10=75"
assert_contains "$out" "vllm_dtype=float16"

out="$(run_case override-gpus \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=163840 \
  GPUS=2,4 \
  SLURM_STEP_GPUS=0,1,2,3,4 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_detection_source=GPUS"
assert_contains "$out" "gpu_profile=l40 gpu_count=2 cuda_visible_devices=2,4"

resume_root="$(mktemp -d)"
mkdir -p "$resume_root/proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_20260528T235310Z"
mkdir -p "$resume_root/proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_20260529T010203Z"
mkdir -p "$resume_root/proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_20260529T010203Z/full"
touch "$resume_root/proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_20260529T010203Z/full/results_bundle.tar.gz"
out="$(run_case auto-resume-incomplete \
  "${common_env[@]}" \
  RUN_ROOT="$resume_root" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "resume_source=auto run_timestamp=20260528T235310Z"
assert_contains "$out" "progress_interval_seconds=60"
assert_contains "$out" "--run-timestamp 20260528T235310Z"
out="$(run_case resume-latest-cli \
  "${common_env[@]}" \
  RUN_ROOT="$resume_root" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment.sh" l40 --resume latest --progress-interval 60)"
assert_contains "$out" "resume_source=latest run_timestamp=20260529T010203Z"
assert_contains "$out" "progress_interval_seconds=60"
assert_contains "$out" "--run-timestamp 20260529T010203Z"
out="$(run_case fresh-overrides-resume \
  "${common_env[@]}" \
  RUN_ROOT="$resume_root" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment.sh" l40 --fresh)"
assert_contains "$out" "resume_source=fresh run_timestamp=new"
assert_not_contains "$out" "--run-timestamp"
rm -rf "$resume_root"

tmpbin="$(mktemp -d)"
cat >"$tmpbin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index --format=csv,noheader,nounits"*)
    printf "0\n1\n2\n"
    ;;
  *"--query-gpu=name --format=csv,noheader"*)
    printf "NVIDIA L40S\nNVIDIA L40S\nNVIDIA L40S\n"
    ;;
  *"--query-gpu=index,memory.total --format=csv,noheader,nounits"*)
    printf "0, 49140\n1, 49140\n2, 49140\n"
    ;;
  *"--query-gpu=index,name,memory.total --format=csv,noheader"*)
    printf "0, NVIDIA L40S, 49140 MiB\n1, NVIDIA L40S, 49140 MiB\n2, NVIDIA L40S, 49140 MiB\n"
    ;;
  -L)
    printf "GPU 0: NVIDIA L40S\nGPU 1: NVIDIA L40S\nGPU 2: NVIDIA L40S\n"
    ;;
  *)
    exit 1
    ;;
esac
SH
chmod +x "$tmpbin/nvidia-smi"
out="$(run_case nvidia-smi-fallback \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=200000 \
  PATH="$tmpbin:$PATH" \
  bash "$ROOT/scripts/run_experiment.sh" l40)"
assert_contains "$out" "gpu_profile=l40 gpu_count=3"
assert_contains "$out" "gpu_detection_source=nvidia-smi"
assert_contains "$out" "max_parallel_cells=2 smoke_max_parallel_cells=1"

cat >"$tmpbin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index,uuid --format=csv,noheader,nounits"*)
    printf "0, GPU-aaaa\n"
    ;;
  *"--query-gpu=index --format=csv,noheader,nounits"*)
    printf "0\n"
    ;;
  *"--query-gpu=name --format=csv,noheader"*)
    printf "NVIDIA L40S\n"
    ;;
  *"--query-gpu=index,memory.total --format=csv,noheader,nounits"*)
    printf "0, 49140\n"
    ;;
  *"--query-gpu=index,name,memory.total --format=csv,noheader"*)
    printf "0, NVIDIA L40S, 49140 MiB\n"
    ;;
  -L)
    printf "GPU 0: NVIDIA L40S\n"
    ;;
  *)
    exit 1
    ;;
esac
SH
chmod +x "$tmpbin/nvidia-smi"
out="$(run_case nvidia-visible-devices-uuid \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=200000 \
  NVIDIA_VISIBLE_DEVICES=GPU-aaaa \
  PATH="$tmpbin:$PATH" \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_profile=l40 gpu_count=1 cuda_visible_devices=0"
assert_contains "$out" "gpu_detection_source=NVIDIA_VISIBLE_DEVICES"

echo "launch profile tests passed"
