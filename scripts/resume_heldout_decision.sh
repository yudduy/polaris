#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Resume the interrupted 20260529T211308Z held-out ProICL run.

Modes:
  status         Print the current run status using scripts/run_experiment.sh.
  acre-decision  Resume ACRE sps_only, then run ACRE prorl_v2_greedy.
  full-resume    Exact full-panel resume through scripts/run_experiment.sh.

Examples:
  bash scripts/resume_heldout_decision.sh status
  bash scripts/resume_heldout_decision.sh acre-decision
  bash scripts/resume_heldout_decision.sh full-resume

Environment overrides:
  RUN_ROOT       Series directory. Default: runs/experiment
  RUN_TIMESTAMP  UTC run timestamp. Default: 20260529T211308Z
  GPU_PROFILE    run_experiment profile. Default: h100
  GPUS           Comma-separated or space-separated GPU ids. Default: first visible GPU/0
  PYTHON         Python interpreter. Default: .venv-eval/bin/python if present, else python3
  FULL_ROOT      Existing full artifact root. Default: $RUN_ROOT/<run-id>/full
EOF
}

MODE="${1:-acre-decision}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ROOT="${RUN_ROOT:-runs/experiment}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-20260529T211308Z}"
GPU_PROFILE="${GPU_PROFILE:-h100}"
RUN_ID="proicl_small-real-slice_custom-4t_cross-family-curriculum_vllm_heldout_${RUN_TIMESTAMP}"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"
FULL_ROOT="${FULL_ROOT:-$RUN_DIR/full}"
CALIB_ARTIFACT="${VLLM_PARITY_ARTIFACT:-${RUN_ROOT%/}/calibration/deepseek_r1_distill_qwen_1p5b_vllm_float32/calibration_summary.json}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x ".venv-eval/bin/python" ]]; then
  PY=".venv-eval/bin/python"
else
  PY="python3"
fi

case "$MODE" in
  status)
    RUN_ROOT="$RUN_ROOT" RUN_TAG=heldout NUM_SHARDS=1 MAX_PARALLEL_CELLS=1 \
      bash scripts/run_experiment.sh "$GPU_PROFILE" --status "$RUN_TIMESTAMP"
    ;;

  full-resume)
    echo "Exact full-panel resume. This is not a 4-6 hour single-H100 run."
    echo "Run root: $RUN_DIR"
    RUN_ROOT="$RUN_ROOT" RUN_TAG=heldout NUM_SHARDS=1 MAX_PARALLEL_CELLS="${MAX_PARALLEL_CELLS:-1}" \
      bash scripts/run_experiment.sh "$GPU_PROFILE" --resume "$RUN_TIMESTAMP"
    ;;

  acre-decision)
    echo "ACRE decision resume. This targets the 4-6 hour single-H100 signal path."
    echo "Full root: $FULL_ROOT"
    echo "Cells: reasoning_gym_acre/sps_only, reasoning_gym_acre/prorl_v2_greedy"
    if [[ ! -d "$FULL_ROOT" ]]; then
      echo "Missing full run root: $FULL_ROOT" >&2
      echo "Run this from the original cluster checkout, or set RUN_ROOT to the series dir." >&2
      exit 1
    fi
    if [[ ! -f "$FULL_ROOT/proicl_signal_plan.json" ]]; then
      echo "Missing plan: $FULL_ROOT/proicl_signal_plan.json" >&2
      exit 1
    fi
    if [[ ! -f "$CALIB_ARTIFACT" ]]; then
      echo "Missing vLLM calibration artifact: $CALIB_ARTIFACT" >&2
      echo "Set VLLM_PARITY_ARTIFACT or run the normal launcher once to build calibration." >&2
      exit 1
    fi

    export PROICL_CELL_HEARTBEAT_SECONDS="${PROICL_CELL_HEARTBEAT_SECONDS:-60}"
    "$PY" - "$FULL_ROOT" "$CALIB_ARTIFACT" "${GPUS:-}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo_root = Path.cwd()
full_root = Path(sys.argv[1]).resolve()
calib_artifact = Path(sys.argv[2]).resolve()
raw_gpus = sys.argv[3].replace(",", " ").split()
if not raw_gpus:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    raw_gpus = [part for part in visible.replace(",", " ").split() if part]
gpus = raw_gpus[:1] or ["0"]

sys.path.insert(0, str(repo_root / "src"))
from polaris.proicl.launcher import LaunchCell, run_cells

plan = json.loads((full_root / "proicl_signal_plan.json").read_text(encoding="utf-8"))
wanted = [("reasoning_gym_acre", "sps_only"), ("reasoning_gym_acre", "prorl_v2_greedy")]
by_key = {(cell["track"], cell["proicl_condition"]): cell for cell in plan}
missing = [key for key in wanted if key not in by_key]
if missing:
    raise SystemExit(f"plan is missing expected ACRE decision cells: {missing}")

def launch_cell(payload: dict) -> LaunchCell:
    item = dict(payload)
    item["split"] = tuple(item["split"])
    item["archive_train_tracks"] = tuple(item.get("archive_train_tracks") or ())
    item["archive_heldout_tracks"] = tuple(item.get("archive_heldout_tracks") or ())
    item["extra_args"] = tuple(item.get("extra_args") or ())
    return LaunchCell(**item)

cells = [launch_cell(by_key[key]) for key in wanted]
for cell in cells:
    out = Path(cell.artifact_dir)
    launch_path = out / "proicl_launch_cell.json"
    if not out.exists() or (out / "metrics.json").exists():
        continue
    expected = cell.to_jsonable()
    if launch_path.exists():
        actual = json.loads(launch_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(
                "refusing to resume a partial cell with a mismatched "
                f"launch contract: {launch_path}"
            )
    else:
        launch_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[decision] seeded missing launch contract to preserve partial cell: {launch_path}")

print("[decision] running cells:")
for cell in cells:
    print(f"  {cell.track}/{cell.proicl_condition}/shard-{cell.shard_id} -> {cell.artifact_dir}")

run_cells(
    repo_root=repo_root,
    cells=cells,
    gpus=gpus,
    events_path=full_root / "events.jsonl",
    backend="vllm",
    local_files_only=False,
    cost_cap_dollars=0.0,
    estimated_dollar_cost=0.0,
    estimated_wall_clock_seconds=3600.0,
    run_kind="local",
    run_stage="small_real_slice",
    max_new_tokens=1024,
    power_sampler="sps",
    mcmc_steps=None,
    mcmc_block_num=6,
    sps_top_k=8,
    sps_candidate_pool_size=8,
    sps_rollouts_per_candidate=8,
    sps_rollout_horizon=128,
    vllm_dtype="bfloat16",
    vllm_model_impl="transformers",
    vllm_gpu_memory_utilization=0.88,
    vllm_max_model_len=None,
    vllm_scoring_mode="forced_decode_v0",
    vllm_parity_artifact=str(calib_artifact),
    vllm_enable_prefix_caching=True,
)

print("[decision] ACRE metrics:")
for condition in ("base_greedy", "sps_only", "prorl_v2_greedy"):
    path = full_root / "runs" / "reasoning_gym_acre" / condition / "shard-0" / "metrics.json"
    if not path.exists():
        print(f"  {condition}: missing")
        continue
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(
        "  "
        f"{condition}: accuracy={metrics.get('accuracy')} "
        f"mean_selected_score={metrics.get('mean_selected_score')} "
        f"n_problems={metrics.get('n_problems')} "
        f"n_candidates={metrics.get('n_candidates')}"
    )
PY
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac
