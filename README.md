# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment.sh`

Use `bash scripts/run_experiment.sh h100`, `bash scripts/run_experiment.sh a100`, or `bash scripts/run_experiment.sh l40` only to force a profile. The script auto-detects assigned GPUs, resumes the latest incomplete matching run, emits active progress updates, caps safe concurrency, selects a compatible vLLM dtype, requires Python 3.11/3.12 for the pinned vLLM stack, logs W&B when `WANDB_API_KEY` is set, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Force a new run with `bash scripts/run_experiment.sh --fresh`.
Check the node and package resolver before using a reserved GPU with `bash scripts/run_experiment.sh --doctor`.
Check a live/resumable run with `bash scripts/run_experiment.sh --status latest`.

On Sherlock, use the Python module only and let the repo create its own vLLM environment:
`ml reset && ml python/3.12.1 && bash scripts/run_experiment.sh --doctor && bash scripts/run_experiment.sh`
Do not load Sherlock's central `py-vllm`, `py-pytorch`, or `py-transformers` modules for this experiment; their advertised versions do not match the pinned vLLM stack. The repo pins the minimal eval resolver stack in `constraints/proicl-eval.txt` and fails fast if Linux binary wheels are unavailable.

For a 12-hour H100 reservation, run `--doctor` first, start the normal command once it passes, and use `bash scripts/run_experiment.sh --status latest` from another shell to check PID, stderr growth, GEPA archive state, and checkpoint counts. On a single GPU, non-GEPA cells run before the GEPA archive build so the run makes useful resumable progress before the longest GEPA step. If the reservation ends, rerun the same command; completed cells and completed problems are skipped.

GEPA checkpoints live under `full/archives/<scope>/gepa_run/`. To request a graceful GEPA checkpoint stop from another shell, run `bash scripts/run_experiment.sh --stop-gepa latest`; then resume the same timestamp with `bash scripts/run_experiment.sh --resume latest` or `bash scripts/run_experiment.sh --resume <timestamp>`. On SLURM, prefer sending an early user signal before walltime so the Python launcher can write `gepa.stop` and let GEPA save `gepa_state.bin` before the allocation is killed.

`scripts/package_results.py` only creates `results_bundle.tar.gz` when every planned cell is complete. For an interrupted run, resume with the same timestamp; if you need a diagnostic snapshot, run `python scripts/package_results.py --run-root <run-dir> --allow-partial`, which writes `partial_results_bundle.tar.gz` and includes per-cell checkpoint status.

Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
