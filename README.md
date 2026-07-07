# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment.sh`

Use `bash scripts/run_experiment.sh h100`, `bash scripts/run_experiment.sh a100`, or `bash scripts/run_experiment.sh l40` only to force a profile. The script auto-detects assigned GPUs, resumes the latest incomplete matching run, emits active progress updates, caps safe concurrency, selects a compatible vLLM dtype, requires Python 3.11/3.12 for the pinned vLLM stack, logs W&B when `WANDB_API_KEY` is set, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Force a new run with `bash scripts/run_experiment.sh --fresh`.
Check the node and package resolver before using a reserved GPU with `bash scripts/run_experiment.sh --doctor`.
Check a live/resumable run with `bash scripts/run_experiment.sh --status latest`.

On Modal, launch the four-H100 held-out RF goal with:
`python -m modal run "scripts/modal/modal_vllm_app.py::run_heldout_goal_h100x4" --estimated-dollar-cost <estimate> --cost-cap-dollars <cap> --user-authorized-paid-run`
This uses the persistent `proicl-hf-cache` and `proicl-runs` Modal volumes, runs the SPS MATH500 sampled-baseline gate (`bon_temp1`, N=100, matched `B=8`, concise MATH prompt, selected pass@8 gate, candidate-accuracy diagnostics, 4096 calibration generation tokens, 6144 calibration context tokens, and selected cap-failure guard by default), evaluates the four held-out conditions at N=50 per cell, and returns bundle paths plus SPS/confound summaries.

On Sherlock, use the Python module only and let the repo create its own vLLM environment:
`ml reset && ml python/3.12.1 && bash scripts/run_experiment.sh --doctor && bash scripts/run_experiment.sh`
Do not load Sherlock's central `py-vllm`, `py-pytorch`, or `py-transformers` modules for this experiment; their advertised versions do not match the repo's vLLM stack.

For a 12-hour H100 reservation, run `--doctor` first, start the normal command once it passes, and use `bash scripts/run_experiment.sh --status latest` from another shell to check PID, stderr growth, GEPA archive state, and checkpoint counts. On a single GPU, non-GEPA cells run before the GEPA archive build so the run makes useful resumable progress before the longest GEPA step. If the reservation ends, rerun the same command; completed cells and completed problems are skipped.

Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
