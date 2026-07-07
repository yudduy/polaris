"""Isolated Modal app for vLLM scorer and MCMC checks."""

from __future__ import annotations

from pathlib import Path

import modal

PROICL_ROOT = Path(__file__).resolve().parents[2]

_IGNORE = [
    ".venv*",
    "__pycache__",
    "*.pyc",
    "upstream",
    "runs/_archive",
    "runs/rws_official_full164_mcmc_ckpt",
    ".git",
    ".pytest_cache",
    ".claude",
    "tests",
]

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.9.2")
    .entrypoint([])
    .run_commands(
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "python -m pip install "
        "hf-transfer==0.1.8 "
        "'huggingface-hub[hf_xet]>=0.33.0,<1.0' "
        "'numpy>=1.26,<2.3' "
        "accelerate datasets pandas pyarrow pylatexenc scipy sympy tqdm wandb "
        "python-dotenv reasoning-gym==0.1.25 transformers==4.51.3",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(str(PROICL_ROOT), remote_path="/proicl", copy=True, ignore=_IGNORE)
)

app = modal.App("proicl-vllm-smokes", image=vllm_image)
hf_cache = modal.Volume.from_name("proicl-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("proicl-runs", create_if_missing=True)


def _setup_paths() -> None:
    import os
    import sys

    if "/proicl/src" not in sys.path:
        sys.path.insert(0, "/proicl/src")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _require_modal_preflight(
    *,
    backend: str,
    estimated_dollar_cost: float | None,
    cost_cap_dollars: float | None,
    user_authorized_paid_run: bool,
    artifact_dir: str = "/proicl-modal-smoke",
    cache_path: str = "/cache/huggingface",
    model_id: str | None = None,
) -> dict:
    from pathlib import Path as _Path

    from polaris.config import MODEL_ID
    from polaris.infra.preflight import PaidRunPreflight, validate_paid_run_preflight

    return validate_paid_run_preflight(
        PaidRunPreflight(
            run_kind="modal",
            artifact_dir=_Path(artifact_dir),
            cache_path=_Path(cache_path),
            split=(0, 1),
            seed=0,
            model_id=model_id or MODEL_ID,
            backend=backend,
            estimated_dollar_cost=estimated_dollar_cost,
            cost_cap_dollars=cost_cap_dollars,
            user_authorized=user_authorized_paid_run,
        )
    )


def _ensure_model_cached(model_id: str, revision: str | None = None) -> str:
    import os

    from huggingface_hub import snapshot_download

    # Snapshot download is a setup step, not the timed inference path. Avoid
    # hf_transfer's opaque failures on Xet-backed repos and let hf_xet/HTTP
    # handle the model cache deterministically.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    return snapshot_download(model_id, revision=revision)


def _run_python_json(script: str, *, timeout: int = 1200) -> dict:
    import json
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/proicl/src"
    proc = subprocess.run(
        ["python", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    marker = "PROICL_JSON:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise RuntimeError(
        "vLLM helper failed or did not emit JSON\n"
        f"returncode={proc.returncode}\nstdout={proc.stdout[-4000:]}\nstderr={proc.stderr[-4000:]}"
    )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_parity(
    segment_len: int = 32,
    temperature: float = 0.25,
    model_impl: str = "transformers",
    dtype: str = "float32",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """R5 parity smoke: vLLM V0 forced-token logits vs HF oracle."""
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    _ensure_model_cached(MODEL_ID)
    summary = _run_python_json(
        f"""
import gc
import json
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from transformers import LogitsProcessor

from polaris.config import MODEL_ID
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.infra.serving.vllm import VLLMGenerator

segment_len = {int(segment_len)}
temperature = {float(temperature)!r}
model_impl = {model_impl!r}
dtype = {dtype!r}
model_path = snapshot_download(MODEL_ID, local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=False, local_files_only=True
)
direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
problem_text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
answer = r"\\left( 3, \\frac{{\\pi}}{{2}} \\right)"
prompt_text = direct.compose(problem_text)
prefix_ids = tokenizer.encode(prompt_text)
target_text = " The final answer is \\\\boxed{{" + answer + "}}."
target_ids = tokenizer.encode(target_text, add_special_tokens=False)[:segment_len]

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32 if dtype in ("float", "float32") else torch.bfloat16,
    trust_remote_code=False,
    local_files_only=True,
).to("cuda").eval()
full_ids = prefix_ids + target_ids
input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
rows = out.logits[0, len(prefix_ids) - 1 : len(full_ids) - 1, :].float()
targets = torch.tensor(target_ids, dtype=torch.long, device=rows.device)
hf_norm_tokens = (
    F.log_softmax(rows / temperature, dim=-1)
    .gather(-1, targets.unsqueeze(-1))
    .squeeze(-1)
)
hf_unnorm_tokens = (
    (1.0 / temperature)
    * F.log_softmax(rows, dim=-1)
    .gather(-1, targets.unsqueeze(-1))
    .squeeze(-1)
)
hf_lp_norm = float(hf_norm_tokens.sum().detach().cpu().item())
hf_lp_unnorm = float(hf_unnorm_tokens.sum().detach().cpu().item())

class HFGenerateRecorder(LogitsProcessor):
    def __init__(self, target_ids, temperature):
        self.target_ids = [int(x) for x in target_ids]
        self.temperature = float(temperature)
        self.lp_norm = []
        self.lp_unnorm = []

    def __call__(self, input_ids, scores):
        pos = len(self.lp_norm)
        if pos >= len(self.target_ids):
            return scores
        target = self.target_ids[pos]
        row = scores[0].float()
        self.lp_norm.append(
            float(F.log_softmax(row / self.temperature, dim=-1)[target].detach().cpu())
        )
        self.lp_unnorm.append(
            float(
                (
                    (1.0 / self.temperature)
                    * F.log_softmax(row, dim=-1)[target]
                ).detach().cpu()
            )
        )
        forced = torch.full_like(scores, -torch.inf)
        forced[:, target] = 0.0
        return forced

hf_proc = HFGenerateRecorder(target_ids, temperature)
forced_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
hf_forced_output = model.generate(
    input_ids=forced_input_ids,
    attention_mask=torch.ones_like(forced_input_ids),
    max_new_tokens=len(target_ids),
    min_new_tokens=len(target_ids),
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    output_logits=True,
    logits_processor=[hf_proc],
)
hf_gen_token_ids = [
    int(x) for x in hf_forced_output.sequences[0][len(prefix_ids) :].detach().cpu().tolist()
]
hf_gen_lp_norm = float(sum(hf_proc.lp_norm))
hf_gen_lp_unnorm = float(sum(hf_proc.lp_unnorm))
del model, out, rows, input_ids, forced_input_ids, hf_forced_output
torch.cuda.empty_cache()
gc.collect()

vllm_gen = VLLMGenerator(
    model_id=model_path,
    dtype=dtype,
    gpu_memory_utilization=0.55,
    model_impl=model_impl,
    max_model_len=4096,
    local_files_only=True,
)
vllm_scores = vllm_gen.score_segments([prefix_ids], [target_ids], temperature=temperature)
vllm_token_ids = target_ids
vllm_lp_norm = float(vllm_scores.lp_norm[0])
vllm_lp_unnorm = float(vllm_scores.lp_unnorm[0])

forward_vs_generate_diff = max(
    abs(hf_lp_norm - hf_gen_lp_norm),
    abs(hf_lp_unnorm - hf_gen_lp_unnorm),
)
vllm_vs_generate_diff = max(
    abs(hf_gen_lp_norm - vllm_lp_norm),
    abs(hf_gen_lp_unnorm - vllm_lp_unnorm),
)
vllm_vs_forward_diff = max(
    abs(hf_lp_norm - vllm_lp_norm),
    abs(hf_lp_unnorm - vllm_lp_unnorm),
)
score_segments_passed = vllm_vs_forward_diff < 1e-3
hf_generate_consistency_passed = forward_vs_generate_diff < 5e-3
summary = {{
    "model_id": MODEL_ID,
    "segment_len": len(target_ids),
    "temperature": temperature,
    "model_impl": model_impl,
    "dtype": dtype,
    "hf_forward_lp_norm": hf_lp_norm,
    "hf_generate_lp_norm": hf_gen_lp_norm,
    "vllm_lp_norm": vllm_lp_norm,
    "hf_forward_lp_unnorm": hf_lp_unnorm,
    "hf_generate_lp_unnorm": hf_gen_lp_unnorm,
    "vllm_lp_unnorm": vllm_lp_unnorm,
    "hf_forward_vs_generate_max_abs_diff": forward_vs_generate_diff,
    "vllm_vs_hf_generate_max_abs_diff": vllm_vs_generate_diff,
    "vllm_vs_hf_forward_max_abs_diff": vllm_vs_forward_diff,
    "score_segments_passed": score_segments_passed,
    "hf_generate_consistency_passed": hf_generate_consistency_passed,
    "passed": score_segments_passed and hf_generate_consistency_passed,
    "acceptance_threshold": 1e-3,
    "hf_forward_generate_tolerance": 5e-3,
    "target_ids_head": target_ids[:5],
    "hf_generate_token_ids_head": hf_gen_token_ids[:5],
    "vllm_token_ids_head": vllm_token_ids[:5],
    "hf_generate_norm_head": hf_proc.lp_norm[:5],
    "vllm_norm_head": vllm_scores.lp_norm_tokens[0][:5],
    "hf_generate_unnorm_head": hf_proc.lp_unnorm[:5],
    "vllm_unnorm_head": vllm_scores.lp_unnorm_tokens[0][:5],
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_vllm_parity: {summary}")
    return summary


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_power_path(
    max_new_tokens: int = 64,
    block_num: int = 4,
    mcmc_steps: int = 1,
    temperature: float = 0.25,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Exercise the real vLLM generate_power path on GPU with a tiny budget."""
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    _ensure_model_cached(MODEL_ID)
    summary = _run_python_json(
        f"""
import json
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

from huggingface_hub import snapshot_download

from polaris.config import MODEL_ID
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.evals.verifiers.math import score_math
from polaris.infra.serving.vllm import VLLMGenerator

model_path = snapshot_download(MODEL_ID, local_files_only=True)
direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
problem_text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
answer = r"\\left( 3, \\frac{{\\pi}}{{2}} \\right)"
prompt_text = direct.compose(problem_text)

gen = VLLMGenerator(
    model_id=model_path,
    dtype="float32",
    gpu_memory_utilization=0.55,
    model_impl="transformers",
    max_model_len=4096,
    local_files_only=True,
)
out = gen.generate_power(
    prompt_text,
    temperature={float(temperature)!r},
    max_new_tokens={int(max_new_tokens)},
    block_num={int(block_num)},
    mcmc_steps={int(mcmc_steps)},
)
verifier = score_math(prompt_text + out.generation, answer)
summary = {{
    "model_id": MODEL_ID,
    "backend": "vllm",
    "dtype": "float32",
    "model_impl": "transformers",
    "temperature": {float(temperature)!r},
    "max_new_tokens": {int(max_new_tokens)},
    "block_num": {int(block_num)},
    "mcmc_steps": {int(mcmc_steps)},
    "generation_token_count": out.generation_token_count,
    "logprobs_norm_count": len(out.logprobs_norm or []),
    "logprobs_unnorm_count": len(out.logprobs_unnorm or []),
    "acceptance_ratio": out.acceptance_ratio,
    "wall_clock_seconds": out.wall_clock_seconds,
    "verifier_passed": verifier.get("passed"),
    "verifier_score": verifier.get("score"),
    "verifier_extracted": verifier.get("extracted"),
    "generation_head": out.generation[:200],
    "generation_tail": out.generation[-300:],
    "passed": (
        out.generation_token_count == {int(max_new_tokens)}
        and len(out.logprobs_norm or []) == out.generation_token_count
        and len(out.logprobs_unnorm or []) == out.generation_token_count
        and out.acceptance_ratio is not None
        and 0.0 <= out.acceptance_ratio <= 1.0
    ),
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_vllm_power_path: {summary}")
    return summary


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_batched_power_path(
    batch_size: int = 8,
    max_new_tokens: int = 512,
    block_num: int = 4,
    mcmc_steps: int = 1,
    temperature: float = 0.25,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Exercise batched vLLM MCMC mechanics on GPU before scale-up."""
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    _ensure_model_cached(MODEL_ID)
    summary = _run_python_json(
        f"""
import json
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

from huggingface_hub import snapshot_download

from polaris.config import MODEL_ID
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.evals.verifiers.math import score_math
from polaris.infra.serving.vllm import VLLMGenerator

batch_size = {int(batch_size)}
max_new_tokens = {int(max_new_tokens)}
block_num = {int(block_num)}
mcmc_steps = {int(mcmc_steps)}
temperature = {float(temperature)!r}

model_path = snapshot_download(MODEL_ID, local_files_only=True)
direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
problem_text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
answer = r"\\left( 3, \\frac{{\\pi}}{{2}} \\right)"
prompt_text = direct.compose(problem_text)
prompt_texts = [prompt_text for _ in range(batch_size)]

gen = VLLMGenerator(
    model_id=model_path,
    dtype="float32",
    gpu_memory_utilization=0.55,
    model_impl="transformers",
    max_model_len=4096,
    local_files_only=True,
)
outs = gen.generate_power_batch(
    prompt_texts,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    block_num=block_num,
    mcmc_steps=mcmc_steps,
    seed_base=17,
    seed_offsets=list(range(batch_size)),
)
verifiers = [score_math(prompt_text + out.generation, answer) for out in outs]
alignment_passed = all(
    out.generation_token_count == len(out.token_ids or [])
    and out.generation_token_count == len(out.logprobs_norm or [])
    and out.generation_token_count == len(out.logprobs_unnorm or [])
    and out.acceptance_ratio is not None
    and 0.0 <= out.acceptance_ratio <= 1.0
    for out in outs
)
summary = {{
    "model_id": MODEL_ID,
    "backend": "vllm",
    "dtype": "float32",
    "model_impl": "transformers",
    "batch_size": batch_size,
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
    "block_num": block_num,
    "mcmc_steps": mcmc_steps,
    "generation_token_counts": [out.generation_token_count for out in outs],
    "logprobs_norm_counts": [len(out.logprobs_norm or []) for out in outs],
    "logprobs_unnorm_counts": [len(out.logprobs_unnorm or []) for out in outs],
    "acceptance_ratios": [out.acceptance_ratio for out in outs],
    "allocated_wall_clock_seconds_sum": sum(out.wall_clock_seconds for out in outs),
    "verifier_passed_count": sum(1 for verifier in verifiers if verifier.get("passed")),
    "verifier_extracted": [verifier.get("extracted") for verifier in verifiers],
    "generation_heads": [out.generation[:120] for out in outs[:2]],
    "generation_tails": [out.generation[-180:] for out in outs[:2]],
    "passed": alignment_passed and len(outs) == batch_size,
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_vllm_batched_power_path: {summary}")
    return summary


@app.function(
    gpu="H100",
    timeout=2400,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_logprobs_probe(
    segment_len: int = 64,
    temperature: float = 0.25,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Probe whether vLLM native logprobs can replace forced decode scoring."""
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    _ensure_model_cached(MODEL_ID)
    summary = _run_python_json(
        f"""
import json
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

from huggingface_hub import snapshot_download

from polaris.config import MODEL_ID
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.infra.serving.vllm import VLLMGenerator

def get_lp(entry, token_id):
    if entry is None:
        raise RuntimeError("missing logprob entry")
    if int(token_id) in entry:
        item = entry[int(token_id)]
    elif str(int(token_id)) in entry:
        item = entry[str(int(token_id))]
    else:
        item = next(iter(entry.values()))
        found = getattr(item, "decoded_token", None)
        raise RuntimeError(f"token {{token_id}} absent from logprob entry; first={{found}} keys={{list(entry)[:5]}}")
    return float(getattr(item, "logprob", item))

segment_len = {int(segment_len)}
temperature = {float(temperature)!r}
model_path = snapshot_download(MODEL_ID, local_files_only=True)
direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
problem_text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
prompt_text = direct.compose(problem_text)

gen = VLLMGenerator(
    model_id=model_path,
    dtype="float32",
    gpu_memory_utilization=0.55,
    model_impl="transformers",
    max_model_len=4096,
    local_files_only=True,
)
prefix_ids = gen._encode(prompt_text)
native_params = gen._sampling_params(
    max_tokens=segment_len,
    min_tokens=segment_len,
    temperature=temperature,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    seed=123,
    ignore_eos=True,
    skip_special_tokens=False,
    detokenize=False,
    logprobs=1,
)
native_output = gen.llm.generate(
    prompts=[{{"prompt_token_ids": prefix_ids}}],
    sampling_params=native_params,
    use_tqdm=False,
)[0]
native_completion = native_output.outputs[0]
token_ids = [int(x) for x in native_completion.token_ids]
native_lp_norm_tokens = [
    get_lp(entry, token_id)
    for entry, token_id in zip(native_completion.logprobs, token_ids)
]

forced = gen.score_segments([prefix_ids], [token_ids], temperature=temperature)
full_ids = prefix_ids + token_ids
prompt_params = gen._sampling_params(
    max_tokens=1,
    min_tokens=1,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    seed=999,
    ignore_eos=True,
    skip_special_tokens=False,
    detokenize=False,
    prompt_logprobs=1,
)
prompt_output = gen.llm.generate(
    prompts=[{{"prompt_token_ids": full_ids}}],
    sampling_params=prompt_params,
    use_tqdm=False,
)[0]
prompt_logprobs = prompt_output.prompt_logprobs
if prompt_logprobs is None:
    raise RuntimeError("prompt_logprobs missing")
base_lp_tokens = []
for offset, token_id in enumerate(token_ids):
    pos = len(prefix_ids) + offset
    base_lp_tokens.append(get_lp(prompt_logprobs[pos], token_id))
prompt_lp_unnorm_tokens = [(1.0 / temperature) * x for x in base_lp_tokens]

native_norm_diff = max(
    abs(a - b)
    for a, b in zip(native_lp_norm_tokens, forced.lp_norm_tokens[0])
) if token_ids else 0.0
prompt_unnorm_diff = max(
    abs(a - b)
    for a, b in zip(prompt_lp_unnorm_tokens, forced.lp_unnorm_tokens[0])
) if token_ids else 0.0
summary = {{
    "model_id": MODEL_ID,
    "segment_len": len(token_ids),
    "temperature": temperature,
    "native_norm_max_abs_diff_vs_forced": native_norm_diff,
    "prompt_unnorm_max_abs_diff_vs_forced": prompt_unnorm_diff,
    "native_norm_sum": float(sum(native_lp_norm_tokens)),
    "forced_norm_sum": float(forced.lp_norm[0]),
    "prompt_unnorm_sum": float(sum(prompt_lp_unnorm_tokens)),
    "forced_unnorm_sum": float(forced.lp_unnorm[0]),
    "native_norm_head": native_lp_norm_tokens[:5],
    "forced_norm_head": forced.lp_norm_tokens[0][:5],
    "prompt_unnorm_head": prompt_lp_unnorm_tokens[:5],
    "forced_unnorm_head": forced.lp_unnorm_tokens[0][:5],
    "token_ids_head": token_ids[:8],
    "passed": native_norm_diff < 1e-3 and prompt_unnorm_diff < 1e-3,
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_vllm_logprobs_probe: {summary}")
    return summary


def _run_calibration_artifact_smoke(
    *,
    gate: str,
    temperature: float,
    scoring_mode: str,
    dtype: str,
    model_impl: str,
    estimated_dollar_cost: float | None,
    cost_cap_dollars: float | None,
    user_authorized_paid_run: bool,
) -> dict:
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    _ensure_model_cached(MODEL_ID)
    summary = _run_python_json(
        f"""
import json
import subprocess
from pathlib import Path

out = Path("/tmp/proicl-vllm-calibration-{gate}")
cmd = [
    "python",
    "/proicl/scripts/preflight/vllm_hf_calibration.py",
    "--model-id",
    {MODEL_ID!r},
    "--out",
    str(out),
    "--temperature",
    str({float(temperature)!r}),
    "--vllm-scoring-mode",
    {scoring_mode!r},
    "--vllm-dtype",
    {dtype!r},
    "--vllm-model-impl",
    {model_impl!r},
    "--local-files-only",
]
subprocess.run(cmd, check=True, text=True)
summary = json.loads((out / "calibration_summary.json").read_text())
summary["requested_gate"] = {gate!r}
summary["artifact_dir"] = str(out)
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_vllm_{gate}: {summary}")
    return summary


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_score_parity(
    temperature: float = 0.25,
    scoring_mode: str = "forced_decode_v0",
    dtype: str = "float32",
    model_impl: str = "transformers",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Write score_parity.jsonl plus the full HF-vLLM calibration bundle."""
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    return _run_calibration_artifact_smoke(
        gate="score_parity",
        temperature=temperature,
        scoring_mode=scoring_mode,
        dtype=dtype,
        model_impl=model_impl,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_mh_replay_parity(
    temperature: float = 0.25,
    scoring_mode: str = "forced_decode_v0",
    dtype: str = "float32",
    model_impl: str = "transformers",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Write mh_replay_parity.jsonl plus the full HF-vLLM calibration bundle."""
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    return _run_calibration_artifact_smoke(
        gate="mh_replay_parity",
        temperature=temperature,
        scoring_mode=scoring_mode,
        dtype=dtype,
        model_impl=model_impl,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_vllm_full_chain_replay(
    temperature: float = 0.25,
    scoring_mode: str = "forced_decode_v0",
    dtype: str = "float32",
    model_impl: str = "transformers",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Write full_chain_replay.json plus the full HF-vLLM calibration bundle."""
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )
    return _run_calibration_artifact_smoke(
        gate="full_chain_replay",
        temperature=temperature,
        scoring_mode=scoring_mode,
        dtype=dtype,
        model_impl=model_impl,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


def _smoke_proicl_one_problem_vllm_native_impl(
    *,
    condition: str = "single_prompt_power",
    power_sampler: str = "mcmc",
    scoring_mode: str = "forced_decode_v0",
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run one ProICL/Reasoning-Gym problem through run_condition with vLLM."""
    _setup_paths()
    from polaris.registry import resolve_model

    model = resolve_model("deepseek-r1-distill-qwen-1.5b")
    model_id = model.hf_id
    model_revision = model.revision
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id=model_id,
    )
    _ensure_model_cached(model_id, revision=model_revision)
    summary = _run_python_json(
        f"""
import json
import subprocess
from pathlib import Path

root = Path("/tmp/proicl-one-problem-vllm")
subprocess.run(
    [
        "python",
        "/proicl/scripts/proicl/run_proicl.py",
        "write-direct-archives",
        "--root",
        str(root),
        "--tracks",
        "reasoning_gym_boxnet",
    ],
    check=True,
    text=True,
)
calibration = root / "calibration"
subprocess.run(
    [
        "python",
        "/proicl/scripts/preflight/vllm_hf_calibration.py",
        "--model-id",
        {model_id!r},
        "--model-revision",
        {model_revision!r},
        "--out",
        str(calibration),
        "--temperature",
        "0.25",
        "--hf-dtype",
        "float32",
        "--hf-scoring-mode",
        "cached_decode",
        "--segment-lens",
        "1",
        "2",
        "8",
        "--vllm-scoring-mode",
        {scoring_mode!r},
        "--vllm-dtype",
        "float32",
        "--vllm-model-impl",
        "transformers",
        "--vllm-max-model-len",
        "4096",
    ],
    check=True,
    text=True,
)
out = root / "run"
cmd = [
    "python",
    "/proicl/scripts/run_condition.py",
    "--track",
    "reasoning_gym_boxnet",
    "--model-key",
    "deepseek-r1-distill-qwen-1.5b",
    "--model-revision",
    {model_revision!r},
    "--condition",
    {condition!r},
    "--archive",
    str(root / "archives" / "reasoning_gym_boxnet" / "direct.json"),
    "--split",
    "20",
    "21",
    "--seed",
    "17",
    "--proicl-source-hash",
    "modal-vllm-smoke",
    "--preregistration-anchor",
    "TODO.PROICL.md#proicl-fast-weight-recovery-audit",
    "--out",
    str(out),
    "--backend",
    "vllm",
    "--samples-per-problem",
    "1",
    "--max-new-tokens",
    str({int(max_new_tokens)}),
    "--sps-block-num",
    "2",
    "--power-sampler",
    {power_sampler!r},
    "--sps-top-k",
    str({int(sps_top_k)}),
    "--sps-candidate-pool-size",
    str({int(sps_candidate_pool_size)}),
    "--sps-rollouts-per-candidate",
    str({int(sps_rollouts_per_candidate)}),
    "--sps-rollout-horizon",
    str({int(sps_rollout_horizon)}),
    "--vllm-scoring-mode",
    {scoring_mode!r},
    "--vllm-dtype",
    "float32",
    "--vllm-model-impl",
    "transformers",
    "--vllm-max-model-len",
    "4096",
    "--vllm-parity-artifact",
    str(calibration / "calibration_summary.json"),
    "--run-stage",
    "smoke",
    "--run-kind",
    "modal",
]
subprocess.run(cmd, check=True, text=True)
metrics = json.loads((out / "metrics.json").read_text())
manifest = json.loads((out / "manifest.json").read_text())
summary = {{
    "passed": metrics.get("n_problems") == 1,
    "metrics": metrics,
    "serving_backend": manifest.get("config", {{}}).get("serving_backend"),
    "artifact_dir": str(out),
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_proicl_one_problem_vllm_native: {summary}")
    return summary


def _smoke_archive_sps_one_problem_vllm_impl(
    *,
    scoring_mode: str = "forced_decode_v0",
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    sps_chain_batch_size: int = 1,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the archive-conditioned SPS path that backs gepa_sps_fixed."""
    _setup_paths()
    from polaris.registry import resolve_model

    model = resolve_model("deepseek-r1-distill-qwen-1.5b")
    model_id = model.hf_id
    model_revision = model.revision
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-archive-sps-smoke",
        model_id=model_id,
    )
    _ensure_model_cached(model_id, revision=model_revision)
    summary = _run_python_json(
        f"""
import json
import os
import subprocess
from pathlib import Path

root = Path("/tmp/proicl-archive-sps-smoke")
root.mkdir(parents=True, exist_ok=True)
archive = root / "archive.json"
archive.write_text(json.dumps({{
    "entries": [
        {{
            "id": "prompt_a",
            "prefix": "Solve carefully. Track every state transition. ",
            "suffix": "\\nReturn the final answer only after checking constraints.",
            "descriptor_hint": "state_tracking",
        }},
        {{
            "id": "prompt_b",
            "prefix": "Use a compact step-by-step plan and verify each move. ",
            "suffix": "\\nMake sure the answer satisfies the task verifier.",
            "descriptor_hint": "constraint_verification",
        }},
    ],
    "cell_fitness": {{}},
}}, sort_keys=True), encoding="utf-8")
calibration = root / "calibration"
subprocess.run(
    [
        "python",
        "/proicl/scripts/preflight/vllm_hf_calibration.py",
        "--model-id",
        {model_id!r},
        "--model-revision",
        {model_revision!r},
        "--out",
        str(calibration),
        "--temperature",
        "0.25",
        "--hf-dtype",
        "float32",
        "--hf-scoring-mode",
        "cached_decode",
        "--segment-lens",
        "1",
        "2",
        "8",
        "--vllm-scoring-mode",
        {scoring_mode!r},
        "--vllm-dtype",
        "float32",
        "--vllm-model-impl",
        "transformers",
        "--vllm-max-model-len",
        "4096",
        "--vllm-gpu-memory-utilization",
        "0.50",
    ],
    check=True,
    text=True,
)
out = root / "run"
cmd = [
    "python",
    "/proicl/scripts/run_condition.py",
    "--track",
    "reasoning_gym_boxnet",
    "--model-key",
    "deepseek-r1-distill-qwen-1.5b",
    "--model-revision",
    {model_revision!r},
    "--condition",
    "full_archive_fixed",
    "--archive",
    str(archive),
    "--split",
    "20",
    "21",
    "--seed",
    "17",
    "--proicl-source-hash",
    "modal-archive-sps-smoke",
    "--preregistration-anchor",
    "TODO.PROICL.md#proicl-fast-weight-recovery-audit",
    "--out",
    str(out),
    "--backend",
    "vllm",
    "--samples-per-problem",
    "2",
    "--max-new-tokens",
    str({int(max_new_tokens)}),
    "--sps-block-num",
    "2",
    "--power-sampler",
    "sps",
    "--sps-top-k",
    str({int(sps_top_k)}),
    "--sps-candidate-pool-size",
    str({int(sps_candidate_pool_size)}),
    "--sps-rollouts-per-candidate",
    str({int(sps_rollouts_per_candidate)}),
    "--sps-rollout-horizon",
    str({int(sps_rollout_horizon)}),
    "--vllm-scoring-mode",
    {scoring_mode!r},
    "--vllm-dtype",
    "float32",
    "--vllm-model-impl",
    "transformers",
    "--vllm-max-model-len",
    "4096",
    "--vllm-gpu-memory-utilization",
    "0.80",
    "--vllm-parity-artifact",
    str(calibration / "calibration_summary.json"),
    "--run-stage",
    "smoke",
    "--run-kind",
    "modal",
]
env = os.environ.copy()
env["PROICL_SPS_CHAIN_BATCH_SIZE"] = str({int(sps_chain_batch_size)})
subprocess.run(cmd, check=True, text=True, env=env)
metrics = json.loads((out / "metrics.json").read_text())
manifest = json.loads((out / "manifest.json").read_text())
candidates = [
    json.loads(line)
    for line in (out / "candidates.jsonl").read_text().splitlines()
    if line.strip()
]
summary = {{
    "passed": metrics.get("n_problems") == 1 and metrics.get("n_candidates") == 2,
    "metrics": metrics,
    "n_candidates_jsonl": len(candidates),
    "prompt_ids": sorted({{row.get("prompt_id") for row in candidates}}),
    "sps_chain_batch_size": {int(sps_chain_batch_size)},
    "serving_backend": manifest.get("config", {{}}).get("serving_backend"),
    "artifact_dir": str(out),
}}
print("PROICL_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_archive_sps_one_problem_vllm: {summary}")
    return summary


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_proicl_one_problem_vllm_native(
    condition: str = "single_prompt_power",
    power_sampler: str = "mcmc",
    scoring_mode: str = "forced_decode_v0",
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run one ProICL/Reasoning-Gym problem through run_condition with vLLM on H100."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_proicl_one_problem_vllm_native_impl(
        condition=condition,
        power_sampler=power_sampler,
        scoring_mode=scoring_mode,
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_proicl_one_problem_vllm_a100_40gb(
    condition: str = "single_prompt_power",
    power_sampler: str = "mcmc",
    scoring_mode: str = "forced_decode_v0",
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the exact ProICL/vLLM power-sampling smoke on Modal A100 40GB."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_proicl_one_problem_vllm_native_impl(
        condition=condition,
        power_sampler=power_sampler,
        scoring_mode=scoring_mode,
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_proicl_one_problem_vllm_a100_80gb(
    condition: str = "single_prompt_power",
    power_sampler: str = "mcmc",
    scoring_mode: str = "forced_decode_v0",
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the exact ProICL/vLLM power-sampling smoke on Modal A100 80GB."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_proicl_one_problem_vllm_native_impl(
        condition=condition,
        power_sampler=power_sampler,
        scoring_mode=scoring_mode,
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_experiment_one_problem_a100_80gb(
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the release-facing experiment smoke on one held-out boxnet problem."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_proicl_one_problem_vllm_native_impl(
        condition="single_prompt_power",
        power_sampler="sps",
        scoring_mode="forced_decode_v0",
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_experiment_one_problem_h100(
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the release-facing experiment smoke on one held-out boxnet problem."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-one-problem-vllm",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_proicl_one_problem_vllm_native_impl(
        condition="single_prompt_power",
        power_sampler="sps",
        scoring_mode="forced_decode_v0",
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_archive_sps_one_problem_a100_40gb(
    max_new_tokens: int = 64,
    sps_top_k: int = 2,
    sps_candidate_pool_size: int = 2,
    sps_rollouts_per_candidate: int = 1,
    sps_rollout_horizon: int = 16,
    sps_chain_batch_size: int = 1,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run archive-conditioned SPS on a constrained 40GB GPU."""
    _setup_paths()
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-archive-sps-smoke",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    return _smoke_archive_sps_one_problem_vllm_impl(
        max_new_tokens=max_new_tokens,
        sps_top_k=sps_top_k,
        sps_candidate_pool_size=sps_candidate_pool_size,
        sps_rollouts_per_candidate=sps_rollouts_per_candidate,
        sps_rollout_horizon=sps_rollout_horizon,
        sps_chain_batch_size=sps_chain_batch_size,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


def _smoke_run_experiment_script_impl(
    *,
    gpu_profile: str,
    max_new_tokens: int = 64,
    vllm_max_model_len: int = 2048,
    smoke_only: bool = True,
    extra_env: dict[str, str] | None = None,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    _setup_paths()
    from polaris.registry import resolve_model

    model = resolve_model("deepseek-r1-distill-qwen-1.5b")
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/tmp/proicl-run-experiment-smoke",
        model_id=model.hf_id,
    )
    summary = _run_python_json(
        f"""
import json
import os
import subprocess
from pathlib import Path

env = os.environ.copy()
env.update({{
    "PYTHONPATH": "/proicl/src",
    "RUN_ROOT": "/tmp/proicl-run-experiment-smoke",
    "SMOKE_ONLY": {("1" if smoke_only else "0")!r},
    "SKIP_INSTALL": "1",
    "SKIP_SPS_MATH500_CALIBRATION": "1",
    "MAX_NEW_TOKENS": str({int(max_new_tokens)}),
    "VLLM_MAX_MODEL_LEN": str({int(vllm_max_model_len)}),
    "REFLECTION_PROVIDER": "local-hf",
}})
env.update({extra_env or {}!r})
proc = subprocess.run(
    ["bash", "/proicl/scripts/run_experiment.sh", {gpu_profile!r}],
    cwd="/proicl",
    env=env,
    text=True,
    capture_output=True,
    timeout=3300,
)
run_root = Path("/tmp/proicl-run-experiment-smoke")
launch_config = run_root / "launch_config.json"
resource_probe = run_root / "resource_probe.json"
backend_preflight = run_root / "backend_preflight" / "backend_preflight.json"
runtime_profile = run_root / "runtime_profile.json"
payload = {{
    "passed": proc.returncode == 0,
    "returncode": proc.returncode,
    "stdout_tail": proc.stdout[-4000:],
    "stderr_tail": proc.stderr[-4000:],
    "launch_config": json.loads(launch_config.read_text()) if launch_config.exists() else None,
    "resource_probe": json.loads(resource_probe.read_text()) if resource_probe.exists() else None,
    "backend_preflight": json.loads(backend_preflight.read_text()) if backend_preflight.exists() else None,
    "runtime_profile": json.loads(runtime_profile.read_text()) if runtime_profile.exists() else None,
}}
if proc.returncode == 0:
    bundles = sorted(run_root.rglob("results_bundle.tar.gz"))
    payload["bundle_count"] = len(bundles)
    payload["bundle_paths"] = [str(path) for path in bundles]
print("PROICL_JSON:" + json.dumps(payload, sort_keys=True))
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
""",
        timeout=3900,
    )
    print(f"smoke_run_experiment_script: {summary}")
    return summary


@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_run_experiment_script_l40(
    max_new_tokens: int = 64,
    vllm_max_model_len: int = 2048,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run scripts/run_experiment.sh l40 in SMOKE_ONLY mode on one Modal L40S."""
    # _require_modal_preflight is enforced inside _smoke_run_experiment_script_impl.
    return _smoke_run_experiment_script_impl(
        gpu_profile="l40",
        max_new_tokens=max_new_tokens,
        vllm_max_model_len=vllm_max_model_len,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_backend_matrix_l40s(
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run production-shaped SPS/vLLM backend preflight on one Modal L40S."""
    # _require_modal_preflight is enforced inside _smoke_run_experiment_script_impl.
    return _smoke_run_experiment_script_impl(
        gpu_profile="l40",
        max_new_tokens=1024,
        vllm_max_model_len=4096,
        smoke_only=True,
        extra_env={
            "TRACKS": "reasoning_gym_boxnet",
            "EVAL_START": "20",
            "EVAL_END": "21",
            "GEPA_DEV_END": "1",
            "ROLLOUT_BUDGET": "1",
            "ARCHIVE_SIZE": "2",
            "MAX_METRIC_CALLS": "4",
            "SPS_TOP_K": "8",
            "SPS_CANDIDATE_POOL_SIZE": "8",
            "SPS_ROLLOUTS_PER_CANDIDATE": "8",
            "SPS_ROLLOUT_HORIZON": "128",
            "SPS_VLLM_BATCH_SIZE": "32",
        },
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="L40S:2",
    timeout=7200,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_run_experiment_script_l40s_2gpu(
    max_new_tokens: int = 64,
    vllm_max_model_len: int = 4096,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run a release-shaped two-GPU L40S smoke that must produce a bundle."""
    # _require_modal_preflight is enforced inside _smoke_run_experiment_script_impl.
    return _smoke_run_experiment_script_impl(
        gpu_profile="l40",
        max_new_tokens=max_new_tokens,
        vllm_max_model_len=vllm_max_model_len,
        smoke_only=False,
        extra_env={
            "TRACKS": "reasoning_gym_boxnet",
            "ARCHIVE_TRAIN_TRACKS": "reasoning_gym_family_relationships",
            "ARCHIVE_HELDOUT_TRACKS": "reasoning_gym_boxnet",
            "CONDITIONS": "base_greedy sps_only gepa_sps_fixed prorl_v2_greedy",
            "EVAL_START": "20",
            "EVAL_END": "22",
            "GEPA_DEV_START": "0",
            "GEPA_DEV_END": "1",
            "ROLLOUT_BUDGET": "1",
            "ARCHIVE_SIZE": "2",
            "MAX_METRIC_CALLS": "4",
            "SPS_TOP_K": "2",
            "SPS_CANDIDATE_POOL_SIZE": "2",
            "SPS_ROLLOUTS_PER_CANDIDATE": "1",
            "SPS_ROLLOUT_HORIZON": "16",
            "SPS_VLLM_BATCH_SIZE": "32",
        },
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_run_experiment_script_a100_40gb(
    max_new_tokens: int = 64,
    vllm_max_model_len: int = 2048,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run scripts/run_experiment.sh a100 in SMOKE_ONLY mode on one A100-40GB."""
    # _require_modal_preflight is enforced inside _smoke_run_experiment_script_impl.
    return _smoke_run_experiment_script_impl(
        gpu_profile="a100",
        max_new_tokens=max_new_tokens,
        vllm_max_model_len=vllm_max_model_len,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache/huggingface": hf_cache},
)
def smoke_run_experiment_script_h100(
    max_new_tokens: int = 64,
    vllm_max_model_len: int = 2048,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run scripts/run_experiment.sh h100 in SMOKE_ONLY mode on one H100."""
    # _require_modal_preflight is enforced inside _smoke_run_experiment_script_impl.
    return _smoke_run_experiment_script_impl(
        gpu_profile="h100",
        max_new_tokens=max_new_tokens,
        vllm_max_model_len=vllm_max_model_len,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )


@app.function(
    gpu="H100:4",
    timeout=60 * 60 * 24,
    volumes={"/cache/huggingface": hf_cache, "/proicl-runs": runs_vol},
)
def run_heldout_goal_h100x4(
    run_root: str = "/proicl-runs/heldout",
    run_tag: str = "modal-goal",
    max_new_tokens: int = 1024,
    math_calib_end: int = 100,
    math_calib_max_new_tokens: int = 4096,
    math_calib_vllm_max_model_len: int = 6144,
    sps_calibration_samples_per_problem: int = 8,
    vllm_max_model_len: int = 4096,
    rollout_budget: int = 8,
    archive_size: int = 8,
    max_metric_calls: int = 1500,
    num_shards: int = 4,
    max_parallel_cells: int = 4,
    sps_alpha: float = 4.0,
    sps_top_k: int = 8,
    sps_candidate_pool_size: int = 8,
    sps_rollouts_per_candidate: int = 8,
    sps_rollout_horizon: int = 128,
    sps_min_sharpening_gain: float = 0.01,
    sps_max_capped_failed_rate: float = 0.01,
    run_backend_preflight: bool = True,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run the paper-aligned held-out RF goal on one 4-GPU Modal H100 node."""
    _setup_paths()
    from polaris.registry import resolve_model

    base_model = resolve_model("deepseek-r1-distill-qwen-1.5b")
    prorl_model = resolve_model("nemotron-prorl-v2")
    _require_modal_preflight(
        backend="vllm",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir=run_root,
        model_id=base_model.hf_id,
    )
    _ensure_model_cached(base_model.hf_id, revision=base_model.revision)
    _ensure_model_cached(prorl_model.hf_id, revision=prorl_model.revision)

    sps_calibration_dirname = (
        f"sps_math500_bon_temp1_k{int(sps_calibration_samples_per_problem)}"
    )
    env_overrides = {
        "PYTHON": "python",
        "PYTHONPATH": "/proicl/src",
        "RUN_ROOT": run_root,
        "RUN_TAG": run_tag,
        "RUN_STAGE": "small_real_slice",
        "RUN_KIND": "modal",
        "SMOKE_ONLY": "0",
        "SKIP_INSTALL": "1",
        "SKIP_SPS_MATH500_CALIBRATION": "0",
        "MATH_CALIB_END": str(int(math_calib_end)),
        "MATH_CALIB_MAX_NEW_TOKENS": str(int(math_calib_max_new_tokens)),
        "MATH_CALIB_VLLM_MAX_MODEL_LEN": str(int(math_calib_vllm_max_model_len)),
        "SPS_CALIBRATION_BASELINE_CONDITION": "bon_temp1",
        "SPS_CALIBRATION_SAMPLES_PER_PROBLEM": str(int(sps_calibration_samples_per_problem)),
        "SPS_REQUIRE_MCMC_APPROXIMATION": "0",
        "SPS_MIN_SHARPENING_GAIN": str(float(sps_min_sharpening_gain)),
        "SPS_MAX_CAPPED_FAILED_RATE": str(float(sps_max_capped_failed_rate)),
        "SPS_ALPHA": str(float(sps_alpha)),
        "RUN_BACKEND_PREFLIGHT": "1" if run_backend_preflight else "0",
        "TRACKS": (
            "reasoning_gym_boxnet reasoning_gym_acre "
            "reasoning_gym_game_of_life_halting reasoning_gym_graph_color_n12"
        ),
        "ARCHIVE_TRAIN_TRACKS": (
            "reasoning_gym_family_relationships reasoning_gym_graph_color_n10 "
            "reasoning_gym_maze reasoning_gym_palindrome_generation "
            "reasoning_gym_letter_counting"
        ),
        "ARCHIVE_HELDOUT_TRACKS": (
            "reasoning_gym_boxnet reasoning_gym_acre "
            "reasoning_gym_game_of_life_halting reasoning_gym_graph_color_n12"
        ),
        "CONDITIONS": "base_greedy sps_only gepa_sps_fixed prorl_v2_greedy",
        "EVAL_START": "20",
        "EVAL_END": "70",
        "GEPA_DEV_START": "0",
        "GEPA_DEV_END": "50",
        "ROLLOUT_BUDGET": str(int(rollout_budget)),
        "ARCHIVE_SIZE": str(int(archive_size)),
        "MAX_METRIC_CALLS": str(int(max_metric_calls)),
        "MAX_NEW_TOKENS": str(int(max_new_tokens)),
        "VLLM_MAX_MODEL_LEN": str(int(vllm_max_model_len)),
        "NUM_SHARDS": str(int(num_shards)),
        "MAX_PARALLEL_CELLS": str(int(max_parallel_cells)),
        "PROFILE_MAX_PARALLEL_CELLS": str(int(max_parallel_cells)),
        "SPS_TOP_K": str(int(sps_top_k)),
        "SPS_CANDIDATE_POOL_SIZE": str(int(sps_candidate_pool_size)),
        "SPS_ROLLOUTS_PER_CANDIDATE": str(int(sps_rollouts_per_candidate)),
        "SPS_ROLLOUT_HORIZON": str(int(sps_rollout_horizon)),
        "SPS_VLLM_BATCH_SIZE": "64",
        "REFLECTION_PROVIDER": "local-hf",
        "INCLUDE_CANDIDATES": "0",
    }
    try:
        summary = _run_python_json(
            f"""
import csv
import json
import os
import subprocess
from pathlib import Path

env = os.environ.copy()
env.update({env_overrides!r})
run_root = Path({run_root!r})
run_root.mkdir(parents=True, exist_ok=True)
log_path = run_root / "modal_subprocess.log"
proc = subprocess.Popen(
    ["bash", "/proicl/scripts/run_experiment.sh", "h100"],
    cwd="/proicl",
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
)
with log_path.open("a", encoding="utf-8") as log:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = proc.wait(timeout={int(60 * 60 * 24 - 600)})
    except Exception:
        proc.kill()
        raise

def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def tail_text(path, limit=6000):
    path = Path(path)
    if not path.exists():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="replace")

run_dirs = sorted(run_root.glob("proicl_*"), key=lambda p: p.stat().st_mtime)
latest = run_dirs[-1] if run_dirs else run_root
full = latest / "full"
bundle_root = latest / "results_bundle"
sps_calibration_dir = (
    latest
    / "calibration"
    / {sps_calibration_dirname!r}
)
summary_paths = {{
    "launch_config": latest / "launch_config.json",
    "resource_probe": latest / "resource_probe.json",
    "backend_preflight": latest / "backend_preflight" / "backend_preflight.json",
    "runtime_profile": latest / "runtime_profile.json",
    "sps_calibration": sps_calibration_dir / "sps_calibration_summary.json",
    "confound_checks": bundle_root / "summary" / "confound_checks.csv",
    "metrics": bundle_root / "summary" / "metrics.csv",
    "decomposition": full / "analysis" / "proicl_decomposition.json",
}}
bundles = sorted(latest.rglob("results_bundle.tar.gz"))
payload = {{
    "passed": returncode == 0,
    "returncode": returncode,
    "run_root": str(run_root),
    "latest_run_dir": str(latest),
    "subprocess_log": str(log_path),
    "stdout_tail": tail_text(log_path),
    "stderr_tail": "",
    "bundle_paths": [str(path) for path in bundles],
    "launch_config": read_json(summary_paths["launch_config"]),
    "resource_probe": read_json(summary_paths["resource_probe"]),
    "backend_preflight": read_json(summary_paths["backend_preflight"]),
    "runtime_profile": read_json(summary_paths["runtime_profile"]),
    "sps_calibration_summary": read_json(summary_paths["sps_calibration"]),
    "confound_checks_path": str(summary_paths["confound_checks"]),
    "confound_checks": read_csv(summary_paths["confound_checks"]),
    "metrics_path": str(summary_paths["metrics"]),
    "decomposition_path": str(summary_paths["decomposition"]),
    "decomposition": read_json(summary_paths["decomposition"]),
}}
print("PROICL_JSON:" + json.dumps(payload, sort_keys=True))
if returncode != 0:
    raise SystemExit(returncode)
""",
            timeout=60 * 60 * 24,
        )
    finally:
        runs_vol.commit()
    print(f"run_heldout_goal_h100x4: {summary}")
    return summary
