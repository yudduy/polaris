from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, MutableMapping


LOCAL_HF_REFLECTION_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"


def load_env_file(path: Path, *, env: MutableMapping[str, str] | None = None) -> None:
    target = env if env is not None else os.environ
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = value


@dataclass(frozen=True)
class LocalHFReflectionConfig:
    model_id: str = LOCAL_HF_REFLECTION_MODEL_DEFAULT
    revision: str | None = None
    temperature: float = 0.7
    max_new_tokens: int = 1024
    local_files_only: bool = False
    device: str = "cuda"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


class LocalHFReflectionLM:
    """Local Hugging Face reflection model for GEPA without paid API calls."""

    def __init__(self, config: LocalHFReflectionConfig) -> None:
        import torch
        import transformers

        self.config = config
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._total_cost = 0.0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if callable(getattr(self.tokenizer, "apply_chat_template", None)):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = str(prompt)
        input_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)
        started_tokens = int(input_ids["input_ids"].shape[-1])
        with self.torch.no_grad():
            output = self.model.generate(
                **input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][started_tokens:].detach().to("cpu")
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self._total_tokens_in += started_tokens
        self._total_tokens_out += int(generated.numel())
        return text


def make_local_hf_reflection_lm(config: LocalHFReflectionConfig) -> LocalHFReflectionLM:
    return LocalHFReflectionLM(config)


def reflection_manifest(
    *,
    provider: str,
    config: LocalHFReflectionConfig | None,
    status: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "provider": provider,
        "status": status,
        "usage": usage or {},
    }
    if config is not None:
        payload["config"] = config.to_manifest()
    return payload
