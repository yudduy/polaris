from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polaris.infra.preflight import RunKind
from polaris.io.artifact_audit import PRODUCTION_ARTIFACTS, SEVEN_ARTIFACTS


REQUIRED_PRODUCTION_ARTIFACTS = SEVEN_ARTIFACTS + PRODUCTION_ARTIFACTS
DEFAULT_RESOURCE_PROFILE_PAYLOAD = {
    "profiles": [
        {
            "key": "farmshare_l40_free",
            "label": "FarmShare 4x L40S free shards",
            "run_kind": "farmshare",
            "role": "free bulk sampling and queued Phase 0/Phase 1 continuation",
            "gpu_model": "l40s",
            "gpu_count": 4,
            "max_concurrent_jobs": 4,
            "allowed_phases": ["phase0", "phase1", "phase2_low"],
            "free": True,
            "fallback_only": False,
            "hourly_rate_dollars": 0.0,
            "initial_spend_cap_dollars": 0.0,
            "notes": "Use four independent one-GPU Slurm jobs; no tensor parallelism.",
        },
        {
            "key": "flow_a100_weekend",
            "label": "Mithril/Flow weekend 4x A100 80GB",
            "run_kind": "flow",
            "role": "weekend accelerator for Phase 1 catch-up and Phase 2 MCMC/GEPA/memory",
            "gpu_model": "a100-80gb.sxm",
            "gpu_count": 4,
            "max_concurrent_jobs": 4,
            "allowed_phases": ["phase1", "phase2", "phase3_debug"],
            "free": False,
            "fallback_only": False,
            "max_bid_dollars_per_gpu_hour": 0.025,
            "initial_spend_cap_dollars": 10.0,
            "notes": "Preferred 4x total price <= $0.10/hr; check live Flow pricing before launch.",
        },
        {
            "key": "modal_burst",
            "label": "Modal burst/debug",
            "run_kind": "modal",
            "role": "Phase 3 mechanistic debugging and short vLLM/HF parity smokes",
            "gpu_model": "l40s-or-a100",
            "gpu_count": 1,
            "max_concurrent_jobs": 1,
            "allowed_phases": ["phase3", "debug"],
            "free": False,
            "fallback_only": False,
            "initial_spend_cap_dollars": 25.0,
            "notes": "Use only for bursty debug sessions; keep scale-to-zero discipline.",
        },
        {
            "key": "cloudrift_fallback",
            "label": "CloudRift RTX 4090 fallback",
            "run_kind": "cloudrift",
            "role": "fallback if FarmShare and Flow block",
            "gpu_model": "rtx4090",
            "gpu_count": 1,
            "max_concurrent_jobs": 1,
            "allowed_phases": ["probe", "phase1", "phase2_low"],
            "free": False,
            "fallback_only": True,
            "hourly_rate_dollars": 0.25,
            "initial_spend_cap_dollars": 25.0,
            "notes": "Use explicit launch-time UI rate in costs.json.",
        },
    ]
}


@dataclass(frozen=True)
class ResourceProfile:
    key: str
    label: str
    run_kind: RunKind
    role: str
    gpu_model: str
    gpu_count: int
    max_concurrent_jobs: int
    allowed_phases: tuple[str, ...]
    free: bool
    fallback_only: bool
    initial_spend_cap_dollars: float
    hourly_rate_dollars: float | None = None
    max_bid_dollars_per_gpu_hour: float | None = None
    notes: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_phases"] = list(self.allowed_phases)
        return payload


def _profile_from_dict(payload: dict[str, Any]) -> ResourceProfile:
    return ResourceProfile(
        key=str(payload["key"]),
        label=str(payload["label"]),
        run_kind=payload["run_kind"],
        role=str(payload["role"]),
        gpu_model=str(payload["gpu_model"]),
        gpu_count=int(payload["gpu_count"]),
        max_concurrent_jobs=int(payload["max_concurrent_jobs"]),
        allowed_phases=tuple(str(item) for item in payload["allowed_phases"]),
        free=bool(payload["free"]),
        fallback_only=bool(payload.get("fallback_only", False)),
        initial_spend_cap_dollars=float(payload["initial_spend_cap_dollars"]),
        hourly_rate_dollars=(
            float(payload["hourly_rate_dollars"])
            if payload.get("hourly_rate_dollars") is not None
            else None
        ),
        max_bid_dollars_per_gpu_hour=(
            float(payload["max_bid_dollars_per_gpu_hour"])
            if payload.get("max_bid_dollars_per_gpu_hour") is not None
            else None
        ),
        notes=str(payload.get("notes", "")),
    )


def validate_resource_profile(profile: ResourceProfile) -> None:
    if not profile.key:
        raise ValueError("profile key is required")
    if profile.gpu_count <= 0:
        raise ValueError(f"{profile.key}: gpu_count must be positive")
    if profile.max_concurrent_jobs <= 0:
        raise ValueError(f"{profile.key}: max_concurrent_jobs must be positive")
    if not profile.allowed_phases:
        raise ValueError(f"{profile.key}: allowed_phases is required")
    if profile.initial_spend_cap_dollars < 0:
        raise ValueError(f"{profile.key}: initial_spend_cap_dollars must be non-negative")
    if profile.hourly_rate_dollars is not None and profile.hourly_rate_dollars < 0:
        raise ValueError(f"{profile.key}: hourly_rate_dollars must be non-negative")
    if (
        profile.max_bid_dollars_per_gpu_hour is not None
        and profile.max_bid_dollars_per_gpu_hour < 0
    ):
        raise ValueError(f"{profile.key}: max_bid_dollars_per_gpu_hour must be non-negative")
    if profile.free and profile.initial_spend_cap_dollars != 0.0:
        raise ValueError(f"{profile.key}: free profiles must have zero spend cap")
    if profile.free and profile.run_kind != "farmshare":
        raise ValueError(f"{profile.key}: only FarmShare is marked free")


def load_resource_profiles(
    path: Path | None = None,
) -> dict[str, ResourceProfile]:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path is not None
        else DEFAULT_RESOURCE_PROFILE_PAYLOAD
    )
    profiles: dict[str, ResourceProfile] = {}
    for raw in payload.get("profiles", []):
        profile = _profile_from_dict(raw)
        if profile.key in profiles:
            raise ValueError(f"duplicate resource profile: {profile.key}")
        validate_resource_profile(profile)
        profiles[profile.key] = profile
    if not profiles:
        raise ValueError(f"no resource profiles found in {path or 'built-in defaults'}")
    return profiles


def get_resource_profile(
    key: str,
    *,
    path: Path | None = None,
) -> ResourceProfile:
    profiles = load_resource_profiles(path)
    try:
        return profiles[key]
    except KeyError as exc:
        raise ValueError(f"unknown resource profile: {key}") from exc


def validate_profile_cost(
    profile: ResourceProfile,
    *,
    estimated_dollar_cost: float,
) -> dict[str, Any]:
    if estimated_dollar_cost < 0:
        raise ValueError("estimated_dollar_cost must be non-negative")
    if profile.free and estimated_dollar_cost != 0.0:
        raise ValueError("FarmShare profile must remain zero-cost")
    if estimated_dollar_cost > profile.initial_spend_cap_dollars:
        raise ValueError(
            "estimated_dollar_cost exceeds profile initial_spend_cap_dollars "
            f"({estimated_dollar_cost} > {profile.initial_spend_cap_dollars})"
        )
    return {
        "passed": True,
        "profile": profile.key,
        "estimated_dollar_cost": estimated_dollar_cost,
        "initial_spend_cap_dollars": profile.initial_spend_cap_dollars,
    }


def artifact_contract() -> tuple[str, ...]:
    return REQUIRED_PRODUCTION_ARTIFACTS
