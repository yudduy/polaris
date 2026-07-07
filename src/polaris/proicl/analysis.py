from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_REQUIRED_KEYS = (
    "base",
    "prorl_v2_greedy",
)

_DIR_TO_ACCURACY_KEY = {
    "base_greedy": "base",
    "bon_temp1": "bon_temp1",
    "sps_only": "sps_only",
    "mcmc_only": "mcmc_only",
    "mixed_alpha_mcmc": "mixed_alpha_mcmc",
    "fork_search": "fork_search",
    "gepa_only": "gepa_only",
    "gepa_sps_fixed": "gepa_sps_fixed",
    "gepa_mcmc": "gepa_mcmc",
    "gepa_mcmc_repair": "gepa_mcmc_repair",
    "gepa_mcmc_fork_repair": "gepa_mcmc_fork_repair",
    "gepa_mcmc_fork_repair_memory": "gepa_mcmc_fork_repair_memory",
    "gepa_mcmc_memory": "gepa_mcmc_memory",
    "prorl_v2_greedy": "prorl_v2_greedy",
}


def _accuracy(source: Mapping[str, float], key: str) -> float:
    try:
        return float(source[key])
    except KeyError as exc:
        raise ValueError(f"missing ProICL accuracy: {key}") from exc


def _rf(*, base: float, recovered: float, trained: float) -> float | None:
    denom = trained - base
    if denom <= 0.0:
        return None
    return max(0.0, min(1.0, (recovered - base) / denom))


def compute_proicl_decomposition(accuracies: Mapping[str, float]) -> dict[str, Any]:
    """Compute the ProICL fast-weight decomposition from condition accuracies."""

    missing = [key for key in _REQUIRED_KEYS if key not in accuracies]
    if missing:
        raise ValueError("missing ProICL accuracies: " + ", ".join(missing))

    base = _accuracy(accuracies, "base")
    bon = float(accuracies.get("bon_temp1", base))
    sps = float(accuracies.get("sps_only", accuracies.get("mcmc_only", bon)))
    mcmc = float(accuracies.get("mcmc_only", accuracies.get("sps_only", bon)))
    mixed = float(accuracies.get("mixed_alpha_mcmc", mcmc))
    fork = float(accuracies.get("fork_search", bon))
    gepa = float(accuracies.get("gepa_only", bon))
    gepa_sps_fixed = float(
        accuracies.get("gepa_sps_fixed", accuracies.get("gepa_mcmc", max(gepa, sps)))
    )
    gepa_mcmc = float(accuracies.get("gepa_mcmc", accuracies.get("gepa_sps_fixed", max(gepa, mixed))))
    repair = float(accuracies.get("gepa_mcmc_repair", gepa_mcmc))
    fork_repair = float(accuracies.get("gepa_mcmc_fork_repair", max(fork, repair)))
    full_memory = float(
        accuracies.get(
            "gepa_mcmc_fork_repair_memory",
            accuracies.get("gepa_mcmc_memory", fork_repair),
        )
    )
    preliminary_memory = float(accuracies.get("gepa_mcmc_memory", full_memory))
    prorl = _accuracy(accuracies, "prorl_v2_greedy")

    rf_denominator = prorl - base
    rf_valid = rf_denominator > 0.0
    confound_delta = gepa_sps_fixed - sps
    confound_status = (
        "gepa_archive_sps_above_base_sps"
        if confound_delta > 0.0
        else "no_gepa_archive_sps_gain_over_base_sps"
    )
    report = {
        "A_base": base,
        "A_bon": bon,
        "A_sps": sps,
        "A_mcmc": mcmc,
        "A_mixed": mixed,
        "A_fork": fork,
        "A_gepa": gepa,
        "A_gepa_sps_fixed": gepa_sps_fixed,
        "A_gepa_mcmc": gepa_mcmc,
        "A_repair": repair,
        "A_fork_repair": fork_repair,
        "A_full_memory": full_memory,
        "A_memory": preliminary_memory,
        "A_prorl_v2": prorl,
        "rf_denominator": rf_denominator,
        "rf_valid": rf_valid,
        "RF_bon": _rf(base=base, recovered=bon, trained=prorl),
        "RF_sps": _rf(base=base, recovered=sps, trained=prorl),
        "RF_mcmc": _rf(base=base, recovered=mcmc, trained=prorl),
        "RF_mixed": _rf(base=base, recovered=mixed, trained=prorl),
        "RF_fork": _rf(base=base, recovered=fork, trained=prorl),
        "RF_gepa": _rf(base=base, recovered=gepa, trained=prorl),
        "RF_gepa_sps_fixed": _rf(base=base, recovered=gepa_sps_fixed, trained=prorl),
        "RF_gepa_mcmc": _rf(base=base, recovered=gepa_mcmc, trained=prorl),
        "RF_repair": _rf(base=base, recovered=repair, trained=prorl),
        "RF_fork_repair": _rf(base=base, recovered=fork_repair, trained=prorl),
        "RF_full_memory": _rf(base=base, recovered=full_memory, trained=prorl),
        "RF_memory": _rf(base=base, recovered=preliminary_memory, trained=prorl),
        "slow_weight_residual": prorl - full_memory,
        "sharpening_gain": mcmc - bon,
        "sps_gain": sps - bon,
        "mixed_alpha_gain": mixed - mcmc,
        "fork_gain": fork - bon,
        "prompt_archive_gain": gepa - bon,
        "repair_gain": repair - gepa_mcmc,
        "fork_repair_gain": fork_repair - repair,
        "memory_gain": full_memory - fork_repair,
        "preliminary_memory_gain": preliminary_memory - gepa_mcmc,
        "discovery_gain": gepa - mcmc,
        "composition_gain": gepa_mcmc - max(mixed, gepa),
        "sps_composition_gain": gepa_sps_fixed - max(sps, gepa),
        "confound_delta_gepa_archive_sps_vs_base_sps": confound_delta,
        "confound_check_base_sps_vs_gepa_archive_sps": {
            "base_sps_accuracy": sps,
            "gepa_archive_sps_accuracy": gepa_sps_fixed,
            "delta": confound_delta,
            "status": confound_status,
        },
    }
    if not rf_valid:
        report["analysis_warning"] = (
            "prorl_v2_greedy accuracy did not exceed base_greedy on this slice; "
            "RF is undefined for this track/smoke slice."
        )
    return report


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.6f}"


def write_proicl_decomposition(
    *,
    accuracies: Mapping[str, float],
    out_dir: Path,
) -> dict[str, Any]:
    report = compute_proicl_decomposition(accuracies)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proicl_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ProICL decomposition",
        "",
        f"- A_base: {report['A_base']:.6f}",
        f"- A_bon: {report['A_bon']:.6f}",
        f"- A_sps: {report['A_sps']:.6f}",
        f"- A_mcmc: {report['A_mcmc']:.6f}",
        f"- A_mixed: {report['A_mixed']:.6f}",
        f"- A_fork: {report['A_fork']:.6f}",
        f"- A_gepa: {report['A_gepa']:.6f}",
        f"- A_gepa_sps_fixed: {report['A_gepa_sps_fixed']:.6f}",
        f"- A_gepa_mcmc: {report['A_gepa_mcmc']:.6f}",
        f"- A_repair: {report['A_repair']:.6f}",
        f"- A_fork_repair: {report['A_fork_repair']:.6f}",
        f"- A_full_memory: {report['A_full_memory']:.6f}",
        f"- A_memory: {report['A_memory']:.6f}",
        f"- A_prorl_v2: {report['A_prorl_v2']:.6f}",
        "",
        f"- rf_denominator: {report['rf_denominator']:.6f}",
        f"- rf_valid: {report['rf_valid']}",
        f"- RF_bon: {_fmt(report['RF_bon'])}",
        f"- RF_sps: {_fmt(report['RF_sps'])}",
        f"- RF_mcmc: {_fmt(report['RF_mcmc'])}",
        f"- RF_mixed: {_fmt(report['RF_mixed'])}",
        f"- RF_fork: {_fmt(report['RF_fork'])}",
        f"- RF_gepa: {_fmt(report['RF_gepa'])}",
        f"- RF_gepa_sps_fixed: {_fmt(report['RF_gepa_sps_fixed'])}",
        f"- RF_gepa_mcmc: {_fmt(report['RF_gepa_mcmc'])}",
        f"- RF_repair: {_fmt(report['RF_repair'])}",
        f"- RF_fork_repair: {_fmt(report['RF_fork_repair'])}",
        f"- RF_full_memory: {_fmt(report['RF_full_memory'])}",
        f"- RF_memory: {_fmt(report['RF_memory'])}",
        "",
        f"- slow_weight_residual: {report['slow_weight_residual']:.6f}",
        f"- sharpening_gain: {report['sharpening_gain']:.6f}",
        f"- sps_gain: {report['sps_gain']:.6f}",
        f"- mixed_alpha_gain: {report['mixed_alpha_gain']:.6f}",
        f"- fork_gain: {report['fork_gain']:.6f}",
        f"- prompt_archive_gain: {report['prompt_archive_gain']:.6f}",
        f"- repair_gain: {report['repair_gain']:.6f}",
        f"- fork_repair_gain: {report['fork_repair_gain']:.6f}",
        f"- memory_gain: {report['memory_gain']:.6f}",
        f"- discovery_gain: {report['discovery_gain']:.6f}",
        f"- composition_gain: {report['composition_gain']:.6f}",
        f"- sps_composition_gain: {report['sps_composition_gain']:.6f}",
        f"- confound_delta_gepa_archive_sps_vs_base_sps: {report['confound_delta_gepa_archive_sps_vs_base_sps']:.6f}",
        "- confound_status_base_sps_vs_gepa_archive_sps: "
        f"{report['confound_check_base_sps_vs_gepa_archive_sps']['status']}",
    ]
    if report.get("analysis_warning"):
        lines.extend(["", f"- warning: {report['analysis_warning']}"])
    (out_dir / "proicl_decomposition.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def collect_track_accuracies(*, root: Path, track: str) -> dict[str, float]:
    """Collect weighted condition accuracies from ProICL shard artifacts."""

    track_dir = root / "runs" / track
    totals: dict[str, tuple[float, int]] = {}
    for dirname, key in _DIR_TO_ACCURACY_KEY.items():
        condition_dir = track_dir / dirname
        if not condition_dir.exists():
            continue
        weighted = 0.0
        n_total = 0
        for metrics_path in sorted(condition_dir.glob("shard-*/metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            n = int(metrics.get("n_problems", 0))
            weighted += float(metrics.get("accuracy", 0.0)) * n
            n_total += n
        if n_total > 0:
            totals[key] = (weighted, n_total)
    return {key: weighted / n for key, (weighted, n) in totals.items()}


def write_proicl_decomposition_by_track(
    *,
    root: Path,
    tracks: list[str] | tuple[str, ...],
    out_dir: Path,
) -> dict[str, Any]:
    """Write per-track ProICL decomposition from canonical run artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for track in tracks:
        accuracies = collect_track_accuracies(root=root, track=track)
        report = compute_proicl_decomposition(accuracies)
        reports[track] = {"accuracies": accuracies, "decomposition": report}

    (out_dir / "proicl_decomposition.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# ProICL decomposition", ""]
    for track, payload in reports.items():
        report = payload["decomposition"]
        lines.extend(
            [
                f"## {track}",
                "",
                f"- A_base: {report['A_base']:.6f}",
                f"- A_bon: {report['A_bon']:.6f}",
                f"- A_sps: {report['A_sps']:.6f}",
                f"- A_mcmc: {report['A_mcmc']:.6f}",
                f"- A_mixed: {report['A_mixed']:.6f}",
                f"- A_fork: {report['A_fork']:.6f}",
                f"- A_gepa: {report['A_gepa']:.6f}",
                f"- A_gepa_sps_fixed: {report['A_gepa_sps_fixed']:.6f}",
                f"- A_gepa_mcmc: {report['A_gepa_mcmc']:.6f}",
                f"- A_repair: {report['A_repair']:.6f}",
                f"- A_fork_repair: {report['A_fork_repair']:.6f}",
                f"- A_full_memory: {report['A_full_memory']:.6f}",
                f"- A_memory: {report['A_memory']:.6f}",
                f"- A_prorl_v2: {report['A_prorl_v2']:.6f}",
                f"- rf_denominator: {report['rf_denominator']:.6f}",
                f"- rf_valid: {report['rf_valid']}",
                f"- RF_bon: {_fmt(report['RF_bon'])}",
                f"- RF_sps: {_fmt(report['RF_sps'])}",
                f"- RF_mcmc: {_fmt(report['RF_mcmc'])}",
                f"- RF_mixed: {_fmt(report['RF_mixed'])}",
                f"- RF_fork: {_fmt(report['RF_fork'])}",
                f"- RF_gepa: {_fmt(report['RF_gepa'])}",
                f"- RF_gepa_sps_fixed: {_fmt(report['RF_gepa_sps_fixed'])}",
                f"- RF_gepa_mcmc: {_fmt(report['RF_gepa_mcmc'])}",
                f"- RF_repair: {_fmt(report['RF_repair'])}",
                f"- RF_fork_repair: {_fmt(report['RF_fork_repair'])}",
                f"- RF_full_memory: {_fmt(report['RF_full_memory'])}",
                f"- RF_memory: {_fmt(report['RF_memory'])}",
                f"- slow_weight_residual: {report['slow_weight_residual']:.6f}",
                f"- sharpening_gain: {report['sharpening_gain']:.6f}",
                f"- sps_gain: {report['sps_gain']:.6f}",
                f"- mixed_alpha_gain: {report['mixed_alpha_gain']:.6f}",
                f"- fork_gain: {report['fork_gain']:.6f}",
                f"- prompt_archive_gain: {report['prompt_archive_gain']:.6f}",
                f"- repair_gain: {report['repair_gain']:.6f}",
                f"- fork_repair_gain: {report['fork_repair_gain']:.6f}",
                f"- memory_gain: {report['memory_gain']:.6f}",
                f"- discovery_gain: {report['discovery_gain']:.6f}",
                f"- composition_gain: {report['composition_gain']:.6f}",
                f"- sps_composition_gain: {report['sps_composition_gain']:.6f}",
                f"- confound_delta_gepa_archive_sps_vs_base_sps: {report['confound_delta_gepa_archive_sps_vs_base_sps']:.6f}",
                "- confound_status_base_sps_vs_gepa_archive_sps: "
                f"{report['confound_check_base_sps_vs_gepa_archive_sps']['status']}",
                *(
                    [f"- warning: {report['analysis_warning']}"]
                    if report.get("analysis_warning")
                    else []
                ),
                "",
            ]
        )
    (out_dir / "proicl_decomposition.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return reports
