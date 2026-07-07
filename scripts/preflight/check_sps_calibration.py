"""Validate a saved SPS sharpening calibration summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.path
    if path.is_dir():
        path = path / "sps_calibration_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_diff = payload.get("accuracy_abs_diff")
    diff = None if raw_diff is None else float(raw_diff)
    approximation_failed = payload.get("approximation_passed") is False
    diff_failed = diff is not None and diff > args.tolerance
    truncation_failed = payload.get("truncation_passed") is False
    if payload.get("passed") is not True or approximation_failed or diff_failed or truncation_failed:
        gain = payload.get("sps_sharpening_gain")
        min_gain = payload.get("min_sharpening_gain")
        raise SystemExit(
            f"SPS calibration failed at {path}: passed={payload.get('passed')} "
            f"baseline_condition={payload.get('baseline_condition')} "
            f"gate_metric={payload.get('gate_metric')} "
            f"baseline_accuracy={payload.get('baseline_accuracy')} "
            f"sps_accuracy={payload.get('sps_accuracy')} "
            f"accuracy_abs_diff={diff} tolerance={args.tolerance} "
            f"sps_sharpening_gain={gain} min_sharpening_gain={min_gain} "
            f"truncation_passed={payload.get('truncation_passed')} "
            f"baseline_capped_failed_rate={payload.get('baseline_capped_failed_rate')} "
            f"sps_capped_failed_rate={payload.get('sps_capped_failed_rate')} "
            f"max_capped_failed_rate={payload.get('max_capped_failed_rate')}"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "path": str(path),
                "gate_metric": payload.get("gate_metric"),
                "baseline_condition": payload.get("baseline_condition"),
                "baseline_accuracy": payload.get("baseline_accuracy"),
                "sps_accuracy": payload.get("sps_accuracy"),
                "baseline_candidate_accuracy": payload.get("baseline_candidate_accuracy"),
                "sps_candidate_accuracy": payload.get("sps_candidate_accuracy"),
                "truncation_passed": payload.get("truncation_passed"),
                "baseline_capped_failed_rate": payload.get("baseline_capped_failed_rate"),
                "sps_capped_failed_rate": payload.get("sps_capped_failed_rate"),
                "max_capped_failed_rate": payload.get("max_capped_failed_rate"),
                "accuracy_abs_diff": diff,
                "mcmc_approximation_skipped": payload.get("mcmc_approximation_skipped", False),
                "tolerance": args.tolerance,
                "sps_sharpening_gain": payload.get("sps_sharpening_gain"),
                "min_sharpening_gain": payload.get("min_sharpening_gain"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
