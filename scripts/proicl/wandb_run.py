#!/usr/bin/env python
"""Start or finalize optional W&B tracking for a ProICL run.

This script is intentionally best-effort. Experiment execution should never
fail just because W&B is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SECRET_KEY_NAMES = {"WANDB_API_KEY"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start")
    start.add_argument("--run-root", type=Path, required=True)
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--resource-probe", type=Path, default=None)
    start.add_argument("--env-file", type=Path, default=Path(".env"))

    finish = sub.add_parser("finish")
    finish.add_argument("--run-root", type=Path, required=True)
    finish.add_argument("--bundle", type=Path, required=True)
    finish.add_argument("--summary", type=Path, default=None)
    finish.add_argument("--events", type=Path, default=None)
    finish.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wandb_key_available() -> bool:
    return bool(os.environ.get("WANDB_API_KEY")) and os.environ.get("WANDB_MODE") != "disabled"


def _metadata_path(run_root: Path) -> Path:
    return run_root / "wandb.json"


def _base_metadata(*, status: str, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "proicl_wandb.v1",
        "enabled": enabled,
        "status": status,
        "project": os.environ.get("WANDB_PROJECT", "proicl"),
        "entity": os.environ.get("WANDB_ENTITY"),
        "group": os.environ.get("WANDB_RUN_GROUP"),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "api_key_saved": False,
    }


def _start(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    run_root = args.run_root.resolve()
    config = _read_json(args.config)
    resource_probe = _read_json(args.resource_probe)

    if not _wandb_key_available():
        payload = _base_metadata(status="disabled_no_api_key", enabled=False)
        _write_json(_metadata_path(run_root), payload)
        print(json.dumps(payload, sort_keys=True))
        return

    try:
        import wandb  # type: ignore
    except Exception as exc:
        payload = _base_metadata(status="disabled_import_failed", enabled=False)
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(_metadata_path(run_root), payload)
        print(json.dumps(payload, sort_keys=True))
        return

    payload = _base_metadata(status="starting", enabled=True)
    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        run = wandb.init(
            project=payload["project"],
            entity=payload["entity"] or None,
            group=payload["group"] or config.get("run_tag"),
            name=os.environ.get("WANDB_NAME") or run_root.name,
            job_type="proicl-experiment",
            config={
                "launch_config": config,
                "resource_probe": resource_probe,
            },
        )
        if run is not None:
            wandb.log(
                {
                    "proicl/gpu_count": config.get("gpu_count"),
                    "proicl/max_parallel_cells": config.get("max_parallel_cells"),
                    "proicl/smoke_max_parallel_cells": config.get("smoke_max_parallel_cells"),
                    "proicl/host_memory_mib": config.get("host_memory_mib"),
                }
            )
            payload.update(
                {
                    "status": "started",
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_url": run.get_url(),
                }
            )
        wandb.finish()
    except Exception as exc:
        payload.update(
            {
                "enabled": False,
                "status": "disabled_start_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    _write_json(_metadata_path(run_root), payload)
    print(json.dumps(payload, sort_keys=True))


def _finish(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    run_root = args.run_root.resolve()
    meta_path = _metadata_path(run_root)
    payload = _read_json(meta_path)
    if not payload:
        payload = _base_metadata(status="missing_start_metadata", enabled=False)
    if not payload.get("enabled") or not payload.get("run_id") or not _wandb_key_available():
        payload["finalize_status"] = "skipped"
        payload["result_bundle"] = str(args.bundle.resolve())
        _write_json(meta_path, payload)
        print(json.dumps(payload, sort_keys=True))
        return

    try:
        import wandb  # type: ignore
    except Exception as exc:
        payload["finalize_status"] = "import_failed"
        payload["finalize_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(meta_path, payload)
        print(json.dumps(payload, sort_keys=True))
        return

    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        run = wandb.init(
            project=payload.get("project") or "proicl",
            entity=payload.get("entity") or None,
            id=payload["run_id"],
            resume="allow",
            job_type="proicl-experiment",
        )
        if args.summary and args.summary.exists():
            summary = _read_json(args.summary)
            run.summary["aggregate_summary"] = summary
        if args.events and args.events.exists():
            artifact = wandb.Artifact(f"{run.name}-events", type="proicl-events")
            artifact.add_file(str(args.events.resolve()))
            run.log_artifact(artifact)
        if args.bundle.exists():
            artifact = wandb.Artifact(f"{run.name}-results", type="proicl-results")
            artifact.add_file(str(args.bundle.resolve()))
            run.log_artifact(artifact)
            run.summary["result_bundle"] = str(args.bundle.resolve())
        payload.update(
            {
                "finalize_status": "finished",
                "result_bundle": str(args.bundle.resolve()),
                "run_url": run.get_url(),
            }
        )
        wandb.finish()
    except Exception as exc:
        payload["finalize_status"] = "failed"
        payload["finalize_error"] = f"{type(exc).__name__}: {exc}"
    _write_json(meta_path, payload)
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    args = _parse_args()
    if args.cmd == "start":
        _start(args)
    elif args.cmd == "finish":
        _finish(args)


if __name__ == "__main__":
    main()
