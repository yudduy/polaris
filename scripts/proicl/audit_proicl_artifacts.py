"""Audit ProICL run artifacts against a saved plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()

    from polaris.proicl.artifact_audit import write_proicl_artifact_audit

    report = write_proicl_artifact_audit(args.plan, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_passed and not report.get("passed"):
        raise SystemExit("ProICL artifact audit failed")


if __name__ == "__main__":
    main()
