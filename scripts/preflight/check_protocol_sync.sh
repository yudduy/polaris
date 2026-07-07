#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROPOSAL="${ROOT_DIR}/PROPOSAL.md"
TODO="${ROOT_DIR}/TODO.md"
PROGRESS="${ROOT_DIR}/runs/progress.md"

missing=0

require_contains() {
  local file=$1
  local pattern=$2
  local label=$3

  if ! grep -qE "${pattern}" "${file}"; then
    echo "[drift-guard] MISSING (${label}) :: ${file}"
    missing=1
  else
    echo "[drift-guard] OK (${label}) :: ${file}"
  fi
}

for f in "${PROPOSAL}" "${TODO}" "${PROGRESS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[drift-guard] SKIP: release checkout has no protocol docs (${f})"
    exit 0
  fi
done

if [[ ${missing} -ne 0 ]]; then
  echo "[drift-guard] protocol sync check failed: required files missing"
  exit 1
fi

require_contains "${PROPOSAL}" "^Protocol version: POLARIS-v3\\.1" "proposal: active protocol version"
require_contains "${PROPOSAL}" "^## 13\\.1 R5 Infrastructure Contract" "proposal: R5 infrastructure contract"
require_contains "${TODO}" "^## Protocol Drift Guard" "TODO: drift guard section"
require_contains "${TODO}" "^## Team Roles and Edit Surfaces" "TODO: edit-surface ownership"
require_contains "${TODO}" "^Last protocol sync:" "TODO: explicit last sync marker"
require_contains "${TODO}" "^## POLARIS-v3\\.1 R5 Infrastructure Contract" "TODO: R5 infrastructure contract"
require_contains "${PROGRESS}" "^## Protocol Integrity Contract" "progress: integrity block"
require_contains "${PROGRESS}" "^## POLARIS-v3\\.1 R5 Infrastructure Contract" "progress: R5 decision block"
require_contains "${PROGRESS}" "^### What remains in this section" "progress: structured checkpoint format"

if [[ ${missing} -ne 0 ]]; then
  echo "[drift-guard] protocol sync check failed"
  exit 1
fi

echo "[drift-guard] Protocol documents are present and have required coordination scaffolding."
