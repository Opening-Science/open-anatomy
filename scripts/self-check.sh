#!/usr/bin/env bash
# Runs scripts/self_check.py in headless Blender. Set BLENDER to choose a
# binary. Without Blender it skips and exits 0, except in CI (CI=true), where a
# missing Blender is an error so the check cannot pass without running.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blender="${BLENDER:-$(command -v blender || true)}"

if [[ -z "$blender" || ! -x "$blender" ]]; then
  if [[ "${CI:-}" == "true" ]]; then
    echo "self-check failed: Blender not found in CI (set BLENDER to its path)" >&2
    exit 1
  fi
  echo "self-check skipped: Blender not found (set BLENDER to its path)"
  exit 0
fi

exec "$blender" --background --factory-startup --python-exit-code 1 \
  --python "$here/self_check.py"
