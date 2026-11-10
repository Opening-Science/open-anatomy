#!/usr/bin/env bash
# Runs every Blender self-check on synthetic meshes (scripts/blender.py
# self-check). Set BLENDER_BIN, or BLENDER, to choose a binary. Without Blender
# it skips and exits 0, except in CI (CI=true), where a missing Blender is an
# error so the check cannot pass without running.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BLENDER_BIN="${BLENDER_BIN:-${BLENDER:-}}"
blender="${BLENDER_BIN:-$(command -v blender || true)}"

if [[ "${CI:-}" == "true" && (-z "$blender" || ! -x "$blender") ]]; then
  echo "self-check failed: Blender not found in CI (set BLENDER_BIN to its path)" >&2
  exit 1
fi

exec python3 "$here/blender.py" self-check
