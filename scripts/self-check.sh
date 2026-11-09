#!/usr/bin/env bash
# Runs scripts/self_check.py in headless Blender. Skips, and exits 0, when no
# Blender is found. Set BLENDER to choose a binary.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blender="${BLENDER:-$(command -v blender || true)}"

if [[ -z "$blender" || ! -x "$blender" ]]; then
  echo "self-check skipped: Blender not found (set BLENDER to its path)"
  exit 0
fi

exec "$blender" --background --factory-startup --python-exit-code 1 \
  --python "$here/self_check.py"
