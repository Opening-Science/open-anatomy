#!/usr/bin/env bash
# Runs every Blender self-check on synthetic meshes (scripts/blender.py
# self-check). Skips, and exits 0, when no Blender is found. Set BLENDER_BIN to
# choose a binary.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/blender.py" self-check
