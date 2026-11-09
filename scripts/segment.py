"""
Segments the v0 structures in one CT with TotalSegmentator's Apache-2.0 "total"
task, and scores the result against the dataset's reference segmentations.

  python scripts/segment.py --case s1086 [--device gpu|cpu] [--fast]

Reads data/cases/<case>/ct.nii.gz, written by fetch.py. Writes one mask per
structure to data/work/<case>/seg/<label>.nii.gz, and seg.json with each
structure's voxel count and Dice score against the reference segmentation.
Only tasks in structures.ALLOWED_TASKS run; TotalSegmentator's licensed tasks
are never used. GPU inference is not bit-exact, so masks can differ by a few
boundary voxels between machines; the Dice scores show by how much.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import ALLOWED_TASKS, STRUCTURES  # noqa: E402

TASK = "total"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / total if total else 1.0


def mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--fast", action="store_true", help="3 mm model, quicker and coarser")
    args = parser.parse_args()
    assert TASK in ALLOWED_TASKS

    from totalsegmentator.python_api import totalsegmentator

    case_dir = ROOT / "data" / "cases" / args.case
    out = ROOT / "data" / "work" / args.case / "seg"
    out.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        case_dir / "ct.nii.gz",
        out,
        task=TASK,
        roi_subset=[s.label for s in STRUCTURES],
        fast=args.fast,
        device=args.device,
        quiet=True,
    )

    report = {"case": args.case, "task": TASK, "fast": args.fast, "structures": {}}
    for s in STRUCTURES:
        found = mask(out / f"{s.label}.nii.gz")
        reference = mask(case_dir / "segmentations" / f"{s.label}.nii.gz")
        report["structures"][s.label] = {
            "uberon": s.uberon,
            "voxels": int(found.sum()),
            "dice": round(dice(found, reference), 4),
        }
        print(f"{s.label:20} {int(found.sum()):>9} voxels  dice {dice(found, reference):.3f}")
    (out.parent / "seg.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
