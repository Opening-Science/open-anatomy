"""
Segments a set of structures (scripts/structures.py) in one CT with
TotalSegmentator's Apache-2.0 "total" task, and scores the result against the
dataset's reference segmentations.

  python scripts/segment.py --case s1086 [--set organs|reference] [--device gpu|cpu] [--fast]

Reads data/cases/<case>/ct.nii.gz, written by fetch.py. Writes one mask per
structure to data/work/<case>/seg/<label>.nii.gz, and seg.json with each
structure's voxel count and Dice score against the reference segmentation (null
where the dataset has none). For --set reference it also runs the Apache-2.0
"body" task, whose trunk outline assemble.py measures, and also segments the
skull and brain, which imaging.py's head guard needs.
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

from structures import ALLOWED_TASKS, HEAD_GUARD, SETS  # noqa: E402

TASK = "total"
BODY_TASK = "body"  # trunk outline, for the reference set only


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / total if total else 1.0


def mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.Nifti1Image.from_filename(path).dataobj) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--fast", action="store_true", help="3 mm model, quicker and coarser")
    parser.add_argument("--set", choices=SETS, default="organs", help="structures to segment")
    args = parser.parse_args()
    structures = SETS[args.set]
    for task in (TASK, BODY_TASK):
        if task not in ALLOWED_TASKS:
            sys.exit(
                f'TotalSegmentator task "{task}" is not Apache-2.0; only these tasks may run: '
                + ", ".join(sorted(ALLOWED_TASKS))
            )

    # TotalSegmentator is in the pipeline extra only, which the type check does not install.
    from totalsegmentator.python_api import (  # pyright: ignore[reportMissingImports]
        totalsegmentator,
    )

    case_dir = ROOT / "data" / "cases" / args.case
    out = ROOT / "data" / "work" / args.case / "seg"
    out.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        case_dir / "ct.nii.gz",
        out,
        task=TASK,
        roi_subset=[s.label for s in structures]
        + (list(HEAD_GUARD) if args.set == "reference" else []),
        fast=args.fast,
        device=args.device,
        quiet=True,
    )

    if args.set == "reference":
        # The trunk outline (body_trunc) is used only to measure the body for the
        # fit in blender/assemble.py; it is never exported.
        totalsegmentator(
            case_dir / "ct.nii.gz", out, task=BODY_TASK, fast=args.fast, device=args.device,
            quiet=True,
        )  # fmt: skip

    report = {"case": args.case, "task": TASK, "fast": args.fast, "set": args.set, "structures": {}}
    for s in structures:
        found = mask(out / f"{s.label}.nii.gz")
        reference = mask(case_dir / "segmentations" / f"{s.label}.nii.gz")
        # Some cases lack a reference mask for a structure (s0903 has no sacrum).
        score = round(dice(found, reference), 4) if reference.any() else None
        report["structures"][s.label] = {"term": s.term, "voxels": int(found.sum()), "dice": score}
        shown = "n/a (no reference)" if score is None else f"{score:.3f}"
        print(f"{s.label:20} {int(found.sum()):>9} voxels  dice {shown}")
    (out.parent / "seg.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
