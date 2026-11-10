"""
Segments a set of structures (scripts/structures.py) in one CT with
TotalSegmentator's Apache-2.0 tasks, and scores the result against the
dataset's reference segmentations.

  python scripts/segment.py --case s1086 [--set organs|reference|skeleton|<system>|body]
      [--device gpu|cpu] [--fast]

Reads data/cases/<case>/ct.nii.gz, written by fetch.py. Writes one mask per
label to data/work/<case>/seg/<label>.nii.gz, and seg.json with each label's
voxel count and Dice score against the reference segmentation (null where the
dataset has none). The sets built under the cut rules (structures.RULED_SETS)
also run the "body" task, whose trunk outline assemble.py measures.
--set body keeps every label of the "total" and "head_glands_cavities" tasks,
not only the set's structures, so each system's masks are ready for mesh.py;
the head task works inside the skull, so its masks are empty in a scan without
one. Only tasks in structures.ALLOWED_TASKS run; TotalSegmentator's licensed
tasks are never used. GPU inference is not bit-exact, so masks can differ by a
few boundary voxels between machines; the Dice scores show by how much.
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

from structures import ALLOWED_TASKS, RULED_SETS, SETS  # noqa: E402

TASK = "total"
HEAD_TASK = "head_glands_cavities"  # eyes, salivary glands, airways and palate, --set body only
BODY_TASK = "body"  # trunk outline, for the ruled sets
EVERYTHING = "body"  # the set that keeps every label of TASK and HEAD_TASK


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / total if total else 1.0


def mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj) > 0


def score(found_path: Path, reference_path: Path) -> tuple[int, float | None]:
    """A mask's voxel count and its Dice against the reference, None without one."""
    found = mask(found_path)
    # Some cases lack a reference mask for a structure (s0903 has no sacrum),
    # and the dataset has none for the head task's labels.
    reference = mask(reference_path) if reference_path.exists() else None
    if reference is None or not reference.any():
        return int(found.sum()), None
    return int(found.sum()), round(dice(found, reference), 4)


def task_labels(task: str) -> list[str]:
    """Every label the task writes, in TotalSegmentator's order."""
    from totalsegmentator.map_to_binary import class_map

    return list(class_map[task].values())


def tasks_for(which: str) -> list[str]:
    """The tasks a set runs, in their order; stops before any that is not Apache-2.0."""
    tasks = [
        TASK,
        *([HEAD_TASK] if which == EVERYTHING else []),
        *([BODY_TASK] if which in RULED_SETS else []),
    ]
    for task in tasks:
        if task not in ALLOWED_TASKS:
            sys.exit(
                f'TotalSegmentator task "{task}" is not Apache-2.0; only these tasks may run: '
                + ", ".join(sorted(ALLOWED_TASKS))
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--fast", action="store_true", help="3 mm model, quicker and coarser")
    parser.add_argument(
        "--set", choices=SETS, default="organs",
        help="structures to segment; body keeps every label of the open tasks",
    )  # fmt: skip
    args = parser.parse_args()
    structures = SETS[args.set]
    if not structures:
        sys.exit(f'the "{args.set}" set has no structures yet')
    tasks = tasks_for(args.set)

    from totalsegmentator.python_api import totalsegmentator

    case_dir = ROOT / "data" / "cases" / args.case
    out = ROOT / "data" / "work" / args.case / "seg"
    out.mkdir(parents=True, exist_ok=True)
    everything = args.set == EVERYTHING
    totalsegmentator(
        case_dir / "ct.nii.gz",
        out,
        task=TASK,
        roi_subset=None if everything else [s.label for s in structures],
        fast=args.fast,
        device=args.device,
        quiet=True,
    )
    if HEAD_TASK in tasks:
        # The head task has no fast model. It crops to the skull first, so a
        # scan without one gets empty masks, not an error.
        totalsegmentator(case_dir / "ct.nii.gz", out, task=HEAD_TASK, device=args.device,
                         quiet=True)  # fmt: skip
    if BODY_TASK in tasks:
        # The trunk outline (body_trunc) is used only to measure the body for the
        # fit in blender/assemble.py; it is never exported.
        totalsegmentator(
            case_dir / "ct.nii.gz", out, task=BODY_TASK, fast=args.fast, device=args.device,
            quiet=True,
        )  # fmt: skip

    labels = (
        task_labels(TASK) + task_labels(HEAD_TASK) if everything else [s.label for s in structures]
    )
    terms = {s.label: s.term for s in structures}
    report = {"case": args.case, "task": TASK, "tasks": tasks, "fast": args.fast, "set": args.set,
              "structures": {}}  # fmt: skip
    for label in labels:
        voxels, dice_score = score(
            out / f"{label}.nii.gz", case_dir / "segmentations" / f"{label}.nii.gz"
        )
        report["structures"][label] = {
            "term": terms.get(label),
            "voxels": voxels,
            "dice": dice_score,
        }
        shown = "n/a (no reference)" if dice_score is None else f"{dice_score:.3f}"
        print(f"{label:30} {voxels:>9} voxels  dice {shown}")
    (out.parent / "seg.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
