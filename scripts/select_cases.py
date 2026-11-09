"""
Lists the CT cases in a TotalSegmentator archive that can make a reference
body, from the dataset's own reference segmentations (no inference).

  python scripts/select_cases.py [--source totalsegmentator-v201] [--out <json>]

A case qualifies when, in its reference masks:
- every organ in structures.ORGANS is present and not cut by the scan's top
  or bottom slice;
- the head is not in the scan (under 1000 skull voxels: a jaw tip or a stray
  label at most, which imaging.py's head guard cuts away anyway);
- each femur shows at least 6 cm, so its head is there for the hip landmark;
- each humerus shows at least 8 cm, its top is inside the scan and its long
  axis is within 45 degrees of vertical with the elbow below the shoulder, so
  the arm hangs as ANNY's does;
- ribs 5 to 8 are there to measure the ribcage;
- the study covers the trunk and the record names no trauma (fractures).
  meta.csv's pathology column takes a fixed set of values (PATHOLOGIES); the
  match ignores case, "trauma" and anything naming a fracture are excluded,
  and a value outside the set stops the run rather than slipping through.

For each it prints sex, age, study, the recorded pathology, the ribcage width
and the hip-to-shoulder rise, and with --out writes them as JSON. Masks are
read in canonical (RAS) orientation, so "top" is superior in every case.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import ORGANS  # noqa: E402

TRUNK_STUDIES = {
    "ct thorax-abdomen-pelvis", "ct neck-thorax-abdomen-pelvis", "ct polytrauma",
    "ct whole body", "ct angiography neck-thx-abd-pelvis-leg", "ct thorax-abdomen",
}  # fmt: skip
MAX_SKULL_VOXELS = 1000
MIN_FEMUR_MM = 60
MIN_HUMERUS_MM = 80
MAX_ARM_ANGLE_DEG = 45
# Every pathology value in v2.0.1's meta.csv, lower-cased ("" when none is given).
PATHOLOGIES = {
    "no_pathology", "tumor", "unclear", "vascular", "trauma", "inflammation", "other",
    "bleeding", "",
}  # fmt: skip
EXCLUDED = {"trauma"}  # injuries, fractures among them


def excluded(pathology: str) -> bool:
    """True for a record that names trauma or a fracture, whatever its case."""
    value = pathology.strip().lower()
    return value in EXCLUDED or "trauma" in value or "fractur" in value


class Case:
    def __init__(self, archive: zipfile.ZipFile, case: str):
        self.archive, self.case = archive, case

    def mask(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        data = self.archive.read(f"{self.case}/segmentations/{label}.nii.gz")
        image = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(gzip.decompress(data)))
        return np.asanyarray(image.dataobj) > 0, image.affine

    def points(self, label: str) -> np.ndarray:
        mask, affine = self.mask(label)
        return np.argwhere(mask) @ affine[:3, :3].T + affine[:3, 3]


def cut(mask: np.ndarray) -> bool:
    k = np.flatnonzero(mask.any((0, 1)))
    return bool(k[0] == 0 or k[-1] == mask.shape[2] - 1)


def check(case: Case) -> str | dict:
    """The first rule the case fails, or its measures."""
    for s in ORGANS:
        mask, _ = case.mask(s.label)
        if not mask.any():
            return f"no {s.label}"
        if cut(mask):
            return f"{s.label} cut"
    skull, _ = case.mask("skull")
    if skull.sum() >= MAX_SKULL_VOXELS:
        return "head in scan"
    femurs = []
    for side in ("left", "right"):
        f = case.points(f"femur_{side}")
        if len(f) == 0 or np.ptp(f[:, 2]) < MIN_FEMUR_MM:
            return "femoral head missing"
        femurs.append(f)
    shoulders = []
    for side in ("left", "right"):
        mask, affine = case.mask(f"humerus_{side}")
        h = np.argwhere(mask) @ affine[:3, :3].T + affine[:3, 3]
        if len(h) == 0 or np.linalg.norm(np.ptp(h, 0)) < MIN_HUMERUS_MM:
            return "humerus too short"
        if np.flatnonzero(mask.any((0, 1)))[-1] == mask.shape[2] - 1:
            return "humeral head cut"
        _, _, vt = np.linalg.svd(h - h.mean(0), full_matrices=False)
        if np.degrees(np.arccos(abs(vt[0][2]))) > MAX_ARM_ANGLE_DEG:
            return "arm not hanging"
        # The end nearer the scapula is the shoulder; the elbow must be below it.
        glenoid = case.points(f"scapula_{side}").mean(0)
        along = (h - h.mean(0)) @ vt[0]
        ends = [
            h[along >= np.percentile(along, 97)].mean(0),
            h[along <= np.percentile(along, 3)].mean(0),
        ]
        shoulder, elbow = sorted(ends, key=lambda e: np.linalg.norm(e - glenoid))
        if elbow[2] >= shoulder[2]:
            return "arm raised"
        shoulders.append(h[:, 2].max())
    ribs = [case.points(f"rib_{s}_{i}") for s in ("left", "right") for i in range(5, 9)]
    ribs = [r for r in ribs if len(r)]
    if len(ribs) < 6:
        return "ribs missing"
    hips = max(f[:, 2].max() for f in femurs)
    return {
        "ribcage_width_mm": round(float(np.ptp(np.vstack(ribs)[:, 0]))),
        "hip_to_shoulder_mm": round(float(np.mean(shoulders) - hips)),
        "skull_voxels": int(skull.sum()),
    }


def main() -> None:
    sources = json.loads((ROOT / "sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="totalsegmentator-v201")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = ROOT / "data" / "sources" / sources["sources"][args.source]["file"]
    archive = zipfile.ZipFile(path)
    meta = archive.read("meta.csv").decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(meta), delimiter=";"))
    values = collections.Counter(r["pathology"].strip().lower() for r in rows)
    print("# pathology values: " + ", ".join(f"{v or '(none)'} {n}" for v, n in values.items()))
    unknown = sorted(set(values) - PATHOLOGIES)
    if unknown:
        sys.exit(f"meta.csv has pathology values select_cases.py does not know: {unknown}")
    reasons, found = collections.Counter(), []
    for row in rows:
        if row["study_type"] not in TRUNK_STUDIES or row["gender"] not in ("m", "f"):
            continue
        if excluded(row["pathology"]):
            reasons[(row["gender"], "trauma")] += 1
            continue
        result = check(Case(archive, row["image_id"]))
        if isinstance(result, str):
            reasons[(row["gender"], result)] += 1
            continue
        result.update({
            "case": row["image_id"], "sex": row["gender"], "age": float(row["age"] or 0),
            "study": row["study_type"], "pathology": row["pathology"],
            "location": row["pathology_location"],
        })  # fmt: skip
        found.append(result)
        print(json.dumps(result), flush=True)
    for (sex, reason), count in sorted(reasons.items()):
        print(f"# {sex} {reason}: {count}")
    print(f"{len(found)} cases qualify")
    if args.out:
        args.out.write_text(json.dumps(found, indent=1) + "\n")


if __name__ == "__main__":
    main()
