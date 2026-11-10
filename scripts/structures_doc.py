"""
Writes docs/structures.md: every structure TotalSegmentator's Apache-2.0 tasks
can give, and what open-anatomy does with it.

  python scripts/structures_doc.py [--body anny-adult-f]

For each label of the "total" and "head_glands_cavities" tasks (the class map
comes from the installed TotalSegmentator) it gives the group, the status, the
ontology ID where built, and why: not built yet, outside the scan (with the
reason mesh.py logged), or the QA result from blender/qa.py. Statuses:

- missing: not built yet;
- built: meshed in the reference set, but not in this body (outside its scan);
- passed QA: checked close up, no flags;
- flagged: checked, with flags (reviewed in the QA PR).

A table counts the same for every reference body built on this machine.
Everything comes from scripts/structures.py, the QA reports and the mesh
manifests, so the document cannot claim more than the code does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import REFERENCE  # noqa: E402

HEAD_GUARD = ("skull", "brain")  # segmented only for the imaging layer's head guard (#14)
SUBTASKS = "https://github.com/wasserth/TotalSegmentator#subtasks"
GROUPS = [
    ("Organs: abdomen", ("spleen", "kidney", "gallbladder", "liver", "stomach", "pancreas",
                         "adrenal", "duodenum", "small_bowel", "colon", "esophagus")),
    ("Organs: pelvis", ("urinary_bladder", "prostate")),
    ("Organs: chest and neck", ("lung", "trachea", "thyroid")),
    ("Heart and vessels", ("heart", "aorta", "pulmonary", "brachiocephalic", "subclavian",
                           "carotid", "atrial", "vena", "portal", "iliac")),
    ("Brain, spinal cord and head", ("brain", "spinal_cord", "skull", "eye", "optic", "parotid",
                                     "submandibular", "pharynx", "nasal", "auditory", "palate")),
    ("Spine", ("vertebrae", "sacrum")),
    ("Ribcage", ("rib_", "sternum", "costal")),
    ("Shoulder girdle and arm", ("clavicula", "scapula", "humerus")),
    ("Pelvis and thigh", ("hip_", "femur")),
    ("Muscles", ("gluteus", "autochthon", "iliopsoas")),
    ("Cysts", ("cyst",)),
]  # fmt: skip
NOT_AVAILABLE = [
    ("Heart chambers (left and right atria and ventricles, myocardium)",
     "only in TotalSegmentator's licensed heartchambers_highres task; the open total task has "
     "the heart as one structure and the left atrial appendage. No open route is verified: "
     "one would be an nnU-Net model trained on a chamber-labelled CT dataset under CC BY or "
     "CC0, and no such dataset's licence has been checked yet"),
    ("Forearm, wrist, hand, lower leg and foot bones",
     "only in the licensed appendicular_bones task"),
    ("Fat and muscle tissue layers", "only in the licensed tissue_types task"),
]  # fmt: skip


def group_of(label: str) -> str:
    if "cyst" in label:  # before the organs, whose "kidney" would match first
        return "Cysts"
    for name, keys in GROUPS:
        if any(k in label for k in keys):
            return name
    return "Other"


def body_files(name: str, case: str) -> tuple[dict | None, dict | None]:
    qa = ROOT / "data" / "qa" / name / "qa.json"
    manifest = ROOT / "data" / "work" / case / f"{case}.reference.manifest.json"
    return (
        json.loads(qa.read_text()) if qa.exists() else None,
        json.loads(manifest.read_text()) if manifest.exists() else None,
    )


def per_body(references: dict) -> list[str]:
    lines = [
        "| Reference | Case | In the body | Passed QA | Flagged | Left out (cut or absent) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, ref in references.items():
        qa, manifest = body_files(name, ref["case"])
        if qa is None or manifest is None:
            lines.append(f"| {name} | {ref['case']} | not built on this machine | | | |")
            continue
        flagged = sum(1 for q in qa["structures"] if q["flags"])
        left_out = len(manifest.get("dropped", [])) + len(manifest.get("missing", []))
        lines.append(
            f"| {name} | {ref['case']} | {len(manifest['nodes'])} | "
            f"{len(qa['structures']) - flagged} | {flagged} | {left_out} |"
        )
    return lines


def main() -> None:
    # TotalSegmentator is in the pipeline extra only, which the type check does not install.
    from totalsegmentator.map_to_binary import (  # pyright: ignore[reportMissingImports]
        class_map,
    )

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--body", default="anny-adult-f", help="the body the rows describe")
    args = parser.parse_args()
    references = json.loads((ROOT / "sources.json").read_text()).get("references", {})
    case = references[args.body]["case"]
    qa_report, manifest = body_files(args.body, case)
    qa = {s["name"]: s for s in qa_report["structures"]} if qa_report else {}
    left_out = {}
    if manifest:
        left_out = {d["label"]: d["reason"] for d in manifest.get("dropped", [])}
        left_out.update({m: "absent from this scan" for m in manifest.get("missing", [])})
    built = {s.label: s for s in REFERENCE}

    labels = [(lab, "total") for lab in class_map["total"].values()]
    labels += [(lab, "head_glands_cavities") for lab in class_map["head_glands_cavities"].values()]
    rows: dict[str, list[str]] = {}
    counts = {"missing": 0, "built": 0, "passed QA": 0, "flagged": 0}
    for label, task in labels:
        s = built.get(label)
        if s is None:
            status, term = "missing", ""
            why = (
                "segmented only to keep faces out of the imaging layer"
                if label in HEAD_GUARD
                else "not built yet"
            )
        else:
            term = s.term
            entry = qa.get(term.replace(":", "_"))
            if entry is None:
                status = "built"
                why = f"outside this scan: {left_out.get(label, 'not in the QA report')}"
            elif entry["flags"]:
                status, why = "flagged", "QA flags, reviewed: " + "; ".join(entry["flags"])
            else:
                status, why = "passed QA", "no flags"
        counts[status] += 1
        rows.setdefault(group_of(label), []).append(
            f"| `{label}` | {task} | {status} | {term} | {why} |"
        )

    lines = [
        "# Structures",
        "",
        "Every structure TotalSegmentator's Apache-2.0 tasks can give, and what",
        "open-anatomy does with it. Generated by `scripts/structures_doc.py`; do not",
        "edit by hand.",
        "",
        "Both tasks used are open: `total` and `head_glands_cavities` are on",
        f"TotalSegmentator's Apache-2.0 list ([Subtasks]({SUBTASKS})).",
        "",
        "## Per reference body",
        "",
        *per_body(references),
        "",
        f"The rows below describe `{args.body}` (case {case}): {counts['passed QA']} passed QA, "
        f"{counts['flagged']} flagged, {counts['built']} built but outside its scan, "
        f"{counts['missing']} missing, of {len(labels)}.",
        "",
        "A flag is not a failure: the contact sheets show each one, and the QA PR",
        'explains the known false positives (single-vertex "spikes" on folds, and',
        '"rough" on the sacrum and costal cartilages, which is real sharp anatomy).',
        "",
    ]
    for group in [g for g, _ in GROUPS] + ["Other"]:
        if group not in rows:
            continue
        lines += [
            f"## {group}",
            "",
            "| Label | Task | Status | ID | Why |",
            "| --- | --- | --- | --- | --- |",
            *rows[group],
            "",
        ]
    lines += ["## Not available from open tasks", "", "| Structures | Why |", "| --- | --- |"]
    lines += [f"| {what} | {why} |" for what, why in NOT_AVAILABLE] + [""]
    path = ROOT / "docs" / "structures.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"{path.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
