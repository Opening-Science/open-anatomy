"""Tests for structures_doc.py, on synthetic QA reports and manifests under tmp_path."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import structures_doc
from structures import REFERENCE
from structures_doc import group_of, per_body

TERMS = {s.label: s.term for s in REFERENCE}


def node(label: str) -> str:
    return TERMS[label].replace(":", "_")


@pytest.mark.parametrize(
    ("label", "group"),
    [
        ("liver", "Organs: abdomen"),
        ("adrenal_gland_left", "Organs: abdomen"),
        ("urinary_bladder", "Organs: pelvis"),
        ("lung_upper_lobe_left", "Organs: chest and neck"),
        ("thyroid_gland", "Organs: chest and neck"),
        ("inferior_vena_cava", "Heart and vessels"),
        ("iliac_artery_left", "Heart and vessels"),
        ("portal_vein_and_splenic_vein", "Heart and vessels"),
        ("spinal_cord", "Brain, spinal cord and head"),
        ("oropharynx", "Brain, spinal cord and head"),
        ("vertebrae_L5", "Spine"),
        ("sacrum", "Spine"),
        ("rib_left_1", "Ribcage"),
        ("costal_cartilages", "Ribcage"),
        ("clavicula_right", "Shoulder girdle and arm"),
        ("hip_left", "Pelvis and thigh"),
        ("femur_right", "Pelvis and thigh"),
        ("iliopsoas_left", "Muscles"),
        ("gluteus_maximus_right", "Muscles"),
        ("something_new", "Other"),
    ],
)
def test_group_of(label: str, group: str) -> None:
    assert group_of(label) == group


def test_kidney_cysts_are_grouped_as_cysts() -> None:
    assert group_of("kidney_cyst_left") == "Cysts"


def test_every_built_structure_has_a_group() -> None:
    assert [s.label for s in REFERENCE if group_of(s.label) == "Other"] == []


def write_body(
    root: Path,
    name: str,
    case: str,
    qa: list[dict[str, Any]] | None,
    manifest: dict[str, Any] | None,
) -> None:
    if qa is not None:
        path = root / "data" / "qa" / name / "qa.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"structures": qa}))
    if manifest is not None:
        path = root / "data" / "work" / case / f"{case}.reference.manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))


def test_per_body_counts_nodes_passes_flags_and_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(structures_doc, "ROOT", tmp_path)
    write_body(
        tmp_path,
        "anny-adult-f",
        "s0001",
        [
            {"name": node("liver"), "flags": []},
            {"name": node("spleen"), "flags": ["rough"]},
            {"name": node("sternum"), "flags": []},
        ],
        {
            "nodes": [{}, {}, {}],
            "dropped": [{"label": "femur_left", "reason": "cut"}],
            "missing": ["humerus_left", "humerus_right"],
        },
    )
    lines = per_body({"anny-adult-f": {"case": "s0001"}})
    assert lines[0].startswith("| Reference | Case | In the body |")
    assert lines[2] == "| anny-adult-f | s0001 | 3 | 2 | 1 | 3 |"


def test_per_body_without_dropped_or_missing_counts_none_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(structures_doc, "ROOT", tmp_path)
    write_body(tmp_path, "b", "s0002", [{"name": "x", "flags": []}], {"nodes": [{}]})
    assert per_body({"b": {"case": "s0002"}})[2] == "| b | s0002 | 1 | 1 | 0 | 0 |"


@pytest.mark.parametrize("have", ["qa only", "manifest only", "neither"])
def test_a_body_not_built_here_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, have: str
) -> None:
    monkeypatch.setattr(structures_doc, "ROOT", tmp_path)
    write_body(
        tmp_path,
        "b",
        "s0003",
        [] if have == "qa only" else None,
        {"nodes": []} if have == "manifest only" else None,
    )
    assert (
        per_body({"b": {"case": "s0003"}})[2] == "| b | s0003 | not built on this machine | | | |"
    )


def test_per_body_keeps_the_order_of_the_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(structures_doc, "ROOT", tmp_path)
    lines = per_body({"z": {"case": "c1"}, "a": {"case": "c2"}})
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| z", "| a"]


# The whole document, with TotalSegmentator's class map replaced by a small one


def generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, labels: list[str], head: list[str]
) -> str:
    module = types.ModuleType("totalsegmentator.map_to_binary")
    module.__dict__["class_map"] = {
        "total": dict(enumerate(labels, 1)),
        "head_glands_cavities": dict(enumerate(head, 1)),
    }
    monkeypatch.setitem(sys.modules, "totalsegmentator", types.ModuleType("totalsegmentator"))
    monkeypatch.setitem(sys.modules, "totalsegmentator.map_to_binary", module)
    monkeypatch.setattr(structures_doc, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["structures_doc.py"])
    structures_doc.main()
    return (tmp_path / "docs" / "structures.md").read_text()


def rows(doc: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in doc.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[cells[0].strip("`")] = cells[1:]
    return out


def test_the_document_gives_each_label_its_status_and_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = {"anny-adult-f": {"case": "s0001"}, "anny-adult-m": {"case": "s0002"}}
    (tmp_path / "sources.json").write_text(json.dumps({"references": references}))
    write_body(
        tmp_path,
        "anny-adult-f",
        "s0001",
        [
            {"name": node("liver"), "flags": []},
            {"name": node("spleen"), "flags": ["rough", "spike"]},
        ],
        {
            "nodes": [{}, {}],
            "dropped": [{"label": "femur_left", "reason": "cut by the scan's edge (axis 2)"}],
            "missing": ["humerus_left"],
        },
    )
    labels = [
        "liver",
        "spleen",
        "femur_left",
        "humerus_left",
        "kidney_left",
        "lung_upper_lobe_left",
    ]
    doc = generate(tmp_path, monkeypatch, labels, ["brain", "eye_left"])
    got = rows(doc)

    def row(task: str, status: str, label: str, why: str) -> list[str]:
        return [task, status, TERMS.get(label, ""), why]

    assert got["liver"] == row("total", "passed QA", "liver", "no flags")
    assert got["spleen"] == row("total", "flagged", "spleen", "QA flags, reviewed: rough; spike")
    assert got["femur_left"] == row(
        "total", "built", "femur_left", "outside this scan: cut by the scan's edge (axis 2)"
    )
    assert got["humerus_left"] == row(
        "total", "built", "humerus_left", "outside this scan: absent from this scan"
    )
    assert got["kidney_left"] == row(
        "total", "built", "kidney_left", "outside this scan: not in the QA report"
    )
    assert got["lung_upper_lobe_left"] == row(
        "total", "missing", "lung_upper_lobe_left", "not built yet"
    )
    assert got["brain"] == row(
        "head_glands_cavities",
        "missing",
        "brain",
        "segmented only to keep faces out of the imaging layer",
    )
    assert got["eye_left"] == row("head_glands_cavities", "missing", "eye_left", "not built yet")

    assert "1 passed QA, 1 flagged, 3 built but outside its scan, 3 missing, of 8." in doc
    assert "| anny-adult-f | s0001 | 2 | 1 | 1 | 2 |" in doc
    assert "| anny-adult-m | s0002 | not built on this machine | | | |" in doc
    # Groups appear in their fixed order, each only when it has rows.
    headings = [line for line in doc.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Per reference body",
        "## Organs: abdomen",
        "## Organs: chest and neck",
        "## Brain, spinal cord and head",
        "## Shoulder girdle and arm",
        "## Pelvis and thigh",
        "## Not available from open tasks",
    ]


def test_without_a_qa_report_every_built_label_is_outside_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sources.json").write_text(
        json.dumps({"references": {"anny-adult-f": {"case": "s0001"}}})
    )
    doc = generate(tmp_path, monkeypatch, ["liver", "colon"], [])
    got = rows(doc)
    assert got["liver"][1:] == ["built", TERMS["liver"], "outside this scan: not in the QA report"]
    assert got["colon"][1] == "missing"
    assert "0 passed QA, 0 flagged, 1 built but outside its scan, 1 missing, of 2." in doc


def test_the_document_is_the_same_on_a_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sources.json").write_text(
        json.dumps({"references": {"anny-adult-f": {"case": "s0001"}}})
    )
    first = generate(tmp_path, monkeypatch, ["liver", "spleen"], ["brain"])
    assert generate(tmp_path, monkeypatch, ["liver", "spleen"], ["brain"]) == first
