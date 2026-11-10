"""Tests for segment.py with a stand-in for TotalSegmentator: which tasks a set
runs, which labels it keeps, the licence guard and seg.json. No model, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import nibabel as nib
import numpy as np
import pytest

import segment
from structures import ORGANS, SKELETON

CLASS_MAP = {
    "total": {1: "spleen", 2: "liver", 90: "brain", 51: "heart"},
    "head_glands_cavities": {1: "eye_left", 2: "eye_right"},
    "body": {1: "body_trunc", 2: "body_extremities"},
}
EMPTY = {"brain", "eye_left", "eye_right"}  # labels the stand-in finds nothing of
Call = dict[str, Any]


def save(path: Path, voxels: slice | None) -> None:
    """A 4 x 4 x 4 mask with the slab `voxels` of its first axis set, or empty."""
    data = np.zeros((4, 4, 4), np.uint8)
    if voxels is not None:
        data[voxels] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Stands in for TotalSegmentator: notes each call and writes the masks it would."""
    made: list[Call] = []

    def totalsegmentator(
        ct: Path,
        out: Path,
        task: str = "total",
        roi_subset: list[str] | None = None,
        fast: bool = False,
        device: str = "gpu",
        quiet: bool = False,
    ) -> None:
        made.append({"task": task, "roi_subset": roi_subset, "fast": fast, "device": device})
        assert ct == tmp_path / "data" / "cases" / "s0001" / "ct.nii.gz"
        for label in roi_subset or CLASS_MAP[task].values():
            save(out / f"{label}.nii.gz", None if label in EMPTY else slice(0, 2))

    package = ModuleType("totalsegmentator")
    api = ModuleType("totalsegmentator.python_api")
    api.totalsegmentator = totalsegmentator  # type: ignore[attr-defined]
    mapping = ModuleType("totalsegmentator.map_to_binary")
    mapping.class_map = CLASS_MAP  # type: ignore[attr-defined]
    for module in (package, api, mapping):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(segment, "ROOT", tmp_path)
    return made


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", ["segment.py", "--case", "s0001", *args])
    segment.main()
    return json.loads((tmp_path / "data" / "work" / "s0001" / "seg.json").read_text())


def test_dice_is_the_overlap_share_and_one_for_two_empty_masks() -> None:
    a = np.zeros(8, bool)
    b = np.zeros(8, bool)
    assert segment.dice(a, b) == 1.0
    a[:4] = True
    b[2:6] = True
    assert segment.dice(a, b) == 0.5
    assert segment.dice(a, a) == 1.0


def test_the_body_set_keeps_every_label_of_both_open_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call]
) -> None:
    reference = tmp_path / "data" / "cases" / "s0001" / "segmentations"
    save(reference / "liver.nii.gz", slice(0, 2))  # the same slab: Dice 1
    save(reference / "spleen.nii.gz", slice(1, 3))  # half of it: Dice 0.5
    save(reference / "heart.nii.gz", None)  # an empty reference scores nothing
    report = run(tmp_path, monkeypatch, "--set", "body")
    assert [(c["task"], c["roi_subset"]) for c in calls] == [
        ("total", None),
        ("head_glands_cavities", None),
        ("body", None),
    ]
    assert report["set"] == "body"
    assert report["tasks"] == ["total", "head_glands_cavities", "body"]
    found = report["structures"]
    assert list(found) == ["spleen", "liver", "brain", "heart", "eye_left", "eye_right"]
    assert found["liver"] == {"term": "UBERON:0002107", "voxels": 32, "dice": 1.0}
    assert found["spleen"] == {"term": "UBERON:0002106", "voxels": 32, "dice": 0.5}
    assert found["heart"] == {"term": "UBERON:0000948", "voxels": 32, "dice": None}
    assert (found["brain"]["voxels"], found["brain"]["dice"]) == (0, None)
    # A label outside structures.py is kept and listed all the same, without a term.
    assert found["eye_left"] == {"term": None, "voxels": 0, "dice": None}
    seg = tmp_path / "data" / "work" / "s0001" / "seg"
    assert {p.name for p in seg.iterdir()} == {
        f"{label}.nii.gz" for task in CLASS_MAP.values() for label in task.values()
    }


def test_the_organ_set_asks_for_its_ten_labels_and_no_other_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call]
) -> None:
    report = run(tmp_path, monkeypatch)
    assert [(c["task"], c["roi_subset"]) for c in calls] == [("total", [s.label for s in ORGANS])]
    assert report["tasks"] == ["total"]
    assert list(report["structures"]) == [s.label for s in ORGANS]
    assert all(found["term"] for found in report["structures"].values())


@pytest.mark.parametrize("which", ["skeleton", "reference", "digestive"])
def test_a_ruled_set_asks_for_its_labels_and_the_trunk_outline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call], which: str
) -> None:
    report = run(tmp_path, monkeypatch, "--set", which)
    labels = [s.label for s in segment.SETS[which]]
    assert [(c["task"], c["roi_subset"]) for c in calls] == [("total", labels), ("body", None)]
    assert report["tasks"] == ["total", "body"]
    assert list(report["structures"]) == labels
    if which == "skeleton":
        assert labels == [s.label for s in SKELETON]


def test_fast_and_the_device_reach_the_tasks_that_take_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call]
) -> None:
    report = run(tmp_path, monkeypatch, "--set", "body", "--fast", "--device", "cpu")
    assert report["fast"] is True
    # The head task has no fast model.
    assert [(c["task"], c["fast"]) for c in calls] == [
        ("total", True),
        ("head_glands_cavities", False),
        ("body", True),
    ]
    assert {c["device"] for c in calls} == {"cpu"}


@pytest.mark.parametrize(
    ("which", "barred"),
    [("body", "head_glands_cavities"), ("skeleton", "body"), ("organs", "total")],
)
def test_a_task_outside_the_apache_list_stops_the_run_before_any_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call], which: str, barred: str
) -> None:
    monkeypatch.setattr(segment, "ALLOWED_TASKS", segment.ALLOWED_TASKS - {barred})
    with pytest.raises(SystemExit, match=f'task "{barred}" is not Apache-2.0'):
        run(tmp_path, monkeypatch, "--set", which)
    assert calls == []


def test_every_task_segment_py_names_is_on_the_apache_list() -> None:
    assert {segment.TASK, segment.HEAD_TASK, segment.BODY_TASK} <= segment.ALLOWED_TASKS
    for which in segment.SETS:
        assert set(segment.tasks_for(which)) <= segment.ALLOWED_TASKS


def test_an_empty_set_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Call]
) -> None:
    monkeypatch.setitem(segment.SETS, "nothing", ())
    with pytest.raises(SystemExit, match='the "nothing" set has no structures yet'):
        run(tmp_path, monkeypatch, "--set", "nothing")
    assert calls == []
