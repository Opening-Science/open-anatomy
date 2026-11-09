"""Tests for scan_quality.py on synthetic bodies made of tubes: no scan is read."""

from __future__ import annotations

import csv
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import scan_quality
from scan_quality import closest_on_triangles, cross_section, measures, regions, shape_change

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Mesh = tuple[FloatArray, IntArray]

STATURE = 1700.0


def tube(cx: float, rx: float, rz: float, y0: float, y1: float, around: int, rings: int) -> Mesh:
    """A closed elliptic tube standing along +Y, in mm, with fan caps."""
    theta = 2 * math.pi * np.arange(around) / around
    heights = np.linspace(y0, y1, rings)
    ring = np.stack([cx + rx * np.cos(theta), np.zeros(around), rz * np.sin(theta)], 1)
    side = np.vstack([ring + [0.0, y, 0.0] for y in heights])
    points = np.vstack([side, [[cx, y0, 0.0], [cx, y1, 0.0]]])
    triangles: list[list[int]] = []
    for r in range(rings - 1):
        for k in range(around):
            a, b = r * around + k, r * around + (k + 1) % around
            triangles += [[a, b, b + around], [a, b + around, a + around]]
    bottom, top = len(side), len(side) + 1
    last = (rings - 1) * around
    for k in range(around):
        triangles += [[bottom, (k + 1) % around, k], [top, last + k, last + (k + 1) % around]]
    return points, np.array(triangles, dtype=np.int64)


def join(*meshes: Mesh) -> Mesh:
    points: list[FloatArray] = []
    triangles: list[IntArray] = []
    offset = 0
    for p, t in meshes:
        points.append(p)
        triangles.append(t + offset)
        offset += len(p)
    return np.vstack(points), np.vstack(triangles)


def standing_body(grow: float = 0.0, arm_x: float = 220.0, scale_y: float = 1.0) -> Mesh:
    """A trunk 300 mm wide and 200 mm deep from the floor to 1700 mm, with an arm
    of 60 mm hanging 40 mm clear on each side from 700 to 1300 mm. `grow` widens
    every tube by that many mm."""
    trunk = tube(0.0, 150.0 + grow, 100.0 + grow, 0.0, STATURE * scale_y, 96, 35)
    left = tube(arm_x, 30.0 + grow, 30.0 + grow, 700.0, 1300.0, 24, 13)
    right = tube(-arm_x, 30.0 + grow, 30.0 + grow, 700.0, 1300.0, 24, 13)
    return join(trunk, left, right)


# Geometry


def test_cross_section_of_a_tube_is_its_outline() -> None:
    points, triangles = tube(0.0, 150.0, 100.0, 0.0, 1000.0, 96, 11)
    outline = cross_section(points, triangles, 432.0)
    assert np.allclose(outline[:, 1], 432.0)
    assert np.ptp(outline[:, 0]) == pytest.approx(300.0)
    assert np.ptp(outline[:, 2]) == pytest.approx(200.0, rel=1e-3)


def test_cross_section_above_the_body_is_empty() -> None:
    points, triangles = tube(0.0, 150.0, 100.0, 0.0, 1000.0, 16, 3)
    assert cross_section(points, triangles, 1500.0).shape == (0, 3)


def test_measures_take_stature_and_the_trunk_alone_at_hip_height() -> None:
    got = measures(*standing_body())
    assert got["stature"] == pytest.approx(STATURE)
    assert got["hipbreadth"] == pytest.approx(300.0)


def test_arms_touching_the_trunk_are_counted_in_the_breadth() -> None:
    # Without a GAP_MM gap there is nothing to split on: the documented reason
    # chest breadth is not taken.
    got = measures(*standing_body(arm_x=165.0))
    assert got["hipbreadth"] == pytest.approx(2 * (165.0 + 30.0))


def test_regions_split_head_feet_hands_and_trunk() -> None:
    points, _ = standing_body()
    masks = regions(points)
    h = points[:, 1] / STATURE
    np.testing.assert_array_equal(masks["head"], h > scan_quality.HEAD_ABOVE)
    np.testing.assert_array_equal(masks["feet"], h < scan_quality.FEET_BELOW)
    arm = np.abs(points[:, 0]) > 160
    np.testing.assert_array_equal(masks["hands"], arm & (h > 0.3) & (h < 0.5))
    np.testing.assert_array_equal(masks["trunk"], ~(masks["head"] | masks["feet"] | masks["hands"]))
    assert all(m.any() for m in masks.values())


A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.0])
C = np.array([0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("point", "closest"),
    [
        ((0.2, 0.2, 5.0), (0.2, 0.2, 0.0)),  # over the face
        ((-1.0, -1.0, 0.0), (0.0, 0.0, 0.0)),  # beyond corner a
        ((2.0, -1.0, 1.0), (1.0, 0.0, 0.0)),  # beyond corner b
        ((-1.0, 2.0, -1.0), (0.0, 1.0, 0.0)),  # beyond corner c
        ((0.5, -1.0, 0.0), (0.5, 0.0, 0.0)),  # beside edge ab
        ((-1.0, 0.5, 0.0), (0.0, 0.5, 0.0)),  # beside edge ac
        ((1.0, 1.0, 0.0), (0.5, 0.5, 0.0)),  # beside edge bc
    ],
)
def test_closest_on_triangles_in_each_voronoi_region(
    point: tuple[float, float, float], closest: tuple[float, float, float]
) -> None:
    got = closest_on_triangles(np.array([point]), A[None], B[None], C[None])
    np.testing.assert_allclose(got[0], closest, atol=1e-12)


def test_closest_on_triangles_beats_dense_sampling() -> None:
    rng = np.random.default_rng(1)
    n = 200
    a, b, c = (rng.normal(size=(n, 3)) for _ in range(3))
    p = rng.normal(size=(n, 3)) * 2
    got = closest_on_triangles(p, a, b, c)
    s, t = np.meshgrid(np.linspace(0, 1, 81), np.linspace(0, 1, 81))
    keep = s + t <= 1
    s, t = s[keep], t[keep]
    for i in range(n):
        samples = a[i] + np.outer(s, b[i] - a[i]) + np.outer(t, c[i] - a[i])
        sampled = np.linalg.norm(samples - p[i], axis=1).min()
        exact = np.linalg.norm(got[i] - p[i])
        assert exact <= sampled + 1e-9
        # And the point found lies on the triangle: non-negative barycentrics.
        m = np.stack([b[i] - a[i], c[i] - a[i]], 1)
        u, v = np.linalg.lstsq(m, got[i] - a[i], rcond=None)[0]
        assert u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9


def test_no_change_is_within_every_tolerance() -> None:
    body = standing_body()
    result = shape_change(body, body)
    assert set(result) == {"trunk", "head", "hands", "feet"}
    for region in result.values():
        assert region == {"meanMm": 0.0, "p95Mm": 0.0, "within": True}


def test_a_small_change_is_within_and_a_large_one_is_not() -> None:
    source = standing_body()
    small = shape_change(standing_body(grow=0.5), source)
    assert all(r["within"] for r in small.values())
    assert 0.3 < small["trunk"]["p95Mm"] <= 0.5 + 1e-3
    large = shape_change(standing_body(grow=4.0), source)
    assert not large["head"]["within"]
    assert not large["trunk"]["within"]
    assert large["trunk"]["p95Mm"] == pytest.approx(4.0, abs=0.1)


# main(), on run directories written under tmp_path


def write_glb(path: Path, mesh: Mesh) -> None:
    """A plain GLB with one mesh, positions in metres."""
    points = (mesh[0] / 1000).astype(np.float32)
    indices = mesh[1].astype(np.uint32).ravel()
    blob = points.tobytes() + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": points.nbytes},
            {"buffer": 0, "byteOffset": points.nbytes, "byteLength": indices.nbytes},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


def test_surface_reads_a_glb_back_in_millimetres(tmp_path: Path) -> None:
    body = standing_body()
    write_glb(tmp_path / "b.glb", body)
    points, triangles = scan_quality.surface(tmp_path / "b.glb")
    np.testing.assert_allclose(points, body[0], atol=1e-3)
    np.testing.assert_array_equal(triangles, body[1])


VALIDITY = {"holes": 0, "openEdges": 0, "nonManifoldEdges": 0, "inconsistentEdges": 0}


def make_run(
    root: Path,
    name: str,
    body: Mesh,
    face_protected: bool = False,
    source: Mesh | None = None,
    undamaged: bool = True,
) -> Path:
    run = root / name
    run.mkdir()
    report: dict[str, Any] = {
        "before": {**VALIDITY, "holes": 3, "inconsistentEdges": 12},
        "after": VALIDITY,
        "inputToOutputMm": {"meanMm": 0.4, "p95Mm": 1.1},
        "faceProtected": face_protected,
    }
    if undamaged:
        report["surfaceErrorBefore"] = {"meanMm": 2.5}
        report["surfaceErrorAfter"] = {"meanMm": 0.9}
    (run / "report.json").write_text(json.dumps(report))
    write_glb(run / "demo.lod0.glb", body)
    if source is not None:
        write_glb(run / "demo.input.glb", source)
    return run


@pytest.fixture
def ansur_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic percentile tables (not ANSUR's data): stature 1500 to 1900 mm,
    hip breadth 320 to 420 mm, evenly spread."""
    sources = tmp_path / "data" / "sources"
    sources.mkdir(parents=True)
    for file in scan_quality.ANSUR.values():
        with (sources / file).open("w", newline="", encoding="latin-1") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subjectid", "stature", "hipbreadth"])
            for i in range(101):
                writer.writerow([i, 1500 + 4 * i, 320 + i])
    monkeypatch.setattr(scan_quality, "ROOT", tmp_path)
    return tmp_path


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_quality.py", "--sex", "female", *args])
    scan_quality.main()


def test_ansur_reads_the_1st_50th_and_99th_percentiles(ansur_root: Path) -> None:
    ranges = scan_quality.ansur("male")
    assert ranges["stature"] == pytest.approx([1504.0, 1700.0, 1896.0])
    assert ranges["hipbreadth"] == pytest.approx([321.0, 370.0, 419.0])


def test_real_refuses_a_run_whose_face_was_not_protected(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(ansur_root, "r1", standing_body())
    out = ansur_root / "out.json"
    with pytest.raises(SystemExit, match="r1: refused: the cleanup did not protect a face"):
        run_main(monkeypatch, "--real", "--runs", str(run), "--out", str(out))
    assert not out.exists()


def test_real_refuses_when_the_report_does_not_say(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(ansur_root, "r1", standing_body())
    report = json.loads((run / "report.json").read_text())
    del report["faceProtected"]
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="refused"):
        run_main(monkeypatch, "--real", "--runs", str(run))


def test_real_refuses_if_any_one_of_several_runs_is_unprotected(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok = make_run(ansur_root, "r1", standing_body(), face_protected=True)
    bad = make_run(ansur_root, "r2", standing_body())
    with pytest.raises(SystemExit, match="r2: refused"):
        run_main(monkeypatch, "--real", "--runs", str(ok), str(bad))


def test_real_accepts_a_protected_run(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_run(ansur_root, "r1", standing_body(), face_protected=True)
    run_main(monkeypatch, "--real", "--runs", str(run))
    assert "r1: holes 3 -> 0, flipped 12 -> 0" in capsys.readouterr().out


def test_a_demo_run_is_measured_and_written(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_run(ansur_root, "r1", standing_body(grow=0.5), source=standing_body())
    out = ansur_root / "out.json"
    run_main(monkeypatch, "--runs", str(run), "--out", str(out))
    summary = json.loads(out.read_text())
    (entry,) = summary["runs"]
    assert entry["run"] == "r1"
    assert entry["validity"]["before"]["holes"] == 3
    assert entry["validity"]["after"] == VALIDITY
    assert entry["errorAgainstUndamagedMm"] == {"before": {"meanMm": 2.5}, "after": {"meanMm": 0.9}}
    assert set(entry["shapeChangeByRegionMm"]) == {"trunk", "head", "hands", "feet"}
    assert all(r["within"] for r in entry["shapeChangeByRegionMm"].values())
    # Stature 1700 is inside the table; a 301 mm hip breadth is under its 1st percentile.
    stature, hips = entry["bodyMeasuresMm"]["stature"], entry["bodyMeasuresMm"]["hipbreadth"]
    assert stature == {"mm": 1700, "ansur1": 1504, "ansur99": 1896, "plausible": True}
    assert hips == {"mm": 301, "ansur1": 321, "ansur99": 419, "plausible": False}
    assert summary["repeatabilitySpreadMm"] == {"stature": 0.0, "hipbreadth": 0.0}
    assert summary["repeatable"] is True
    printed = capsys.readouterr().out
    assert "hipbreadth 301 (OUT)" in printed
    assert "ok  trunk" in printed


def test_a_run_without_its_input_has_no_regional_shape_change(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(ansur_root, "r1", standing_body())
    out = ansur_root / "out.json"
    run_main(monkeypatch, "--runs", str(run), "--out", str(out))
    assert json.loads(out.read_text())["runs"][0]["shapeChangeByRegionMm"] is None


def test_repeat_runs_far_apart_are_not_repeatable(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_run(ansur_root, "r1", standing_body())
    second = make_run(ansur_root, "r2", standing_body(scale_y=1.01))  # 17 mm taller
    out = ansur_root / "out.json"
    run_main(monkeypatch, "--runs", str(first), str(second), "--out", str(out))
    summary = json.loads(out.read_text())
    assert summary["repeatabilitySpreadMm"]["stature"] == pytest.approx(17.0, abs=0.1)
    assert summary["repeatable"] is False


def test_repeat_runs_close_together_are_repeatable(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_run(ansur_root, "r1", standing_body())
    second = make_run(ansur_root, "r2", standing_body(scale_y=1.003))  # 5 mm taller
    out = ansur_root / "out.json"
    run_main(monkeypatch, "--runs", str(first), str(second), "--out", str(out))
    assert json.loads(out.read_text())["repeatable"] is True


def test_a_scan_without_an_undamaged_reference_is_reported(
    ansur_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(ansur_root, "r1", standing_body(), face_protected=True, undamaged=False)
    run_main(monkeypatch, "--real", "--runs", str(run))
