"""Tests for size_body.py's measures, on synthetic masks and a synthetic glTF body."""

from __future__ import annotations

import ast
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

import size_body

FloatArray = npt.NDArray[np.float64]


def sphere_points(centre: FloatArray, radius: float, n: int = 200) -> FloatArray:
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return centre + radius * directions


def test_fit_sphere_finds_the_centre() -> None:
    centre = np.array([12.0, -30.0, 140.0])
    np.testing.assert_allclose(size_body.fit_sphere(sphere_points(centre, 22.0)), centre)


def test_fit_sphere_works_on_a_cap() -> None:
    # Only the medial or upper part of a femoral or humeral head is used.
    centre = np.array([-60.0, 5.0, 900.0])
    points = sphere_points(centre, 24.0, 400)
    cap = points[points[:, 2] >= centre[2] + 6]
    np.testing.assert_allclose(size_body.fit_sphere(cap), centre, atol=1e-6)


def test_surface_points_are_the_boundary_in_ras_millimetres(tmp_path: Path) -> None:
    shape = (30, 30, 30)
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    ball = ((grid - 15) ** 2).sum(-1) <= 8**2
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [100.0, -20.0, 300.0]
    nib.save(nib.Nifti1Image(ball.astype(np.uint8), affine), tmp_path / "b.nii.gz")
    points = size_body.surface_points(tmp_path / "b.nii.gz")
    centre = affine[:3, :3] @ [15, 15, 15] + affine[:3, 3]
    distance = np.linalg.norm(points - centre, axis=1)
    assert distance.max() <= 8 * 2 + 1e-9
    assert distance.min() >= 7 * 2 - 1e-9
    assert len(points) == len(np.unique(points, axis=0))


# A small rigged glTF, written by hand


def glb_bytes(gltf: dict[str, Any], blob: bytes) -> bytes:
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


class Builder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add(self, array: npt.NDArray[Any], component: int, kind: str, **extra: Any) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": array.nbytes})
        self.blob.extend(array.tobytes())
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[kind]
        self.accessors.append(
            {
                "bufferView": len(self.views) - 1,
                "componentType": component,
                "count": array.size // width,
                "type": kind,
                **extra,
            }
        )
        return len(self.accessors) - 1


def test_accessor_reads_typed_arrays_and_normalised_bytes() -> None:
    b = Builder()
    floats = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = b.add(floats, 5126, "VEC3")
    second = b.add(np.array([[0, 51, 255, 0]], dtype=np.uint8), 5121, "VEC4", normalized=True)
    gltf = {"accessors": b.accessors, "bufferViews": b.views}
    np.testing.assert_array_equal(size_body.accessor(gltf, bytes(b.blob), first), floats)
    np.testing.assert_allclose(
        size_body.accessor(gltf, bytes(b.blob), second), [[0.0, 0.2, 1.0, 0.0]]
    )


def test_world_positions_compose_parent_rotation_scale_and_translation() -> None:
    quarter_turn_z = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    gltf = {
        "nodes": [
            {"name": "root", "translation": [0, 1, 0], "rotation": quarter_turn_z, "children": [1]},
            {"name": "mid", "translation": [1, 0, 0], "scale": [2, 2, 2], "children": [2]},
            {"name": "tip", "translation": [1, 0, 0]},
        ]
    }
    world = size_body.world_positions(gltf)
    np.testing.assert_allclose(world["root"], [0, 1, 0], atol=1e-12)
    # mid sits at root + R(+x) = (0, 1, 0) + (0, 1, 0); tip adds R(2 * +x).
    np.testing.assert_allclose(world["mid"], [0, 2, 0], atol=1e-12)
    np.testing.assert_allclose(world["tip"], [0, 4, 0], atol=1e-12)


def rigged_body(path: Path) -> None:
    """A trunk 300 mm wide and 200 mm deep, with arms off to the side, hip joints
    at y = 0.9 m and shoulder joints at y = 1.4 m."""
    joints = ["spine", "upperleg01.L", "upperleg01.R", "upperarm01.L", "upperarm01.R"]
    heights = {"upperleg01.L": 0.9, "upperleg01.R": 0.9, "upperarm01.L": 1.4, "upperarm01.R": 1.4}
    nodes: list[dict[str, Any]] = [{"name": "spine", "translation": [0, 1.0, 0], "children": []}]
    for name in joints[1:]:
        nodes[0]["children"].append(len(nodes))
        x = 0.1 if name.endswith("L") else -0.1
        nodes.append({"name": name, "translation": [x, heights[name] - 1.0, 0]})
    nodes.append({"name": "body", "mesh": 0, "skin": 0})

    trunk = [
        (x, y, z) for x in (-0.15, 0.15) for z in (-0.1, 0.1) for y in np.linspace(1.0, 1.3, 31)
    ]
    arms = [(x, y, 0.0) for x in (-0.5, 0.5) for y in np.linspace(1.0, 1.3, 31)]
    positions = np.array(trunk + arms, dtype=np.float32)
    which = np.zeros((len(positions), 4), dtype=np.uint8)
    which[len(trunk) :, 0] = 3  # the arm vertices follow upperarm01.L
    weights = np.zeros((len(positions), 4), dtype=np.float32)
    weights[:, 0] = 1.0

    b = Builder()
    attributes = {
        "POSITION": b.add(positions, 5126, "VEC3"),
        "JOINTS_0": b.add(which, 5121, "VEC4"),
        "WEIGHTS_0": b.add(weights, 5126, "VEC4"),
    }
    gltf = {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "skins": [{"joints": list(range(len(joints)))}],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.blob)}],
    }
    path.write_bytes(glb_bytes(gltf, bytes(b.blob)))


def test_read_glb_splits_json_and_binary(tmp_path: Path) -> None:
    rigged_body(tmp_path / "body.glb")
    gltf, blob = size_body.read_glb(tmp_path / "body.glb")
    assert gltf["buffers"][0]["byteLength"] == len(blob)


def test_measure_anny_takes_rise_from_joints_and_breadth_from_the_trunk(tmp_path: Path) -> None:
    rigged_body(tmp_path / "body.glb")
    got = size_body.measure_anny(tmp_path / "body.glb")
    assert got["rise"] == pytest.approx(500.0, abs=1e-3)
    assert got["width"] == pytest.approx(300.0, abs=1e-3)
    assert got["depth"] == pytest.approx(200.0, abs=1e-3)


# The CT measures, on a synthetic segmentation in data/work/<case>/seg under tmp_path


def ball(
    shape: tuple[int, int, int], centre: tuple[int, int, int], radius: int
) -> npt.NDArray[np.bool_]:
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    return ((grid - np.array(centre)) ** 2).sum(-1) <= radius**2


def test_measure_ct_on_a_synthetic_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 2 mm voxels, x flipped as CTs often are: canonical RAS must undo it.
    shape = (100, 60, 160)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    seg = tmp_path / "data" / "work" / "synthetic" / "seg"
    seg.mkdir(parents=True)

    def save(label: str, mask: npt.NDArray[np.bool_]) -> None:
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), seg / f"{label}.nii.gz")

    # Hip joints at k = 20, shoulder joints at k = 140: a rise of 240 mm.
    save("femur_left", ball(shape, (30, 30, 20), 8))
    save("femur_right", ball(shape, (70, 30, 20), 8))
    save("humerus_left", ball(shape, (10, 30, 140), 6))
    save("humerus_right", ball(shape, (90, 30, 140), 6))
    trunk = np.zeros(shape, bool)
    trunk[20:80, 10:50, 5:155] = True  # 59 x 39 voxel steps between the outer faces
    save("body_trunc", trunk)

    monkeypatch.setattr(size_body, "ROOT", tmp_path)
    got = size_body.measure_ct("synthetic")
    assert got["rise"] == pytest.approx(240.0, abs=0.5)
    assert got["width"] == pytest.approx(59 * 2.0)
    assert got["depth"] == pytest.approx(39 * 2.0)


# The choice of body in main(), with the CT and ANNY measures stubbed and every
# body the grid and its refinement may ask for already cached under tmp_path,
# so nothing is baked.

VALUES = sorted({*size_body.GRID, *(round(k * size_body.FINE_STEP, 2) for k in range(21))})

CT = {"rise": 500.0, "width": 300.0, "depth": 200.0}
OTHER = {"rise": 400.0, "width": 330.0, "depth": 150.0}
BODIES = {
    # The same size as the CT, nearly the same shape.
    (0.5, 0.5): {"rise": 500.0, "width": 290.0, "depth": 205.0},
    # Smaller, but exactly the CT's shape once scaled to its rise (x 1.25).
    (0.3, 0.7): {"rise": 400.0, "width": 240.0, "depth": 160.0},
}


def choose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *flags: str,
    bodies: dict[tuple[float, float], dict[str, float]] | None = None,
) -> tuple[dict[str, Any], str]:
    bodies = BODIES if bodies is None else bodies
    grid = tmp_path / "data" / "bodies" / "grid"
    grid.mkdir(parents=True)
    looked_up: dict[str, dict[str, float]] = {}
    for weight in VALUES:
        for height in VALUES:
            path = grid / f"anny-male-w{weight}-h{height}.rigged.glb"
            path.write_bytes(b"")
            looked_up[path.name] = bodies.get((weight, height), OTHER)

    def fake_anny(path: Path) -> dict[str, float]:
        return looked_up[path.name]

    def fake_ct(case: str) -> dict[str, float]:
        return CT

    monkeypatch.setattr(size_body, "ROOT", tmp_path)
    monkeypatch.setattr(size_body, "measure_ct", fake_ct)
    monkeypatch.setattr(size_body, "measure_anny", fake_anny)
    argv = ["size_body.py", "--case", "synthetic", "--sex", "male", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    size_body.main()
    out = capsys.readouterr().out
    chosen: dict[str, Any] = json.loads(out.strip().splitlines()[-1])
    return chosen, out


def test_by_default_the_body_of_the_same_size_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chosen, _ = choose(tmp_path, monkeypatch, capsys)
    assert (chosen["weight"], chosen["height"]) == (0.5, 0.5)
    assert chosen["ct"] == {"rise": 500, "width": 300, "depth": 200}
    assert chosen["anny"] == {"rise": 500, "width": 290, "depth": 205}


def test_scale_free_picks_the_body_of_the_same_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chosen, _ = choose(tmp_path, monkeypatch, capsys, "--scale-free")
    assert (chosen["weight"], chosen["height"]) == (0.3, 0.7)
    # The reported ANNY measures are its own, unscaled.
    assert chosen["anny"] == {"rise": 400, "width": 240, "depth": 160}


def test_scale_free_reports_a_zero_error_for_an_exact_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = choose(tmp_path, monkeypatch, capsys, "--scale-free")
    best = next(line for line in out.splitlines() if line.startswith("  weight"))
    assert best.startswith("  weight 0.3 height 0.7:")
    assert best.endswith("error 0.0000")


def test_a_best_body_between_grid_points_is_found_by_the_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    near = {"rise": 495.0, "width": 295.0, "depth": 202.0}
    bodies = {(0.5, 0.5): near, (0.55, 0.45): CT}
    chosen, out = choose(tmp_path, monkeypatch, capsys, bodies=bodies)
    assert (chosen["weight"], chosen["height"]) == (0.55, 0.45)
    assert chosen["edge"] == []
    assert "warning" not in out


def test_a_best_body_at_annys_limit_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chosen, out = choose(tmp_path, monkeypatch, capsys, bodies={(1.0, 0.5): CT})
    assert (chosen["weight"], chosen["height"]) == (1.0, 0.5)
    assert chosen["edge"] == ["weight 1.0 is at ANNY's limit; the CT may lie beyond it"]
    assert "warning: weight 1.0 is at ANNY's limit" in out


# --scale-free and blender/fit_body.py must use the same scale.


def test_uniform_scale_is_the_geometric_mean_of_the_three_ratios() -> None:
    scale, ratios = size_body.uniform_scale(CT, {"rise": 400.0, "width": 250.0, "depth": 200.0})
    assert ratios == {"rise": 1.25, "width": 1.2, "depth": 1.0}
    assert scale == pytest.approx((1.25 * 1.2 * 1.0) ** (1 / 3))


# Two bodies that the old rule (scale to the rise alone, then compare width and
# depth) and fit_body's rule (the geometric mean of all three ratios) rank the
# other way round. Rise-only: 0.0200 against 0.0126; geometric mean: 0.0059
# against 0.0125.
EVEN = {"rise": 400.0, "width": 264.0, "depth": 176.0}  # trunk 10% broad for its rise
UNEVEN = {"rise": 400.0, "width": 260.0, "depth": 148.0}  # rise right, wide and shallow


def test_scale_free_scores_bodies_at_the_scale_fit_body_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bodies = {(0.5, 0.5): UNEVEN, (0.3, 0.7): EVEN}
    chosen, out = choose(tmp_path, monkeypatch, capsys, "--scale-free", bodies=bodies)
    assert (chosen["weight"], chosen["height"]) == (0.3, 0.7)
    k, _ = size_body.uniform_scale(CT, EVEN)
    expected = sum(((EVEN[m] * k - CT[m]) / CT[m]) ** 2 for m in ("rise", "width", "depth"))
    best = next(line for line in out.splitlines() if line.startswith("  weight"))
    assert best.endswith(f"error {expected:.4f}")


def test_fit_body_scales_anny_with_size_bodys_uniform_scale() -> None:
    # fit_body.py runs only inside Blender, so check its source: it must take
    # its scale from size_body.uniform_scale and compute none of its own.
    source = (Path(__file__).resolve().parent.parent / "blender" / "fit_body.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "size_body"
        for alias in n.names
    }
    assert "uniform_scale" in imported
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "uniform_scale"
    ]
    assert len(calls) == 1
    assert "np.exp(np.mean(np.log" not in source
