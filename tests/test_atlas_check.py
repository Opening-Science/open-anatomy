"""Tests for atlas_check.py on analytic shapes and small synthetic scenes
(spheres, boxes and tubes): no CT, no Blender."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

import atlas_check as ac
import mesh as mesh_py
from structures import SKELETON

Floats = npt.NDArray[np.float64]

VOXEL = 1.5  # mm, as the Stage 1 scans
# x flipped, as real CTs have, and an origin off the grid.
AFFINE = np.array(
    [[-VOXEL, 0, 0, 61.3], [0, VOXEL, 0, -47.9], [0, 0, VOXEL, -50.2], [0, 0, 0, 1.0]]
)
SHAPE = (80, 64, 64)
CENTRE = np.array([1.6, 0.4, -2.3])  # RAS mm: the middle of the grid, off the voxel centres


def cube(low: tuple[float, float, float], high: tuple[float, float, float]) -> ac.Mesh:
    """A box as twelve triangles wound outward."""
    lo, hi = np.array(low, dtype=np.float64), np.array(high, dtype=np.float64)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]  # fmt: skip
    return ac.Mesh(corners * (hi - lo) + lo, np.array(faces, dtype=np.int64))


def icosphere(radius: float, centre: Floats, level: int = 5) -> ac.Mesh:
    """A sphere as 20 * 4 ** level triangles with every vertex on it."""
    g = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g],
         [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1]]
    )  # fmt: skip
    faces = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    )  # fmt: skip
    for _ in range(level):
        ends = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
        edges, which = np.unique(ends, axis=0, return_inverse=True)
        middle = len(vertices) + which.reshape(3, -1)  # per face: the middles of its three edges
        vertices = np.vstack([vertices, vertices[edges].mean(1)])
        a, b, c = faces.T
        ab, bc, ca = middle
        faces = np.vstack([np.stack(f, 1) for f in ((a, ab, ca), (b, bc, ab), (c, ca, bc),
                                                    (ab, bc, ca))])  # fmt: skip
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    return ac.Mesh(vertices * radius + centre, faces.astype(np.int64))


def scaled(mesh: ac.Mesh, centre: Floats, factors: Floats) -> ac.Mesh:
    return ac.Mesh((mesh.vertices - centre) * factors + centre, mesh.triangles)


def moved(mesh: ac.Mesh, by: tuple[float, float, float]) -> ac.Mesh:
    return ac.Mesh(mesh.vertices + np.array(by), mesh.triangles)


def joined(*meshes: ac.Mesh) -> ac.Mesh:
    """Several meshes as one, each keeping its own vertices."""
    starts = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return ac.Mesh(
        np.vstack([m.vertices for m in meshes]),
        np.vstack([m.triangles + start for m, start in zip(meshes, starts, strict=True)]),
    )


def tube(path: list[tuple[float, float, float]], radius: float, sides: int = 24) -> ac.Mesh:
    """A closed round tube along a path of straight stretches, with flat
    ends, wound outward. Each stretch is cut into rings about 1 mm apart."""
    corners = np.array(path, dtype=np.float64)
    points = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:], strict=True):
        steps = max(int(np.ceil(np.linalg.norm(b - a))), 1)
        points += [a + (b - a) * k / steps for k in range(1, steps + 1)]
    line = np.array(points)
    tangent = np.gradient(line, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    across = np.cross(tangent[0], [0.3, 0.5, 0.81])
    rings = []
    for point, along in zip(line, tangent, strict=True):
        across = across - (across @ along) * along
        across /= np.linalg.norm(across)
        angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)[:, None]
        rings.append(point + radius * (np.cos(angle) * across
                                       + np.sin(angle) * np.cross(along, across)))  # fmt: skip
    vertices = np.vstack([*rings, line[0], line[-1]])
    k = np.arange(sides)
    faces = []
    for ring in range(len(line) - 1):
        a, b = ring * sides + k, ring * sides + (k + 1) % sides
        faces += [np.stack([a, b, b + sides], 1), np.stack([a, b + sides, a + sides], 1)]
    last = (len(line) - 1) * sides
    faces.append(np.stack([np.full(sides, len(vertices) - 2), (k + 1) % sides, k], 1))
    faces.append(np.stack([np.full(sides, len(vertices) - 1), last + k, last + (k + 1) % sides], 1))
    out = ac.Mesh(vertices, np.vstack(faces).astype(np.int64))
    return out if ac.signed_volume(out) > 0 else ac.Mesh(vertices, out.triangles[:, ::-1])


def voxel_centres() -> Floats:
    index = np.indices(SHAPE).transpose(1, 2, 3, 0).astype(np.float64)
    return index @ AFFINE[:3, :3].T + AFFINE[:3, 3]


def ball(radius: float, centre: Floats = CENTRE) -> npt.NDArray[np.bool_]:
    """The voxels whose centres lie in a sphere."""
    return ((voxel_centres() - centre) ** 2).sum(-1) <= radius**2


def as_mask(data: npt.NDArray[np.bool_]) -> ac.Mask:
    return ac.Mask(data, AFFINE)


def save(path: Path, data: npt.NDArray[np.bool_]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), AFFINE), path)
    return path


# The frame and the GLB


def test_the_frame_is_mesh_pys() -> None:
    np.testing.assert_array_equal(ac.RAS_TO_GLTF, mesh_py.RAS_TO_GLTF)


def write_skeleton(path: Path, meshes: dict[str, ac.Mesh], centre: Floats | None = None) -> None:
    """A GLB as mesh.py writes it, from meshes in RAS mm by label."""
    structures = {s.label: s for s in SKELETON}
    centre = np.zeros(3) if centre is None else centre
    parts = [
        {
            "structure": structures[label],
            "vertices": (m.vertices @ ac.RAS_TO_GLTF.T - centre).astype(np.float32),
            "triangles": m.triangles.astype(np.uint32),
            "normals": np.zeros((len(m.vertices), 3), dtype=np.float32),
            "modelled": label == "costal_cartilages",
        }
        for label, m in meshes.items()
    ]
    mesh_py.write_glb(path, parts)


def test_a_glb_from_mesh_py_reads_back_into_the_scans_millimetres(tmp_path: Path) -> None:
    box = cube((10, -20, 30), (25, -5, 60))
    centre = np.array([0.011, 0.25, -0.04])
    write_skeleton(tmp_path / "s.glb", {"sternum": box}, centre)
    read = ac.read_glb(tmp_path / "s.glb")
    assert list(read) == ["FMA_7485"]
    np.testing.assert_array_equal(read["FMA_7485"].triangles, box.triangles)
    np.testing.assert_allclose(ac.to_ras(read["FMA_7485"].vertices, centre), box.vertices,
                               atol=1e-3)  # fmt: skip


def glb_bytes(gltf: dict[str, Any], blob: bytes) -> bytes:
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(blob))
            + struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(blob), 0x004E4942) + blob)  # fmt: skip


def test_read_glb_applies_node_transforms_and_reads_short_indices(tmp_path: Path) -> None:
    """As Blender's exporter writes: 16-bit indices, and a node may be moved,
    turned and scaled, under a parent that is moved too."""
    position = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    index = np.array([0, 1, 2], dtype=np.uint16)
    blob = position.tobytes() + index.tobytes()
    half = float(np.sqrt(0.5))
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "parent", "translation": [0, 0, 10], "children": [1]},
            # A quarter turn about z, then twice the size, then 5 along x.
            {
                "name": "child",
                "mesh": 0,
                "translation": [5, 0, 0],
                "rotation": [0, 0, half, half],
                "scale": [2, 2, 2],
            },
        ],  # fmt: skip
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [{"byteLength": 42}],
    }
    (tmp_path / "t.glb").write_bytes(glb_bytes(gltf, blob))
    read = ac.read_glb(tmp_path / "t.glb")
    assert list(read) == ["child"]
    np.testing.assert_allclose(read["child"].vertices, [[5, 0, 10], [5, 2, 10], [3, 0, 10]],
                               atol=1e-6)  # fmt: skip
    np.testing.assert_array_equal(read["child"].triangles, [[0, 1, 2]])


def test_a_glb_that_is_not_triangles_is_refused(tmp_path: Path) -> None:
    gltf = {
        "asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "n", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1, "indices": 0}]}],
    }  # fmt: skip
    (tmp_path / "t.glb").write_bytes(glb_bytes(gltf, b""))
    with pytest.raises(ValueError, match="not indexed triangles"):
        ac.read_glb(tmp_path / "t.glb")


def test_welding_joins_vertices_written_once_per_corner() -> None:
    box = cube((0, 0, 0), (1, 1, 1))
    loose = ac.Mesh(box.vertices[box.triangles].reshape(-1, 3), np.arange(36).reshape(-1, 3))
    assert len(ac.welded(loose).vertices) == 8
    assert ac.topology_faults(ac.topology(loose), several=False) == []


# Distances


def test_point_triangle_distance_over_the_face_an_edge_and_a_corner() -> None:
    a, b, c = (np.array([p], dtype=np.float64) for p in ((0, 0, 0), (4, 0, 0), (0, 4, 0)))

    def distance(*p: float) -> float:
        return float(ac.point_triangle_distance(np.array([p], dtype=np.float64), a, b, c)[0])

    assert distance(1, 1, 3) == pytest.approx(3)  # over the face
    assert distance(2, -2, 0) == pytest.approx(2)  # beside the edge a-b
    assert distance(3, 3, 0) == pytest.approx(np.sqrt(2))  # beside the slanted edge
    assert distance(-3, -4, 0) == pytest.approx(5)  # beyond the corner a
    assert distance(1, 1, 0) == pytest.approx(0)
    # A triangle with no area is its edges.
    flat = ac.point_triangle_distance(np.array([[2.0, 3.0, 0.0]]), a, b, b)
    assert float(flat[0]) == pytest.approx(3)


def box_distance(points: Floats, low: float, high: float | Floats) -> Floats:
    """The distance from points to the surface of a cube, worked out by hand."""
    outside = np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0), axis=1)
    inside = np.minimum(points - low, high - points).min(1)
    return np.where(outside > 0, outside, inside)


def test_the_distance_to_a_surface_is_exact_for_a_cube() -> None:
    surface = ac.Surface(cube((0, 0, 0), (10, 10, 10)))
    points = np.random.default_rng(1).uniform(-8, 18, (5000, 3))
    np.testing.assert_allclose(surface.distance(points), box_distance(points, 0, 10), atol=1e-9)


def test_inside_a_cube_even_along_its_faces_and_edges() -> None:
    surface = ac.Surface(cube((0, 0, 0), (10, 10, 10)))
    points = np.random.default_rng(2).uniform(-5, 15, (20000, 3))
    np.testing.assert_array_equal(surface.inside(points), ((points > 0) & (points < 10)).all(1))
    # Whole numbers: rays along the axes would run through edges and corners.
    grid = np.stack(np.meshgrid(*[np.arange(-2.0, 13.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    clear = (np.abs(grid) > 1e-9).all(1) & (np.abs(grid - 10) > 1e-9).all(1)  # not on the surface
    np.testing.assert_array_equal(
        surface.inside(grid[clear]), ((grid[clear] > 0) & (grid[clear] < 10)).all(1)
    )


def test_inside_a_sphere() -> None:
    surface = ac.Surface(icosphere(20, CENTRE, 4))
    points = CENTRE + np.random.default_rng(3).uniform(-30, 30, (20000, 3))
    off = np.linalg.norm(points - CENTRE, axis=1)
    sure = np.abs(off - 20) > 0.1  # the mesh lies a hair inside the sphere
    np.testing.assert_array_equal(surface.inside(points[sure]), off[sure] < 20)


def test_points_spread_over_a_surface_follow_its_area() -> None:
    box = cube((0, 0, 0), (40, 20, 10))
    points = ac.sample_points(box)
    assert len(points) == int(np.ceil(2800 / ac.SAMPLE_MM**2))
    np.testing.assert_allclose(box_distance(points, 0, np.array([40, 20, 10])), 0, atol=1e-9)
    top = np.isclose(points[:, 2], 10).mean()  # the 40 x 20 top is 800 of 2800 mm2
    assert top == pytest.approx(800 / 2800, abs=0.01)
    np.testing.assert_array_equal(points, ac.sample_points(box))  # the same on every run


# (b) Fidelity


def test_a_voxel_next_to_the_boundary_is_half_a_voxel_from_it() -> None:
    inside = np.zeros((12, 12, 12), dtype=bool)
    inside[4:8, 4:8, 4:8] = True
    distance = ac.signed_distance(inside, np.array([0.8, 0.8, 1.5]))
    assert distance[4, 5, 5] == pytest.approx(-0.4) and distance[3, 5, 5] == pytest.approx(0.4)
    assert distance[5, 5, 7] == pytest.approx(-0.75) and distance[5, 5, 8] == pytest.approx(0.75)
    assert distance[5, 5, 9] == pytest.approx(2.25)  # one voxel further out
    # Diagonally off a corner: to the corner of the box, half the way to its centre.
    assert distance[3, 3, 5] == pytest.approx(np.hypot(0.4, 0.4))


def test_a_mask_without_a_boundary_has_no_surface() -> None:
    with pytest.raises(ValueError, match="no boundary"):
        ac.signed_distance(np.zeros((4, 4, 4), dtype=bool), np.ones(3))
    with pytest.raises(ValueError, match="empty"):
        ac.mask_surface(as_mask(np.zeros(SHAPE, dtype=bool)))


def test_the_zero_level_of_a_spheres_distance_is_the_sphere() -> None:
    index = np.indices((40, 40, 40)).transpose(1, 2, 3, 0)
    field = np.asarray(np.linalg.norm(index - 19.3, axis=-1) - 12.0, dtype=np.float64)
    triangles = ac.zero_level(field)
    off = np.linalg.norm(triangles.reshape(-1, 3) - 19.3, axis=1) - 12.0
    assert np.abs(off).max() < 0.05  # straight between two samples of a curved field
    soup = ac.Mesh(triangles.reshape(-1, 3), np.arange(len(triangles) * 3).reshape(-1, 3))
    assert ac.triangle_areas(soup).sum() == pytest.approx(4 * np.pi * 144, rel=0.01)
    # Closed: every edge of the welded triangles is used by exactly two.
    found = ac.topology(soup)
    assert found["openEdges"] == 0 and found["nonManifoldEdges"] == 0


def test_a_masks_surface_is_the_shape_that_was_voxelised() -> None:
    """Not the voxel steps, which lie up to half a voxel (0.75 mm) off it."""
    for radius in (30.0, 8.0, 4.0):
        surface = ac.mask_surface(as_mask(ball(radius)))
        off = np.abs(np.linalg.norm(ac.sample_points(surface) - CENTRE, axis=1) - radius)
        assert off.mean() < 0.25 and np.percentile(off, 95) < 0.6, radius


def flat_ellipsoid() -> tuple[ac.Mesh, ac.Mask]:
    """A flattened ellipsoid, 60 x 44 x 12 mm, meshed exactly and voxelised."""
    radii = np.array([30.0, 22.0, 6.0])
    exact = scaled(icosphere(1.0, CENTRE), CENTRE, radii)
    return exact, as_mask((((voxel_centres() - CENTRE) / radii) ** 2).sum(-1) <= 1)


def test_a_sphere_meshed_exactly_scores_near_zero() -> None:
    found = ac.fidelity(icosphere(30.0, CENTRE), as_mask(ball(30.0)))
    assert found["pass"]
    for way in ("meshToMask", "maskToMesh"):
        assert found[way]["mean"] < 0.25 and found[way]["p95"] < 0.6, found


def test_a_flattened_ellipsoid_meshed_exactly_scores_near_zero() -> None:
    found = ac.fidelity(*flat_ellipsoid())
    assert found["pass"]
    for way in ("meshToMask", "maskToMesh"):
        assert found[way]["mean"] < 0.3 and found[way]["p95"] < 0.75, found


def test_a_mesh_moved_two_millimetres_fails_both_ways() -> None:
    exact, mask = flat_ellipsoid()
    found = ac.fidelity(moved(exact, (0, 0, 2.0)), mask)  # along its short axis
    assert not found["pass"]
    for way in ("meshToMask", "maskToMesh"):
        assert 1.4 < found[way]["mean"] < 2.1, found


def test_a_moved_sphere_reads_its_true_mean_distance() -> None:
    """A sphere moved 2 mm lies 2 |cos| off itself: 1 mm on average, 2 at most."""
    found = ac.fidelity(moved(icosphere(30.0, CENTRE), (2.0, 0, 0)), as_mask(ball(30.0)))
    for way in ("meshToMask", "maskToMesh"):
        assert found[way]["mean"] == pytest.approx(1.0, abs=0.12), found
        assert found[way]["p95"] == pytest.approx(1.9, abs=0.15), found


def test_a_mesh_shrunk_by_two_millimetres_fails_both_ways() -> None:
    found = ac.fidelity(icosphere(28.0, CENTRE), as_mask(ball(30.0)))
    assert not found["pass"]
    for way in ("meshToMask", "maskToMesh"):
        assert found[way]["mean"] == pytest.approx(2.0, abs=0.2), found
        assert found[way]["p95"] > ac.MAX_P95_MM, found


LOBE = CENTRE + np.array([24.0, 0, 0])  # a 10 mm ball standing 14 mm out of a 20 mm one


def test_a_mesh_missing_a_lobe_fails_from_the_mask_to_the_mesh() -> None:
    found = ac.fidelity(icosphere(20.0, CENTRE), as_mask(ball(20.0) | ball(10.0, LOBE)))
    assert not found["pass"]
    out, back = found["meshToMask"], found["maskToMesh"]
    assert out["mean"] <= ac.MAX_MEAN_MM and out["p95"] <= ac.MAX_P95_MM, found
    assert back["p95"] > ac.MAX_P95_MM and back["max"] > 10, found


def test_a_mesh_with_a_lobe_the_mask_lacks_fails_from_the_mesh_to_the_mask() -> None:
    lobed = joined(icosphere(20.0, CENTRE), icosphere(10.0, LOBE))
    found = ac.fidelity(lobed, as_mask(ball(20.0)))
    assert not found["pass"]
    out, back = found["meshToMask"], found["maskToMesh"]
    assert out["p95"] > ac.MAX_P95_MM and out["max"] > 10, found
    assert back["mean"] <= ac.MAX_MEAN_MM and back["p95"] <= ac.MAX_P95_MM, found


def test_declared_change_adds_up_what_the_manifest_says() -> None:
    node = {"knobs_trimmed_ml": 0.3, "end_lumps_trimmed_ml": 0.2, "end_filled_ml": 0.3,
            "head_grown_ml": 0.1, "bridged_ml": 0.1, "dropped_share": 0.02}  # fmt: skip
    fragments = [
        {"label": "rib_left_4", "ml": 1.0, "action": "dropped"},
        {"label": "rib_left_4", "ml": 0.5, "action": "moved", "to": "rib_left_5"},
        {"label": "rib_left_3", "ml": 0.7, "action": "moved", "to": "rib_left_4"},
        {"label": "rib_left_5", "ml": 9.0, "action": "moved", "to": "rib_left_6"},
    ]
    added, removed = ac.declared_change(node, "rib_left_4", fragments, kept_ml=9.8)
    assert added == pytest.approx(0.4 + 0.1 + 0.7)
    # 2% of the mask lay outside the 9.8 ml that were kept: 0.2 ml.
    assert removed == pytest.approx(0.3 + 0.2 + 1.0 + 0.5 + 0.2)
    assert ac.declared_change({}, "sternum", fragments, kept_ml=50.0) == (0.0, 0.0)


def rib_node(tmp_path: Path, raw: npt.NDArray[np.bool_], contoured: npt.NDArray[np.bool_] | None,
             **declared: Any) -> dict[str, Any]:  # fmt: skip
    """check_fidelity of a 20 mm ball meshed exactly, as the node of the left
    first rib, with its raw mask and (if given) the mask that was contoured."""
    save(tmp_path / "seg" / "rib_left_1.nii.gz", raw)
    node: dict[str, Any] = {"name": "FMA_7987", **declared}
    if contoured is not None:
        save(tmp_path / "masks" / "rib_left_1.nii.gz", contoured)
        node["contoured_mask"] = "masks/rib_left_1.nii.gz"
    body = ac.Body("FMA_7987", icosphere(20.0, CENTRE))
    manifest = {"fragments": declared.pop("fragments", [])}
    return ac.check_fidelity(body, node, manifest, tmp_path, tmp_path / "seg")


KNOB = ball(20.0) | ball(6.0, CENTRE + np.array([27.0, 0, 0]))  # a loose 0.9 ml lump beside it


def test_without_a_contoured_mask_the_mesh_is_held_to_the_raw_mask(tmp_path: Path) -> None:
    found = rib_node(tmp_path, KNOB, None, knobs_trimmed_ml=0.9)
    assert found["against"] == "raw mask" and not found["pass"]  # the lump shows as missing
    assert "change" not in found


def test_a_declared_trim_is_measured_against_the_contoured_mask(tmp_path: Path) -> None:
    found = rib_node(tmp_path, KNOB, ball(20.0), knobs_trimmed_ml=0.9)
    assert found["against"] == "contoured mask" and found["pass"]
    change = found["change"]
    assert change["pass"] and change["addedMl"] == 0
    assert change["removedMl"] == pytest.approx(4 / 3 * np.pi * 6**3 / 1000, abs=0.05)


def test_a_change_the_manifest_does_not_declare_fails(tmp_path: Path) -> None:
    found = rib_node(tmp_path, KNOB, ball(20.0), knobs_trimmed_ml=0.5)
    assert found["pass"] and not found["change"]["pass"]  # true to its mask, 0.4 ml unaccounted
    grown = rib_node(tmp_path, ball(20.0), KNOB, knobs_trimmed_ml=0.9)
    assert not grown["change"]["pass"] and grown["change"]["addedMl"] > 0.8
    filled = rib_node(tmp_path, ball(20.0), KNOB, end_filled_ml=0.9)
    assert filled["change"]["pass"]


def test_a_piece_moved_in_from_a_neighbour_is_declared_by_its_fragment(tmp_path: Path) -> None:
    note = {"label": "rib_left_2", "ml": 0.9, "action": "moved", "to": "rib_left_1"}
    assert rib_node(tmp_path, ball(20.0), KNOB, fragments=[note])["change"]["pass"]
    left_out = {**note, "exported": False}  # never arrived: its neighbour was left out
    assert not rib_node(tmp_path, ball(20.0), KNOB, fragments=[left_out])["change"]["pass"]


def test_a_mirrored_node_is_not_held_to_its_own_sides_mask(tmp_path: Path) -> None:
    found = rib_node(tmp_path, ball(20.0, CENTRE + np.array([9.0, 0, 0])), ball(20.0),
                     mirrored=True)  # fmt: skip
    assert found["pass"] and found["change"]["pass"] and found["change"]["mirrored"]
    assert found["change"]["addedMl"] > 5


def test_a_missing_or_misplaced_mask_fails_the_node(tmp_path: Path) -> None:
    body = ac.Body("FMA_7987", icosphere(20.0, CENTRE))
    seg = tmp_path / "seg"
    seg.mkdir()
    found = ac.check_fidelity(body, {}, {}, tmp_path, seg)
    assert not found["pass"] and "rib_left_1.nii.gz is missing" in found["why"]
    save(seg / "rib_left_1.nii.gz", ball(20.0))
    found = ac.check_fidelity(body, {"contoured_mask": "gone.nii.gz"}, {}, tmp_path, seg)
    assert not found["pass"] and "gone.nii.gz is missing" in found["why"]
    nib.save(nib.Nifti1Image(ball(20.0)[:-2].astype(np.uint8), AFFINE), tmp_path / "cut.nii.gz")
    found = ac.check_fidelity(body, {"contoured_mask": "cut.nii.gz"}, {}, tmp_path, seg)
    assert not found["pass"] and "another grid" in found["why"]


def test_a_modelled_node_is_listed_and_not_measured(tmp_path: Path) -> None:
    body = ac.Body("FMA_85213", icosphere(20.0, CENTRE))
    found = ac.check_fidelity(body, {"modelled": True}, {}, tmp_path, tmp_path)
    assert found == {"against": "modelled", "pass": True}


# (c) Topology


def faults(mesh: ac.Mesh, several: bool = False) -> list[str]:
    return ac.topology_faults(ac.topology(mesh), several)


def tri(*corners: tuple[float, float, float]) -> Floats:
    return np.array([corners], dtype=np.float64)


FLOOR = tri((0, 0, 0), (4, 0, 0), (0, 4, 0))  # in the plane z = 0


def test_two_triangles_that_pass_through_each_other_cross() -> None:
    wall = tri((1, 1, -1), (1, 1, 1), (2, 1, 1))  # stands in y = 1, through the floor
    assert ac.crossing(FLOOR, wall).tolist() == [True]
    assert ac.crossing(wall, FLOOR).tolist() == [True]
    # Through the plane of the floor, but beside the floor itself.
    assert ac.crossing(FLOOR, tri((5, 5, -1), (5, 5, 1), (6, 5, 1))).tolist() == [False]


def test_triangles_that_only_touch_do_not_cross() -> None:
    corner_on_face = tri((1, 1, 0), (1, 1, 2), (2, 1, 2))
    edge_on_face = tri((1, 1, 0), (2, 1, 0), (1, 1, 2))
    # Through the floor's plane, exactly along the floor's own border x = 0.
    along_border = tri((0, 1, -1), (0, 3, -1), (0, 2, 1))
    # Its edge passes through the floor's border edge at one point only.
    through_border = tri((0, 2, -1), (0, 2, 1), (-3, 2, 1))
    apart = tri((1, 1, 0.5), (1, 1, 2), (2, 1, 2))
    for other in (corner_on_face, edge_on_face, along_border, through_border, apart):
        assert ac.crossing(FLOOR, other).tolist() == [False], other
        assert ac.crossing(other, FLOOR).tolist() == [False], other


def test_triangles_in_one_plane_cross_when_they_share_area() -> None:
    assert ac.crossing(FLOOR, tri((1, 1, 0), (3, 1, 0), (1, 3, 0))).tolist() == [True]
    assert ac.crossing(FLOOR, tri((3, 3, 0), (-1, 1, 0), (1, -1, 0))).tolist() == [True]
    beside = tri((4, 0, 0), (0, 4, 0), (4, 4, 0))  # shares the slanted edge, no area
    assert ac.crossing(FLOOR, beside).tolist() == [False]
    assert ac.crossing(FLOOR, tri((5, 5, 0), (7, 5, 0), (5, 7, 0))).tolist() == [False]


def test_a_triangle_without_area_crosses_nothing() -> None:
    assert ac.crossing(FLOOR, tri((1, 1, -1), (1, 1, 1), (1, 1, 1))).tolist() == [False]


def test_neighbouring_triangles_are_not_counted_as_crossing() -> None:
    # Two triangles folded flat onto each other along a shared edge, and a
    # third standing through both: only the pairs without a common vertex count.
    vertices = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [1, 3, 0],
                         [1, 1, -1], [1, 1, 1], [2, 1, 1]], dtype=np.float64)  # fmt: skip
    folded = ac.Mesh(vertices, np.array([[0, 1, 2], [1, 0, 3]]))
    assert len(ac.crossings(folded)) == 0
    pierced = ac.Mesh(vertices, np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6]]))
    assert ac.crossings(pierced).tolist() == [[0, 2], [1, 2]]


def test_crossings_of_large_meshes() -> None:
    """A node of 160 000 triangles: a clean sphere has none, and two spheres
    pushed into each other cross along the circle they share."""
    one = icosphere(30.0, CENTRE, 6)
    assert len(one.triangles) == 81920 and len(ac.crossings(one)) == 0
    two = joined(one, moved(one, (25.0, 3.0, 1.0)))
    pairs = ac.crossings(two)
    assert len(pairs) > 300
    assert ((pairs[:, 0] < 81920) & (pairs[:, 1] >= 81920)).all()  # one from each sphere
    # Every crossing lies on the circle: the same distance from both centres.
    middle = two.vertices[two.triangles[pairs.reshape(-1)]].mean(1)
    for centre in (CENTRE, CENTRE + np.array([25.0, 3.0, 1.0])):
        assert np.abs(np.linalg.norm(middle - centre, axis=1) - 30).max() < 1.5


def test_a_clean_cube_passes() -> None:
    found = ac.topology(cube((0, 0, 0), (10, 10, 10)))
    assert ac.topology_faults(found, several=False) == []
    assert found["triangles"] == 12 and found["volumeMl"] == pytest.approx(1.0)
    assert found["pieces"] == [
        {"triangles": 12, "volumeMl": 1.0, "openEdges": 0, "crossingFaces": 0}
    ]


def test_a_cube_with_a_hole_is_open() -> None:
    box = cube((0, 0, 0), (10, 10, 10))
    found = faults(ac.Mesh(box.vertices, box.triangles[1:]))
    assert found == ["3 open edges (holes)"]


def test_a_cube_with_a_flipped_face_is_wound_against_itself() -> None:
    box = cube((0, 0, 0), (10, 10, 10))
    triangles = box.triangles.copy()
    triangles[0] = triangles[0][::-1]
    assert faults(ac.Mesh(box.vertices, triangles)) == [
        "3 edges whose faces are wound against each other"
    ]


def test_a_cube_with_a_duplicated_inner_face_is_not_manifold() -> None:
    box = cube((0, 0, 0), (10, 10, 10))
    doubled = ac.Mesh(box.vertices, np.vstack([box.triangles, box.triangles[:1, ::-1]]))
    assert faults(doubled)[0] == "3 edges used by more than two faces"
    # A wall across the inside, on the cube's own corners.
    walled = ac.Mesh(box.vertices, np.vstack([box.triangles, [[0, 3, 7], [0, 7, 4]]]))
    assert "edges used by more than two faces" in faults(walled)[0]


def test_a_cube_with_a_second_loose_shell_is_two_pieces() -> None:
    both = joined(cube((0, 0, 0), (10, 10, 10)), cube((20, 0, 0), (21, 1, 1)))
    assert faults(both) == ["2 pieces, not one (the others: [0.001] ml)"]
    # In a set of pieces it is a loose blob, under MIN_PIECE_ML.
    assert faults(both, several=True) == ["1 loose pieces under 0.5 ml (smallest 0.001 ml)"]
    two = joined(cube((0, 0, 0), (10, 10, 10)), cube((20, 0, 0), (30, 10, 10)))
    assert faults(two, several=True) == []


def test_two_shells_pushed_into_each_other_cross() -> None:
    both = joined(cube((0, 0, 0), (10, 10, 10)), cube((5, 5, 5), (15, 15, 15)))
    found = ac.topology(both)
    assert found["crossingPairs"] > 0 and found["crossingBetweenPieces"] == found["crossingPairs"]
    assert all(piece["crossingFaces"] > 0 for piece in found["pieces"])
    crossing = f"{found['crossingPairs']} pairs of faces cross"
    assert ac.topology_faults(found, several=True) == [crossing]
    assert ac.topology_faults(found, several=False) == [
        crossing, "2 pieces, not one (the others: [1.0] ml)"
    ]  # fmt: skip


def test_a_cube_turned_inside_out_has_no_volume() -> None:
    box = cube((0, 0, 0), (10, 10, 10))
    assert faults(ac.Mesh(box.vertices, box.triangles[:, ::-1])) == [
        "1 pieces inside out or without volume"
    ]


def test_two_shells_joined_at_one_corner_are_pinched() -> None:
    both = joined(cube((0, 0, 0), (10, 10, 10)), cube((10, 10, 10), (20, 20, 20)))
    assert faults(both, several=True) == ["1 vertices where two sheets are pinched together"]


def test_a_face_without_area_is_counted() -> None:
    box = cube((0, 0, 0), (10, 10, 10))
    # A sliver along an edge: three corners in a line.
    vertices = np.vstack([box.vertices, [[0, 0, 5]]])
    sliver = ac.Mesh(vertices, np.vstack([box.triangles, [[0, 8, 1]]]))
    assert ac.topology(sliver)["degenerateFaces"] == 1
    assert "1 faces without area" in faults(sliver)
    assert faults(ac.Mesh(box.vertices, np.zeros((0, 3), dtype=np.int64))) == ["no faces"]


# Between two nodes


def body(label: str, mesh: ac.Mesh) -> ac.Body:
    return ac.Body(ac.NAMES[label], mesh)


def test_the_gap_between_two_boxes() -> None:
    one = body("sternum", cube((0, 0, 0), (10, 10, 10)))
    assert ac.gap(one, body("rib_left_1", cube((12, 2, 2), (14, 6, 6)))) == pytest.approx(2.0)
    assert ac.gap(body("rib_left_1", cube((12, 2, 2), (14, 6, 6))), one) == pytest.approx(2.0)
    assert ac.gap(one, body("rib_left_1", cube((8, 2, 2), (14, 6, 6)))) == 0.0  # it enters
    # Narrowed to one end of the first: its far face, 10 mm further off.
    far = one.points[one.points[:, 0] < 0.5]
    assert ac.gap(one, body("rib_left_1", cube((12, 2, 2), (14, 6, 6))), far) == pytest.approx(
        np.hypot(12 - 0.5, 0), abs=0.6
    )


def test_two_boxes_that_enter_each_other_give_depth_and_volume() -> None:
    one = body("sternum", cube((0, 0, 0), (10, 10, 10)))
    found = ac.entered(one, body("rib_left_1", cube((8, 2, 2), (14, 6, 6))))
    assert found is not None
    assert found["sharedMl"] == pytest.approx(2 * 4 * 4 / 1000, rel=0.05)
    # The rib's corners lie 2 mm inside the sternum; the sternum's face, at
    # most 2 mm inside the rib (the middle of a 4 mm wide box).
    assert found["secondInFirstMm"] == pytest.approx(2.0, abs=0.05)
    assert found["firstInSecondMm"] == pytest.approx(2.0, abs=0.4)
    assert ac.entered(one, body("rib_left_1", cube((10.5, 2, 2), (14, 6, 6)))) is None
    assert ac.entered(one, body("rib_left_1", cube((40, 2, 2), (44, 6, 6)))) is None
    # A small box wholly inside: no surface near the other's, yet it is in.
    inner = ac.entered(one, body("rib_left_1", cube((4.5, 4.5, 4.5), (5.5, 5.5, 5.5))))
    assert inner is not None and inner["sharedMl"] == pytest.approx(0.001, rel=0.2)


# (d) Anatomy, on a chest of boxes and tubes. RAS: x to the body's right, y
# forward, z up. Rib n lies level at height level(n); the vertebrae are boxes
# behind, the sternum a box in front, every cartilage a tube.

SIDES = {"left": -1.0, "right": 1.0}


def level(n: int) -> float:
    return 150.0 - 14.0 * (n - 1)


def rib(side: str, n: int, off: float = 0.0) -> ac.Mesh:
    """A rib: back along the vertebra's flank, 1 mm (and off) from it, then
    round to its front end at x 62, y 92."""
    x, z = SIDES[side], level(n)
    return tube([(x * (40 + off), -5, z), (x * (40 + off), 30, z), (x * 62, 92, z)], 4.0)


def vertebra(n: int) -> ac.Mesh:
    return cube((-35, -25, level(n) - 9.5), (35, 5, level(n) + 3.5))


def cartilage(side: str, n: int, to: int | None = None, start: float = 3.0) -> ac.Mesh:
    """The cartilage of one of ribs 1 to 7: from start mm inside its rib's
    end to 3 mm inside the sternum's flank, at the height of rib `to`."""
    x = SIDES[side]
    back = np.array([x * 62, 92, level(n)]) - start * np.array([x * 22, 62, 0]) / np.hypot(22, 62)
    return tube([tuple(back), (x * 12, 106, level(n if to is None else to))], 3.0)


LAP_STARTS = {8: 4.0, 9: 12.0, 10: 20.0}  # mm along its rib's line before a band turns up


def lapping(side: str, n: int, run: float = 24.0, under: float = 5.5) -> ac.Mesh:
    """The cartilage of one of ribs 8 to 10 as a piece of its own: out from
    its rib's end, up, and then run mm along under the cartilage above,
    its middle `under` below that one's (two 3 mm tubes: 6 is touching)."""
    x = SIDES[side]
    along = np.array([x * -50.0, 14.0, 0.0]) / np.hypot(50, 14)  # the seventh's line
    tip = np.array([x * 62, 92, level(n)])
    height = level(7) - 5.5 * (n - 8) - under
    start = tip + LAP_STARTS[n] * along
    turn = np.array([start[0], start[1], height])
    return tube([tuple(tip), tuple(start), tuple(turn), tuple(turn + run * along)], 3.0)


def arch(side: str, ribs: tuple[int, ...] = (10, 9, 8, 7)) -> ac.Mesh:
    """One piece of cartilage up through the front ends of the lower ribs
    and on from the highest of them into the sternum."""
    x = SIDES[side]
    path = [(x * 62, 92, level(n)) for n in ribs]
    return tube([(x * 62, 92, level(ribs[0]) - 1), *path, (x * 12, 106, level(ribs[-1]))], 3.0)


def chest(cartilages: list[ac.Mesh] | None = None, **changed: ac.Mesh) -> dict[str, ac.Body]:
    """The whole chest by label; cartilages replaces the cartilage pieces,
    and any other keyword replaces that bone."""
    meshes = {"sternum": cube((-15, 100, -20), (15, 112, 160))}
    for n in range(1, 13):
        meshes[f"vertebrae_T{n}"] = vertebra(n)
        for side in SIDES:
            meshes[f"rib_{side}_{n}"] = rib(side, n)
    if cartilages is None:
        cartilages = whole_cartilages()
    meshes["costal_cartilages"] = joined(*cartilages)
    meshes.update(changed)
    return {label: body(label, mesh) for label, mesh in meshes.items()}


def whole_cartilages(but: tuple[tuple[str, int], ...] = ()) -> list[ac.Mesh]:
    """Cartilages 1 to 7 to the sternum and 8 to 10 lapping onto the one
    above, each a piece of its own, on both sides, but for some."""
    return [
        cartilage(side, n) if n <= 7 else lapping(side, n)
        for side in SIDES
        for n in range(1, 11)
        if (side, n) not in but
    ]


def test_a_ribs_two_ends_are_told_apart() -> None:
    back, front = ac.rib_tips(body("rib_left_3", rib("left", 3)))
    assert np.linalg.norm(back - [-40, -5, level(3)]) < 4.5  # on the rim of its flat end
    assert np.linalg.norm(front - [-62, 92, level(3)]) < 4.5


def test_every_rib_of_the_chest_meets_its_vertebrae() -> None:
    rows, failures = ac.rib_joints(chest(), {})
    assert failures == [] and len(rows) == 24 and all(row["pass"] for row in rows)
    by_rib = {row["rib"]: row for row in rows}
    assert by_rib["rib_left_1"]["gaps"] == {"T1": pytest.approx(1.0, abs=0.05)}
    # Ribs 2 to 9 also meet the vertebra above, across the corner of its body.
    assert list(by_rib["rib_right_5"]["gaps"]) == ["T5", "T4"]
    assert by_rib["rib_right_5"]["gaps"]["T4"] == pytest.approx(np.hypot(5, 4.5) - 4, abs=0.1)
    assert list(by_rib["rib_left_10"]["gaps"]) == ["T10"]
    assert by_rib["rib_left_7"]["nearest"] == "T7"


def test_a_rib_standing_off_its_vertebra_fails() -> None:
    rows, failures = ac.rib_joints(chest(rib_left_5=rib("left", 5, off=5.0)), {})
    assert [f.rsplit(" ", 4)[0] for f in failures] == ["rib_left_5 is", "rib_left_5 is"]
    gaps = next(row["gaps"] for row in rows if row["rib"] == "rib_left_5")
    assert gaps["T5"] == pytest.approx(6.0, abs=0.1)
    assert gaps["T4"] == pytest.approx(np.hypot(10, 4.5) - 4, abs=0.1)
    assert [row["rib"] for row in rows if not row["pass"]] == ["rib_left_5"]


def test_a_dropped_vertebra_fails_no_rib_but_a_missing_one_does() -> None:
    scene = chest()
    del scene["vertebrae_T1"], scene["vertebrae_T4"]
    cut = {"vertebrae_T1": "cut by the scan's edge (unscanned padding)"}
    rows, failures = ac.rib_joints(scene, cut)
    by_rib = {row["rib"]: row for row in rows}
    assert by_rib["rib_left_1"]["gaps"] == {"T1": "vertebra not in the build"}
    assert by_rib["rib_left_1"]["pass"] and by_rib["rib_left_2"]["pass"]
    assert by_rib["rib_left_2"]["gaps"]["T1"] == "vertebra not in the build"
    # T4 is neither built nor listed as dropped: ribs 4 and 5 cannot meet it.
    assert not by_rib["rib_right_4"]["pass"] and not by_rib["rib_right_5"]["pass"]
    assert "rib_right_4: T4 is neither built nor listed as dropped" in failures
    assert len(failures) == 4


def test_the_chests_cartilages_pass_every_rule() -> None:
    report, failures = ac.cartilages(chest(), None)
    assert failures == {"cartilage to rib": [], "floating ribs": [], "sternum contacts": [],
                        "costal arch": []}  # fmt: skip
    for side in SIDES:
        mine = [p for p in report["patches"] if p["side"] == side]
        assert [p["rib"] for p in mine] == [1, 2, 3, 4, 5, 6, 7]
        for n, patch in enumerate(mine, start=1):
            assert patch["centre"][2] == pytest.approx(level(n), abs=0.5)
            # A 3 mm tube through the flank, and CONTACT_MM round it.
            assert patch["areaMm2"] == pytest.approx(np.pi * 3.5**2, rel=0.25)
    reach = {row["rib"]: row for row in report["ribs"]}
    assert all(reach[f"rib_left_{n}"]["gapMm"] == 0 for n in range(1, 11))
    assert len({reach[f"rib_left_{n}"]["piece"] for n in range(1, 11)}) == 10
    assert [(row["side"], row["cartilage"], row["state"]) for row in report["arch"]] == [
        (side, n, "lapped") for side in SIDES for n in (8, 9, 10)
    ]
    eighth, ninth = report["arch"][0], report["arch"][1]
    assert eighth["gapMm"] == 0.0  # 5.5 mm between the middles of two 3 mm tubes
    assert eighth["lapMm"] == pytest.approx(24.0, abs=4.0)
    assert ninth["lapMm"] == pytest.approx(18.0, abs=5.0)  # it starts 8 mm further along
    # The eighth ends 28 mm along the seventh, whose patch is 50 mm along it.
    assert eighth["tipToEndAboveMm"] == pytest.approx(24.0, abs=4.0)
    assert ninth["tipToEndAboveMm"] == pytest.approx(8.0, abs=4.0)
    assert reach["rib_right_11"]["piece"] is None and reach["rib_right_11"]["gapMm"] > 5
    assert report["joined"] == []


def test_a_cartilage_that_stops_short_of_its_rib_fails() -> None:
    short = whole_cartilages(but=(("left", 3),)) + [cartilage("left", 3, start=-6.0)]
    report, failures = ac.cartilages(chest(short), None)
    assert len(failures["cartilage to rib"]) == 1
    assert failures["cartilage to rib"][0].startswith("rib_left_3: the cartilage is 2.")
    # Its patch on the sternum is then no rib's.
    assert failures["sternum contacts"] == ["left: no patch of its own for cartilage [3]"]
    assert failures["floating ribs"] == [] and failures["costal arch"] == []


def test_cartilage_on_a_floating_rib_fails() -> None:
    stub = tube([(-62, 90, level(11)), (-50, 100, level(11))], 3.0)
    _, failures = ac.cartilages(chest(whole_cartilages() + [stub]), None)
    assert failures["floating ribs"] == ["rib_left_11: cartilage 0.00 mm from its front end"]
    assert failures["sternum contacts"] == []


def test_two_cartilages_at_one_place_on_the_sternum_fail() -> None:
    # The fifth runs down to where the sixth ends, a millimetre beside it.
    fifth = tube([(60.5, 88, level(5)), (12, 106.5, level(6) + 1)], 3.0)
    shared = whole_cartilages(but=(("right", 5),)) + [fifth]
    report, failures = ac.cartilages(chest(shared), None)
    assert [p.get("rib") for p in report["patches"] if p["side"] == "right"] == [
        1, 2, 3, 4, None, 7
    ]  # fmt: skip
    both = next(p for p in report["patches"] if len(p["pieces"]) == 2)
    assert failures["sternum contacts"] == [
        f"right: the patch at z {both['centre'][2]} is shared by 2 cartilage pieces",
        "right: the piece of cartilages [5] makes 0 patches, not 1",
        "right: the piece of cartilages [6] makes 0 patches, not 1",
        "right: 6 contact patches, not 7",
        "right: no patch of its own for cartilage [5, 6]",
    ]


def test_cartilages_that_reach_the_sternum_out_of_order_fail() -> None:
    swapped = whole_cartilages(but=(("left", 3), ("left", 4)))
    swapped += [cartilage("left", 3, to=4), cartilage("left", 4, to=3)]
    _, failures = ac.cartilages(chest(swapped), None)
    assert failures["sternum contacts"] == [
        "left: the patches run [1, 2, 4, 3, 5, 6, 7] from the top"
    ]


def test_a_lower_cartilage_must_lie_against_the_one_above() -> None:
    off = whole_cartilages(but=(("left", 8),)) + [lapping("left", 8, under=7.5)]
    report, failures = ac.cartilages(chest(off), None)
    row = next(r for r in report["arch"] if r["side"] == "left" and r["cartilage"] == 8)
    assert row["state"] == "apart" and row["gapMm"] == pytest.approx(1.5, abs=0.3)
    assert failures["costal arch"] == [
        f"left: cartilage 8 lies against cartilage 7 over 0.0 mm (least gap {row['gapMm']} mm), "
        "under 10.0"
    ]
    assert failures["sternum contacts"] == []


def test_a_lap_shorter_than_a_cartilage_is_wide_fails() -> None:
    short = whole_cartilages(but=(("right", 10),)) + [lapping("right", 10, run=1.0)]
    report, failures = ac.cartilages(chest(short), None)
    row = report["arch"][-1]
    assert row["state"] == "apart" and row["gapMm"] == 0.0 and row["lapMm"] < 10
    assert failures["costal arch"] == [
        f"right: cartilage 10 lies against cartilage 9 over {row['lapMm']} mm "
        "(least gap 0.0 mm), under 10.0"
    ]


def test_cartilages_fused_into_one_piece_are_reported_as_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused = [cartilage(side, n) for side in SIDES for n in range(1, 7)]
    fused += [arch("left")] + [cartilage("right", 7)] + [lapping("right", n) for n in (8, 9, 10)]
    report, failures = ac.cartilages(chest(fused), None)
    left = [row for row in report["arch"] if row["side"] == "left"]
    assert [row["state"] for row in left] == ["fused"] * 3
    assert all(row["with"] == [7, 8, 9, 10] and row["lapMm"] is None for row in left)
    assert failures["costal arch"] == [
        f"left: cartilage {n} is fused with cartilage {n - 1} (one piece: cartilages [7, 8, 9, 10])"
        for n in (8, 9, 10)
    ]
    assert failures["sternum contacts"] == []
    monkeypatch.setattr(ac, "FUSED_FAILS", False)
    report, failures = ac.cartilages(chest(fused), None)
    assert failures["costal arch"] == [] and report["arch"][0]["state"] == "fused"


def test_a_lower_cartilage_without_cartilage_cannot_lap() -> None:
    _, failures = ac.cartilages(chest(whole_cartilages(but=(("left", 9),))), None)
    assert failures["costal arch"] == [
        "left: cartilage 9 is missing, so 9 cannot lap onto 8",
        "left: cartilage 9 is missing, so 10 cannot lap onto 9",
    ]


def test_an_eighth_cartilage_on_the_sternum_fails() -> None:
    own = whole_cartilages(but=(("right", 8),)) + [arch("right", ribs=(8,))]
    _, failures = ac.cartilages(chest(own), None)
    assert failures["sternum contacts"] == ["right: 8 contact patches, not 7"]
    assert failures["costal arch"][0].startswith(
        "right: cartilage 8 lies against cartilage 7 over 0.0 mm"
    )


def test_upper_cartilages_in_one_piece_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # One tube from the sternum round through the ends of ribs 1 and 2 and back.
    hairpin = tube([(-12, 106, level(1)), (-62, 92, level(1)), (-62, 92, level(2)),
                    (-12, 106, level(2))], 3.0)  # fmt: skip
    pieces = whole_cartilages(but=(("left", 1), ("left", 2))) + [hairpin]
    report, failures = ac.cartilages(chest(pieces), None)
    assert report["joined"] == [{"side": "left", "cartilages": [1, 2]}]
    assert failures["sternum contacts"] == []  # two patches, in order: reported only
    monkeypatch.setattr(ac, "JOINED_FAILS", True)
    _, failures = ac.cartilages(chest(pieces), None)
    assert failures["sternum contacts"] == ["left: cartilages [1, 2] are one piece"]


def test_no_cartilage_or_no_sternum_fails_the_rules_that_need_them() -> None:
    scene = chest()
    del scene["costal_cartilages"]
    report, failures = ac.cartilages(scene, None)
    assert report["patches"] == [] and failures["floating ribs"] == []
    for rule in ("cartilage to rib", "sternum contacts", "costal arch"):
        assert failures[rule] == ["no costal cartilages in the build"]
    scene = chest()
    del scene["sternum"]
    assert ac.cartilages(scene, None)[1]["sternum contacts"] == ["no sternum in the build"]


def band(side: str, n: int, joins: str, end: tuple[float, float, float]) -> dict[str, Any]:
    return {"rib": f"rib_{side}_{n}", "side": side, "number": n, "length_mm": 50.0,
            "joins": joins, "tip_ras_mm": [SIDES[side] * 62, 92, level(n)],
            "end_ras_mm": list(end)}  # fmt: skip


def test_the_manifests_band_records_are_held_to_the_rules() -> None:
    good = [band("left", 3, "sternum", (-12, 106, level(3))),
            band("left", 8, "rib_left_7", (-48.5, 95.9, level(7) - 5.5))]  # fmt: skip
    report, failures = ac.cartilages(chest(), good)
    assert failures["sternum contacts"] == [] and failures["costal arch"] == []
    assert report["bands"] == good
    bad = [
        band("left", 3, "rib_left_2", (-30, 100, level(3))),  # joins a band, ends in the open
        band("left", 8, "sternum", (-14, 106, level(7))),  # ends in the sternum
        band("right", 9, "rib_right_8", (80, 92, level(8))),  # ends in the air
    ]
    _, failures = ac.cartilages(chest(), bad)
    assert failures["sternum contacts"] == [
        "rib_left_3: the manifest says it joins rib_left_2",
        "rib_left_3: its band ends 15.0 mm from the sternum",
    ]
    assert failures["costal arch"] == [
        "rib_left_8: the manifest says it joins sternum",
        "rib_left_8: its band ends 1.0 mm from the sternum",
        "rib_right_9: its band ends 17.4 mm out of the cartilage",
    ]


# The sternum: a plate 10 mm thick, 50 mm of manubrium, 100 mm of body and a
# xiphoid process, bent in the sagittal plane where manubrium and body meet.

TOP = 160.0
MANUBRIUM, BODY = 50.0, 100.0


def plate_axis(lean_manubrium: float, lean_body: float, length: float) -> tuple[Floats, Floats]:
    """The plate's mid line in the sagittal plane, every half millimetre from
    the top down: its points (y, z) and its forward normals. It leans back
    by lean_manubrium degrees above the joint and lean_body below, turning
    from one to the other over 10 mm."""
    s = np.arange(0.0, length + 0.25, 0.5)
    lean = np.radians(np.interp(s, [MANUBRIUM - 5, MANUBRIUM + 5], [lean_manubrium, lean_body]))
    step = np.stack([np.sin(lean), -np.cos(lean)], 1) * 0.5  # down, and forward as it leans
    line = np.array([100.0, TOP]) + np.vstack([[0, 0], np.cumsum(step[:-1], 0)])
    return line, np.stack([np.cos(lean), np.sin(lean)], 1)


def sternum_plate(
    lean_manubrium: float = 30.0, lean_body: float = 10.0, notch: float = 4.0,
    xiphoid: float = 30.0, xiphoid_half_width: float = 6.0,
) -> ac.Mesh:  # fmt: skip
    """The plate as a closed mesh: 30 mm half width at the top narrowing to
    18 at the joint, 18 down the body, then the xiphoid process; its upper
    border dips by notch mm in the middle, between two level corners."""
    length = MANUBRIUM + BODY + xiphoid
    line, normal = plate_axis(lean_manubrium, lean_body, length)
    u = np.linspace(-1, 1, 61)[:, None]
    dip = notch * np.exp(-((u * 30 / 8) ** 2))  # the notch: 8 mm wide at the top
    s = dip + np.linspace(0, 1, 2 * int(length) + 1)[None, :] * (length - dip)
    half = np.interp(s, [0, MANUBRIUM, MANUBRIUM + BODY, MANUBRIUM + BODY + 3, length],
                     [30, 18, 18, xiphoid_half_width, xiphoid_half_width])  # fmt: skip
    k = s / 0.5
    middle = np.stack([np.interp(k, np.arange(len(line)), line[:, d]) for d in range(2)], -1)
    out = np.stack([np.interp(k, np.arange(len(line)), normal[:, d]) for d in range(2)], -1)
    sheets = [
        np.stack(
            [u * half, middle[..., 0] + side * out[..., 0], middle[..., 1] + side * out[..., 1]], -1
        )
        for side in (5.0, -5.0)
    ]
    rows, columns = s.shape
    number = np.arange(rows * columns).reshape(rows, columns)
    a, b, c, d = number[:-1, :-1], number[1:, :-1], number[1:, 1:], number[:-1, 1:]
    front = np.vstack(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    )
    # The rim: round the border of the front sheet, joined to the back sheet's.
    border = np.concatenate([number[:, 0], number[-1, 1:], number[-2::-1, -1], number[0, -2:0:-1]])
    following = np.roll(border, -1)
    back = rows * columns
    rim = np.vstack([np.stack([border, following, following + back], 1),
                     np.stack([border, following + back, border + back], 1)])  # fmt: skip
    mesh = ac.Mesh(
        np.vstack([sheet.reshape(-1, 3) for sheet in sheets]),
        np.vstack([front, front[:, ::-1] + back, rim[:, ::-1]]).astype(np.int64),
    )
    return mesh if ac.signed_volume(mesh) > 0 else ac.Mesh(mesh.vertices, mesh.triangles[:, ::-1])


def joint_z(lean_manubrium: float = 30.0) -> float:
    return TOP - MANUBRIUM * float(np.cos(np.radians(lean_manubrium)))


def seventh(
    lean_manubrium: float = 30.0, lean_body: float = 10.0, rib: int = 7
) -> list[dict[str, Any]]:
    """The two lowest contact patches, at the lower end of the body."""
    line, _ = plate_axis(lean_manubrium, lean_body, MANUBRIUM + BODY)
    y, z = line[-1]
    return [{"side": side, "rib": rib, "centre": [x, y, z], "zLow": z - 4, "zHigh": z + 4}
            for side, x in (("left", -18.0), ("right", 18.0))]  # fmt: skip


def shape(
    mesh: ac.Mesh, second: list[float] | None = None, lowest: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    return ac.sternum_shape(
        body("sternum", mesh),
        [joint_z()] * 2 if second is None else second,
        seventh() if lowest is None else lowest,
    )


def test_the_plate_is_a_clean_closed_surface() -> None:
    assert faults(sternum_plate()) == []


def test_a_sternum_with_notch_angle_and_xiphoid_passes() -> None:
    report, failures = shape(sternum_plate())
    assert failures == {"jugular notch": [], "sternal angle": [], "xiphoid": []}
    angle = report["sternalAngle"]
    assert angle["frontSurfaceDeg"] == pytest.approx(20.0, abs=1.0)
    assert angle["midLineDeg"] == pytest.approx(20.0, abs=1.0)
    assert angle["about"] == "the second cartilages' contact"
    assert angle["bendZ"] == pytest.approx(joint_z(), abs=3.0)
    assert angle["bendAboveSecondCartilageMm"] == pytest.approx(0.0, abs=3.0)
    notch = report["jugularNotch"]
    assert notch["leanDeg"] == pytest.approx(30.0, abs=3.0)
    # 4 mm down the plate; seen from the front, the leaning plate shows less of it.
    assert notch["enFace"]["depthMm"] == pytest.approx(4.0, abs=0.4)
    assert notch["frontView"]["depthMm"] == pytest.approx(4.0 * np.cos(np.radians(30)), abs=0.4)
    assert notch["enFace"]["spanMm"] > 16  # the corners: beyond the 8 mm wide dip
    xiphoid = report["xiphoid"]
    assert xiphoid["below"] == [7, 7]
    assert xiphoid["lengthMm"] == pytest.approx(30.0, abs=3.0)
    assert xiphoid["widthMm"] == pytest.approx(12.0, abs=1.0)
    assert xiphoid["bodyWidthMm"] == pytest.approx(36.0, abs=1.0)


def test_a_straight_or_a_sharply_bent_sternum_fails_the_angle() -> None:
    report, failures = shape(sternum_plate(lean_manubrium=10.0), second=[joint_z(10.0)] * 2,
                             lowest=seventh(10.0))  # fmt: skip
    assert report["sternalAngle"]["frontSurfaceDeg"] == pytest.approx(0.0, abs=0.5)
    assert len(failures["sternal angle"]) >= 1
    assert failures["sternal angle"][0].startswith("the front surface bends ")
    assert failures["sternal angle"][0].endswith(" deg, outside 3.0 to 31.0")
    report, failures = shape(sternum_plate(lean_manubrium=50.0), second=[joint_z(50.0)] * 2,
                             lowest=seventh(50.0))  # fmt: skip
    assert report["sternalAngle"]["frontSurfaceDeg"] == pytest.approx(40.0, abs=1.5)
    assert failures["sternal angle"] == [
        f"the front surface bends {report['sternalAngle']['frontSurfaceDeg']} deg, "
        "outside 3.0 to 31.0"
    ]


def test_the_bend_must_lie_at_the_second_cartilages() -> None:
    report, failures = shape(sternum_plate(), second=[joint_z() - 20.0] * 2)
    off = report["sternalAngle"]["bendAboveSecondCartilageMm"]
    assert off == pytest.approx(20.0, abs=3.0)
    assert (
        f"the bend lies {off:+.1f} mm from the second cartilages' contact"
        in failures["sternal angle"]
    )
    report, failures = shape(sternum_plate(), second=[])
    assert report["sternalAngle"]["about"] == "the bend"
    assert report["sternalAngle"]["frontSurfaceDeg"] == pytest.approx(20.0, abs=1.5)
    assert failures["sternal angle"] == ["no second cartilage contact to place the bend by"]


def test_a_sternum_without_a_notch_fails() -> None:
    report, failures = shape(sternum_plate(notch=0.0))
    assert report["jugularNotch"]["enFace"]["depthMm"] < 0.2
    assert failures["jugular notch"] == [
        f"{report['jugularNotch']['enFace']['depthMm']} mm deep en face, under 1.5"
    ]
    assert shape(sternum_plate(notch=1.0))[1]["jugular notch"] != []
    assert shape(sternum_plate(notch=2.0))[1]["jugular notch"] == []


def test_a_short_or_a_broad_xiphoid_fails() -> None:
    report, failures = shape(sternum_plate(xiphoid=12.0))
    assert report["xiphoid"]["lengthMm"] == pytest.approx(12.0, abs=3.0)
    assert failures["xiphoid"] == [
        f"{report['xiphoid']['lengthMm']} mm below the lowest contacts, under 20.0"
    ]
    report, failures = shape(sternum_plate(xiphoid_half_width=18.0))
    assert failures["xiphoid"] == [
        f"{report['xiphoid']['widthMm']} mm wide, not narrower than the body's "
        f"{report['xiphoid']['bodyWidthMm']} mm"
    ]


def test_the_xiphoid_is_measured_below_the_seventh_cartilages() -> None:
    _, failures = shape(sternum_plate(), lowest=seventh(rib=6))
    assert failures["xiphoid"] == [
        "the lowest contacts are not the two seventh cartilages': [('left', 6), ('right', 6)]"
    ]
    _, failures = shape(sternum_plate(), lowest=seventh()[:1])
    assert failures["xiphoid"] == [
        "the lowest contacts are not the two seventh cartilages': [('left', 7)]"
    ]
    report, failures = shape(sternum_plate(), lowest=[])
    assert failures["xiphoid"] == ["no cartilage contact to measure below"]
    assert "lengthMm" not in report["xiphoid"]


def test_the_deepest_dip_of_a_profile() -> None:
    across = np.arange(9.0)
    found = ac.dip(np.array([0, 2, 5, 3, 1, 2, 4, 6, 3.0]), across)
    # Between the corners at 2 (5 high) and 7 (6 high): the line joining them is 5.4 at 4.
    assert found == {"depthMm": 4.4, "spanMm": 5.0}
    assert ac.dip(np.array([0, 1, 3, 5, 3, 1, 0.0]), np.arange(7.0))["depthMm"] == 0.0  # a dome
    assert ac.dip(np.array([np.nan, 1.0, np.nan]), np.arange(3.0))["depthMm"] == 0.0


# The whole check


def build(tmp_path: Path, raw_lump: bool = False, contoured: bool = False,
          declared: float = 0.0) -> tuple[Path, Path, Path]:  # fmt: skip
    """A build of three nodes in tmp_path: a sternum (a flattened ellipsoid)
    and a left first rib (a ball) meshed exactly from their masks, and a
    modelled cartilage set. Returns the GLB, the manifest and the masks."""
    exact, mask = flat_ellipsoid()
    meshes = {
        "sternum": exact,
        "rib_left_1": icosphere(10.0, CENTRE + [0, 0, 22.0], 4),
        "costal_cartilages": icosphere(5.0, CENTRE + [0, 0, -25.0], 3),
    }
    centre = np.array([0.01, -0.02, 0.03])
    tmp_path.mkdir(parents=True, exist_ok=True)
    glb = tmp_path / "c.skeleton.glb"
    write_skeleton(glb, meshes, centre)
    seg = tmp_path / "seg"
    save(seg / "sternum.nii.gz", mask.data)
    rib_mask = ball(10.0, CENTRE + [0, 0, 22.0])
    lump = ball(4.0, CENTRE + [16.0, 0, 30.0])  # 0.27 ml beside the rib, in the raw mask only
    save(seg / "rib_left_1.nii.gz", rib_mask | lump if raw_lump else rib_mask)
    nodes: list[dict[str, Any]] = [
        {"name": ac.NAMES[label], "label": label, "modelled": label == "costal_cartilages",
         "pieces": "several" if label == "costal_cartilages" else "one"}
        for label in meshes
    ]  # fmt: skip
    if contoured:
        save(tmp_path / "c.skeleton.contoured" / "rib_left_1.nii.gz", rib_mask)
        save(tmp_path / "c.skeleton.contoured" / "sternum.nii.gz", mask.data)
        nodes[0]["contoured_mask"] = "c.skeleton.contoured/sternum.nii.gz"
        nodes[1]["contoured_mask"] = "c.skeleton.contoured/rib_left_1.nii.gz"
        nodes[1]["knobs_trimmed_ml"] = declared
    manifest = tmp_path / "c.skeleton.manifest.json"
    manifest.write_text(json.dumps({
        "centre_offset_m": centre.tolist(), "nodes": nodes, "fragments": [],
        "dropped": [{"label": "rib_left_12", "reason": "cut by the scan's edge (axis 2)"},
                    {"label": "rib_right_12", "reason": "its pair rib_left_12 is missing"}],
    }))  # fmt: skip
    return glb, manifest, seg


def rules(report: dict[str, Any]) -> dict[str, Any]:
    return {r["rule"]: r for r in report["rules"]}


def test_check_reports_every_node_and_every_rule(tmp_path: Path) -> None:
    report = ac.check(*build(tmp_path))
    assert list(report["nodes"]) == ["FMA_7485", "FMA_7987", "FMA_85213"]
    sternum, rib, cartilage = report["nodes"].values()
    assert sternum["label"] == "sternum" and sternum["fidelity"]["against"] == "raw mask"
    assert sternum["fidelity"]["pass"] and rib["fidelity"]["pass"]
    assert rib["fidelity"]["maskToMesh"]["mean"] < 0.3
    assert cartilage["fidelity"] == {"against": "modelled", "pass": True}
    assert all(node["topology"]["pass"] for node in report["nodes"].values())
    found = rules(report)
    assert list(found) == [
        "nodes", "fidelity", "declared changes", "topology", "entering", "rib pairs",
        "rib joints", "cartilage to rib", "floating ribs", "sternum contacts", "costal arch",
        "jugular notch", "sternal angle", "xiphoid",
    ]  # fmt: skip
    assert found["nodes"]["pass"] and found["fidelity"]["pass"] and found["topology"]["pass"]
    assert "2 against the raw mask; 1 modelled, not measured" in found["fidelity"]["detail"]
    assert found["declared changes"]["pass"] is None
    assert "measured against the raw masks" in found["declared changes"]["detail"]
    assert found["entering"]["pass"] is None and report["entering"] == []
    assert found["rib pairs"]["pass"] is False
    assert "pair 1 is missing from the build;" in found["rib pairs"]["detail"]
    assert ("pair 12 is cut by the scan (manifest: cut by the scan's edge (axis 2); "
            "its pair rib_left_12 is missing)") in found["rib pairs"]["detail"]  # fmt: skip
    assert report["ribPairs"] == {"present": [], "cutByScan": [12], "missing": list(range(1, 12))}
    assert report["notes"] == []
    assert found["sternum contacts"]["pass"] is False
    assert report["pass"] is False
    assert report["limits"]["meanMm"] == 1.0 and report["limits"]["p95Mm"] == 2.0


def test_check_against_contoured_masks_holds_them_to_the_manifest(tmp_path: Path) -> None:
    one = tmp_path / "raw"
    report = ac.check(*build(one, raw_lump=True))
    assert not report["nodes"]["FMA_7987"]["fidelity"]["pass"]  # the trimmed lump shows
    assert "FMA_7987 rib_left_1: mesh to mask" in rules(report)["fidelity"]["detail"]
    two = tmp_path / "declared"
    report = ac.check(*build(two, raw_lump=True, contoured=True, declared=0.27))
    rib = report["nodes"]["FMA_7987"]["fidelity"]
    assert rib["against"] == "contoured mask" and rib["pass"] and rib["change"]["pass"]
    assert rules(report)["fidelity"]["pass"] and rules(report)["declared changes"]["pass"]
    assert "2 against the contoured mask" in rules(report)["fidelity"]["detail"]
    three = tmp_path / "undeclared"
    report = ac.check(*build(three, raw_lump=True, contoured=True, declared=0.1))
    assert rules(report)["fidelity"]["pass"]
    assert rules(report)["declared changes"]["pass"] is False
    assert rules(report)["declared changes"]["detail"].startswith(
        "FMA_7987 rib_left_1: undeclared change, +0.0 ml of +0.0 declared, -0.2"
    )


def test_check_names_nodes_the_manifest_and_the_glb_disagree_on(tmp_path: Path) -> None:
    glb, manifest, seg = build(tmp_path)
    listed = json.loads(manifest.read_text())
    listed["nodes"] = listed["nodes"][:2] + [{"name": "FMA_9165", "label": "T1"}]
    manifest.write_text(json.dumps(listed))
    found = rules(ac.check(glb, manifest, seg))["nodes"]
    assert found["pass"] is False
    assert found["detail"] == (
        "FMA_85213 is not in the manifest or not a skeleton structure; "
        "FMA_9165 is in the manifest but not in the GLB"
    )


def test_a_broken_node_fails_topology(tmp_path: Path) -> None:
    glb, manifest, seg = build(tmp_path)
    exact, _ = flat_ellipsoid()
    holed = ac.Mesh(exact.vertices, exact.triangles[1:])
    write_skeleton(glb, {"sternum": holed}, np.array([0.01, -0.02, 0.03]))
    found = rules(ac.check(glb, manifest, seg))
    assert found["topology"]["pass"] is False
    assert found["topology"]["detail"] == "FMA_7485 sternum: 3 open edges (holes)"


def test_main_prints_a_line_per_node_and_rule_and_exits_1_on_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    glb, manifest, seg = build(tmp_path)
    out = tmp_path / "report.json"
    argv = ["atlas_check.py", "--glb", str(glb), "--manifest", str(manifest)]
    monkeypatch.setattr(sys, "argv", [*argv, "--seg-dir", str(seg), "--out", str(out)])
    assert ac.main() == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[1:4]] == ["FMA_7485", "FMA_7987", "FMA_85213"]
    assert "raw mask" in lines[1] and "clean" in lines[1] and " ok " in lines[1]
    assert "modelled" in lines[3]
    assert sum(line.startswith("rule ") for line in lines) == 14
    assert any(line.startswith("rule       fidelity          pass") for line in lines)
    assert lines[-2].startswith("atlas check failed: rib pairs, ")
    assert lines[-1] == f"report: {out}"
    assert json.loads(out.read_text())["pass"] is False


def test_main_with_a_case_reads_what_atlas_py_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "data" / "work" / "c"
    work.mkdir(parents=True)
    glb, _, _ = build(work)
    atlas = tmp_path / "data" / "atlas" / "c"
    atlas.mkdir(parents=True)
    glb.rename(atlas / "c.skeleton.glb")
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["atlas_check.py", "c"])
    assert ac.main() == 1
    assert (atlas / "c.skeleton.check.json").exists()
    assert "FMA_7987" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["atlas_check.py", "--glb", str(atlas / "c.skeleton.glb")])
    with pytest.raises(SystemExit):
        ac.main()


def test_ribs_numbered_one_apart_from_the_vertebrae_are_noted_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every rib nearest the vertebra above its own: the gaps still fail as
    they are, and a note at the end says what it looks like."""

    def one_apart(bodies: dict[str, ac.Body], dropped: dict[str, str]) -> tuple[Any, list[str]]:
        rows = [{"rib": f"rib_left_{n}", "node": "n", "gaps": {f"T{n}": 20.0}, "pass": False,
                 "nearest": f"T{n - 1}", "nearestMm": 1.0} for n in (2, 3, 4)]  # fmt: skip
        return rows, [f"rib_left_{n} is 20.0 mm from T{n}" for n in (2, 3, 4)]

    monkeypatch.setattr(ac, "rib_joints", one_apart)
    report = ac.check(*build(tmp_path))
    assert report["notes"] == [
        "3 of 3 ribs lie nearest another vertebra than their own: the segmentation may "
        "number ribs and vertebrae one apart"
    ]
    assert rules(report)["rib joints"]["detail"].count("note") == 0
    ac.print_report(report)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "note       " + report["notes"][0]
    assert any("T2 20.00 mm" in line and "nearest T1 1.00 mm" in line for line in lines)
