"""Tests for scan_damage.py on a figure made of capsules: no scan is read."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import scan_damage
import scan_quality
from scan_damage import (
    FINGER_FUSED,
    FINGER_MISSING,
    HAND,
    UNSEEN,
    Mesh,
    Settings,
    blocked,
    damage,
    depth_map,
    facing,
    read_mesh,
    resample,
    scanner_heads,
    signed_volume,
    topology,
    vertex_normals,
    write_glb,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Everything off: each test switches on what it tests.
OFF = Settings(resample=False, fingers=False, holes=False, banding=False, noise=False, seams=False)
ARM_X, ARM_R, TRUNK_R, LEG_X, LEG_R, HEAD_Y, HEAD_R = 0.21, 0.04, 0.14, 0.08, 0.07, 1.62, 0.1
FINGER_R = 0.008


def capsule(x: float, z: float, y0: float, y1: float, r: float, around: int = 32) -> Mesh:
    """A closed capsule standing along +Y in metres: a tube from y0 to y1 with a
    half sphere on each end, the normals pointing outward."""
    half = max(3, around // 4)
    rings = max(1, round((y1 - y0) / (2 * math.pi * r / around)))
    lat = np.linspace(0, math.pi / 2, half + 1)[1:]  # from just off the pole to the equator
    heights = np.concatenate([y0 - r * np.cos(lat), np.linspace(y0, y1, rings + 1)[1:],
                              y1 + r * np.cos(lat[::-1])[1:]])  # fmt: skip
    radii = np.concatenate([r * np.sin(lat), np.full(rings, r), r * np.sin(lat[::-1])[1:]])
    theta = 2 * math.pi * np.arange(around) / around
    points = [[x, y0 - r, z]]
    for h, rad in zip(heights, radii, strict=True):
        points += [[x + rad * math.sin(t), h, z + rad * math.cos(t)] for t in theta]
    points.append([x, y1 + r, z])
    triangles = []
    for k in range(around):
        triangles.append([0, 1 + (k + 1) % around, 1 + k])
    for ring in range(len(heights) - 1):
        a0, b0 = 1 + ring * around, 1 + (ring + 1) * around
        for k in range(around):
            a, b, c, d = a0 + k, a0 + (k + 1) % around, b0 + (k + 1) % around, b0 + k
            triangles += [[a, b, c], [a, c, d]]
    top, last = len(points) - 1, 1 + (len(heights) - 1) * around
    for k in range(around):
        triangles.append([top, last + k, last + (k + 1) % around])
    return Mesh(np.array(points, dtype=np.float64), np.array(triangles, dtype=np.int64))


def join(*meshes: Mesh) -> Mesh:
    points, triangles, fingers, offset = [], [], [], 0
    for m in meshes:
        points.append(m.points)
        triangles.append(m.triangles + offset)
        fingers.append(m.finger if m.finger is not None else np.full(len(m.points), -1))
        offset += len(m.points)
    return Mesh(np.vstack(points), np.vstack(triangles), None, np.concatenate(fingers))


def finger(x: float, number: int) -> Mesh:
    """A thin capsule hanging at the end of the right arm. Its upper end is
    part of the hand, the rest is finger `number`."""
    m = capsule(x, 0.0, 0.70, 0.78, FINGER_R, around=12)
    m.finger = np.where(m.points[:, 1] < 0.775, number, -1)
    return m


def figure(fingers: bool = False) -> Mesh:
    """A standing figure: trunk, head, two arms hanging 3 cm clear of the trunk
    and two legs 2 cm apart, feet on the floor. With `fingers`, three fingers
    4 mm apart under the right arm."""
    parts = [
        capsule(0.0, 0.0, 0.95, 1.40, TRUNK_R),
        capsule(0.0, 0.0, HEAD_Y, HEAD_Y, HEAD_R),
        capsule(ARM_X, 0.0, 0.85, 1.40, ARM_R, around=16),
        capsule(-ARM_X, 0.0, 0.85, 1.40, ARM_R, around=16),
        capsule(LEG_X, 0.0, LEG_R, 0.85, LEG_R, around=24),
        capsule(-LEG_X, 0.0, LEG_R, 0.85, LEG_R, around=24),
    ]
    if fingers:
        parts += [finger(ARM_X + 0.02 * (n - 2), n) for n in (1, 2, 3)]
    out = join(*parts)
    if not fingers:
        out.finger = None
    return out


@pytest.fixture(scope="module")
def holed() -> tuple:
    """The figure with only the holes on, and its vertex normals."""
    body = figure()
    result = damage(body, 3, Settings(**{**vars(OFF), "holes": True}))
    return body, vertex_normals(body.points, body.triangles), result


def switched(**on: object) -> Settings:
    return Settings(**{**vars(OFF), **on})


# The sensors


def test_heads_stand_in_the_corners_of_the_published_footprint() -> None:
    body = figure()
    heads = scanner_heads(body.points, Settings())
    assert heads.shape == (4, 2)
    # 2.4 m square: 1.697 m to a corner, less the 0.2 m the heads are set in.
    assert np.allclose(np.linalg.norm(heads, axis=1), 2.4 / math.sqrt(2) - 0.2, atol=1e-6)
    assert np.allclose(np.abs(heads[:, 0]), np.abs(heads[:, 1]), atol=1e-6)  # on the diagonals


def test_depth_map_is_high_facing_a_head_and_zero_on_top() -> None:
    body = capsule(0.0, 0.0, 0.5, 1.5, 0.1)
    normals = vertex_normals(body.points, body.triangles)
    heads = scanner_heads(body.points, Settings())
    depth = depth_map(body.points, normals, heads, 0.3)
    assert depth.min() >= 0 and depth.max() <= 1
    toward = np.argmax(body.points @ np.array([1.0, 0.0, 1.0]) - 9 * np.abs(body.points[:, 1] - 1))
    assert depth[toward] > 0.95
    assert depth[-1] < 0.25  # the top pole looks straight up, past every camera


def test_a_ray_is_blocked_only_by_a_surface_in_its_way() -> None:
    wall = Mesh(
        np.array([[0.5, 0.0, -0.1], [0.5, 0.0, 0.1], [0.5, 0.4, 0.1], [0.5, 0.4, -0.1]]),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    origins = np.array([[0.0, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.9, 0.0], [0.0, 0.2, 0.0]])
    targets = np.array([[1.0, 0.2, 0.0], [1.0, 0.2, 0.5], [1.0, 0.9, 0.0], [0.4, 0.2, 0.0]])
    hit = blocked(origins, targets, wall.points, wall.triangles, np.array([0.0, 0.0]))
    # Through the wall; past its side; over its top; stopping short of it.
    assert hit.tolist() == [True, False, False, False]


# Holes


def region(body: Mesh, x: float, y: float, z: float, within: float) -> BoolArray:
    return np.linalg.norm(body.points - np.array([x, y, z]), axis=1) < within


def test_holes_open_where_the_sensors_cannot_see(holed: tuple) -> None:
    body, normals, (_, _, labels, _) = holed
    seen = labels["vertexSeen"]
    # A horizontal laser line only grazes what faces straight down or up.
    soles = (normals[:, 1] < -0.98) & (body.points[:, 1] < 0.01)
    crown = (normals[:, 1] > 0.98) & (body.points[:, 1] > HEAD_Y + 0.09)
    assert soles.sum() == 2 and crown.sum() == 1
    assert not seen[soles].any() and not seen[crown].any()
    # Under the arms: the side of each arm that faces the trunk.
    for side in (1, -1):
        inner_arm = region(body, side * (ARM_X - ARM_R), 1.15, 0.0, 0.03)
        assert inner_arm.any() and not seen[inner_arm].any()
    # Between the legs.
    for side in (1, -1):
        inner_leg = region(body, side * (LEG_X - LEG_R), 0.45, 0.0, 0.012)
        assert inner_leg.any() and not seen[inner_leg].any()


def test_exposed_surfaces_stay_whole(holed: tuple) -> None:
    body, _, (damaged, _, labels, _) = holed
    seen, gone = labels["vertexSeen"], labels["vertexRemoved"] != 0
    exposed = [
        region(body, 0.0, 1.2, TRUNK_R, 0.08),  # the chest
        region(body, 0.0, 1.2, -TRUNK_R, 0.08),  # the back
        region(body, ARM_X + ARM_R, 1.15, 0.0, 0.04),  # the outside of an arm
        region(body, -LEG_X - LEG_R, 0.45, 0.0, 0.05),  # the outside of a leg
        region(body, 0.0, HEAD_Y, HEAD_R, 0.05),  # the front of the head
        region(body, LEG_X, 0.3, LEG_R, 0.05),  # a shin
    ]
    for mask in exposed:
        assert mask.sum() > 5
        assert seen[mask].all() and not gone[mask].any()
    assert 0.05 < gone.mean() < 0.5
    assert topology(body.triangles)["holes"] == 0
    assert topology(damaged.triangles)["holes"] >= 4


def test_a_body_outside_the_measuring_range_is_cut_off() -> None:
    tall = capsule(0.0, 0.0, 0.1, 2.3, 0.1)  # 2.5 m: the range is 2.1 m high
    _, _, labels, record = damage(tall, 1, switched(holes=True))
    above = tall.points[:, 1] > 2.1
    assert (labels["vertexRemoved"][above] & scan_damage.OUTSIDE).all()
    assert record["steps"][2]["verticesOutsideRange"] == int(above.sum())


def test_stray_fragments_are_dropped() -> None:
    body = figure()
    kept = damage(body, 1, switched(holes=True, min_island_faces=0))[2]["faceRemoved"]
    cleared = damage(body, 1, switched(holes=True, min_island_faces=10**6))[2]["faceRemoved"]
    assert not (kept & scan_damage.FRAGMENT).any()
    # With the bar above every patch's size, everything left counts as a fragment.
    assert ((cleared & scan_damage.FRAGMENT) > 0).sum() == (kept == 0).sum()


# Labels


def test_labels_say_exactly_what_was_removed(holed: tuple) -> None:
    body, _, (damaged, clean, labels, record) = holed
    assert clean is body  # nothing resampled
    kept = labels["faceRemoved"] == 0
    assert len(damaged.triangles) == kept.sum()
    assert np.array_equal(labels["faceIndex"], np.flatnonzero(kept))
    # The damaged faces are the kept clean faces, vertex for vertex.
    assert np.array_equal(labels["vertexIndex"][damaged.triangles], body.triangles[kept])
    assert np.allclose(damaged.points, body.points[labels["vertexIndex"]])
    # A vertex is removed exactly when no kept face uses it, and its label says why.
    used = np.zeros(len(body.points), dtype=bool)
    used[body.triangles[kept]] = True
    assert np.array_equal(labels["vertexRemoved"] == 0, used)
    assert np.array_equal(np.flatnonzero(used), labels["vertexIndex"])
    assert (labels["vertexRemoved"][~labels["vertexSeen"]] & UNSEEN).all()
    # A face goes as unseen exactly when one of its corners was not measured.
    unseen = ~labels["vertexSeen"][body.triangles].all(1)
    assert np.array_equal((labels["faceRemoved"] & UNSEEN) > 0, unseen)
    assert record["damaged"]["facesRemoved"] == int((~kept).sum())
    assert record["damaged"]["verticesRemoved"] == int((~used).sum())
    assert record["damaged"]["removedByLabel"]["unseen"] == int(unseen.sum())


# Determinism and switches


def test_one_seed_gives_one_result() -> None:
    body = figure(fingers=True)
    first, second = damage(body, 11, Settings(edge_mm=30)), damage(body, 11, Settings(edge_mm=30))
    other = damage(body, 12, Settings(edge_mm=30))
    assert np.array_equal(first[0].points, second[0].points)
    assert np.array_equal(first[0].triangles, second[0].triangles)
    assert all(np.array_equal(first[2][k], second[2][k]) for k in first[2])
    assert json.dumps(first[3], sort_keys=True) == json.dumps(second[3], sort_keys=True)
    assert not np.array_equal(first[2]["displacement"], other[2]["displacement"])


def test_everything_off_changes_nothing() -> None:
    body = figure(fingers=True)
    damaged, clean, labels, record = damage(body, 5, OFF)
    assert np.array_equal(damaged.points, body.points)
    assert np.array_equal(damaged.triangles, body.triangles)
    assert not labels["vertexRemoved"].any() and not labels["faceRemoved"].any()
    assert not labels["displacement"].any()
    assert [s["applied"] for s in record["steps"]] == [False] * 6
    assert [s["step"] for s in record["steps"]] == [
        "resample", "fingers", "holes", "banding", "noise", "seams",
    ]  # fmt: skip


def test_switching_one_artefact_off_leaves_the_others_as_they_were() -> None:
    body = figure()
    both = damage(body, 4, switched(noise=True, banding=True))[2]["displacement"]
    noise = damage(body, 4, switched(noise=True))[2]["displacement"]
    banding = damage(body, 4, switched(banding=True))[2]["displacement"]
    assert np.allclose(both, noise + banding, atol=1e-9)


def test_inward_faces_are_turned_outward_first() -> None:
    body = figure()
    inside_out = Mesh(body.points, body.triangles[:, ::-1].copy())
    _, clean, labels, record = damage(inside_out, 3, switched(holes=True))
    assert record["input"]["facesTurnedOutward"]
    assert signed_volume(clean.points, clean.triangles) > 0
    assert np.array_equal(
        labels["vertexSeen"], damage(body, 3, switched(holes=True))[2]["vertexSeen"]
    )


# Resampling


def test_resampling_brings_the_edges_to_one_length_and_keeps_the_shape() -> None:
    body = capsule(0.0, 0.0, 0.2, 1.0, 0.15, around=12)  # edges of 8 cm
    out = resample(body, 0.02)
    corners = out.points[out.triangles]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    # Halved until under 4/3 of the edge; joining close vertices stretches a few.
    assert np.percentile(edges, 90) <= 0.02 * 4 / 3 and edges.max() < 0.02 * 1.5
    assert edges.min() > 0.02 / 4
    assert topology(out.triangles) == {
        "triangles": len(out.triangles), "openEdges": 0, "holes": 0, "nonManifoldEdges": 0,
    }  # fmt: skip
    before = signed_volume(body.points, body.triangles)
    assert signed_volume(out.points, out.triangles) == pytest.approx(before, rel=1e-3)


def test_resampling_joins_vertices_closer_than_the_scanner_samples() -> None:
    body = capsule(0.0, 0.0, 0.2, 0.3, 0.01, around=32)  # edges of 2 mm
    out = resample(body, 0.02)
    assert len(out.points) < len(body.points) / 4
    assert topology(out.triangles)["openEdges"] == 0


def test_resampling_carries_colours_and_fingers() -> None:
    body = figure(fingers=True)
    body.colour = np.stack([body.points[:, 1] / 2, np.zeros(len(body.points)),
                            np.ones(len(body.points))], 1)  # fmt: skip
    out = resample(body, 0.03)
    assert out.colour is not None and out.finger is not None
    assert np.allclose(out.colour[:, 0], out.points[:, 1] / 2, atol=0.01)  # red follows height
    assert np.allclose(out.colour[:, 2], 1.0)
    assert set(np.unique(out.finger)) == {-1, 1, 2, 3}


def test_an_edge_finer_than_the_scanner_measures_is_refused() -> None:
    with pytest.raises(ValueError, match="finer than the VITUS bodyscan measures"):
        damage(figure(), 1, Settings(edge_mm=0.5))  # 300 points per cm2 is 0.58 mm
    with pytest.raises(ValueError, match="VITUS smart XXL"):
        damage(figure(), 1, Settings(model="smart-xxl", edge_mm=1.5))  # 27 per cm2 is 1.92 mm


# Noise and banding


def along_sight(body: Mesh, labels: dict, settings: Settings) -> FloatArray:
    """Each vertex's displacement measured along the line to its sensor's
    camera; fails if any of it lies across that line."""
    points = body.points[labels["vertexIndex"]]
    normals = vertex_normals(body.points, body.triangles)[labels["vertexIndex"]]
    heads = scanner_heads(body.points, settings)
    sight = facing(points, normals, heads, settings.baseline_m)[2].reshape(len(points), -1, 3)
    line = sight[np.arange(len(points)), labels["sensor"]]
    moved = labels["displacement"].astype(np.float64)
    along = (moved * line).sum(1)
    assert np.allclose(moved, along[:, None] * line, atol=1e-7)
    return along


def test_noise_lies_along_the_line_of_sight_with_the_set_spread() -> None:
    body = resample(figure(), 0.012)  # some 20 000 vertices
    settings = switched(noise=True, noise_mm=0.5, sensor_bias_mm=0.0)
    damaged, _, labels, _ = damage(body, 21, settings)
    along = along_sight(body, labels, settings) * 1000
    assert len(along) > 15000
    assert along.std() == pytest.approx(0.5, rel=0.03)
    assert abs(along.mean()) < 0.02
    assert np.allclose(damaged.points - body.points, labels["displacement"], atol=1e-7)
    louder = damage(body, 21, switched(noise=True, noise_mm=2.0, sensor_bias_mm=0.0))[2]
    assert (along_sight(body, louder, settings) * 1000).std() == pytest.approx(2.0, rel=0.03)


def test_each_sensor_has_its_own_small_offset() -> None:
    body = figure()
    settings = switched(noise=True, noise_mm=0.0, sensor_bias_mm=0.15)
    labels = damage(body, 8, settings)[2]
    along = along_sight(body, labels, settings) * 1000
    offsets = [np.unique(np.round(along[labels["sensor"] == s], 6)) for s in range(8)]
    assert all(len(o) <= 1 for o in offsets)  # one offset per sensor
    values = np.concatenate(offsets)
    assert len(values) >= 4 and len(np.unique(values)) == len(values)
    assert np.abs(values).max() < 0.15 * 4


def test_banding_is_the_same_along_a_line_and_changes_up_the_body() -> None:
    body = capsule(0.0, 0.0, 0.2, 1.8, 0.15, around=48)
    settings = switched(banding=True, band_mm=0.3)
    labels, record = damage(body, 2, settings)[2:]
    along = along_sight(body, labels, settings) * 1000
    assert record["steps"][3]["bandHeightMm"] == pytest.approx(2100 / 8 / 25, abs=0.01)  # 10.5 mm
    head, height = labels["sensor"] // 2, np.round(body.points[:, 1], 6)
    tube = (body.points[:, 1] > 0.2) & (body.points[:, 1] < 1.8)
    spreads = []
    for h in range(4):
        rings = [along[tube & (head == h) & (height == y)] for y in np.unique(height[tube])]
        rings = [r for r in rings if len(r) > 1]
        assert all(np.ptp(r) < 1e-4 for r in rings)  # one head, one height: one offset
        spreads.append(np.std([r[0] for r in rings]))
    assert all(0.1 < s < 0.5 for s in spreads)  # set to 0.3 mm


# Fingers


def test_missing_fingers_lose_their_tips() -> None:
    body = figure(fingers=True)
    settings = switched(fingers=True, finger_missing=1.0, finger_fused=0.0)
    damaged, _, labels, record = damage(body, 6, settings)
    assert body.finger is not None
    step = record["steps"][1]
    assert [f["state"] for f in step["fingers"]] == ["missing"] * 3
    gone = labels["faceRemoved"] == FINGER_MISSING
    assert gone.sum() == sum(f["facesRemoved"] for f in step["fingers"]) > 0
    assert (body.finger[body.triangles[gone]] >= 0).all()  # only finger faces go
    for f in step["fingers"]:
        assert 0.3 <= f["lengthLost"] <= 1.0
        own = body.finger == f["finger"]
        lost = own & (labels["vertexRemoved"] == FINGER_MISSING)
        kept = own & (labels["vertexRemoved"] == 0)
        assert lost.any()
        # The fingers hang down from the hand: what goes is the lower end.
        assert not kept.any() or body.points[lost, 1].max() <= body.points[kept, 1].min() + 1e-9
    assert topology(damaged.triangles)["holes"] == 3  # three open stumps


def test_close_fingers_are_joined_into_one_lump() -> None:
    body = figure(fingers=True)
    settings = switched(fingers=True, finger_missing=0.0, finger_fused=1.0)
    damaged, _, labels, record = damage(body, 6, settings)
    assert body.finger is not None
    (lump,) = record["steps"][1]["fingers"]
    assert lump["state"] == "fused" and lump["fingers"] == [1, 2, 3]
    gone = labels["faceRemoved"] == FINGER_FUSED
    assert gone.sum() == lump["facesRemoved"] > 0
    added = labels["faceIndex"] == -1
    assert added.sum() == lump["facesAdded"] > 0
    # The lump is made of finger vertices.
    corners = labels["vertexIndex"][damaged.triangles[added]]
    assert (body.finger[corners] >= 0).any(1).all()
    # The middle finger lies inside the lump: its corners are the outer two's.
    assert {int(i) for i in np.unique(body.finger[corners]) if i >= 0} == {1, 3}
    lump_faces = damaged.points[damaged.triangles[added]]
    outward = np.cross(lump_faces[:, 1] - lump_faces[:, 0], lump_faces[:, 2] - lump_faces[:, 0])
    middle = body.points[body.finger >= 0].mean(0)
    assert ((outward * (lump_faces.mean(1) - middle)).sum(1) > 0).all()  # facing outward
    # Vertices between the fingers are no longer on the surface.
    between = (body.finger == 2) & (labels["vertexRemoved"] == FINGER_FUSED)
    assert between.any()


def test_fingers_too_far_apart_are_not_joined() -> None:
    body = figure(fingers=True)
    settings = switched(fingers=True, finger_missing=0.0, finger_fused=1.0, fuse_gap_mm=2.0)
    _, _, labels, record = damage(body, 6, settings)  # the gaps are 4 mm
    assert record["steps"][1]["fingers"] == [] and not labels["faceRemoved"].any()


def test_a_mesh_without_fingers_skips_the_step() -> None:
    record = damage(figure(), 6, switched(fingers=True))[3]
    assert record["steps"][1] == {
        "step": "fingers", "applied": False, "skipped": "the mesh names no fingers",
    }  # fmt: skip


# Colour seams


def test_seams_step_the_colour_between_sensors() -> None:
    body = figure()
    body.colour = np.full((len(body.points), 3), 0.5)
    damaged, _, labels, record = damage(body, 9, switched(seams=True))
    assert record["steps"][5]["applied"] and damaged.colour is not None
    shades = [np.unique(np.round(damaged.colour[labels["sensor"] == s], 9), axis=0)
              for s in np.unique(labels["sensor"])]  # fmt: skip
    assert all(len(s) == 1 for s in shades)  # one gain per sensor
    assert len(np.unique(np.vstack(shades), axis=0)) == len(shades) >= 4  # and each its own
    assert np.abs(np.vstack(shades) - 0.5).max() < 0.5 * 0.05 * 5
    assert not labels["displacement"].any()  # only the colour changed


def test_a_mesh_without_colour_skips_the_seams() -> None:
    damaged, _, _, record = damage(figure(), 9, switched(seams=True))
    assert damaged.colour is None
    assert record["steps"][5]["skipped"] == "the mesh has no vertex colours"


# The record


def test_record_names_every_parameter_and_where_it_comes_from() -> None:
    record = damage(figure(), 1, OFF)[3]
    names = {f for f in vars(Settings()) if not isinstance(getattr(Settings(), f), bool)}
    assert set(record["parameters"]) == names == set(scan_damage.PARAMETERS)
    assert {p["basis"] for p in record["parameters"].values()} == {
        "published", "derived", "assumption",
    }  # fmt: skip
    assert record["parameters"]["noise_mm"]["basis"] == "assumption"
    assert record["scanner"]["name"] == "VITUS bodyscan"
    assert (
        "Humanetics" in record["scanner"]["source"] and "Maurer" in record["scanner"]["principle"]
    )
    assert record["scanner"]["pointsPerCm2"] == 300 and record["scanner"]["heads"] == 4
    json.dumps(record)  # everything in it can be written out


# With scan_quality.py: the measures move the right way


def test_scan_quality_sees_the_damage_grow_with_the_noise() -> None:
    body = resample(figure(), 0.02)
    clean_mm = (body.points * 1000, body.triangles)
    means = []
    for noise_mm in (0.0, 0.5, 2.0):
        damaged = damage(body, 13, switched(noise=noise_mm > 0, noise_mm=noise_mm))[0]
        change = scan_quality.shape_change((damaged.points * 1000, damaged.triangles), clean_mm)
        means.append(change["trunk"]["meanMm"])
    assert means[0] == pytest.approx(0.0, abs=1e-3)
    assert means[0] < means[1] < means[2]
    assert (
        0.2 < means[1] < 0.5 and 0.8 < means[2] < 2.0
    )  # the noise, less what lies along the surface


def test_scan_quality_measures_survive_the_holes(holed: tuple) -> None:
    body, _, (damaged, _, _, _) = holed
    before = scan_quality.measures(body.points * 1000, body.triangles)
    after = scan_quality.measures(damaged.points * 1000, damaged.triangles)
    # The crown and the soles are open, so the body measures a little short, never taller.
    assert before["stature"] - 20 < after["stature"] <= before["stature"]
    assert topology(body.triangles)["openEdges"] == 0 < topology(damaged.triangles)["openEdges"]


# Files and the command line


def rigged_glb(path: Path, body: Mesh, joints: list[str], joint_of: npt.NDArray[np.int64]) -> None:
    """A skinned glTF binary: `body` with every vertex bound to one named joint."""
    arrays = [
        body.points.astype("<f4"), body.triangles.astype("<u4").reshape(-1),
        np.stack([joint_of, *[np.zeros_like(joint_of)] * 3], 1).astype("<u1"),
        np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (len(body.points), 1)),
    ]  # fmt: skip
    kinds = [("VEC3", 5126), ("SCALAR", 5125), ("VEC4", 5121), ("VEC4", 5126)]
    blob, views, accessors = b"", [], []
    for array, (kind, component) in zip(arrays, kinds, strict=True):
        data = array.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        accessors.append({"bufferView": len(views) - 1, "componentType": component,
                          "count": len(array), "type": kind})  # fmt: skip
        blob += data + b"\0" * (-len(data) % 4)
    attributes = {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}
    gltf = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "skin": 0}, *[{"name": name} for name in joints]],
        "skins": [{"joints": list(range(1, len(joints) + 1))}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": 1}]}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
    }  # fmt: skip
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    head = struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(blob))
    path.write_bytes(head + struct.pack("<II", len(text), 0x4E4F534A) + text
                     + struct.pack("<II", len(blob), 0x004E4942) + blob)  # fmt: skip


def test_a_written_glb_reads_back_and_scan_quality_reads_it_too(tmp_path: Path) -> None:
    body = figure()
    body.colour = np.random.default_rng(0).random((len(body.points), 3))
    write_glb(tmp_path / "body.glb", body)
    back = read_mesh(tmp_path / "body.glb")
    assert np.allclose(back.points, body.points, atol=1e-6)
    assert np.array_equal(back.triangles, body.triangles)
    assert back.colour is not None and np.allclose(back.colour, body.colour, atol=1e-6)
    assert back.finger is None
    points_mm, triangles = scan_quality.surface(tmp_path / "body.glb")
    assert np.allclose(points_mm, body.points * 1000, atol=1e-3)
    assert np.array_equal(triangles, body.triangles)


def test_fingers_are_read_from_the_joints_of_an_anny_rig(tmp_path: Path) -> None:
    body = figure(fingers=True)
    assert body.finger is not None
    joints = ["root", "finger1-1.L", "finger2-3.L", "finger3-2.R", "toe1-1.L"]
    joint_of = np.select([body.finger == 1, body.finger == 2, body.finger == 3], [1, 2, 3], 0)
    joint_of[0] = 4  # a toe is not a finger
    rigged_glb(tmp_path / "rig.glb", body, joints, joint_of.astype(np.int64))
    back = read_mesh(tmp_path / "rig.glb")
    assert back.finger is not None
    # Left hand 0, right hand 1: the finger's number is HAND * hand + finger.
    expected = np.select([body.finger == 1, body.finger == 2, body.finger == 3],
                         [1, 2, HAND + 3], -1)  # fmt: skip
    assert np.array_equal(back.finger, expected)


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_damage.py", *args])
    scan_damage.main()


def test_command_line_writes_meshes_labels_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_glb(tmp_path / "body.glb", figure())
    args = ("--in", str(tmp_path / "body.glb"), "--seed", "4", "--edge-mm", "30", "--noise-mm", "1")
    run_cli(monkeypatch, *args, "--out", str(tmp_path / "a" / "body.seed4"))
    run_cli(monkeypatch, *args, "--out", str(tmp_path / "b" / "body.seed4"))
    assert "body.glb seed 4" in capsys.readouterr().out
    names = ["body.seed4.clean.glb", "body.seed4.damaged.glb", "body.seed4.json",
             "body.seed4.labels.npz"]  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == names
    for name in names:  # the same seed writes the same bytes
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    record = json.loads((tmp_path / "a" / "body.seed4.json").read_text())
    assert record["source"] == "body.glb" and record["seed"] == 4
    assert record["settings"]["edge_mm"] == 30 and record["settings"]["noise_mm"] == 1
    assert record["outputs"]["damaged"]["sha256"] == scan_damage.sha256(
        tmp_path / "a" / "body.seed4.damaged.glb"
    )
    clean = read_mesh(tmp_path / "a" / "body.seed4.clean.glb")
    damaged = read_mesh(tmp_path / "a" / "body.seed4.damaged.glb")
    labels = np.load(tmp_path / "a" / "body.seed4.labels.npz")
    assert sorted(labels.files) == record["outputs"]["labels"]["arrays"]
    assert len(labels["vertexRemoved"]) == len(clean.points) == record["clean"]["vertices"]
    assert len(labels["vertexIndex"]) == len(damaged.points) == record["damaged"]["vertices"]
    # The clean position plus the recorded displacement is the damaged position.
    rebuilt = clean.points[labels["vertexIndex"]] + labels["displacement"]
    assert np.allclose(damaged.points, rebuilt, atol=1e-6)
    assert record["damaged"]["holes"] > 0 == record["clean"]["holes"]


def test_command_line_switches_turn_artefacts_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_glb(tmp_path / "body.glb", figure())
    off = ["--no-resample", "--no-fingers", "--no-holes", "--no-banding", "--no-noise",
           "--no-seams"]  # fmt: skip
    out = tmp_path / "out" / "body"
    run_cli(monkeypatch, "--in", str(tmp_path / "body.glb"), "--out", str(out), *off)
    record = json.loads(out.with_name("body.json").read_text())
    assert not any(step["applied"] for step in record["steps"])
    assert record["damaged"]["verticesRemoved"] == 0


@pytest.mark.parametrize("which", ["--in", "--out"])
def test_command_line_refuses_the_folder_of_real_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    write_glb(tmp_path / "body.glb", figure())
    real = tmp_path / "data" / "vitronic" / "someone"
    paths = {"--in": str(tmp_path / "body.glb"), "--out": str(tmp_path / "out" / "body")}
    paths[which] = str(real)
    with pytest.raises(SystemExit, match="refused: data/vitronic/ holds real people's scans"):
        run_cli(monkeypatch, "--in", paths["--in"], "--out", paths["--out"])
    assert not (tmp_path / "out").exists() and not real.exists()
