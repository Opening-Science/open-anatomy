"""
Turns a clean synthetic body into what an optical body scanner would give, so
a model that repairs scans can be trained on pairs of damaged and clean bodies
without ever seeing a person's scan.

  python scripts/scan_damage.py --in <glb> --out <stem> --seed 7
      [--model bodyscan|smart-xxl] [--edge-mm 5] [--noise-mm 0.5] ...
      [--no-resample] [--no-fingers] [--no-holes] [--no-banding] [--no-noise] [--no-seams]

The input is a triangle mesh in a glTF binary, standing on the floor with +Y
up and in metres: a rigged ANNY body from data/bodies, or any other mesh. It
writes four files:

  <stem>.clean.glb    the clean body at the scanner's triangle density
  <stem>.damaged.glb  the same body as the scanner would give it
  <stem>.labels.npz   what was removed and moved, per vertex and per face
  <stem>.json         exactly what was done: every parameter, where it comes
                      from, the seed and the counts

The scanner is modelled from its published specification only (SCANNERS
below): pillars around the body, each with a head that travels from top to
bottom and carries a laser that throws a horizontal line and a camera above
and below it. A point is measured when a laser lights it and a camera of the
same head sees it. From that follow, each with its own switch:

- resample   triangles brought to one size all over, as a scanner samples
- fingers    fingers cut short, or neighbours joined into one lump
- holes      whatever no sensor measures is left out: under the arms, the
             crotch, the soles, the crown
- banding    ripple along the direction of travel, the same along each line
- noise      error along the line of sight, and a small offset per sensor
- seams      steps in vertex colour where one sensor's view meets the next

Parameters the specification gives no figure for are assumptions, named as
such in PARAMETERS and in the JSON. One seed gives one result. Nothing here
was taken from a real scan, and a path under data/vitronic/ is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from size_body import accessor, read_glb  # noqa: E402

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Real people's scans are kept under data/vitronic/, here or in another checkout.
PERSONAL = re.compile(r"(^|[\\/])data[\\/]vitronic([\\/]|$)")

FLYER = (
    "Humanetics Digital Europe, flyer 'VITUS bodyscan, high-precision 3D body scanner', "
    "specification table (humaneticsgroup.com, Flyer_VITUSbodyscan.pdf, read 12 Nov 2026)"
)
MAURER = (
    "M. Maurer (Vitronic), 'VITUS 3D Body Scanner', Proc. 3rd Int. Conf. on 3D Body "
    "Scanning Technologies, Lugano, 2012, pp. 99-105 (proc.3dbody.tech, paper 12099)"
)

# What the maker publishes, per model. Lengths in metres, times in seconds.
SCANNERS: dict[str, dict[str, Any]] = {
    "bodyscan": {
        "name": "VITUS bodyscan",
        "source": FLYER,
        "heads": 4,  # "8 sensors"; two cameras to a head as on the smart XXL
        "camerasPerHead": 2,
        "rangeHeightM": 2.1,
        "rangeWidthM": 1.2,
        "rangeDepthM": 1.2,
        "footprintM": 2.4,  # the scanner is 2.4 m square and 2.95 m high
        "scanTimeS": (6.0, 10.0),
        "pointsPerCm2": 300.0,
        "girthErrorMm": 1.0,  # "average max. girth error < 1 mm" on a 110 mm tube
        "colour": True,  # "with color texture output"
    },
    "smart-xxl": {
        "name": "VITUS smart XXL",
        "source": MAURER + ", table 1",
        "heads": 4,  # "4 dual-camera heads"
        "camerasPerHead": 2,
        "rangeHeightM": 2.1,
        "rangeWidthM": 1.2,
        "rangeDepthM": 1.0,
        "footprintM": 2.2,
        "scanTimeS": (12.0, 12.0),
        "pointsPerCm2": 27.0,
        "girthErrorMm": 1.0,
        "colour": False,  # grey levels only
    },
}
# How the heads work, the same for both models (Maurer 2012, sections 2.1 to 2.3).
PRINCIPLE = (
    "light stripe method: each head carries a laser that throws a horizontal line and a "
    "camera above and one below the laser plane, looking slightly down and up; the heads "
    "start level, one in each pillar, and travel from top to bottom; a part no camera sees "
    "is not measured (" + MAURER + ")"
)

# Bits of the face and vertex labels: why something on the clean body is not
# in the damaged one. 0 means it was kept.
UNSEEN, FINGER_MISSING, FINGER_FUSED, FRAGMENT, OUTSIDE = 1, 2, 4, 8, 16
LABELS = {
    "unseen": UNSEEN,
    "fingerMissing": FINGER_MISSING,
    "fingerFused": FINGER_FUSED,
    "fragment": FRAGMENT,
    "outsideRange": OUTSIDE,
}

RAY_MARGIN_M = 0.0005  # a ray counts as blocked only this far short of its target
RAY_BINS = 512  # slices of bearing about a pillar that rays and triangles are sorted into
RAY_PAIRS = 4_000_000  # ray and triangle pairs tested at once
FINGER = re.compile(r"^finger(\d)-\d\.(L|R)$")  # ANNY's finger joints
HAND = 8  # a finger's number is 8 * hand + finger, so one hand's fingers sit together


@dataclass(frozen=True)
class Settings:
    """Every switch and parameter. PARAMETERS says where each default comes from."""

    model: str = "bodyscan"
    resample: bool = True
    fingers: bool = True
    holes: bool = True
    banding: bool = True
    noise: bool = True
    seams: bool = True
    edge_mm: float = 5.0
    pillar_inset_m: float = 0.2
    ring_turn_deg: float = 45.0
    baseline_m: float = 0.3
    max_incidence_deg: float = 75.0
    min_island_faces: int = 30
    finger_missing: float = 0.25
    finger_fused: float = 0.5
    finger_lost_least: float = 0.3
    fuse_gap_mm: float = 10.0
    ripple_hz: float = 25.0
    band_mm: float = 0.3
    noise_mm: float = 0.5
    sensor_bias_mm: float = 0.15
    seam_gain: float = 0.05


# Where each default comes from: "published" (a figure in SCANNERS), "derived"
# (worked out from published figures, the working given) or "assumption" (the
# specification gives no figure).
PARAMETERS: dict[str, tuple[str, str]] = {
    "model": ("published", "which scanner's figures are used (SCANNERS)"),
    "edge_mm": (
        "assumption",
        "edge length of the exported mesh. Published is the density of measured points "
        "(300 per cm2, a spacing of 0.58 mm, which no edge may go below); how far the "
        "export thins them is not published",
    ),
    "pillar_inset_m": (
        "assumption",
        "how far in from the corner of the published footprint a head sits; with the "
        "2.4 m footprint it puts the heads 1.5 m from the middle",
    ),
    "ring_turn_deg": (
        "assumption",
        "bearing of the first pillar from the direction the body faces: the body faces "
        "between two pillars",
    ),
    "baseline_m": (
        "assumption",
        "height of each camera above or below its laser. Published is only that one "
        "camera sits above and one below, and that the pillars are 0.85 m taller than "
        "the range they measure",
    ),
    "max_incidence_deg": (
        "assumption",
        "a surface turned further than this from the laser or the camera gives no "
        "reading; it is what leaves the crown, the soles and the shoulder tops open "
        "under a horizontal laser line",
    ),
    "min_island_faces": (
        "assumption",
        "a measured patch with fewer faces than this is dropped as a stray fragment",
    ),
    "finger_missing": ("assumption", "chance that a finger is cut short"),
    "finger_fused": ("assumption", "chance that two neighbouring fingers are joined"),
    "finger_lost_least": ("assumption", "smallest share of a cut finger's length that goes"),
    "fuse_gap_mm": (
        "assumption",
        "only neighbouring fingers closer than this can be joined",
    ),
    "ripple_hz": (
        "assumption",
        "rate at which the travelling head shakes. With the published travel (2.1 m in "
        "6 to 10 s, taken as 8 s) it sets the height of one band: 262 mm/s at 25 Hz is "
        "10.5 mm",
    ),
    "band_mm": ("assumption", "standard deviation of the band offsets along the line of sight"),
    "noise_mm": (
        "assumption",
        "standard deviation of a single point along its line of sight. Published is "
        "only the girth error, which says nothing about single points",
    ),
    "sensor_bias_mm": (
        "derived",
        "standard deviation of each sensor's own offset along its line of sight: a "
        "girth error of 1 mm is a radius error of 1 / (2 pi) = 0.16 mm",
    ),
    "seam_gain": (
        "assumption",
        "standard deviation of each sensor's gain per colour channel; applies only to "
        "a mesh with vertex colours",
    ),
}


@dataclass
class Mesh:
    """A triangle mesh in metres, +Y up. `colour` is per vertex, 0 to 1; `finger`
    numbers each vertex's finger (HAND * hand + finger) or is -1."""

    points: FloatArray
    triangles: IntArray
    colour: FloatArray | None = None
    finger: IntArray | None = None


# Geometry


def vertex_normals(points: FloatArray, triangles: IntArray) -> FloatArray:
    """Unit normals per vertex, each face weighted by its area."""
    corners = points[triangles]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length == 0, 1.0, length)


def signed_volume(points: FloatArray, triangles: IntArray) -> float:
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return float((a * np.cross(b, c)).sum() / 6.0)


def topology(triangles: IntArray) -> dict[str, int]:
    """Open edges, holes (loops of open edges) and edges with more than two faces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(triangles):
        return {"triangles": 0, "openEdges": 0, "holes": 0, "nonManifoldEdges": 0}
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    rim = unique[counts == 1]
    holes = 0
    if len(rim):
        ids, flat = np.unique(rim, return_inverse=True)
        a, b = flat.reshape(-1, 2).T
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
        holes = int(connected_components(graph, directed=False)[0])
    return {
        "triangles": int(len(triangles)),
        "openEdges": int(len(rim)),
        "holes": holes,
        "nonManifoldEdges": int((counts > 2).sum()),
    }


# The sensors


def head_positions(centre: FloatArray, radius: float, heads: int, turn_deg: float) -> FloatArray:
    """Where the pillars stand: (heads, 2) as x and z, in a ring about `centre`
    (x, z). The body faces +Z; the first pillar stands `turn_deg` round from
    there."""
    angle = math.radians(turn_deg) + 2 * math.pi * np.arange(heads) / heads
    return np.asarray(centre)[None, :] + radius * np.stack([np.sin(angle), np.cos(angle)], 1)


def scanner_heads(points: FloatArray, settings: Settings) -> FloatArray:
    """The pillars of the chosen scanner around a body standing in its middle:
    in the corners of the published footprint, `pillar_inset_m` in."""
    scanner = SCANNERS[settings.model]
    centre = (points[:, [0, 2]].min(0) + points[:, [0, 2]].max(0)) / 2
    radius = scanner["footprintM"] / math.sqrt(2) - settings.pillar_inset_m
    return head_positions(centre, radius, scanner["heads"], settings.ring_turn_deg)


def camera_points(points: FloatArray, heads: FloatArray, baseline: float) -> FloatArray:
    """The cameras that could measure each point: (points, heads, 2, 3). A head is
    level with the point when its laser lights it, with one camera `baseline`
    above and one below."""
    out = np.empty((len(points), len(heads), 2, 3))
    out[:, :, :, 0] = heads[None, :, None, 0]
    out[:, :, :, 2] = heads[None, :, None, 1]
    out[:, :, 0, 1] = points[:, None, 1] + baseline
    out[:, :, 1, 1] = points[:, None, 1] - baseline
    return out


def facing(points: FloatArray, normals: FloatArray, heads: FloatArray, baseline: float):
    """How squarely each camera faces each point, and each head's laser: the
    cosines (points, heads, 2) and (points, heads), and the unit lines of sight
    to the cameras (points, heads, 2, 3). Occlusion is not looked at."""
    sight = camera_points(points, heads, baseline) - points[:, None, None, :]
    sight /= np.linalg.norm(sight, axis=3, keepdims=True)
    camera = (sight * normals[:, None, None, :]).sum(3)
    beam = np.stack([heads[None, :, 0] - points[:, None, 0],
                     heads[None, :, 1] - points[:, None, 2]], 2)  # fmt: skip
    beam /= np.linalg.norm(beam, axis=2, keepdims=True)
    laser = beam[:, :, 0] * normals[:, None, 0] + beam[:, :, 1] * normals[:, None, 2]
    return camera, laser, sight


def depth_map(points: FloatArray, normals: FloatArray, heads: FloatArray, baseline: float):
    """A stand-in for the scanner's per-vertex _DepthMap: how squarely the best
    camera faces each vertex, 0 to 1. blender/make_demo_scan.py uses it."""
    return np.clip(facing(points, normals, heads, baseline)[0].max(axis=(1, 2)), 0.0, 1.0)


def blocked(
    origins: FloatArray, targets: FloatArray, points: FloatArray, triangles: IntArray,
    axis: FloatArray,
) -> BoolArray:  # fmt: skip
    """Whether the surface blocks each straight line from origins[i] to
    targets[i]. Every origin stands on the vertical line through `axis` (x, z),
    so each ray lies in a vertical plane through that line; rays and triangles
    are sorted by the bearing of that plane, and only triangles that the plane
    cuts, at the ray's height and nearer than its target, are tested."""
    out = np.zeros(len(origins), dtype=bool)
    if not len(origins) or not len(triangles):
        return out
    corners = points[triangles]
    middle = points[:, [0, 2]].mean(0) - axis
    turn = math.atan2(middle[1], middle[0])

    def bearing(p: FloatArray) -> FloatArray:
        a = np.arctan2(p[..., 2] - axis[1], p[..., 0] - axis[0]) - turn
        return (a + math.pi) % (2 * math.pi) - math.pi

    tri_bearing = bearing(corners)
    low, high = tri_bearing.min(1), tri_bearing.max(1)
    ray_bearing = bearing(targets)
    span = (float(low.min()), float(high.max()))
    width = (span[1] - span[0]) / RAY_BINS or 1.0

    def slot(a: FloatArray) -> IntArray:
        return np.clip(((a - span[0]) / width).astype(np.int64), 0, RAY_BINS - 1)

    first, last = slot(low), slot(high)
    count = last - first + 1
    tri_of = np.repeat(np.arange(len(triangles)), count)
    slot_of = np.repeat(first, count) + (np.arange(count.sum()) - np.repeat(
        np.cumsum(count) - count, count))  # fmt: skip
    order = np.argsort(slot_of, kind="stable")
    tri_of, starts = tri_of[order], np.searchsorted(slot_of[order], np.arange(RAY_BINS + 1))
    tri_y = corners[:, :, 1]
    y_low, y_high = tri_y.min(1), tri_y.max(1)
    reach = np.hypot(corners[:, :, 0] - axis[0], corners[:, :, 2] - axis[1]).min(1)
    ray_reach = np.hypot(targets[:, 0] - axis[0], targets[:, 2] - axis[1])
    ray_low = np.minimum(origins[:, 1], targets[:, 1])
    ray_high = np.maximum(origins[:, 1], targets[:, 1])
    ray_slot = slot(ray_bearing)
    ray_order = np.argsort(ray_slot, kind="stable")
    ray_starts = np.searchsorted(ray_slot[ray_order], np.arange(RAY_BINS + 1))
    v0, e1, e2 = corners[:, 0], corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    for b in range(RAY_BINS):
        tris = tri_of[starts[b] : starts[b + 1]]
        rays = ray_order[ray_starts[b] : ray_starts[b + 1]]
        if not len(tris) or not len(rays):
            continue
        step = max(1, RAY_PAIRS // len(tris))
        for at in range(0, len(rays), step):
            r = rays[at : at + step]
            near = (
                (low[tris][None, :] <= ray_bearing[r][:, None])
                & (high[tris][None, :] >= ray_bearing[r][:, None])
                & (y_low[tris][None, :] <= ray_high[r][:, None])
                & (y_high[tris][None, :] >= ray_low[r][:, None])
                & (reach[tris][None, :] < ray_reach[r][:, None])
            )
            ri, ti = np.nonzero(near)
            if not len(ri):
                continue
            ray, tri = r[ri], tris[ti]
            o, d = origins[ray], targets[ray] - origins[ray]
            h = np.cross(d, e2[tri])
            a = (e1[tri] * h).sum(1)
            ok = np.abs(a) > 1e-18
            f = 1.0 / np.where(ok, a, 1.0)
            s = o - v0[tri]
            u = f * (s * h).sum(1)
            q = np.cross(s, e1[tri])
            v = f * (d * q).sum(1)
            t = f * (e2[tri] * q).sum(1)
            length = np.linalg.norm(d, axis=1)
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            hit &= t * length < length - RAY_MARGIN_M
            out[ray[hit]] = True
    return out


def measured(mesh: Mesh, normals: FloatArray, heads: FloatArray, settings: Settings):
    """Which sensor measures which vertex: (points, heads, 2), true where the
    head's laser lights the vertex, that camera sees it and neither meets the
    surface too slantingly. Also returns the vertices outside the range."""
    points, triangles = mesh.points, mesh.triangles
    scanner = SCANNERS[settings.model]
    limit = math.cos(math.radians(settings.max_incidence_deg))
    camera, laser, _ = facing(points, normals, heads, settings.baseline_m)
    cameras = camera_points(points, heads, settings.baseline_m)
    floor = points[:, 1].min()
    centre = (points[:, [0, 2]].min(0) + points[:, [0, 2]].max(0)) / 2
    outside = (
        (points[:, 1] - floor > scanner["rangeHeightM"])
        | (np.abs(points[:, 0] - centre[0]) > scanner["rangeWidthM"] / 2)
        | (np.abs(points[:, 2] - centre[1]) > scanner["rangeDepthM"] / 2)
    )
    seen = np.zeros(camera.shape, dtype=bool)
    for k, head in enumerate(heads):
        lit = np.flatnonzero((laser[:, k] > limit) & ~outside)
        source = np.stack([np.full(len(lit), head[0]), points[lit, 1],
                           np.full(len(lit), head[1])], 1)  # fmt: skip
        lit = lit[~blocked(source, points[lit], points, triangles, head)]
        for c in range(2):
            faced = lit[camera[lit, k, c] > limit]
            clear = ~blocked(cameras[faced, k, c], points[faced], points, triangles, head)
            seen[faced[clear], k, c] = True
    return seen, outside


# Resampling


def split_long_edges(mesh: Mesh, limit: float, extra: FloatArray) -> tuple[Mesh, FloatArray]:
    """Halves every edge longer than `limit` until none is left; the triangles
    stay joined edge to edge. `extra` holds per-vertex values, carried along."""
    points, triangles = mesh.points, mesh.triangles
    while True:
        pairs = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges, inverse = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)
        length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        long = length > limit
        if not long.any():
            return Mesh(points, triangles), extra
        new = np.full(len(edges), -1, dtype=np.int64)
        new[long] = len(points) + np.arange(int(long.sum()))
        points = np.vstack([points, (points[edges[long, 0]] + points[edges[long, 1]]) / 2])
        extra = np.vstack([extra, (extra[edges[long, 0]] + extra[edges[long, 1]]) / 2])
        mid = new[inverse.reshape(-1, 3)]  # midpoints of edges 01, 12 and 20, or -1
        marked = (mid >= 0).sum(1)
        out = [triangles[marked == 0]]
        t, m = triangles[marked == 3], mid[marked == 3]
        out += [np.stack([t[:, 0], m[:, 0], m[:, 2]], 1), np.stack([m[:, 0], t[:, 1], m[:, 1]], 1),
                np.stack([m[:, 2], m[:, 1], t[:, 2]], 1), m]  # fmt: skip
        # One long edge: turn the triangle so that it is edge 01, and cut in two.
        t, m = triangles[marked == 1], mid[marked == 1]
        roll = (np.argmax(m >= 0, axis=1)[:, None] + np.arange(3)[None, :]) % 3
        t, m = np.take_along_axis(t, roll, 1), np.take_along_axis(m, roll, 1)
        out += [np.stack([t[:, 0], m[:, 0], t[:, 2]], 1), np.stack([m[:, 0], t[:, 1], t[:, 2]], 1)]
        # Two long edges: turn it so that they are 01 and 12, and cut in three.
        t, m = triangles[marked == 2], mid[marked == 2]
        roll = (np.argmax(m < 0, axis=1)[:, None] + 1 + np.arange(3)[None, :]) % 3
        t, m = np.take_along_axis(t, roll, 1), np.take_along_axis(m, roll, 1)
        out += [np.stack([m[:, 0], t[:, 1], m[:, 1]], 1), np.stack([t[:, 0], m[:, 0], m[:, 1]], 1),
                np.stack([t[:, 0], m[:, 1], t[:, 2]], 1)]  # fmt: skip
        triangles = np.vstack(out)


def merge_close_vertices(mesh: Mesh, cell: float, extra: FloatArray) -> tuple[Mesh, FloatArray]:
    """Joins vertices that share a cube of side `cell` and are linked by edges
    inside it into one at their mean, and drops the triangles that collapse:
    detail finer than the scanner samples goes. Only linked vertices are
    joined, so two surfaces that lie close together stay apart."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points, triangles = mesh.points, mesh.triangles
    cells = np.floor((points - points.min(0)) / cell).astype(np.int64)
    pairs = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    pairs = pairs[(cells[pairs[:, 0]] == cells[pairs[:, 1]]).all(1)]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    group = connected_components(graph, directed=False)[1].astype(np.int64)
    size = np.bincount(group).astype(float)

    def mean(values: FloatArray) -> FloatArray:
        total = np.zeros((len(size), values.shape[1]))
        np.add.at(total, group, values)
        return total / size[:, None]

    points, extra = mean(points), mean(extra)
    triangles = group[triangles]
    a, b, c = triangles.T
    triangles = triangles[(a != b) & (b != c) & (c != a)]
    # Triangles left on the same three vertices are a fold with nothing inside.
    _, inverse, count = np.unique(
        np.sort(triangles, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    triangles = triangles[count[inverse.reshape(-1)] == 1]
    used = np.unique(triangles)
    renumber = np.full(len(points), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return Mesh(points[used], renumber[triangles]), extra[used]


def resample(mesh: Mesh, edge: float) -> Mesh:
    """The mesh at about one edge length all over: longer edges are halved until
    under 4/3 of `edge`, then linked vertices closer than half of it are joined
    (which moves them a little, so a few edges end up longer again).
    Colours are carried across; each new vertex takes the finger of the nearest
    old one."""
    from scipy.spatial import KDTree

    extra = mesh.colour if mesh.colour is not None else np.zeros((len(mesh.points), 0))
    out, extra = split_long_edges(mesh, edge * 4 / 3, extra)
    out, extra = merge_close_vertices(out, edge / 2, extra)
    if mesh.colour is not None:
        out.colour = extra
    if mesh.finger is not None:
        out.finger = mesh.finger[KDTree(mesh.points).query(out.points)[1]]
    return out


# Fingers


def face_fingers(triangles: IntArray, finger: IntArray) -> IntArray:
    """Each face's finger: the one two or more of its corners belong to, or -1."""
    f = finger[triangles]
    pair = (f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2])
    return np.where(pair, f[:, 0], np.where(f[:, 1] == f[:, 2], f[:, 1], -1))


def damage_fingers(mesh: Mesh, rng: np.random.Generator, settings: Settings):
    """Cuts fingers short and joins neighbours. Returns the label of every face
    (0, FINGER_MISSING or FINGER_FUSED), the faces added to close joined
    fingers into one lump (their convex hull), and a record of each finger."""
    from scipy.spatial import ConvexHull, KDTree

    points, triangles = mesh.points, mesh.triangles
    label = np.zeros(len(triangles), dtype=np.uint8)
    added = np.zeros((0, 3), dtype=np.int64)
    record: list[dict[str, Any]] = []
    if mesh.finger is None:
        return label, added, record
    of_face = face_fingers(triangles, mesh.finger)
    ids = [int(i) for i in np.unique(of_face) if i >= 0]
    faces = {i: np.flatnonzero(of_face == i) for i in ids}
    verts = {i: np.unique(triangles[faces[i]]) for i in ids}
    # The same draws for every finger whatever comes of them, so one finger's
    # fate does not shift the next one's.
    draws = {i: rng.random(3) for i in ids}
    state = {i: "missing" if draws[i][0] < settings.finger_missing else "kept" for i in ids}
    # Where each finger meets the rest of the body: the vertices it shares.
    other = np.zeros(len(points), dtype=bool)
    other[np.unique(triangles[of_face < 0])] = True
    for i in ids:
        if state[i] != "missing":
            continue
        own = points[verts[i]]
        root = own[other[verts[i]]]
        base = root.mean(0) if len(root) else own[np.argmin(own[:, 1])]
        tip = own[np.argmax(np.linalg.norm(own - base, axis=1))]
        along = ((points - base) @ (tip - base)) / max(float((tip - base) @ (tip - base)), 1e-12)
        lost = settings.finger_lost_least + (1 - settings.finger_lost_least) * draws[i][1]
        gone = faces[i][(along[triangles[faces[i]]] >= 1 - lost).all(1)]
        label[gone] |= FINGER_MISSING
        record.append({"finger": i, "state": "missing", "lengthLost": round(float(lost), 3),
                       "facesRemoved": int(len(gone))})  # fmt: skip
    # Neighbours on one hand, both kept and close enough, may be joined; chains
    # of joined neighbours make one lump.
    chains: list[list[int]] = []
    for i in ids:
        j = i + 1
        if j not in faces or j // HAND != i // HAND or "missing" in (state[i], state[j]):
            continue
        gap = float(KDTree(points[verts[i]]).query(points[verts[j]])[0].min())
        if gap * 1000 > settings.fuse_gap_mm or draws[i][2] >= settings.finger_fused:
            continue
        if chains and chains[-1][-1] == i:
            chains[-1].append(j)
        else:
            chains.append([i, j])
    for chain in chains:
        own = np.unique(np.concatenate([verts[i] for i in chain]))
        if len(own) < 4:
            continue
        hull = ConvexHull(points[own])
        simplices = own[hull.simplices]
        a, b, c = (points[simplices[:, k]] for k in range(3))
        inward = (np.cross(b - a, c - a) * hull.equations[:, :3]).sum(1) < 0
        simplices[inward] = simplices[inward][:, ::-1]
        # The lump stays open where the fingers meet the hand.
        simplices = simplices[~other[simplices].all(1)]
        gone = np.concatenate([faces[i] for i in chain])
        label[gone] |= FINGER_FUSED
        added = np.vstack([added, simplices])
        record.append({"fingers": chain, "state": "fused", "facesRemoved": int(len(gone)),
                       "facesAdded": int(len(simplices))})  # fmt: skip
    return label, added, record


# The whole model


def small_islands(triangles: IntArray, vertices: int, least: int) -> BoolArray:
    """Faces of connected patches with fewer than `least` faces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(triangles):
        return np.zeros(0, dtype=bool)
    a = np.concatenate([triangles[:, 0], triangles[:, 1]])
    b = np.concatenate([triangles[:, 1], triangles[:, 2]])
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(vertices, vertices))
    part = connected_components(graph, directed=False)[1][triangles[:, 0]]
    return np.bincount(part, minlength=vertices)[part] < least


def stream(seed: int, step: str) -> np.random.Generator:
    """Each artefact has its own random stream, so switching one off leaves the
    others as they were."""
    return np.random.default_rng([seed, int(hashlib.sha256(step.encode()).hexdigest()[:8], 16)])


def band_pitch_m(settings: Settings) -> float:
    """Height of one band: how far the head travels in one shake."""
    scanner = SCANNERS[settings.model]
    return scanner["rangeHeightM"] / float(np.mean(scanner["scanTimeS"])) / settings.ripple_hz


def point_spacing_m(settings: Settings) -> float:
    """Spacing of the measured points, from the published density."""
    return 0.01 / math.sqrt(SCANNERS[settings.model]["pointsPerCm2"])


def damage(mesh: Mesh, seed: int, settings: Settings | None = None):
    """Applies the damage model. Returns the damaged mesh, the clean mesh it
    was made from (resampled, the training target), the labels and the record.

    Labels, with n vertices and m faces in the clean mesh and N and M in the
    damaged one:
      vertexRemoved (n) and faceRemoved (m): 0 if kept, else LABELS bits
      vertexSeen (n): whether any sensor measured the vertex
      vertexIndex (N): the clean vertex each damaged vertex is
      faceIndex (M): the clean face each damaged face is, -1 for an added one
      sensor (N): the sensor each vertex was taken from, 2 * head + camera
      displacement (N, 3): what was added to the clean position, metres
    """
    settings = settings or Settings()
    scanner = SCANNERS[settings.model]
    steps: list[dict[str, Any]] = []
    flipped = signed_volume(mesh.points, mesh.triangles) < 0
    if flipped:  # the normals must point out of the body
        mesh = replace(mesh, triangles=mesh.triangles[:, ::-1].copy())
    source = topology(mesh.triangles) | {"vertices": int(len(mesh.points))}

    spacing = point_spacing_m(settings)
    if settings.resample:
        edge = settings.edge_mm / 1000
        if edge < spacing:
            raise ValueError(
                f"edge_mm {settings.edge_mm} is finer than the {scanner['name']} measures "
                f"({spacing * 1000:.2f} mm between points)"
            )
        clean = resample(mesh, edge)
    else:
        clean = mesh
    steps.append({"step": "resample", "applied": settings.resample, "edgeMm": settings.edge_mm,
                  "publishedPointSpacingMm": round(spacing * 1000, 3)})  # fmt: skip
    points, triangles = clean.points, clean.triangles
    normals = vertex_normals(points, triangles)
    heads = scanner_heads(points, settings)
    radius = float(np.linalg.norm(heads[0] - heads.mean(0)))

    face_label = np.zeros(len(triangles), dtype=np.uint8)
    added = np.zeros((0, 3), dtype=np.int64)
    step: dict[str, Any] = {"step": "fingers", "applied": False}
    if settings.fingers and clean.finger is None:
        step["skipped"] = "the mesh names no fingers"
    elif settings.fingers:
        face_label, added, record = damage_fingers(clean, stream(seed, "fingers"), settings)
        step |= {"applied": True, "missingChance": settings.finger_missing,
                 "fusedChance": settings.finger_fused, "fuseGapMm": settings.fuse_gap_mm,
                 "leastLengthLost": settings.finger_lost_least, "fingers": record}  # fmt: skip
    steps.append(step)

    seen = np.ones((len(points), len(heads), 2), dtype=bool)
    outside = np.zeros(len(points), dtype=bool)
    if settings.holes:
        seen, outside = measured(clean, normals, heads, settings)
    vertex_seen = seen.any(axis=(1, 2))
    unseen_face = ~vertex_seen[triangles].all(1)
    face_label[unseen_face] |= UNSEEN
    face_label[outside[triangles].any(1)] |= OUTSIDE
    added = added[vertex_seen[added].all(1)]
    fragments = 0
    if settings.holes and settings.min_island_faces > 1:
        left = np.vstack([triangles[face_label == 0], added])
        small = small_islands(left, len(points), settings.min_island_faces)
        kept = np.flatnonzero(face_label == 0)
        face_label[kept[small[: len(kept)]]] |= FRAGMENT
        added = added[~small[len(kept) :]]
        fragments = int(small.sum())
    steps.append({
        "step": "holes", "applied": settings.holes, "heads": int(len(heads)),
        "camerasPerHead": 2, "headRadiusM": round(radius, 4), "ringTurnDeg": settings.ring_turn_deg,
        "baselineM": settings.baseline_m, "maxIncidenceDeg": settings.max_incidence_deg,
        "minIslandFaces": settings.min_island_faces,
        "verticesUnseen": int((~vertex_seen).sum()), "verticesOutsideRange": int(outside.sum()),
        "facesUnseen": int(unseen_face.sum()), "facesInFragments": fragments,
    })  # fmt: skip

    kept_faces = np.flatnonzero(face_label == 0)
    faces = np.vstack([triangles[kept_faces], added])
    face_index = np.concatenate([kept_faces, np.full(len(added), -1, dtype=np.int64)])
    vertex_index = np.unique(faces)
    renumber = np.full(len(points), -1, dtype=np.int64)
    renumber[vertex_index] = np.arange(len(vertex_index))
    vertex_label = np.zeros(len(points), dtype=np.uint8)
    for k in range(3):
        np.bitwise_or.at(vertex_label, triangles[:, k], face_label)
    vertex_label[vertex_index] = 0

    # Each vertex comes from the sensor that sees it most squarely.
    camera, _, sight = facing(points[vertex_index], normals[vertex_index], heads,
                              settings.baseline_m)  # fmt: skip
    sensors = 2 * len(heads)
    score = np.where(seen[vertex_index], camera, camera - 4.0).reshape(-1, sensors)
    sensor = score.argmax(1)
    line = sight.reshape(-1, sensors, 3)[np.arange(len(vertex_index)), sensor]
    along = np.zeros(len(vertex_index))
    pitch = band_pitch_m(settings)
    if settings.banding:
        rng = stream(seed, "banding")
        height = points[vertex_index, 1] - points[:, 1].min()
        rows = int(math.ceil(scanner["rangeHeightM"] / pitch)) + 2
        offsets = rng.normal(scale=settings.band_mm / 1000, size=(len(heads), rows))
        at = np.clip(height / pitch, 0, rows - 1.001)
        low = at.astype(np.int64)
        blend = (1 - np.cos(math.pi * (at - low))) / 2  # smooth from one band to the next
        head = sensor // 2
        along += offsets[head, low] * (1 - blend) + offsets[head, low + 1] * blend
    step = {"step": "banding", "applied": settings.banding, "bandMm": settings.band_mm}
    steps.append(step | {"rippleHz": settings.ripple_hz, "bandHeightMm": round(pitch * 1000, 2)})
    if settings.noise:
        rng = stream(seed, "noise")
        bias = rng.normal(scale=settings.sensor_bias_mm / 1000, size=2 * len(heads))
        # Drawn for every clean vertex, so a vertex's noise does not depend on
        # which others were removed.
        along += rng.normal(scale=settings.noise_mm / 1000, size=len(points))[vertex_index]
        along += bias[sensor]
    steps.append({"step": "noise", "applied": settings.noise, "noiseMm": settings.noise_mm,
                  "sensorBiasMm": settings.sensor_bias_mm})  # fmt: skip
    displacement = along[:, None] * line
    colour = clean.colour[vertex_index] if clean.colour is not None else None
    step = {"step": "seams", "applied": False, "seamGain": settings.seam_gain}
    if settings.seams and colour is None:
        step["skipped"] = "the mesh has no vertex colours"
    elif settings.seams and colour is not None:
        gain = 1 + stream(seed, "seams").normal(scale=settings.seam_gain, size=(2 * len(heads), 3))
        colour = colour.copy()
        colour[:, :3] = np.clip(colour[:, :3] * gain[sensor], 0.0, 1.0)
        step["applied"] = True
    steps.append(step)

    damaged = Mesh(points[vertex_index] + displacement, renumber[faces], colour)
    labels = {
        "vertexRemoved": vertex_label, "faceRemoved": face_label, "vertexSeen": vertex_seen,
        "vertexIndex": vertex_index, "faceIndex": face_index,
        "sensor": sensor.astype(np.int8), "displacement": displacement.astype(np.float32),
    }  # fmt: skip
    moved = np.linalg.norm(displacement, axis=1) * 1000
    record = {
        "seed": seed,
        "scanner": {k: v for k, v in scanner.items()} | {"principle": PRINCIPLE},
        "settings": asdict(settings),
        "parameters": {
            f.name: {
                "value": getattr(settings, f.name),
                "basis": PARAMETERS[f.name][0],
                "note": PARAMETERS[f.name][1],
            }
            for f in fields(settings)
            if f.name in PARAMETERS
        },  # fmt: skip
        "steps": steps,
        "labelBits": LABELS,
        "input": source | {"facesTurnedOutward": bool(flipped)},
        "clean": topology(triangles) | {"vertices": int(len(points))},
        "damaged": topology(damaged.triangles)
        | {
            "vertices": int(len(vertex_index)),
            "verticesRemoved": int(len(points) - len(vertex_index)),
            "facesRemoved": int((face_label != 0).sum()),
            "facesAdded": int(len(added)),
            "removedByLabel": {k: int((face_label & bit > 0).sum()) for k, bit in LABELS.items()},
            "displacementMm": {
                "mean": round(float(moved.mean()), 4) if len(moved) else 0.0,
                "p95": round(float(np.percentile(moved, 95)), 4) if len(moved) else 0.0,
            },
        },
    }
    return damaged, clean, labels, record


# Files


def read_mesh(path: Path) -> Mesh:
    """The largest mesh of a glTF binary, as it is stored (no node transforms):
    positions, triangles, vertex colours if any, and fingers from the joints
    of an ANNY rig if it has one."""
    gltf, blob = read_glb(path)
    counts = [a["count"] for a in gltf["accessors"]]
    best = max(
        gltf["meshes"],
        key=lambda m: sum(counts[p["attributes"]["POSITION"]] for p in m["primitives"]),
    )
    names = None
    if gltf.get("skins"):
        names = [gltf["nodes"][j].get("name", "") for j in gltf["skins"][0]["joints"]]
    points, triangles, colours, fingers = [], [], [], []
    offset = 0
    for prim in best["primitives"]:
        if prim.get("mode", 4) != 4:
            continue
        attributes = prim["attributes"]
        p = accessor(gltf, blob, attributes["POSITION"]).astype(np.float64)
        if "indices" in prim:
            t = accessor(gltf, blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
        else:
            t = np.arange(len(p), dtype=np.int64).reshape(-1, 3)
        points.append(p)
        triangles.append(t + offset)
        offset += len(p)
        if "COLOR_0" in attributes:
            acc = gltf["accessors"][attributes["COLOR_0"]]
            c = accessor(gltf, blob, attributes["COLOR_0"]).astype(np.float64)
            if acc.get("normalized") and acc["componentType"] == 5123:
                c = c / 65535.0
            colours.append(c[:, :3])
        if names and "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
            joints = accessor(gltf, blob, attributes["JOINTS_0"]).astype(np.int64)
            weights = accessor(gltf, blob, attributes["WEIGHTS_0"])
            strongest = joints[np.arange(len(joints)), weights.argmax(1)]
            number = np.full(len(names), -1, dtype=np.int64)
            for j, name in enumerate(names):
                match = FINGER.match(name)
                if match:
                    number[j] = HAND * "LR".index(match.group(2)) + int(match.group(1))
            fingers.append(number[strongest])
    whole = len(points)
    finger = np.concatenate(fingers) if len(fingers) == whole else None
    if finger is not None and not (finger >= 0).any():
        finger = None
    return Mesh(
        np.vstack(points), np.vstack(triangles),
        np.vstack(colours) if len(colours) == whole else None, finger,
    )  # fmt: skip


def write_glb(path: Path, mesh: Mesh) -> None:
    """A plain glTF binary: one mesh with POSITION, NORMAL, COLOR_0 if the mesh
    has colours, and 32-bit indices."""
    arrays = [
        ("POSITION", mesh.points.astype("<f4"), "VEC3", 5126),
        ("NORMAL", vertex_normals(mesh.points, mesh.triangles).astype("<f4"), "VEC3", 5126),
    ]
    if mesh.colour is not None:
        arrays.append(("COLOR_0", mesh.colour[:, :3].astype("<f4"), "VEC3", 5126))
    arrays.append(("indices", mesh.triangles.astype("<u4").reshape(-1), "SCALAR", 5125))
    blob, views, accessors = b"", [], []
    for name, array, kind, component in arrays:
        data = array.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data),
                      "target": 34963 if name == "indices" else 34962})  # fmt: skip
        entry = {"bufferView": len(views) - 1, "componentType": component,
                 "count": len(array), "type": kind}  # fmt: skip
        if name == "POSITION" and len(array):
            entry |= {"min": array.min(0).tolist(), "max": array.max(0).tolist()}
        accessors.append(entry)
        blob += data + b"\0" * (-len(data) % 4)
    names = [a[0] for a in arrays]
    primitive = {"attributes": {n: i for i, n in enumerate(names) if n != "indices"},
                 "indices": names.index("indices"), "mode": 4}  # fmt: skip
    gltf = {
        "asset": {"version": "2.0", "generator": "open-anatomy scan_damage.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": path.stem}],
        "meshes": [{"name": path.stem, "primitives": [primitive]}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
    }  # fmt: skip
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(text), 0x4E4F534A)
        + text + struct.pack("<II", len(blob), 0x004E4942) + blob
    )  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="path and stem of the outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", choices=SCANNERS, default=defaults.model)
    for f in fields(Settings):
        flag = f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(f"--no-{flag}", dest=f.name, action="store_false")
        elif f.name != "model":
            kind = int if f.type == "int" else float
            basis, note = PARAMETERS[f.name]
            parser.add_argument(f"--{flag}", type=kind, default=getattr(defaults, f.name),
                                help=f"{basis}: {note} (default %(default)s)")  # fmt: skip
    args = parser.parse_args()
    for path in (args.source, args.out):
        if PERSONAL.search(str(path)) or PERSONAL.search(str(path.resolve())):
            sys.exit(f"{path}: refused: data/vitronic/ holds real people's scans, and the "
                     "damage model is only ever run on synthetic bodies")  # fmt: skip
    settings = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)})
    damaged, clean, labels, record = damage(read_mesh(args.source), args.seed, settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.name
    outputs = {}
    for kind, mesh in (("clean", clean), ("damaged", damaged)):
        path = args.out.with_name(f"{stem}.{kind}.glb")
        write_glb(path, mesh)
        outputs[kind] = {"file": path.name, "sha256": sha256(path)}
    path = args.out.with_name(f"{stem}.labels.npz")
    np.savez_compressed(path, **labels)
    outputs["labels"] = {"file": path.name, "arrays": sorted(labels)}
    record = {"source": args.source.name, "sourceSha256": sha256(args.source)} | record
    record["outputs"] = outputs
    args.out.with_name(f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    d = record["damaged"]
    print(f"scan_damage: {args.source.name} seed {args.seed}: {record['clean']['vertices']} "
          f"vertices, {d['verticesRemoved']} removed, {d['holes']} holes, moved "
          f"{d['displacementMm']['mean']} mm on average; wrote {args.out}.*")  # fmt: skip


if __name__ == "__main__":
    main()
