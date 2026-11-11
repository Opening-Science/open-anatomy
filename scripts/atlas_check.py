"""
Checks one built atlas skeleton against the measured parts of the Stage 1 bar:
plain Python, no Blender.

  python scripts/atlas_check.py <case>
  python scripts/atlas_check.py --glb <skeleton.glb> --manifest <skeleton.manifest.json>
      --seg-dir <dir of TotalSegmentator masks> [--out report.json]

With a case it reads what atlas.py built: data/atlas/<case>/<case>.skeleton.glb,
data/work/<case>/<case>.skeleton.manifest.json and the masks in
data/work/<case>/seg. The GLB is taken back into the scan's RAS millimetres
(the manifest's centre_offset_m) and measured there.

(b) Fidelity to the scan. For every node the manifest does not call modelled,
the distance between the mesh and its mask's surface, both ways (mesh to mask,
and mask to mesh, so a missing part shows): the mean at most MAX_MEAN_MM and
the 95th percentile at most MAX_P95_MM. The mask's surface is the zero level
of its smoothed signed distance (smooth_field), which gives a smooth shape
back to about a tenth of a voxel rather than half of one. mesh.py changes
masks in ways it declares (pieces moved between neighbours or dropped, knobs
trimmed, gaps bridged, rib ends evened). With `mesh.py --contoured` the
manifest names the mask each node was contoured from; the mesh is then
measured against that mask, and that mask against the scan's own: what it
gained and lost must be covered by what the manifest declares for the node,
or the node fails as an undeclared change. Without contoured masks the meshes
are measured against the scan's own masks, and the report says so.

(c) Topology. Every node closed (each edge used by exactly two faces),
manifold at its edges and vertices, wound one way with a positive volume,
free of faces without area and of faces that cross (crossings), and in one
piece; the set of costal cartilages in several, none under MIN_PIECE_ML.
Nodes that enter each other are listed with the depth and the shared volume,
and fail only if MAX_ENTERING_MM is set.

(d) Anatomy, on the meshes: twelve pairs of ribs; each rib within RIB_JOINT_MM
of the vertebrae it articulates with (rib_joints); cartilage at the front end
of ribs 1 to 10 and none at ribs 11 and 12; seven separate contacts on each
side of the sternum, cartilages 1 to 7 in order, and cartilages 8 to 10 each
lapping onto the one above (cartilages, arch_laps); the sternum's jugular notch,
sternal angle and xiphoid process (sternum_shape).

It prints one line per node, per pair of nodes that enter each other, per
rib, per contact and per rule, writes the whole report as JSON, and exits 1
when any rule fails, after reporting everything.
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
import sys
from pathlib import Path
from typing import Any, NamedTuple

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import SKELETON, node_name  # noqa: E402

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Bools = npt.NDArray[np.bool_]

# RAS millimetres to glTF metres, as mesh.py writes them (its RAS_TO_GLTF).
RAS_TO_GLTF = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64) * 0.001
COMPONENTS = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
LABELS = {node_name(s.term): s.label for s in SKELETON}  # node name -> TotalSegmentator label
NAMES = {label: name for name, label in LABELS.items()}

# (b) Fidelity. The two limits are the owner's bar of 11 November 2026.
MAX_MEAN_MM = 1.0
MAX_P95_MM = 2.0
FIELD_SMOOTH_VOXELS = 0.75  # the Gaussian that evens out a mask's voxel steps (smooth_field)
FIELD_PAD = 6  # voxels kept round a mask: beyond the wider Gaussian's reach
SAMPLE_MM = 0.75  # a surface is measured at points this far apart: half a voxel of the scans
MIN_SAMPLES = 2000  # and at no fewer than this many, however small it is
MAX_SAMPLES = 400000
LOCATOR_MM = 1.5  # a triangle is covered with points no further apart than this (Surface)
LOCATOR_MAX_PARTS = 24  # an edge is cut into at most this many for that
LOCATOR_NEAREST = 8  # the triangles of this many nearest points are measured exactly
# The manifest rounds each volume to 0.01 ml and the dropped share to 0.0001.
DECLARED_SLACK_ML = 0.05
# The manifest's per-node volumes, in ml, of what mesh.py added to a mask
# before contouring it and of what it took off. A new declared change goes here.
DECLARED_ADDED = ("end_filled_ml", "bridged_ml", "head_grown_ml")
DECLARED_REMOVED = ("knobs_trimmed_ml", "end_lumps_trimmed_ml")

# (c) Topology.
MIN_PIECE_ML = 0.5  # a piece of a set under this is a loose blob (mesh.py's MIN_PIECE_VOXELS)
# Nearer than this is touching, not crossing: a nanometre, far below the
# 0.00002 mm steps of the GLB's 32-bit metres.
TOUCH_MM = 1e-6
DEGENERATE_SHARE = 1e-9  # of its longest edge squared: a face with less area has none
# How deep one node may lie in another. A join needs some overlap (mesh.py
# starts a cartilage inside its rib) and the bar sets no limit, so None: the
# depths are reported and fail nothing.
MAX_ENTERING_MM: float | None = None
OVERLAP_GRID_MM = 0.5  # the shared volume is counted on a grid this fine
OVERLAP_MAX_CELLS = 3e5  # or coarser, so that the grid has no more cells than this
ENTER_PROBE_MM = 3.0  # two nodes are tried for entering where their points come this near

# (d) Anatomy.
# Bone to bone across a rib's joints with its vertebrae. No measured norm for
# the costovertebral joints was found; the facet joints of the thoracic
# spine, small synovial joints like them, measure about 1.3 mm on CT. 3 mm is
# that and one voxel of a 1.5 mm scan, by which two masks stand back from a
# cleft they both border.
RIB_JOINT_MM = 3.0
RIB_END_MM = 10.0  # a rib's front end is its surface within this of its tip
# Rib and cartilage are one continuous piece of tissue with no joint cleft, so
# any gap is the model's; half a millimetre is under what the surfaces are
# sampled at.
COSTOCHONDRAL_MM = 0.5
# Ribs 11 and 12 end free. Cartilage nearer their end than a rib is thick
# would read as joined to it.
FLOAT_CLEAR_MM = 5.0
CONTACT_MM = 0.5  # sternum within this of the cartilage, or inside it, is in contact
PATCH_SAMPLE_MM = 0.5  # the sternum's surface is tried at points this far apart
PATCH_LINK_MM = 1.0  # two contacts are separate when this much bare bone lies between them
PATCH_MIN_MM2 = 5.0  # a contact under this area is a speck
# Interchondral joints lie between cartilages 6 to 9, sometimes 5 and 10
# (Gray's Anatomy, 1918, "Interchondral articulations"). Cartilages 1 to 4
# join nothing but their rib and the sternum.
LONE_CARTILAGES = (1, 2, 3, 4)
JOINED_FAILS = False  # one of those in one piece with another is reported; the bar is silent
# A band of ribs 8 to 10 ends at least this far from the sternum: a band's
# half width (9 mm at most in mesh.py) and the contact's reach.
ARCH_CLEAR_MM = 10.0
# Cartilages 8, 9 and 10 lap onto the one above: a piece of its own each,
# lying against it. Against means within LAP_GAP_MM (the interchondral joints
# are synovial clefts under a millimetre wide), over LAP_MM or more of length:
# the facets are oblong, about as long as a cartilage is wide, and 10 mm is a
# narrow lower cartilage's width. Both figures are a proposal, not a norm.
LAP_GAP_MM = 1.0
LAP_MM = 10.0
FUSED_FAILS = True  # two cartilages in one piece fail: the bar wants each a piece of its own
# The sternum. The definitions follow data/work/tmp/round7/findings.md, which
# measured the same on masks and meshes.
MANUBRIUM_MM = 45.0  # the top of the sternum that gives its midline: a manubrium's length
PLANES_MM = (-6.0, -3.0, 0.0, 3.0, 6.0)  # sagittal planes either side of the midline
PROFILE_BIN_MM = 1.5  # the sagittal outline is read every this much of height
TOP_CLEAR_MM = 9.0  # the rounded top is left out of the straight lines
BEND_FIT_MM = 100.0  # the bend is looked for in this much of the sternum from the top
BEND_BELOW_TOP_MM = (30.0, 75.0)  # and not nearer the top or further down than this
ANGLE_CLEAR_MM = 6.0  # the lines keep this far from the joint either side
ANGLE_BODY_MM = 48.0  # the body's line runs down to this far below the joint
# Between manubrium and body the sternal angle measured 149 to 177 degrees in
# a cadaver series (mean 163.4 in men, 165.0 in women; StatPearls, "Anatomy,
# Angle of Louis"): a bend of 3 to 31 degrees from straight.
STERNAL_ANGLE_DEG = (3.0, 31.0)
# The second cartilage joins the sternum at the angle, on a facet that
# manubrium and body share, so the bend lies within its contact: half a
# cartilage's width (9 mm at most in mesh.py) and a voxel.
ANGLE_AT_SECOND_MM = 10.0
NOTCH_BIN_MM = 0.5  # the upper outline is read every this much of width
NOTCH_HALF_SPAN_MM = 32.0  # out to this far either side of the midline
NOTCH_SAMPLE_MM = 0.2  # from points this far apart on the manubrium
# No measured norm for the jugular notch's depth was found. A dip shallower
# than one voxel of the 1.5 mm scans cannot be told from a mask's steps, so
# anything under that is no notch.
JUGULAR_NOTCH_MM = 1.5
# The xiphoid process measured 20.6 to 81.5 mm (mean 50) in 500 adults on CT
# (Skeletal Radiology 2011, doi:10.1007/s00256-010-1022-1).
XIPHOID_MM = 20.0
WIDTH_BIN_MM = 3.0  # the sternum's width is read every this much of height
LIMITS = {
    "meanMm": MAX_MEAN_MM, "p95Mm": MAX_P95_MM, "declaredSlackMl": DECLARED_SLACK_ML,
    "minPieceMl": MIN_PIECE_ML, "enteringMm": MAX_ENTERING_MM, "ribJointMm": RIB_JOINT_MM,
    "lapGapMm": LAP_GAP_MM, "lapMm": LAP_MM,
    "costochondralMm": COSTOCHONDRAL_MM, "floatClearMm": FLOAT_CLEAR_MM, "contactMm": CONTACT_MM,
    "sternalAngleDeg": STERNAL_ANGLE_DEG, "angleAtSecondMm": ANGLE_AT_SECOND_MM,
    "jugularNotchMm": JUGULAR_NOTCH_MM, "xiphoidMm": XIPHOID_MM,
}  # fmt: skip


def tilted() -> Floats:
    """A fixed turn of space, by 0.7 radians about the axis 1, 2, 3, that
    lines up no axis with another: Surface.inside casts its rays along it."""
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(0.7) * k + (1 - np.cos(0.7)) * k @ k


TILT = tilted()


class Mesh(NamedTuple):
    vertices: Floats  # one row per vertex, in the scan's RAS millimetres
    triangles: Ints  # three vertex numbers per triangle


# Reading the GLB


def node_matrix(node: dict[str, Any]) -> Floats:
    """A glTF node's own 4x4 transform: its matrix, or its translation,
    rotation (a quaternion x, y, z, w) and scale."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(node.get("scale", [1.0, 1.0, 1.0]))
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def read_glb(path: Path) -> dict[str, Mesh]:
    """Every mesh node of a GLB by name, in the GLB's own frame and units,
    with each node's transform (and its parents') applied. A node's
    primitives are joined into one mesh; anything but indexed triangles
    stops the check, since nothing else can be measured as a closed surface."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB")
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + length])
    blob = data[20 + length + 8 :]

    def array(index: int) -> npt.NDArray[Any]:
        accessor = gltf["accessors"][index]
        if "sparse" in accessor or "bufferView" not in accessor:
            raise ValueError(f"{path.name}: accessor {index} is sparse, which is not read")
        view = gltf["bufferViews"][accessor["bufferView"]]
        dtype = np.dtype(COMPONENTS[accessor["componentType"]])
        width = WIDTHS[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(blob, np.uint8, offset=start),
            shape=(accessor["count"], dtype.itemsize * width),
            strides=(stride, 1),
        )
        return np.ascontiguousarray(rows).view(dtype).reshape(accessor["count"], width)

    world: dict[int, Floats] = {}

    def place(index: int, parent: Floats) -> None:
        world[index] = parent @ node_matrix(gltf["nodes"][index])
        for child in gltf["nodes"][index].get("children", []):
            place(child, world[index])

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        place(root, np.eye(4))
    meshes: dict[str, Mesh] = {}
    for index, node in enumerate(gltf["nodes"]):
        if "mesh" not in node or index not in world:
            continue
        vertices, triangles, count = [], [], 0
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4 or "indices" not in primitive:
                raise ValueError(f"{path.name}: {node.get('name')} is not indexed triangles")
            position = array(primitive["attributes"]["POSITION"]).astype(np.float64)
            vertices.append(position @ world[index][:3, :3].T + world[index][:3, 3])
            triangles.append(array(primitive["indices"]).astype(np.int64).reshape(-1, 3) + count)
            count += len(position)
        name = str(node.get("name", f"node{index}"))
        if name in meshes:
            raise ValueError(f"{path.name}: two nodes are named {name}")
        meshes[name] = Mesh(np.vstack(vertices), np.vstack(triangles))
    return meshes


def to_ras(vertices: Floats, centre_offset_m: Floats) -> Floats:
    """GLB metres back to the scan's RAS millimetres: mesh.py wrote
    ras @ RAS_TO_GLTF.T - centre, and the manifest keeps the centre."""
    return (vertices + centre_offset_m) @ np.linalg.inv(RAS_TO_GLTF).T


def welded(mesh: Mesh) -> Mesh:
    """The mesh with vertices at one and the same position made one vertex:
    an exporter may write a vertex once per face corner, which would read as
    a surface full of holes."""
    _, first, inverse = np.unique(
        mesh.vertices.astype(np.float32), axis=0, return_index=True, return_inverse=True
    )
    return Mesh(mesh.vertices[first], inverse.reshape(-1)[mesh.triangles])


# Distances between surfaces


def triangle_areas(mesh: Mesh) -> Floats:
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2


def signed_volume(mesh: Mesh) -> float:
    """The volume a closed surface encloses, in mm3: positive when its faces
    are wound counter-clockwise seen from outside."""
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def sample_points(mesh: Mesh, spacing_mm: float = SAMPLE_MM) -> Floats:
    """Points spread evenly over a surface by area, about one per
    spacing_mm squared (at least MIN_SAMPLES, at most MAX_SAMPLES): each
    triangle gets its share of them by its area, at places drawn with a
    fixed seed, so two runs measure at the same points."""
    areas = triangle_areas(mesh)
    total = float(areas.sum())
    if total <= 0:
        return np.zeros((0, 3))
    count = int(np.clip(np.ceil(total / spacing_mm**2), MIN_SAMPLES, MAX_SAMPLES))
    at = (np.arange(count) + 0.5) / count * total
    which = np.minimum(np.searchsorted(np.cumsum(areas), at), len(areas) - 1)
    rng = np.random.default_rng(0)
    root, v = np.sqrt(rng.random(count)), rng.random(count)
    a, b, c = (mesh.vertices[mesh.triangles[which, i]] for i in range(3))
    return (1 - root)[:, None] * a + (root * (1 - v))[:, None] * b + (root * v)[:, None] * c


def point_triangle_distance(p: Floats, a: Floats, b: Floats, c: Floats) -> Floats:
    """The distance from each point p to its triangle a, b, c (rows): to the
    triangle's plane where the foot of the perpendicular falls inside the
    triangle, and to the nearest of its three edges otherwise. A triangle
    with no area is its edges."""

    def to_edge(u: Floats, v: Floats) -> Floats:
        along = v - u
        t = np.einsum("ij,ij->i", p - u, along) / np.maximum(
            np.einsum("ij,ij->i", along, along), 1e-30
        )
        return np.linalg.norm(p - u - np.clip(t, 0, 1)[:, None] * along, axis=1)

    nearest = np.minimum(np.minimum(to_edge(a, b), to_edge(b, c)), to_edge(c, a))
    normal = np.cross(b - a, c - a)
    size = np.linalg.norm(normal, axis=1)
    inside = size > 1e-12
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ij,ij->i", np.cross(v - u, p - u), normal) >= 0
    plane = np.abs(np.einsum("ij,ij->i", p - a, normal)) / np.maximum(size, 1e-12)
    return np.where(inside, plane, nearest)


class Surface:
    """A triangle surface readied for measuring against: the distance from
    any point to it, and whether a point lies inside it.

    For the distance, every triangle is covered with points no further apart
    than LOCATOR_MM (the middles of the small triangles it divides into); a
    point's LOCATOR_NEAREST nearest of those name the triangles that can be
    its nearest, and the distance to each of them is worked out exactly.
    """

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.corners = tuple(mesh.vertices[mesh.triangles[:, i]] for i in range(3))
        a, b, c = self.corners
        longest = np.maximum(
            np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
            np.linalg.norm(a - c, axis=1),
        )
        parts = np.clip(np.ceil(longest / LOCATOR_MM), 1, LOCATOR_MAX_PARTS).astype(np.int64)
        points, owners = [], []
        for n in np.unique(parts):
            # The n * n small triangles of a triangle with each edge cut in n:
            # their middles, as shares of the three corners.
            i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            up = np.stack([i, j], -1)[i + j <= n - 1] + 1 / 3
            down = np.stack([i, j], -1)[i + j <= n - 2] + 2 / 3
            share = np.vstack([up, down]) / n
            which = np.flatnonzero(parts == n)
            middle = (
                (1 - share.sum(1))[None, :, None] * a[which][:, None, :]
                + share[None, :, 0, None] * b[which][:, None, :]
                + share[None, :, 1, None] * c[which][:, None, :]
            )
            points.append(middle.reshape(-1, 3))
            owners.append(np.repeat(which, len(share)))
        self.owner = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
        self.tree = KDTree(np.vstack(points)) if owners else None
        self._columns: tuple[Ints, Ints, Floats, Floats] | None = None

    def nearest(self, points: Floats) -> tuple[Floats, Ints]:
        """The distance from each point to the surface, in mm, and the
        triangle it is nearest to."""
        if self.tree is None or len(points) == 0:
            return np.full(len(points), np.inf), np.zeros(len(points), dtype=np.int64)
        a, b, c = self.corners
        k = min(LOCATOR_NEAREST, len(self.owner))
        distance = np.empty(len(points))
        triangle = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), 20000):
            chunk = points[start : start + 20000]
            near = self.tree.query(chunk, k=k)[1].reshape(len(chunk), k)
            which = self.owner[near]
            flat = which.reshape(-1)
            d = point_triangle_distance(np.repeat(chunk, k, axis=0), a[flat], b[flat], c[flat])
            d = d.reshape(len(chunk), k)
            best = np.argmin(d, axis=1)
            rows = np.arange(len(chunk))
            distance[start : start + len(chunk)] = d[rows, best]
            triangle[start : start + len(chunk)] = which[rows, best]
        return distance, triangle

    def distance(self, points: Floats) -> Floats:
        """The distance from each point to the surface, in mm."""
        return self.nearest(points)[0]

    def inside(self, points: Floats) -> Bools:
        """Whether each point lies inside the surface, which must be closed:
        a ray from the point crosses it an odd number of times. The rays all
        run one way, along a direction tilted off every axis (TILT), so that
        none runs along a face or through an edge of a mesh laid out on the
        axes. The triangles are sorted into the cells of a grid across the
        rays, and a ray is tried against those in its own cell only."""
        if len(points) == 0 or len(self.mesh.triangles) == 0:
            return np.zeros(len(points), dtype=bool)
        if self._columns is None:
            flat = self.mesh.vertices @ TILT.T
            corners = flat[self.mesh.triangles]
            low, high = corners[:, :, :2].min(1), corners[:, :, :2].max(1)
            size = cell_size(high - low, len(corners))
            origin = low.min(0)
            owner, cells = cells_of(
                np.floor((low - origin) / size).astype(np.int64),
                np.floor((high - origin) / size).astype(np.int64),
            )
            width = int(cells[:, 1].max()) + 1
            key = cells[:, 0] * width + cells[:, 1]
            order = np.argsort(key, kind="stable")
            self._columns = (key[order], owner[order], corners, np.array([*origin, size, width]))
        key, owner, corners, grid = self._columns
        flat = points @ TILT.T
        cell = np.floor((flat[:, :2] - grid[:2]) / grid[2]).astype(np.int64)
        mine = cell[:, 0] * int(grid[3]) + cell[:, 1]
        mine[(cell < 0).any(1) | (cell[:, 1] >= grid[3])] = -1  # beside the grid: outside
        first, last = np.searchsorted(key, mine, "left"), np.searchsorted(key, mine, "right")
        count = last - first
        point = np.repeat(np.arange(len(points)), count)
        entry = (
            np.repeat(first, count)
            + np.arange(int(count.sum()))
            - np.repeat(np.cumsum(count) - count, count)
        )
        a, b, c = (corners[owner[entry], i] for i in range(3))
        p = flat[point]

        def side(u: Floats, v: Floats) -> Floats:
            return (v[:, 0] - u[:, 0]) * (p[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (
                p[:, 0] - u[:, 0]
            )

        ab, bc, ca = side(a, b), side(b, c), side(c, a)
        within = ((ab >= 0) & (bc >= 0) & (ca >= 0)) | ((ab <= 0) & (bc <= 0) & (ca <= 0))
        whole = ab + bc + ca
        within &= np.abs(whole) > 1e-30
        height = (bc * a[:, 2] + ca * b[:, 2] + ab * c[:, 2]) / np.where(within, whole, 1.0)
        above = within & (height > p[:, 2])
        return np.bincount(point[above], minlength=len(points)) % 2 == 1


def cells_of(low: Ints, high: Ints) -> tuple[Ints, Ints]:
    """Every cell of a grid that each box covers, from the boxes' first and
    last cells along each axis (rows): the number of the box, and the cell,
    once per cell covered."""
    span = high - low + 1
    count = span.prod(1)
    box = np.repeat(np.arange(len(low)), count)
    within = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    cells = np.empty((len(box), low.shape[1]), dtype=np.int64)
    for axis in reversed(range(low.shape[1])):
        wide = np.repeat(span[:, axis], count)
        cells[:, axis] = np.repeat(low[:, axis], count) + within % wide
        within = within // wide
    return box, cells


def cell_size(extent: Floats, boxes: int) -> float:
    """A grid's cell size for boxes of these extents (rows): one and a half
    times their median side, doubled until the boxes cover no more than
    40 cells each on average, so one huge triangle cannot fill the memory."""
    size = max(1.5 * float(np.median(extent.max(1))), 1e-6)
    while (np.floor(extent / size) + 2).prod(1).sum() > 40 * boxes + 1e6:
        size *= 2
    return size


def cell_keys(cells: Ints) -> Ints:
    """One number per cell (rows of whole numbers), the same for the same cell."""
    cells = cells - cells.min(0)
    key = np.zeros(len(cells), dtype=np.int64)
    for axis in range(cells.shape[1]):
        key = key * (int(cells[:, axis].max()) + 1) + cells[:, axis]
    return key


def crossing(first: Floats, second: Floats) -> Bools:
    """Whether each pair of triangles (k x 3 corners x 3) passes through the
    other. Two triangles cross when each has corners strictly on both sides
    of the other's plane and the stretches they cut out of the line the two
    planes share overlap: then one runs through the other, on a cut of some
    length. Triangles that only touch (a corner or an edge of one lying on
    the other, without going through) do not cross, and nor do two that keep
    to one side of each other. Two triangles in one plane cross when they
    cover some area in common. TOUCH_MM is the least distance that counts as
    a side or as a length."""
    normals, sizes = [], []
    for t in (first, second):
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        size = np.linalg.norm(n, axis=1)
        normals.append(n / np.maximum(size, 1e-30)[:, None])
        sizes.append(size)
    flat = (sizes[0] > 1e-12) & (sizes[1] > 1e-12)  # a triangle with no area crosses nothing
    off_second = np.einsum("ijk,ik->ij", first - second[:, :1], normals[1])
    off_first = np.einsum("ijk,ik->ij", second - first[:, :1], normals[0])
    line = np.cross(normals[0], normals[1])

    def stretch(t: Floats, off: Floats) -> tuple[Floats, Floats]:
        """Where a triangle crosses the other's plane, as the two ends of its
        cut measured along the planes' common line."""
        along = np.einsum("ijk,ik->ij", t, line)
        low, high = np.full(len(t), np.inf), np.full(len(t), -np.inf)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            cut = off[:, i] * off[:, j] < 0
            at = along[:, i] + (along[:, j] - along[:, i]) * off[:, i] / np.where(
                cut, off[:, i] - off[:, j], 1.0
            )
            low = np.where(cut, np.minimum(low, at), low)
            high = np.where(cut, np.maximum(high, at), high)
        on = np.abs(off) <= TOUCH_MM  # a corner in the other's plane is an end of the cut
        low = np.minimum(low, np.where(on, along, np.inf).min(1))
        high = np.maximum(high, np.where(on, along, -np.inf).max(1))
        return low, high

    low1, high1 = stretch(first, off_second)
    low2, high2 = stretch(second, off_first)
    shared = np.minimum(high1, high2) - np.maximum(low1, low2)
    through = (
        (off_second.min(1) < -TOUCH_MM) & (off_second.max(1) > TOUCH_MM)
        & (off_first.min(1) < -TOUCH_MM) & (off_first.max(1) > TOUCH_MM)
        & (shared > TOUCH_MM * np.linalg.norm(line, axis=1))
    )  # fmt: skip
    level = (np.abs(off_second).max(1) <= TOUCH_MM) & (np.abs(off_first).max(1) <= TOUCH_MM)
    if (level & flat).any():
        # In one plane: they share area unless some edge of either has all of
        # the other on its far side (a separating line).
        u = first[:, 1] - first[:, 0]
        u = u / np.maximum(np.linalg.norm(u, axis=1), 1e-30)[:, None]
        v = np.cross(normals[0], u)
        flat1 = np.stack([np.einsum("ijk,ik->ij", first, d) for d in (u, v)], -1)
        flat2 = np.stack([np.einsum("ijk,ik->ij", second, d) for d in (u, v)], -1)
        apart = np.zeros(len(first), dtype=bool)
        for t in (flat1, flat2):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = t[:, j] - t[:, i]
                across = np.stack([-edge[:, 1], edge[:, 0]], -1)
                across = across / np.maximum(np.linalg.norm(across, axis=1), 1e-30)[:, None]
                one = np.einsum("ijk,ik->ij", flat1, across)
                other = np.einsum("ijk,ik->ij", flat2, across)
                apart |= (one.max(1) <= other.min(1) + TOUCH_MM) | (
                    other.max(1) <= one.min(1) + TOUCH_MM
                )
        through |= level & ~apart
    return through & flat


def crossings(mesh: Mesh) -> Ints:
    """The pairs of triangles of a mesh that cross (crossing) and share no
    vertex, as rows of two triangle numbers. Triangles that share a vertex
    are neighbours and meet there by design. The triangles are sorted into
    the cells of a grid by their bounding boxes, and only two that meet in a
    cell, and whose boxes overlap, are tested."""
    triangles = mesh.triangles
    corners = mesh.vertices[triangles]
    low, high = corners.min(1), corners.max(1)
    if len(triangles) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    size = cell_size(high - low, len(triangles))
    origin = low.min(0)
    owner, cells = cells_of(
        np.floor((low - origin) / size).astype(np.int64),
        np.floor((high - origin) / size).astype(np.int64),
    )
    key = cell_keys(cells)
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    found = []
    at = np.arange(len(key) - 1)
    lag = 1
    while len(at):
        at = at[at + lag < len(key)]
        at = at[key[at + lag] == key[at]]  # still in the same cell, lag entries on
        i, j = owner[at], owner[at + lag]
        near = ((low[i] <= high[j]) & (low[j] <= high[i])).all(1)
        found.append(np.minimum(i[near], j[near]) * len(triangles) + np.maximum(i[near], j[near]))
        lag += 1
    pair = np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)
    i, j = pair // len(triangles), pair % len(triangles)
    apart = (triangles[i][:, :, None] != triangles[j][:, None, :]).all((1, 2))
    i, j = i[apart], j[apart]
    hit = np.zeros(len(i), dtype=bool)
    for start in range(0, len(i), 200000):
        part = slice(start, start + 200000)
        hit[part] = crossing(corners[i[part]], corners[j[part]])
    return np.stack([i[hit], j[hit]], 1)


# (c) Topology


def topology(mesh: Mesh) -> dict[str, Any]:
    """What a node's surface is made of, counted on its welded triangles:
    its connected pieces (joined through shared vertices), each with its
    triangles, enclosed volume in ml and open edges; the edges used by one
    face only (holes) and by more than two (non-manifold); the vertices
    whose faces make more than one fan round them (two sheets pinched
    together at a point); the edges whose two faces are wound against each
    other; the faces with no area; and the pairs of faces that cross
    (crossings), with the pairs that cross between two different pieces."""
    mesh = welded(mesh)
    triangles, count = mesh.triangles, len(mesh.triangles)
    points = len(mesh.vertices)
    if count == 0:
        return {"triangles": 0, "pieces": [], "openEdges": 0, "nonManifoldEdges": 0,
                "pinchedVertices": 0, "flippedEdges": 0, "degenerateFaces": 0,
                "crossingPairs": 0, "crossingBetweenPieces": 0, "volumeMl": 0.0}  # fmt: skip
    # Half edges in three blocks: corner 0 to 1 of every face, then 1 to 2, then 2 to 0.
    start = triangles.T.reshape(-1)
    end = triangles[:, [1, 2, 0]].T.reshape(-1)
    low, high = np.minimum(start, end), np.maximum(start, end)
    _, edge, uses = np.unique(low * points + high, return_inverse=True, return_counts=True)
    edge = edge.reshape(-1)
    turn = np.bincount(edge, np.where(start < end, 1.0, -1.0))
    face = np.tile(np.arange(count), 3)
    block = np.repeat(np.arange(3), count)
    corner_start, corner_end = face * 3 + block, face * 3 + (block + 1) % 3
    at_low = np.where(start < end, corner_start, corner_end)
    at_high = np.where(start < end, corner_end, corner_start)
    # Round a vertex, two faces are in one fan when they share an edge at it
    # that only the two of them use.
    order = np.argsort(edge, kind="stable")
    same = np.flatnonzero((edge[order][1:] == edge[order][:-1]) & (uses[edge[order]][1:] == 2))
    pair, twin = order[same], order[same + 1]
    links = (np.r_[at_low[pair], at_high[pair]], np.r_[at_low[twin], at_high[twin]])
    fans = connected_components(
        coo_matrix((np.ones(len(links[0])), links), shape=(3 * count, 3 * count)), directed=False
    )[1]
    vertex_of = triangles.reshape(-1)  # corner f * 3 + k is at vertex triangles[f, k]
    round_vertex = np.unique(np.stack([vertex_of, fans], 1), axis=0)[:, 0]
    pinched = int((np.bincount(round_vertex, minlength=points) > 1).sum())

    a, b, c = (mesh.vertices[triangles[:, i]] for i in range(3))
    longest = np.maximum(
        np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
        np.linalg.norm(a - c, axis=1),
    )
    area = triangle_areas(mesh)
    degenerate = (
        (triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0]) | (area <= DEGENERATE_SHARE * longest**2)
    )  # fmt: skip
    joined = coo_matrix((np.ones(len(start)), (start, end)), shape=(points, points))
    piece_of_vertex = connected_components(joined, directed=False)[1]
    piece_of_face = piece_of_vertex[triangles[:, 0]]
    names, piece_of_face = np.unique(piece_of_face, return_inverse=True)
    piece_of_face = piece_of_face.reshape(-1)
    volume = np.bincount(piece_of_face, np.einsum("ij,ij->i", a, np.cross(b, c)) / 6)
    open_edge = np.zeros(len(uses), dtype=bool)
    open_edge[edge] = uses[edge] == 1
    edge_piece = np.zeros(len(uses), dtype=np.int64)
    edge_piece[edge] = piece_of_face[face]
    open_by_piece = np.bincount(edge_piece[open_edge], minlength=len(names))
    crossed = crossings(mesh)
    between = piece_of_face[crossed[:, 0]] != piece_of_face[crossed[:, 1]]
    crossed_by_piece = np.bincount(piece_of_face[crossed].reshape(-1), minlength=len(names))
    pieces = [
        {
            "triangles": int((piece_of_face == k).sum()),
            "volumeMl": round(float(volume[k]) / 1000, 3),
            "openEdges": int(open_by_piece[k]),
            "crossingFaces": int(crossed_by_piece[k]),
        }
        for k in np.argsort(-volume, kind="stable")
    ]
    return {
        "triangles": count,
        "pieces": pieces,
        "openEdges": int((uses == 1).sum()),
        "nonManifoldEdges": int((uses > 2).sum()),
        "pinchedVertices": pinched,
        "flippedEdges": int(((uses == 2) & (turn != 0)).sum()),
        "degenerateFaces": int(degenerate.sum()),
        "crossingPairs": len(crossed),
        "crossingBetweenPieces": int(between.sum()),
        "volumeMl": round(float(volume.sum()) / 1000, 3),
    }


def topology_faults(found: dict[str, Any], several: bool) -> list[str]:
    """What is wrong with a node's surface (topology), in words. A bone must
    be one closed piece; a set (several: the costal cartilages) may be many,
    each closed and none under MIN_PIECE_ML."""
    faults = []
    if found["triangles"] == 0:
        return ["no faces"]
    if found["openEdges"]:
        faults.append(f"{found['openEdges']} open edges (holes)")
    if found["nonManifoldEdges"]:
        faults.append(f"{found['nonManifoldEdges']} edges used by more than two faces")
    if found["pinchedVertices"]:
        faults.append(f"{found['pinchedVertices']} vertices where two sheets are pinched together")
    if found["flippedEdges"]:
        faults.append(f"{found['flippedEdges']} edges whose faces are wound against each other")
    if found["degenerateFaces"]:
        faults.append(f"{found['degenerateFaces']} faces without area")
    if found["crossingPairs"]:
        faults.append(f"{found['crossingPairs']} pairs of faces cross")
    inside_out = [p for p in found["pieces"] if p["volumeMl"] <= 0]
    if inside_out:
        faults.append(f"{len(inside_out)} pieces inside out or without volume")
    if several:
        small = [p for p in found["pieces"] if 0 < p["volumeMl"] < MIN_PIECE_ML]
        if small:
            faults.append(
                f"{len(small)} loose pieces under {MIN_PIECE_ML} ml "
                f"(smallest {min(p['volumeMl'] for p in small)} ml)"
            )
    elif len(found["pieces"]) != 1:
        rest = [p["volumeMl"] for p in found["pieces"][1:]]
        faults.append(f"{len(found['pieces'])} pieces, not one (the others: {rest} ml)")
    return faults


# (b) Fidelity to the scan


class Mask(NamedTuple):
    data: Bools  # the mask on its own voxel grid
    affine: Floats  # voxel numbers to RAS millimetres


def read_mask(path: Path) -> Mask:
    image = nib.Nifti1Image.from_filename(path)
    affine = image.affine
    if affine is None:
        raise ValueError(f"{path.name} has no affine")
    return Mask(np.asanyarray(image.dataobj) > 0, np.asarray(affine, dtype=np.float64))


def voxel_ml(mask: Mask) -> float:
    return float(abs(np.linalg.det(mask.affine[:3, :3]))) / 1000


def signed_distance(inside: Bools, spacing: Floats) -> Floats:
    """How far each voxel's centre lies from a mask's boundary, in mm:
    negative inside the mask, positive outside. The boundary is the faces
    between voxels inside and outside, so a centre is measured to the box
    of the nearest voxel of the other kind, not to that voxel's centre: a
    centre next to the boundary is half a voxel from it, on either side."""

    def to_other(here: Bools) -> Floats:
        nearest = ndimage.distance_transform_edt(
            here, sampling=spacing, return_distances=False, return_indices=True
        )
        steps = np.abs(np.asarray(nearest) - np.indices(here.shape))
        reach = np.maximum(steps - 0.5, 0.0) * spacing[:, None, None, None]
        return np.sqrt((reach**2).sum(0))

    if not inside.any() or inside.all():
        raise ValueError("a mask with no boundary has no surface")
    return np.where(inside, -to_other(inside), to_other(~inside))


def smooth_field(mask: Mask) -> tuple[Floats, Floats]:
    """A mask's smooth signed distance, in mm, on its own grid cut down to
    the mask with FIELD_PAD voxels round it, and the voxel number of the
    cut's first sample.

    The signed distance to the voxel faces (signed_distance) is zero on a
    staircase, up to half a voxel off the surface the scan saw. Averaging it
    (a Gaussian of FIELD_SMOOTH_VOXELS voxels) evens the steps out, but an
    average also pulls a curved surface toward its centre of curvature, by
    more than the steps were off on a rib. Twice that average less the
    average over a Gaussian with twice the variance evens the steps out as
    well and has no such pull, to first order (its second moment is zero).
    A smooth shape, voxelised, gets its own surface back as the zero level
    to about a tenth of a voxel; test_atlas_check.py holds it to that."""
    if not mask.data.any():
        raise ValueError("the mask is empty")
    filled = [
        np.flatnonzero(mask.data.any(tuple(a for a in range(3) if a != axis))) for axis in range(3)
    ]
    first = np.array([k[0] for k in filled]) - FIELD_PAD
    last = np.array([k[-1] for k in filled]) + FIELD_PAD + 1
    box = np.zeros(tuple(last - first), dtype=bool)
    lo, hi = np.maximum(first, 0), np.minimum(last, mask.data.shape)
    box[tuple(slice(a - f, b - f) for a, b, f in zip(lo, hi, first, strict=True))] = mask.data[
        tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    ]
    spacing = np.linalg.norm(mask.affine[:3, :3], axis=0)
    distance = signed_distance(box, spacing)
    narrow = ndimage.gaussian_filter(distance, FIELD_SMOOTH_VOXELS, mode="nearest")
    wide = ndimage.gaussian_filter(distance, FIELD_SMOOTH_VOXELS * np.sqrt(2), mode="nearest")
    return 2 * narrow - wide, first.astype(np.float64)


# The six tetrahedra that fill a cube round its diagonal from corner 0,0,0 to
# 1,1,1: each steps along the three axes in one of the six orders.
CUBE_TETRAHEDRA = np.array([
    np.cumsum(np.vstack([np.zeros(3, dtype=np.int64), np.eye(3, dtype=np.int64)[list(order)]]), 0)
    for order in itertools.permutations(range(3))
])  # fmt: skip


def zero_level(field: Floats) -> Floats:
    """The surface where a field sampled on a grid is zero, as loose
    triangles (k x 3 corners x 3) in grid units: marching tetrahedra. Each
    cube of eight samples is cut into six tetrahedra, and in each the
    surface is the triangle (one corner on the other side from the other
    three) or the four-sided patch (two and two) through the points on its
    edges where the field, taken to run straight between two samples, is
    zero. Negative is inside. Neighbouring cubes cut their shared face the
    same way, so the triangles join into a closed surface."""
    below = field < 0
    shape = np.array(field.shape) - 1
    views = [
        below[dx : dx + shape[0], dy : dy + shape[1], dz : dz + shape[2]]
        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
    ]  # fmt: skip
    cubes = np.argwhere(np.any(views, 0) & ~np.all(views, 0))
    out = []
    for tetrahedron in CUBE_TETRAHEDRA:
        at = cubes[:, None, :] + tetrahedron[None, :, :]  # k x 4 corners x 3
        value = field[at[..., 0], at[..., 1], at[..., 2]]
        # The corners inside first, so the patterns are few.
        order = np.argsort(value >= 0, axis=1, kind="stable")
        at = np.take_along_axis(at, order[:, :, None], 1).astype(np.float64)
        value = np.take_along_axis(value, order, 1)
        inside = (value < 0).sum(1)

        def cut(rows: Bools, i: int, j: int, at: Floats = at, value: Floats = value) -> Floats:
            share = value[rows, i] / (value[rows, i] - value[rows, j])
            return at[rows, i] + share[:, None] * (at[rows, j] - at[rows, i])

        one, two, three = inside == 1, inside == 2, inside == 3
        out.append(np.stack([cut(one, 0, 1), cut(one, 0, 2), cut(one, 0, 3)], 1))
        out.append(np.stack([cut(three, 0, 3), cut(three, 1, 3), cut(three, 2, 3)], 1))
        corners = [cut(two, 0, 2), cut(two, 0, 3), cut(two, 1, 3), cut(two, 1, 2)]
        out.append(np.stack(corners[:3], 1))
        out.append(np.stack([corners[0], corners[2], corners[3]], 1))
    return np.vstack(out)


def mask_surface(mask: Mask) -> Mesh:
    """The surface of a mask, in RAS mm: the zero level of its smooth signed
    distance (smooth_field), as loose triangles."""
    field, first = smooth_field(mask)
    corners = (zero_level(field) + first) @ mask.affine[:3, :3].T + mask.affine[:3, 3]
    return Mesh(corners.reshape(-1, 3), np.arange(len(corners) * 3).reshape(-1, 3))


def spread(distances: Floats) -> dict[str, float]:
    """The mean, 95th percentile and largest of some distances, in mm."""
    if len(distances) == 0:
        return {"mean": float("inf"), "p95": float("inf"), "max": float("inf")}
    return {
        "mean": round(float(distances.mean()), 3),
        "p95": round(float(np.percentile(distances, 95)), 3),
        "max": round(float(distances.max()), 3),
    }


def fidelity(mesh: Mesh, mask: Mask) -> dict[str, Any]:
    """How far a mesh lies from its mask's surface (mask_surface), both
    ways: from points spread over the mesh to the mask's surface, and from
    points spread over the mask's surface to the mesh, so a part the mesh
    lacks shows as much as one it adds. It passes when the mean is at most
    MAX_MEAN_MM and the 95th percentile at most MAX_P95_MM, both ways."""
    surface = mask_surface(mask)
    out = spread(Surface(surface).distance(sample_points(mesh)))
    back = spread(Surface(mesh).distance(sample_points(surface)))
    ok = all(d["mean"] <= MAX_MEAN_MM and d["p95"] <= MAX_P95_MM for d in (out, back))
    return {"meshToMask": out, "maskToMesh": back, "pass": ok}


# Between two nodes


class Body:
    """A node of the build readied for measuring against the others: its
    welded mesh in RAS mm, its Surface, and points spread over it."""

    def __init__(self, name: str, mesh: Mesh) -> None:
        self.name = name
        self.label = LABELS.get(name, name)
        self.mesh = welded(mesh)
        self.low, self.high = self.mesh.vertices.min(0), self.mesh.vertices.max(0)
        self._surface: Surface | None = None
        self._points: Floats | None = None
        self._tree: Any = None

    @property
    def surface(self) -> Surface:
        if self._surface is None:
            self._surface = Surface(self.mesh)
        return self._surface

    @property
    def points(self) -> Floats:
        """The mesh's vertices and points spread evenly over it."""
        if self._points is None:
            self._points = np.vstack([self.mesh.vertices, sample_points(self.mesh)])
        return self._points

    @property
    def tree(self) -> Any:
        """A k-d tree of its points, for a first look at what lies near."""
        if self._tree is None:
            self._tree = KDTree(self.points)
        return self._tree

    def near(self, other: Body, reach: float) -> Floats:
        """Its points within reach mm of the other's bounding box."""
        keep = ((self.points >= other.low - reach) & (self.points <= other.high + reach)).all(1)
        return self.points[keep]


def gap(one: Body, other: Body, of_one: Floats | None = None) -> float:
    """The least distance between two bodies' surfaces, in mm, and 0 when
    one enters the other. of_one narrows the first body to some of its
    points (a rib's end). The points of each that lie nearest the other's
    points (within two sample spacings of the nearest of all) are measured
    exactly to the other's surface, so the gap is right to a small part of
    SAMPLE_MM."""
    mine = one.points if of_one is None else of_one
    if len(mine) == 0 or len(other.points) == 0:
        return float("inf")
    rough = other.tree.query(mine)[0]
    mine = mine[rough <= rough.min() + 2 * SAMPLE_MM]
    reach = float(rough.min()) + 2 * SAMPLE_MM
    theirs = other.points[KDTree(mine).query(other.points, distance_upper_bound=reach)[0] <= reach]
    if other.surface.inside(mine).any() or one.surface.inside(theirs).any():
        return 0.0
    least = float(other.surface.distance(mine).min())
    if len(theirs) and of_one is None:
        least = min(least, float(one.surface.distance(theirs).min()))
    return least


def entered(one: Body, other: Body) -> dict[str, float] | None:
    """How far two bodies lie inside each other, or None when they do not:
    the deepest any point of one lies under the other's surface, each way
    in mm, and the volume they share in ml, counted on a grid of
    OVERLAP_GRID_MM round the points found inside. Two bodies are looked at
    closely only when their surfaces come within ENTER_PROBE_MM somewhere
    and a point of one there lies in the other, or one lies wholly in the
    other."""
    if (one.low > other.high).any() or (other.low > one.high).any():
        return None
    mine, theirs = one.near(other, 0.0), other.near(one, 0.0)
    if len(mine) + len(theirs) == 0:
        return None
    close = mine[other.tree.query(mine, distance_upper_bound=ENTER_PROBE_MM)[0] <= ENTER_PROBE_MM]
    back = theirs[one.tree.query(theirs, distance_upper_bound=ENTER_PROBE_MM)[0] <= ENTER_PROBE_MM]
    if not (
        other.surface.inside(np.vstack([close, mine[:1]])).any()
        or one.surface.inside(np.vstack([back, theirs[:1]])).any()
    ):
        return None
    mine, theirs = mine[other.surface.inside(mine)], theirs[one.surface.inside(theirs)]
    both = np.vstack([mine, theirs])
    low, high = both.min(0) - 1.0, both.max(0) + 1.0
    cells = float(np.prod((high - low) / OVERLAP_GRID_MM))
    step = OVERLAP_GRID_MM * max(1.0, cells / OVERLAP_MAX_CELLS) ** (1 / 3)
    axes = [np.arange(a + step / 2, b, step) for a, b in zip(low, high, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    shared = one.surface.inside(grid)
    shared[shared] = other.surface.inside(grid[shared])
    return {
        "firstInSecondMm": round(float(other.surface.distance(mine).max()), 2)
        if len(mine)
        else 0.0,
        "secondInFirstMm": (
            round(float(one.surface.distance(theirs).max()), 2) if len(theirs) else 0.0
        ),
        "sharedMl": round(float(shared.sum()) * step**3 / 1000, 3),
    }


# (d) Anatomy


def rib_label(side: str, number: int) -> str:
    return f"rib_{side}_{number}"


def rib_number(rib: str) -> int:
    return int(rib.rsplit("_", 1)[1])


def rib_tips(body: Body) -> tuple[Floats, Floats]:
    """A rib's two ends, vertebral and front, as points on its surface: one
    is the vertex furthest from the rib's middle, the other the vertex
    furthest from that one along the surface; the front end is the one
    further forward (RAS +A), as in mesh.py's rib_end_of."""
    vertices, triangles = body.mesh.vertices, body.mesh.triangles
    start, end = triangles.T.reshape(-1), triangles[:, [1, 2, 0]].T.reshape(-1)
    length = np.linalg.norm(vertices[start] - vertices[end], axis=1)
    graph = coo_matrix((length, (start, end)), shape=(len(vertices),) * 2).tocsr()
    used = np.unique(triangles)
    one = used[np.argmax(np.linalg.norm(vertices[used] - vertices[used].mean(0), axis=1))]
    along = dijkstra(graph, directed=False, indices=int(one))
    other = used[np.argmax(np.where(np.isfinite(along[used]), along[used], -1.0))]
    back, front = sorted((vertices[one], vertices[other]), key=lambda p: p[1])
    return back, front


def rib_joints(
    bodies: dict[str, Body], dropped: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Each rib's gap to the vertebrae it articulates with, and what fails.

    Rib n articulates with thoracic vertebra n, and ribs 2 to 9 with
    vertebra n - 1 as well (the head sits on two bodies there). A gap is
    the least distance between the rib's vertebral half and the vertebra
    (gap); it passes at RIB_JOINT_MM or less. A vertebra the manifest lists
    as dropped is "not in the build" and fails no rib. The vertebra nearest
    the rib is given too: it shows a segmentation that numbers ribs and
    vertebrae one apart."""
    rows, failures = [], []
    vertebrae = {label: body for label, body in bodies.items() if label.startswith("vertebrae_T")}
    for side in ("left", "right"):
        for number in range(1, 13):
            rib = bodies.get(rib_label(side, number))
            if rib is None:
                continue
            back, front = rib_tips(rib)
            nearer_back = np.linalg.norm(rib.points - back, axis=1) < np.linalg.norm(
                rib.points - front, axis=1
            )
            half = rib.points[nearer_back]
            row: dict[str, Any] = {"rib": rib.label, "node": rib.name, "gaps": {}}
            for n in [number, number - 1] if 2 <= number <= 9 else [number]:
                label = f"vertebrae_T{n}"
                if label in vertebrae:
                    found = round(gap(rib, vertebrae[label], half), 2)
                    row["gaps"][f"T{n}"] = found
                    if found > RIB_JOINT_MM:
                        failures.append(f"{rib.label} is {found} mm from T{n}")
                elif label in dropped:
                    row["gaps"][f"T{n}"] = "vertebra not in the build"
                else:
                    row["gaps"][f"T{n}"] = "vertebra missing"
                    failures.append(f"{rib.label}: T{n} is neither built nor listed as dropped")
            # The nearest vertebra: tried in the order of their boxes' distance
            # from the rib's, until no box left is nearer than the nearest found.
            low, high = half.min(0), half.max(0)
            apart = {
                label: float(np.linalg.norm(np.maximum(np.maximum(low - v.high, v.low - high), 0)))
                for label, v in vertebrae.items()
            }
            nearest, least = None, float("inf")
            for label in sorted(apart, key=lambda label: apart[label]):
                if apart[label] >= least:
                    break
                found = gap(rib, vertebrae[label], half)
                if found < least:
                    nearest, least = label, found
            if nearest:
                row["nearest"] = nearest.removeprefix("vertebrae_")
                row["nearestMm"] = round(least, 2)
            row["pass"] = not any(f.startswith(rib.label + " ") or f.startswith(rib.label + ":")
                                  for f in failures)  # fmt: skip
            rows.append(row)
    return rows, failures


def pieces_of(mesh: Mesh) -> Ints:
    """The number of the connected piece each triangle of a welded mesh belongs to."""
    start, end = mesh.triangles.T.reshape(-1), mesh.triangles[:, [1, 2, 0]].T.reshape(-1)
    points = len(mesh.vertices)
    joined = coo_matrix((np.ones(len(start)), (start, end)), shape=(points, points))
    return connected_components(joined, directed=False)[1][mesh.triangles[:, 0]]


def touching(points: Floats, body: Body, reach: float) -> tuple[Bools, Ints]:
    """Which points lie in a body or within reach mm of its surface, and the
    body's triangle nearest each point."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    distance, triangle = body.surface.nearest(points)
    return (distance <= reach) | body.surface.inside(points), triangle


def contact_patches(sternum: Body, cartilage: Body, piece: Ints) -> list[dict[str, Any]]:
    """The patches of the sternum's surface that the cartilage covers, from
    top to bottom. Points PATCH_SAMPLE_MM apart on the sternum that lie in
    the cartilage, or within CONTACT_MM of it, are in contact; points in
    contact within PATCH_LINK_MM of each other are one patch, so two patches
    are apart when that much bare bone lies between them, and a patch under
    PATCH_MIN_MM2 is a speck that does not count. Each patch has its side
    of the sternum's midline, its height (RAS z), its middle, its area and
    the cartilage pieces that make it."""
    points = sample_points(sternum.mesh, PATCH_SAMPLE_MM)
    near = points[((points >= cartilage.low - 1) & (points <= cartilage.high + 1)).all(1)]
    touch, triangle = touching(near, cartilage, CONTACT_MM)
    near, triangle = near[touch], triangle[touch]
    if len(near) == 0:
        return []
    pairs = KDTree(near).query_pairs(PATCH_LINK_MM, output_type="ndarray")
    linked = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(near),) * 2)
    patch = connected_components(linked, directed=False)[1]
    midline = float(np.median(sternum.mesh.vertices[:, 0]))
    patches = []
    for k in np.unique(patch):
        mine = near[patch == k]
        area = len(mine) * PATCH_SAMPLE_MM**2
        if area < PATCH_MIN_MM2:
            continue
        patches.append({
            # RAS +x is the body's right.
            "side": "left" if float(mine[:, 0].mean()) < midline else "right",
            "zLow": round(float(mine[:, 2].min()), 1),
            "zHigh": round(float(mine[:, 2].max()), 1),
            "centre": [round(float(c), 1) for c in mine.mean(0)],
            "areaMm2": round(area, 1),
            "pieces": sorted(int(p) for p in np.unique(piece[triangle[patch == k]])),
        })  # fmt: skip
    return sorted(patches, key=lambda p: -p["centre"][2])


def arch_laps(
    cartilage: Body,
    piece: Ints,
    side: str,
    ribs: dict[int, int | None],
    tips: dict[str, Floats],
    bands: list[dict[str, Any]] | None,
    seventh: Floats | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """How cartilages 8, 9 and 10 of one side lie on the one above, and
    what fails. ribs gives the cartilage piece at each rib's front end,
    tips the ribs' front tips, seventh the middle of the seventh
    cartilage's contact with the sternum.

    A cartilage in a piece of its own is "lapped" when points spread over
    it lie in the piece above, or within LAP_GAP_MM of it, along at least
    LAP_MM (the longest extent of those points); otherwise it is "apart".
    Its row gives the least gap, the lap's length and how far its tip ends
    from the sternal end of the cartilage above. The tip and that end are
    the manifest's band ends (end_ras_mm) where it has them; otherwise the
    tip is the point of the lap furthest from the cartilage's own rib, and
    the end above is the seventh's contact, or the tip of the one above.
    Two cartilages in one piece are "fused": nothing can be measured
    between them, and the row says which they are."""
    rows, failures = [], []
    ends = {int(b["number"]): np.array(b["end_ras_mm"], dtype=np.float64)
            for b in bands or [] if b["side"] == side}  # fmt: skip
    if 7 not in ends and seventh is not None:
        ends[7] = seventh
    parts: dict[int, tuple[Surface, Floats]] = {}

    def part(k: int) -> tuple[Surface, Floats]:
        """One piece of the cartilage: its surface, and points spread over it."""
        if k not in parts:
            own = Mesh(cartilage.mesh.vertices, cartilage.mesh.triangles[piece == k])
            parts[k] = (Surface(own), sample_points(own))
        return parts[k]

    for n in (8, 9, 10):
        if rib_label(side, n) not in tips:
            continue
        row: dict[str, Any] = {"side": side, "cartilage": n, "gapMm": None, "lapMm": None,
                               "tipToEndAboveMm": None}  # fmt: skip
        rows.append(row)
        mine, above = ribs.get(n), ribs.get(n - 1)
        if mine is None or above is None:
            row["state"] = "no cartilage"
            failures.append(f"{side}: cartilage {n if mine is None else n - 1} is missing, "
                            f"so {n} cannot lap onto {n - 1}")  # fmt: skip
            continue
        if mine == above:
            row["state"] = "fused"
            row["with"] = [m for m in range(1, 11) if ribs.get(m) == mine]
            if FUSED_FAILS:
                failures.append(f"{side}: cartilage {n} is fused with cartilage {n - 1} "
                                f"(one piece: cartilages {row['with']})")  # fmt: skip
            continue
        surface, _ = part(above)
        points = part(mine)[1]
        distance = surface.distance(points)
        distance[surface.inside(points)] = 0.0
        row["gapMm"] = round(float(distance.min()), 2)
        lap = points[distance <= LAP_GAP_MM]
        row["lapMm"] = 0.0
        if len(lap) >= 3:
            along = np.linalg.svd(lap - lap.mean(0), full_matrices=False)[2][0]
            row["lapMm"] = round(float(np.ptp(lap @ along)), 1)
            if n not in ends:
                tip = tips[rib_label(side, n)]
                ends[n] = lap[np.argmax(np.linalg.norm(lap - tip, axis=1))]
        if n in ends and n - 1 in ends:
            row["tipToEndAboveMm"] = round(float(np.linalg.norm(ends[n] - ends[n - 1])), 1)
        row["state"] = "lapped" if row["lapMm"] >= LAP_MM else "apart"
        if row["state"] == "apart":
            failures.append(
                f"{side}: cartilage {n} lies against cartilage {n - 1} over {row['lapMm']} mm "
                f"(least gap {row['gapMm']} mm), under {LAP_MM}"
            )
    return rows, failures


def cartilages(
    bodies: dict[str, Body], bands: list[dict[str, Any]] | None
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The costal cartilages against the ribs and the sternum, and what
    fails, by rule.

    - "cartilage to rib": the cartilage reaches the front end of each of
      ribs 1 to 10 (the rib's surface within RIB_END_MM of its front tip),
      within COSTOCHONDRAL_MM.
    - "floating ribs": no cartilage within FLOAT_CLEAR_MM of the front end
      of ribs 11 and 12.
    - "sternum contacts": on each side the sternum carries seven separate
      contact patches (contact_patches), one for each of cartilages 1 to 7
      and in their order from the top. A patch is a cartilage's when it is
      made by the cartilage piece that reaches that rib; where one piece
      reaches several of ribs 1 to 7 it must make as many patches, taken in
      order. A patch made by two pieces is shared and is nobody's.
    - "costal arch": cartilages 8, 9 and 10 each lap onto the cartilage
      above (arch_laps): a piece of its own, within LAP_GAP_MM of the piece
      above over at least LAP_MM of length. Two of them in one piece are
      reported as fused, and fail when FUSED_FAILS says so. None reaches the
      sternum itself (an eighth patch would fail the rule above).
    With the manifest's band records (mesh.py writes one per band: rib,
    what it joins, where it ends) the declared ends are held to the same:
    bands 1 to 7 join the sternum and end in or on it, bands 8 to 10 join
    the band above and end in the cartilage or on its surface (within
    LAP_GAP_MM), ARCH_CLEAR_MM or more from the sternum.
    Cartilages of ribs 1 to 4 that are of one piece with another are listed
    under "joined" (no interchondral joint lies there), and fail "sternum
    contacts" only when JOINED_FAILS says so."""
    failures: dict[str, list[str]] = {
        "cartilage to rib": [], "floating ribs": [], "sternum contacts": [], "costal arch": [],
    }  # fmt: skip
    report: dict[str, Any] = {"ribs": [], "patches": [], "joined": [], "arch": []}
    cartilage, sternum = bodies.get("costal_cartilages"), bodies.get("sternum")
    if cartilage is None:
        for rule in ("cartilage to rib", "sternum contacts", "costal arch"):
            failures[rule].append("no costal cartilages in the build")
        return report, failures
    piece = pieces_of(cartilage.mesh)
    reached: dict[str, int | None] = {}  # the cartilage piece at each rib's front end
    tips: dict[str, Floats] = {}  # each rib's front tip
    for side in ("left", "right"):
        for number in range(1, 13):
            rib = bodies.get(rib_label(side, number))
            if rib is None:
                continue
            _, front = rib_tips(rib)
            tips[rib.label] = front
            end = rib.points[np.linalg.norm(rib.points - front, axis=1) <= RIB_END_MM]
            found = gap(rib, cartilage, end)
            row = {"rib": rib.label, "gapMm": round(found, 2), "piece": None}
            if number <= 10:
                touch, triangle = touching(end, cartilage, COSTOCHONDRAL_MM)
                if touch.any():
                    mine = np.bincount(piece[triangle[touch]])
                    reached[rib.label] = row["piece"] = int(mine.argmax())
                else:
                    reached[rib.label] = None
                    failures["cartilage to rib"].append(
                        f"{rib.label}: the cartilage is {found:.2f} mm from its front end"
                    )
            elif found < FLOAT_CLEAR_MM:
                failures["floating ribs"].append(
                    f"{rib.label}: cartilage {found:.2f} mm from its front end"
                )
            report["ribs"].append(row)
    if sternum is None:
        failures["sternum contacts"].append("no sternum in the build")
        return report, failures
    patches = contact_patches(sternum, cartilage, piece)
    report["patches"] = patches
    for side in ("left", "right"):
        mine = [p for p in patches if p["side"] == side]
        ribs = {n: reached.get(rib_label(side, n)) for n in range(1, 11)}
        for p in mine:
            if len(p["pieces"]) > 1:
                failures["sternum contacts"].append(
                    f"{side}: the patch at z {p['centre'][2]} is shared by "
                    f"{len(p['pieces'])} cartilage pieces"
                )
        for k in sorted({k for k in ribs.values() if k is not None}):
            owners = [n for n in range(1, 8) if ribs[n] == k]
            made = [p for p in mine if p["pieces"] == [k]]
            if len(owners) == len(made):
                for n, p in zip(owners, made, strict=True):
                    p["rib"] = n
            elif owners:
                failures["sternum contacts"].append(
                    f"{side}: the piece of cartilages {owners} makes {len(made)} patches, "
                    f"not {len(owners)}"
                )
            lone = [n for n in range(1, 11) if ribs[n] == k and n in LONE_CARTILAGES]
            if lone and sum(1 for n in ribs if ribs[n] == k) > 1:
                group = [n for n in range(1, 11) if ribs[n] == k]
                report["joined"].append({"side": side, "cartilages": group})
                if JOINED_FAILS:
                    failures["sternum contacts"].append(f"{side}: cartilages {group} are one piece")
        order = [p.get("rib") for p in mine]
        if len(mine) != 7:
            failures["sternum contacts"].append(f"{side}: {len(mine)} contact patches, not 7")
        missing = [n for n in range(1, 8) if n not in order]
        if missing:
            failures["sternum contacts"].append(
                f"{side}: no patch of its own for cartilage {missing}"
            )
        placed = [n for n in order if n is not None]
        if placed != sorted(placed):
            failures["sternum contacts"].append(f"{side}: the patches run {order} from the top")
        seventh = [p["centre"] for p in mine if p.get("rib") == 7]
        rows, failed = arch_laps(
            cartilage, piece, side, ribs, tips, bands, np.array(seventh[0]) if seventh else None
        )
        report["arch"] += rows
        failures["costal arch"] += failed
    for band in bands or []:
        number, rib = int(band["number"]), str(band["rib"])
        end = np.array([band["end_ras_mm"]], dtype=np.float64)
        expected = "sternum" if number <= 7 else rib.replace(f"_{number}", f"_{number - 1}")
        rule = "sternum contacts" if number <= 7 else "costal arch"
        if band["joins"] != expected:
            failures[rule].append(f"{rib}: the manifest says it joins {band['joins']}")
        off = float(sternum.surface.distance(end)[0])
        if number <= 7 and off > CONTACT_MM and not sternum.surface.inside(end)[0]:
            failures[rule].append(f"{rib}: its band ends {off:.1f} mm from the sternum")
        if number > 7 and (off < ARCH_CLEAR_MM or sternum.surface.inside(end)[0]):
            failures[rule].append(f"{rib}: its band ends {off:.1f} mm from the sternum")
        # A band's centreline ends on its own rounded tip, so on the surface.
        away = float(cartilage.surface.distance(end)[0])
        if number > 7 and away > LAP_GAP_MM and not cartilage.surface.inside(end)[0]:
            failures[rule].append(f"{rib}: its band ends {away:.1f} mm out of the cartilage")
    report["bands"] = bands
    return report, failures


def cut_at(mesh: Mesh, x: float) -> Floats:
    """Points along the outline a mesh makes in the sagittal plane at RAS x:
    seven along each stretch a triangle cuts out of the plane."""
    off = mesh.vertices[:, 0] - x
    below = off[mesh.triangles] < 0
    cut = mesh.triangles[below.any(1) & ~below.all(1)]
    ends = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = off[cut[:, i]], off[cut[:, j]]
        share = a / np.where(a != b, a - b, 1.0)
        point = mesh.vertices[cut[:, i]] + share[:, None] * (
            mesh.vertices[cut[:, j]] - mesh.vertices[cut[:, i]]
        )
        ends.append(np.where(((a < 0) != (b < 0))[:, None], point, np.nan))
    ends = np.stack(ends, 1)  # per triangle: three edges, two of them cut
    order = np.argsort(np.isnan(ends[:, :, 0]), axis=1, kind="stable")
    ends = np.take_along_axis(ends, order[:, :, None], 1)[:, :2]
    share = np.linspace(0, 1, 7)[None, :, None]
    return (ends[:, :1] * (1 - share) + ends[:, 1:] * share).reshape(-1, 3)


def dip(height: Floats, across: Floats) -> dict[str, float]:
    """The deepest dip of a profile (height over across) between two higher
    places: how far it lies under the straight line joining the highest
    points on either side of it (the corners), and how far apart those are.
    0 when the profile nowhere dips between two higher places."""
    known = np.isfinite(height)
    height, across = height[known], across[known]
    if len(height) < 3:
        return {"depthMm": 0.0, "spanMm": 0.0}
    walls = np.minimum(np.maximum.accumulate(height), np.maximum.accumulate(height[::-1])[::-1])
    k = int(np.argmax(walls - height))
    if walls[k] - height[k] <= 0:
        return {"depthMm": 0.0, "spanMm": 0.0}
    a, b = int(np.argmax(height[:k])), k + int(np.argmax(height[k:]))
    chord = height[a] + (height[b] - height[a]) * (across[k] - across[a]) / (across[b] - across[a])
    return {
        "depthMm": round(float(chord - height[k]), 2),
        "spanMm": round(float(abs(across[b] - across[a])), 1),
    }


def top_profile(points: Floats, across: Floats, up: Floats) -> tuple[Floats, Floats]:
    """The upper outline of some points seen along the third direction: the
    highest point (along up) in every NOTCH_BIN_MM of width (along across),
    out to NOTCH_HALF_SPAN_MM either side."""
    u, v = points @ across, points @ up
    column = np.floor((u + NOTCH_HALF_SPAN_MM) / NOTCH_BIN_MM).astype(np.int64)
    count = int(2 * NOTCH_HALF_SPAN_MM / NOTCH_BIN_MM)
    keep = (column >= 0) & (column < count)
    height = np.full(count, -np.inf, dtype=np.float64)
    np.maximum.at(height, column[keep], v[keep])
    height[~np.isfinite(height)] = np.nan
    middles = (np.arange(count, dtype=np.float64) + 0.5) * NOTCH_BIN_MM - NOTCH_HALF_SPAN_MM
    return height, middles


def sternum_shape(
    sternum: Body, second: list[float], lowest: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The sternum's jugular notch, sternal angle and xiphoid process,
    measured on its mesh, and what fails, by rule. second is the height
    (RAS z) of the second cartilages' contact patches, lowest the lowest
    contact patch of each side (the seventh cartilages', when all is well).

    Sternal angle: the sternum's outline is cut in five sagittal planes (the
    midline and PLANES_MM either side) and read every PROFILE_BIN_MM of
    height as a front and a back surface. The bend is where a line of two
    straight parts fits the mid-thickness line best, looked for between
    BEND_BELOW_TOP_MM under the top. The angle is between straight lines
    fitted to the front surface above the bend (from ANGLE_CLEAR_MM over it
    to TOP_CLEAR_MM under the top) and below it (ANGLE_BODY_MM), the mean
    of the five planes; the same on the mid-thickness line is given too. It
    passes within STERNAL_ANGLE_DEG, with the bend within ANGLE_AT_SECOND_MM
    of the second cartilages' contact.

    Jugular notch: the deepest dip (dip) of the manubrium's upper outline,
    seen square to the manubrium's own plane ("en face": the manubrium
    leans back, and a front view hides a notch behind its front rim); the
    front view's is given too. It passes at JUGULAR_NOTCH_MM or more.

    Xiphoid process: from the middle of the lowest contact patches straight
    to the sternum's lowest point; it passes at XIPHOID_MM or more when the
    sternum below those patches is narrower (median left-right width every
    WIDTH_BIN_MM) than the body between the second cartilages and them."""
    failures: dict[str, list[str]] = {"jugular notch": [], "sternal angle": [], "xiphoid": []}
    vertices = sternum.mesh.vertices
    top, bottom = float(vertices[:, 2].max()), float(vertices[:, 2].min())
    upper = sternum.points[sternum.points[:, 2] > top - MANUBRIUM_MM]
    midline = float(np.median(upper[:, 0]))
    report: dict[str, Any] = {"topZ": round(top, 1), "bottomZ": round(bottom, 1),
                              "midlineX": round(midline, 1)}  # fmt: skip

    # The sagittal outline, read in bins of height from the top down.
    count = int((top - bottom) / PROFILE_BIN_MM)
    height = top - (np.arange(count) + 0.5) * PROFILE_BIN_MM
    front = np.full((len(PLANES_MM), count), np.nan)
    back = np.full((len(PLANES_MM), count), np.nan)
    for k, offset in enumerate(PLANES_MM):
        outline = cut_at(sternum.mesh, midline + offset)
        row = np.floor((top - outline[:, 2]) / PROFILE_BIN_MM).astype(np.int64)
        keep = (row >= 0) & (row < count)
        high, low = np.full(count, -np.inf), np.full(count, np.inf)
        np.maximum.at(high, row[keep], outline[keep, 1])
        np.minimum.at(low, row[keep], outline[keep, 1])
        front[k] = np.where(np.isfinite(high), high, np.nan)
        back[k] = np.where(np.isfinite(low), low, np.nan)
    middle = (front + back) / 2

    def slope(line: Floats, low: float, high: float) -> float:
        """The slope (mm forward per mm up) of a straight line fitted to a
        profile between two heights; nan with fewer than four samples."""
        use = (height >= low) & (height <= high) & np.isfinite(line)
        return float(np.polyfit(height[use], line[use], 1)[0]) if use.sum() >= 4 else float("nan")

    # The bend: where a line of two straight parts fits the mid-thickness line best.
    known = np.isfinite(middle)
    mean_middle = np.where(
        known.any(0), np.where(known, middle, 0.0).sum(0) / np.maximum(known.sum(0), 1), np.nan
    )
    use = (height <= top - TOP_CLEAR_MM) & (height >= top - BEND_FIT_MM) & np.isfinite(mean_middle)
    bend, best = float("nan"), np.inf
    if use.sum() >= 8:
        for candidate in np.arange(top - BEND_BELOW_TOP_MM[1], top - BEND_BELOW_TOP_MM[0], 0.5):
            basis = np.stack([np.ones(use.sum()), height[use],
                              np.maximum(height[use] - candidate, 0.0)], 1)  # fmt: skip
            misfit = np.linalg.lstsq(basis, mean_middle[use], rcond=None)[1]
            if len(misfit) and misfit[0] < best:
                bend, best = float(candidate), float(misfit[0])

    def angles(about: float) -> dict[str, float | None]:
        """The bend about a height, on the front surface and on the
        mid-thickness line: the mean of the planes' angles in degrees."""
        out: dict[str, float | None] = {}
        for name, lines in (("frontSurfaceDeg", front), ("midLineDeg", middle)):
            found = [
                np.degrees(
                    np.arctan(slope(line, about - ANGLE_BODY_MM, about - ANGLE_CLEAR_MM))
                    - np.arctan(slope(line, about + ANGLE_CLEAR_MM, top - TOP_CLEAR_MM))
                )
                for line in lines
            ]
            known = np.isfinite(found)
            out[name] = round(float(np.mean(np.array(found)[known])), 1) if known.any() else None
        return out

    joint = float(np.mean(second)) if second else bend
    angle: dict[str, Any] = {
        "about": "the second cartilages' contact" if second else "the bend",
        **(angles(joint) if np.isfinite(joint) else {"frontSurfaceDeg": None, "midLineDeg": None}),
        "bendZ": round(bend, 1) if np.isfinite(bend) else None,
        "aboutBend": angles(bend) if np.isfinite(bend) else None,
        "bendAboveSecondCartilageMm": None,
    }
    if angle["frontSurfaceDeg"] is None:
        failures["sternal angle"].append("the sagittal outline could not be read")
    elif not STERNAL_ANGLE_DEG[0] <= angle["frontSurfaceDeg"] <= STERNAL_ANGLE_DEG[1]:
        failures["sternal angle"].append(
            f"the front surface bends {angle['frontSurfaceDeg']} deg, outside "
            f"{STERNAL_ANGLE_DEG[0]} to {STERNAL_ANGLE_DEG[1]}"
        )
    if not second:
        failures["sternal angle"].append("no second cartilage contact to place the bend by")
    elif np.isfinite(bend):
        angle["bendAboveSecondCartilageMm"] = round(bend - joint, 1)
        if abs(bend - joint) > ANGLE_AT_SECOND_MM:
            failures["sternal angle"].append(
                f"the bend lies {bend - joint:+.1f} mm from the second cartilages' contact"
            )
    report["sternalAngle"] = angle

    # The jugular notch, on the manubrium: what lies above the bend.
    floor = bend if np.isfinite(bend) else top - MANUBRIUM_MM
    manubrium = Mesh(vertices, sternum.mesh.triangles[
        (vertices[sternum.mesh.triangles][:, :, 2] > floor).all(1)
    ])  # fmt: skip
    cloud = np.vstack([vertices[vertices[:, 2] > floor], sample_points(manubrium, NOTCH_SAMPLE_MM)])
    cloud = cloud - np.array([midline, 0.0, 0.0])
    # Its long axis in the sagittal plane, pointing up.
    spread_yz = np.linalg.svd(cloud[:, 1:] - cloud[:, 1:].mean(0), full_matrices=False)[2][0]
    spread_yz = spread_yz if spread_yz[1] > 0 else -spread_yz
    axis = np.array([0.0, spread_yz[0], spread_yz[1]])
    sideways = np.array([1.0, 0.0, 0.0])
    notch = {
        "enFace": dip(*top_profile(cloud, sideways, axis)),
        "frontView": dip(*top_profile(cloud, sideways, np.array([0.0, 0.0, 1.0]))),
        "leanDeg": round(float(np.degrees(np.arctan2(abs(axis[1]), axis[2]))), 1),
    }
    if notch["enFace"]["depthMm"] < JUGULAR_NOTCH_MM:
        failures["jugular notch"].append(
            f"{notch['enFace']['depthMm']} mm deep en face, under {JUGULAR_NOTCH_MM}"
        )
    report["jugularNotch"] = notch

    # The xiphoid process, below the lowest cartilage contacts.
    xiphoid: dict[str, Any] = {"below": [p.get("rib") for p in lowest]}
    if not lowest:
        failures["xiphoid"].append("no cartilage contact to measure below")
    else:
        if any(p.get("rib") != 7 for p in lowest) or len(lowest) != 2:
            failures["xiphoid"].append(
                "the lowest contacts are not the two seventh cartilages': "
                f"{[(p['side'], p.get('rib')) for p in lowest]}"
            )
        seat = np.mean([p["centre"] for p in lowest], 0)
        tip = vertices[np.argmin(vertices[:, 2])]
        xiphoid["lengthMm"] = round(float(np.linalg.norm(tip - seat)), 1)
        under = min(p["zLow"] for p in lowest)

        def width(low: float, high: float) -> float | None:
            """The median left-right width of the sternum between two heights."""
            part = sternum.points[(sternum.points[:, 2] >= low) & (sternum.points[:, 2] < high)]
            if len(part) == 0:
                return None
            level = np.floor((part[:, 2] - low) / WIDTH_BIN_MM).astype(np.int64)
            wide = [float(np.ptp(part[level == k, 0])) for k in np.unique(level)]
            return round(float(np.median(wide)), 1)

        xiphoid["widthMm"] = width(bottom, under)
        body_top = float(np.mean(second)) if second else top - MANUBRIUM_MM
        xiphoid["bodyWidthMm"] = width(float(seat[2]), body_top)
        if xiphoid["lengthMm"] < XIPHOID_MM:
            failures["xiphoid"].append(
                f"{xiphoid['lengthMm']} mm below the lowest contacts, under {XIPHOID_MM}"
            )
        if xiphoid["widthMm"] is None or xiphoid["bodyWidthMm"] is None:
            failures["xiphoid"].append("no sternum below the lowest contacts to measure")
        elif xiphoid["widthMm"] >= xiphoid["bodyWidthMm"]:
            failures["xiphoid"].append(
                f"{xiphoid['widthMm']} mm wide, not narrower than the body's "
                f"{xiphoid['bodyWidthMm']} mm"
            )
    report["xiphoid"] = xiphoid
    return report, failures


# The check


def declared_change(
    node: dict[str, Any], label: str, fragments: list[dict[str, Any]], kept_ml: float
) -> tuple[float, float]:
    """The volume, in ml, that the manifest says mesh.py added to a node's
    mask and took off it before contouring it (kept_ml of mask). Added: a
    rib's end filled in, a gap bridged, a rib's head grown from the CT
    (DECLARED_ADDED), pieces
    moved here from a neighbour (fragments with "to"). Taken off: knobs and
    end lumps trimmed, its own pieces moved away or dropped (fragments),
    and the share outside its largest piece (dropped_share)."""
    added = sum(float(node.get(key, 0.0)) for key in DECLARED_ADDED)
    removed = sum(float(node.get(key, 0.0)) for key in DECLARED_REMOVED)
    share = float(node.get("dropped_share", 0.0))
    if 0 < share < 1:
        removed += kept_ml * share / (1 - share)
    for note in fragments:
        if note["label"] == label:
            removed += float(note["ml"])
        if note.get("to") == label and note.get("exported") is not False:
            added += float(note["ml"])
    return added, removed


def check_fidelity(
    body: Body, node: dict[str, Any], manifest: dict[str, Any], manifest_dir: Path, seg_dir: Path
) -> dict[str, Any]:
    """One node against the scan: not measured when the manifest calls it
    modelled; otherwise its distances to the mask mesh.py contoured, when
    the manifest names one ("contoured_mask", written by mesh.py
    --contoured), with the volume that mask gained and lost against the
    scan's own mask held to what the manifest declares (declared_change,
    within DECLARED_SLACK_ML); or, without one, its distances to the scan's
    own mask. A node the manifest calls mirrored is measured the same way
    but is not held to its own side's mask."""
    if node.get("modelled"):
        return {"against": "modelled", "pass": True}
    raw_path = seg_dir / f"{body.label}.nii.gz"
    if not raw_path.exists():
        return {"against": "nothing", "pass": False, "why": f"{raw_path.name} is missing"}
    raw = read_mask(raw_path)
    if "contoured_mask" not in node:
        try:
            return {"against": "raw mask", **fidelity(body.mesh, raw)}
        except ValueError as error:
            return {"against": "raw mask", "pass": False, "why": str(error)}
    path = manifest_dir / str(node["contoured_mask"])
    if not path.exists():
        return {"against": "nothing", "pass": False, "why": f"{node['contoured_mask']} is missing"}
    contoured = read_mask(path)
    if contoured.data.shape != raw.data.shape or not np.allclose(contoured.affine, raw.affine):
        return {"against": "nothing", "pass": False, "why": f"{path.name} is on another grid"}
    out = {"against": "contoured mask", **fidelity(body.mesh, contoured)}
    one = voxel_ml(raw)
    gained = float((contoured.data & ~raw.data).sum()) * one
    lost = float((raw.data & ~contoured.data).sum()) * one
    may_gain, may_lose = declared_change(
        node, body.label, manifest.get("fragments", []), float(contoured.data.sum()) * one
    )
    out["change"] = {
        "addedMl": round(gained, 3), "removedMl": round(lost, 3),
        "declaredAddedMl": round(may_gain, 3), "declaredRemovedMl": round(may_lose, 3),
        "mirrored": bool(node.get("mirrored")),
    }  # fmt: skip
    out["change"]["pass"] = bool(node.get("mirrored")) or (
        gained <= may_gain + DECLARED_SLACK_ML and lost <= may_lose + DECLARED_SLACK_ML
    )
    return out


def check(glb: Path, manifest_path: Path, seg_dir: Path) -> dict[str, Any]:
    """Measures one built skeleton against parts (b), (c) and (d) of the
    bar and returns the report: per node its fidelity and topology, the
    nodes that enter each other, the ribs' joints, the cartilages' contacts,
    the sternum's shape, and every rule with whether it passed."""
    manifest = json.loads(manifest_path.read_text())
    centre = np.array(manifest["centre_offset_m"], dtype=np.float64)
    entries = {node["name"]: node for node in manifest["nodes"]}
    dropped = {d["label"]: d["reason"] for d in manifest.get("dropped", [])}
    meshes = read_glb(glb)
    rules: list[dict[str, Any]] = []

    def rule(name: str, failures: list[str] | None, passed: str = "") -> None:
        """One rule's outcome: failed with its failures, passed, or (None) not checked."""
        ok = None if failures is None else not failures
        rules.append({"rule": name, "pass": ok, "detail": "; ".join(failures or []) or passed})

    unknown = sorted(set(meshes) - set(entries)) + sorted(set(meshes) - set(LABELS))
    absent = sorted(set(entries) - set(meshes))
    rule(
        "nodes",
        [f"{name} is not in the manifest or not a skeleton structure" for name in unknown]
        + [f"{name} is in the manifest but not in the GLB" for name in absent],
        f"{len(meshes)} nodes, as the manifest lists",
    )
    bodies = {
        LABELS[name]: Body(name, Mesh(to_ras(mesh.vertices, centre), mesh.triangles))
        for name, mesh in meshes.items()
        if name in entries and name in LABELS
    }
    nodes: dict[str, Any] = {}
    unfaithful, undeclared, broken = [], [], []
    against: dict[str, int] = {}
    for label, body in bodies.items():
        node = entries[body.name]
        several = node.get("pieces") == "several"
        found = topology(body.mesh)
        faults = topology_faults(found, several)
        true = check_fidelity(body, node, manifest, manifest_path.parent, seg_dir)
        against[true["against"]] = against.get(true["against"], 0) + 1
        nodes[body.name] = {
            "label": label,
            "name": node.get("label"),
            "fidelity": true,
            "topology": {**found, "faults": faults, "pass": not faults},
        }
        if faults:
            broken.append(f"{body.name} {label}: {', '.join(faults)}")
        if not true["pass"]:
            unfaithful.append(f"{body.name} {label}: {fidelity_words(true)}")
        change = true.get("change")
        if change and not change["pass"]:
            undeclared.append(
                f"{body.name} {label}: undeclared change, +{change['addedMl']} ml of "
                f"+{change['declaredAddedMl']} declared, -{change['removedMl']} ml of "
                f"-{change['declaredRemovedMl']} declared"
            )
    measured = ", ".join(f"{count} against the {kind}" for kind, count in against.items()
                         if kind not in ("modelled", "nothing"))  # fmt: skip
    rule("fidelity", unfaithful,
         f"mean <= {MAX_MEAN_MM} mm and p95 <= {MAX_P95_MM} mm both ways: {measured}; "
         f"{against.get('modelled', 0)} modelled, not measured")  # fmt: skip
    if against.get("contoured mask"):
        rule("declared changes", undeclared, "every contoured mask differs from the scan's "
             "own by no more than the manifest declares")  # fmt: skip
    else:
        rule("declared changes", None, "not checked: the manifest names no contoured masks "
             "(mesh.py --contoured), so the meshes were measured against the raw masks"
             )  # fmt: skip
    rule("topology", broken, f"{len(bodies)} nodes closed, manifold, wound outward, free of "
         "crossing faces and of stray pieces")  # fmt: skip

    labels = list(bodies)
    entering = []
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            found = entered(bodies[first], bodies[second])
            if found:
                entering.append({"first": first, "second": second, **found})
    too_deep = [
        f"{e['first']} and {e['second']} enter each other by "
        f"{max(e['firstInSecondMm'], e['secondInFirstMm'])} mm"
        for e in entering
        if MAX_ENTERING_MM is not None
        and max(e["firstInSecondMm"], e["secondInFirstMm"]) > MAX_ENTERING_MM
    ]
    rule("entering", too_deep if MAX_ENTERING_MM is not None else None,
         f"{len(entering)} pairs of nodes enter each other (reported, no limit set)"
         if MAX_ENTERING_MM is None else f"no pair deeper than {MAX_ENTERING_MM} mm")  # fmt: skip

    pairs = [n for n in range(1, 13)
             if all(rib_label(side, n) in bodies for side in ("left", "right"))]  # fmt: skip
    lacking, cut, missing = [], [], []
    for n in sorted(set(range(1, 13)) - set(pairs)):
        why = {dropped[r] for r in (rib_label("left", n), rib_label("right", n)) if r in dropped}
        if any(w.startswith("cut by") for w in why):
            cut.append(n)
            lacking.append(f"pair {n} is cut by the scan (manifest: {'; '.join(sorted(why))})")
        else:
            missing.append(n)
            lacking.append(f"pair {n} is missing from the build"
                           + (f" (manifest: {'; '.join(sorted(why))})" if why else ""))  # fmt: skip
    rule("rib pairs", lacking, "12 of 12 pairs")
    joints, failures = rib_joints(bodies, dropped)
    shifted = [j for j in joints if j.get("nearest") and j["nearest"] != f"T{rib_number(j['rib'])}"]
    notes = []
    if joints and len(shifted) > len(joints) / 2:
        notes.append(
            f"{len(shifted)} of {len(joints)} ribs lie nearest another vertebra than their "
            "own: the segmentation may number ribs and vertebrae one apart"
        )
    rule("rib joints", failures, f"every rib within {RIB_JOINT_MM} mm of its vertebrae")
    cartilage, failed = cartilages(bodies, entries.get(NAMES["costal_cartilages"], {}).get("bands"))
    passed = {
        "cartilage to rib": f"reaches the front end of ribs 1 to 10 within {COSTOCHONDRAL_MM} mm",
        "floating ribs": f"none within {FLOAT_CLEAR_MM} mm of the front end of ribs 11 and 12",
        "sternum contacts": "seven separate patches a side, cartilages 1 to 7 in order",
        "costal arch": f"cartilages 8 to 10 each lap onto the one above: a piece of their "
        f"own, within {LAP_GAP_MM} mm of it over {LAP_MM} mm or more",
    }
    for name, what in failed.items():
        rule(name, what, passed[name])
    sternum: dict[str, Any] = {}
    if "sternum" in bodies:
        patches = cartilage["patches"]
        sides = [[p for p in patches if p["side"] == side] for side in ("left", "right")]
        sternum, failed = sternum_shape(
            bodies["sternum"],
            [p["centre"][2] for p in patches if p.get("rib") == 2],
            [mine[-1] for mine in sides if mine],
        )
        notch, angle, xiphoid = sternum["jugularNotch"], sternum["sternalAngle"], sternum["xiphoid"]
        passed = {
            "jugular notch": f"{notch['enFace']['depthMm']} mm deep en face "
            f"(front view {notch['frontView']['depthMm']} mm; at least {JUGULAR_NOTCH_MM})",
            "sternal angle": f"{angle['frontSurfaceDeg']} deg on the front surface "
            f"(mid-thickness {angle['midLineDeg']}), bend "
            f"{angle['bendAboveSecondCartilageMm']} mm above the second cartilages' contact",
            "xiphoid": f"{xiphoid.get('lengthMm')} mm below the seventh cartilages' contact, "
            f"{xiphoid.get('widthMm')} mm wide against the body's {xiphoid.get('bodyWidthMm')}",
        }
        for name, what in failed.items():
            rule(name, what, passed[name])
    else:
        for name in ("jugular notch", "sternal angle", "xiphoid"):
            rule(name, ["no sternum in the build"])
    return {
        "glb": str(glb),
        "manifest": str(manifest_path),
        "segDir": str(seg_dir),
        "limits": LIMITS,
        "nodes": nodes,
        "entering": entering,
        "ribPairs": {"present": pairs, "cutByScan": cut, "missing": missing},
        "ribJoints": joints,
        "notes": notes,
        "cartilages": cartilage,
        "sternum": sternum,
        "rules": rules,
        "pass": not any(r["pass"] is False for r in rules),
    }


def fidelity_words(true: dict[str, Any]) -> str:
    """A node's fidelity in a line: mean / 95th percentile, mesh to mask
    and mask to mesh, and what it was measured against."""
    if "meshToMask" not in true:
        return true.get("why", true["against"])
    out, back = true["meshToMask"], true["maskToMesh"]
    return (f"mesh to mask {out['mean']:.2f} / {out['p95']:.2f} mm, mask to mesh "
            f"{back['mean']:.2f} / {back['p95']:.2f} mm ({true['against']})")  # fmt: skip


def print_report(report: dict[str, Any]) -> None:
    """One line per node, per pair of nodes that enter each other, per rib,
    per contact patch and per rule."""
    print("node       structure          mesh>mask mean/p95  mask>mesh mean/p95  against"
          "          change +/- ml (declared)   pieces  volume ml  topology")  # fmt: skip
    for name, node in report["nodes"].items():
        true, found = node["fidelity"], node["topology"]
        if "meshToMask" in true:
            out, back = true["meshToMask"], true["maskToMesh"]
            numbers = (f"{out['mean']:8.2f} /{out['p95']:6.2f}   {back['mean']:8.2f} /"
                       f"{back['p95']:6.2f}   {true['against']:15s}")  # fmt: skip
        else:
            numbers = f"{'':18s}{'':20s}{true.get('why', true['against']):15s}"
        change = true.get("change")
        declared = (f"+{change['addedMl']:.2f} -{change['removedMl']:.2f} (+"
                    f"{change['declaredAddedMl']:.2f} -{change['declaredRemovedMl']:.2f})"
                    if change else "")  # fmt: skip
        ok = true["pass"] and (not change or change["pass"])
        print(f"{name:10s} {node['label']:18s} {numbers} {'ok  ' if ok else 'FAIL'} "
              f"{declared:26s} {len(found['pieces']):6d} {found['volumeMl']:10.2f}  "
              f"{'; '.join(found['faults']) or 'clean'}")  # fmt: skip
    for e in report["entering"]:
        print(f"entering   {e['first']} in {e['second']} {e['firstInSecondMm']:.2f} mm, "
              f"{e['second']} in {e['first']} {e['secondInFirstMm']:.2f} mm, "
              f"shared {e['sharedMl']:.3f} ml")  # fmt: skip
    reach = {row["rib"]: row for row in report["cartilages"].get("ribs", [])}
    for joint in report["ribJoints"]:
        gaps = "  ".join(f"{vertebra} {found if isinstance(found, str) else f'{found:.2f} mm'}"
                         for vertebra, found in joint["gaps"].items())  # fmt: skip
        near = (
            f"nearest {joint['nearest']} {joint['nearestMm']:.2f} mm" if "nearest" in joint else ""
        )
        front = reach.get(joint["rib"])
        cartilage = f"cartilage {front['gapMm']:.2f} mm from its front end" if front else ""
        print(f"rib        {joint['rib']:13s} {gaps:44s} {near:24s} {cartilage}")
    for patch in report["cartilages"].get("patches", []):
        print(f"patch      {patch['side']:5s} cartilage {patch.get('rib', '?')!s:2s} "
              f"z {patch['zLow']:.1f} to {patch['zHigh']:.1f}  {patch['areaMm2']:.0f} mm2  "
              f"cartilage pieces {patch['pieces']}")  # fmt: skip
    for lap in report["cartilages"].get("arch", []):
        what = (f"one piece with cartilages {lap['with']}" if lap["state"] == "fused" else
                f"gap {lap['gapMm']} mm, lap {lap['lapMm']} mm, tip {lap['tipToEndAboveMm']} mm "
                f"from the sternal end of cartilage {lap['cartilage'] - 1}")  # fmt: skip
        if lap["state"] == "no cartilage":
            what = ""
        print(
            f"arch       {lap['side']:5s} cartilage {lap['cartilage']:2d} {lap['state']:7s} {what}"
        )
    for joined in report["cartilages"].get("joined", []):
        print(f"joined     {joined['side']} cartilages {joined['cartilages']} are one piece")
    sternum = report["sternum"]
    if sternum:
        notch, angle, xiphoid = sternum["jugularNotch"], sternum["sternalAngle"], sternum["xiphoid"]
        print(f"sternum    jugular notch {notch['enFace']['depthMm']} mm en face "
              f"(corners {notch['enFace']['spanMm']} mm apart), {notch['frontView']['depthMm']} "
              f"mm in a front view; the manubrium leans {notch['leanDeg']} deg")  # fmt: skip
        print(f"sternum    sternal angle {angle['frontSurfaceDeg']} deg front surface, "
              f"{angle['midLineDeg']} deg mid-thickness, about {angle['about']}; the bend "
              f"is at z {angle['bendZ']}, {angle['bendAboveSecondCartilageMm']} mm above that "
              f"contact (about the bend: {angle['aboutBend']})")  # fmt: skip
        print(f"sternum    xiphoid {xiphoid.get('lengthMm')} mm below the lowest contacts "
              f"(cartilages {xiphoid['below']}), {xiphoid.get('widthMm')} mm wide; the body "
              f"{xiphoid.get('bodyWidthMm')} mm")  # fmt: skip
    for r in report["rules"]:
        state = {True: "pass", False: "FAIL", None: "    "}[r["pass"]]
        print(f"rule       {r['rule']:17s} {state}  {r['detail']}")
    for note in report["notes"]:
        print(f"note       {note}")
    failed = [r["rule"] for r in report["rules"] if r["pass"] is False]
    print(f"atlas check passed: {report['glb']}" if not failed
          else f"atlas check failed: {', '.join(failed)}")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("case", nargs="?", help="a case built by atlas.py, for the default paths")
    parser.add_argument("--glb", type=Path, help="default: data/atlas/<case>/<case>.skeleton.glb")
    parser.add_argument(
        "--manifest", type=Path, help="default: data/work/<case>/<case>.skeleton.manifest.json"
    )
    parser.add_argument("--seg-dir", type=Path, help="default: data/work/<case>/seg")
    parser.add_argument("--out", type=Path, help="default: <glb>.check.json beside the GLB")
    args = parser.parse_args()
    if args.case:
        work = ROOT / "data" / "work" / args.case
        args.glb = args.glb or ROOT / "data" / "atlas" / args.case / f"{args.case}.skeleton.glb"
        args.manifest = args.manifest or work / f"{args.case}.skeleton.manifest.json"
        args.seg_dir = args.seg_dir or work / "seg"
    if not (args.glb and args.manifest and args.seg_dir):
        parser.error("give a case, or --glb, --manifest and --seg-dir")
    report = check(args.glb, args.manifest, args.seg_dir)
    print_report(report)
    out = args.out or args.glb.with_suffix(".check.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
