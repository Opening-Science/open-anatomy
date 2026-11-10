"""
Turns the masks from segment.py into one glTF binary (GLB) with a mesh per
structure, named by its UBERON ID, plus a manifest.

  python scripts/mesh.py --case s1086 [--seg-dir <dir>] [--out-dir <dir>]
      [--max-triangles 40000]

For each structure: crop the mask, upsample it 2x, blur it and contour it at
the level that keeps the mask's voxel volume, smooth lightly, keep the
largest connected piece, decimate it to at most --max-triangles (organs) or
--max-bone-triangles (bones) while it stays closed, and orient the normals
outward. Coordinates go from the CT's RAS millimetres to glTF metres (+Y up,
+Z anterior, +X the body's left), and the whole set is moved so its bounding
box is centred on the origin. The share of a single structure's voxels outside
its largest piece goes into the manifest as dropped_share; above
MAX_DROPPED_SHARE the run stops (a structure in two pieces is a
segmentation problem, not noise), unless the pieces are one bone split over
thin bone, in line and within BRIDGE_GAP_MM: that gap is bridged and noted.

--set organs (the default) needs every organ; --set reference adds the trunk
skeleton and lists, rather than meshes, a bone the scan misses or cuts (the
femurs and humeri may be cut). An organ that is absent or cut by the scan
stops the run. With body_trunc.nii.gz in the masks it also
writes <case>.outline.glb, the trunk outline in the same frame, which
assemble.py measures and never releases.

Writes data/work/<case>/<case>.<set>.glb and <case>.<set>.manifest.json with each node's
UBERON ID, system, licence, triangle count, volume and dropped share, and the
GLB's SHA-256. Every GLB's asset.copyright and asset.extras credit the source
dataset's creator, DOI and case, so a GLB copied on its own keeps its
attribution. It stops if any surface is open, non-manifold or inside out, or if its volume
is more than 5% off the mask's voxel volume. VTK runs on
one thread so the same masks give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("VTK_SMP_BACKEND_IN_USE", "Sequential")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import vtk  # noqa: E402
from scipy import ndimage  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import RULED_SETS, SETS, Structure, node_name  # noqa: E402

# RAS millimetres to glTF metres: X = -R (the body's left), Y = S, Z = A.
# Limb bones run out of a trunk scan by design; their joint ends are what count.
MIN_ML = 0.3  # a mask smaller than this is a stray label, not a bone (volume, not voxels)
PADDING_HU = -1020  # at or below: a scanner's fill beyond what it scanned, never tissue
PADDING_BLEND = 2  # voxels: resampling blends the fill into the last scanned slice this far
SETS_OF_PIECES = {"costal_cartilages"}  # one node, several separate pieces
MIN_PIECE_VOXELS = 150  # a piece smaller than this (0.5 ml) in a set is a loose blob
LEVEL_STEPS = 12
SLICE_SMOOTHED = ("sternum", "rib_", "vertebrae_", "costal_cartilages", "sacrum")
# (in-plane, along slices), in source voxels. Half a voxel in-plane left a
# vertebral body's wall fluted: the voxel steps of each slice, drawn out along
# the slice axis by the stronger blur there.
SLICE_BLURS = ((1.0, 2.0), (0.5, 1.5), (0.25, 1.0))
STERNUM_BLURS = ((1.0, 3.0), (0.75, 2.0), (0.5, 1.5))  # its ripples span several slices
RIB_BLURS = ((0.75, 5.0), (0.75, 3.5), (0.75, 2.0))  # (across, along the rib) in mm: unroll()
CARTILAGE_BLURS = ((1.0, 1.5), (0.75, 1.0), (0.5, 0.75))  # thin, lumpy segmentation
CARTILAGE_MAX_MM = 170  # the 7th costal cartilage runs up to about 15 cm in a tall man
JOIN_OVERLAP_MM = 6.0  # a modelled cartilage begins this far inside its rib's end, out of sight
INSIDE_SHARE = 0.6  # of the rib's own section: the band's where it begins inside the rib
CARTILAGE_FLAT = 2.5  # a cartilage band is this many times wider than deep
CARTILAGE_WIDEN = 0.15  # a cartilage of ribs 1-7 is this much wider at the sternum than at its rib
ARCH_JOIN = {8: 0.75, 9: 0.6, 10: 0.5}  # where along the cartilage above ribs 8-10 merge into it
CARTILAGE_HALF_W_MM = 9.0  # a band is at most 18 mm wide, whatever its rib's end measures
TANGENT_SHARE = 0.7  # of the rib-to-sternum span, for the cartilage curve's end tangents
ARCH_RUN_MM = 24.0  # a band of ribs 8-10 runs on this far inside the band above, thinning out
ARCH_THIN = 0.3  # of the band above's section: what is left of it where it ends in there
ARCH_LEAVE_MM = 30.0  # of its way from its rib before a band of ribs 8-10 lies beside the one above
ARCH_SMOOTH_MM = (
    8.0  # the band above's frame is smoothed this far along before a band is hung on it
)
ARCH_MIN_MM = 5.0  # a band of ribs 8-10 shorter than this is not worth laying
CHEST_AXIS_MM = 100.0  # the chest wall faces away from an axis this far behind the sternum
XIPHOID_SHARE = 0.7  # of the sternum's usual width: what hangs below that is the xiphoid
NOTCH_MM = 2.0  # a cartilage keeps its full width this far into the sternum's border
SET_BACK_MM = 1.0  # a cartilage's front face lies this far behind the sternum's, where they meet
FACE_SLAB_MM = 2.5  # the sternum's front face at a height is read from this far above and below
FACE_SPAN_MM = 6.0  # and its lean from the face this far above and below that
CARTILAGE_VOXEL_MM = 0.9  # the bands' field is sampled this far apart, then contoured
FIELD_REACH_MM = 3.0  # the field tells inside from outside up to this far from a band's surface
UNION_BLEND_MM = 3.0  # bands nearer each other than about half this run together in a fillet
CARTILAGE_BLUR_MM = 0.5  # a light Gaussian blur of the field before it is contoured
EVEN_ROUNDS = 10  # rounds of evening out the contoured cartilage's triangles
EVEN_SLOPE = 0.5  # mm of depth per mm: below it the bands' field has no clear slope to follow
MIN_HALF_DEPTH_MM = 2.0  # neither a rib end nor a band is taken to be thinner than 4 mm
CARTILAGE_SAMPLE_MM = 0.5  # centreline samples this far apart decide which voxels a band holds
CARTILAGE_SQUARE = 3.0  # the band's section: 2 is an ellipse, 3 a flatter face with rounded edges
UP = np.array([0.0, 0.0, 1.0])  # RAS superior
KNOB_MAX_ML = 1.0  # a lump the opening separates from a rib is taken off below this volume
RIB_BIN_MM = 3.0
END_SKIP_BINS = 3  # bins of a rib's centreline at its front end that curl into a corner
END_FIT_BINS = 7  # bins behind those (21 mm) that give the end's direction
END_SECTION_MM = (8.0, 18.0)  # behind a rib's tip: its section is taken clear of the rounding
RIB_SMOOTH_MM = 9.0
RIB_LEAD_MM = 30.0  # a rib's centreline runs on this far past its ends, for the blur's reach
RIB_KEPT_SHARE = 0.9  # of a rib's volume must lie where its centreline puts it (real ribs: 0.96+)
PAD = 2  # empty voxels around a crop, so the surface closes
UPSAMPLE = 2
BLUR_VOXELS = (1.0, 0.5, 0.25)  # Gaussian sigmas to try, in source voxels
BLUR_TOLERANCE = 0.02  # a blur is kept if the surface's volume is this close
MAX_VOLUME_DRIFT = 0.05  # a mesh may differ from its voxel volume by this share
LANDMARK_BONES = {"femur_left", "femur_right", "humerus_left", "humerus_right"}
OUTLINE = Structure("body_trunc", "outline", "trunk outline", "outline", (0.8, 0.8, 0.8))
RAS_TO_GLTF = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64) * 0.001
SOURCE_LICENCE = "CC-BY-4.0"
SOURCES = json.loads((ROOT / "sources.json").read_text())
# Pieces other than the largest may hold at most this share of a single
# structure's voxels: stray voxels are dropped, a split organ stops the run.
MAX_DROPPED_SHARE = 0.02
BRIDGE_GAP_MM = (
    20.0  # a bone in two pieces this close, in line, is one bone with a segmentation gap
)
BRIDGE_END_MM = 15.0  # the length of bone whose direction the gap must follow
BRIDGEABLE = ("rib", "clavicula", "humerus", "femur")  # long bones a gap can split in two
FRAGMENT_SHARE = 0.05  # a detached piece under this share of a bone is a segmentation fragment
REASSIGN_SHARE = 0.25  # up to this share, a piece touching a neighbour of the same kind is its
NEIGHBOUR_CONTACT = 20  # voxels of contact that make a piece a neighbour's


def triangles_of(poly: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)


def decimate(poly: vtk.vtkPolyData, max_triangles: int) -> vtk.vtkPolyData:
    """Reduces to about max_triangles, keeping the surface closed and manifold.

    Either decimator can open a thin or holed bone (a rib, the sacrum), so each
    result is checked: quadric first, then DecimatePro, each at the target and
    then at 1.5 and 2.25 times it. If none stays closed the surface is kept whole.
    """
    for target in (max_triangles, max_triangles * 1.5, max_triangles * 2.25):
        reduction = 1 - target / poly.GetNumberOfCells()
        if reduction <= 0:
            break
        quadric = vtk.vtkQuadricDecimation()
        quadric.SetTargetReduction(reduction)
        quadric.VolumePreservationOn()
        pro = vtk.vtkDecimatePro()
        pro.SetTargetReduction(reduction)
        pro.PreserveTopologyOn()
        pro.SplittingOff()
        pro.BoundaryVertexDeletionOff()
        for decimator in (quadric, pro):
            decimator.SetInputData(poly)
            decimator.Update()
            out = decimator.GetOutput()
            if not open_edges(triangles_of(out)):
                return out
    return poly


def rib_end_of(vertices: np.ndarray, triangles: np.ndarray) -> dict | None:
    """The front end of a rib's meshed surface (RAS mm vertices, triangles):
    its tip, the unit direction the rib runs in there, and the surface's
    points END_SECTION_MM behind the tip, centred and flattened across the
    rib ("section"), with the direction across their width and the
    half-width and half-thickness along that. The section is taken behind
    the tip's rounding, where the rib is as wide as it looks, so a cartilage
    that takes it is as wide as its rib. The rib's centreline is its surface
    walked along the mesh from its back end in RIB_BIN_MM bins; the walk's
    last bins curl into a corner of a blunt end (and its first ones fan out
    from one, which is why it starts at the back), so the direction comes
    from the END_FIT_BINS before the last END_SKIP_BINS, fitted again
    through slices cut straight across it, and the tip is where the surface
    ends along it, on the rib's own axis. Measured on the mesh, the end is
    the one the body shows, whatever was trimmed or smoothed away."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    edges = np.unique(
        np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1),
        axis=0,
    )
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2).tocsr()
    used = np.unique(triangles)
    start = used[np.argmax(np.linalg.norm(vertices[used] - vertices[used].mean(0), axis=1))]
    along = dijkstra(graph, directed=False, indices=int(start))
    far = used[np.argmax(np.where(np.isfinite(along[used]), along[used], -1.0))]
    if vertices[start, 1] > vertices[far, 1]:  # RAS +A is the front: walk from the back end
        along = dijkstra(graph, directed=False, indices=int(far))
    ok = np.isfinite(along)
    ok[np.setdiff1d(np.arange(len(vertices)), used)] = False
    bins = (along[ok] // RIB_BIN_MM).astype(int)
    count = np.bincount(bins)
    filled = count > 0
    line = np.stack([np.bincount(bins, vertices[ok][:, d]) for d in range(3)], 1)
    line = line[filled] / count[filled, None]
    if len(line) < END_SKIP_BINS + 3:
        return None
    line = ndimage.gaussian_filter1d(line, 1.5, axis=0, mode="nearest")
    anchor = line[-1 - END_SKIP_BINS]
    direction = anchor - line[max(len(line) - 1 - END_SKIP_BINS - END_FIT_BINS, 0)]
    direction /= np.linalg.norm(direction)
    near = vertices[ok][
        np.linalg.norm(vertices[ok] - anchor, axis=1) < (END_SKIP_BINS + 6) * RIB_BIN_MM
    ]
    # The walk ends in a corner, and its centreline leans toward it. Slices cut
    # straight across that line have their middles on the rib's axis whichever
    # way it leans, so the direction is fitted again through them.
    ahead = (near - anchor) @ direction
    slices = ((float(ahead.max()) - ahead) // RIB_BIN_MM).astype(int)
    middles = [near[slices == i].mean(0) for i in range(1, END_FIT_BINS + 1) if (slices == i).any()]
    if len(middles) >= 3:
        fitted = np.linalg.svd(np.array(middles) - np.mean(middles, 0), full_matrices=False)[2][0]
        direction = fitted if fitted @ direction > 0 else -fitted
    ahead = (near - anchor) @ direction
    reach = float(ahead.max())
    ring = near[(ahead > reach - END_SECTION_MM[1]) & (ahead < reach - END_SECTION_MM[0])]
    if len(ring) < 10:
        return None
    flat = ring - np.outer((ring - anchor) @ direction, direction)
    centre = flat.mean(0)
    flat -= centre
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    across = vt[0] - direction * (vt[0] @ direction)
    across /= np.linalg.norm(across)
    width, depth = extents(flat, across, np.cross(direction, across))
    return {"tip": centre + direction * reach, "direction": direction, "across": across,
            "halfWidth": width, "halfDepth": depth, "section": flat}  # fmt: skip


def extents(section: np.ndarray, across: np.ndarray, deep: np.ndarray) -> tuple[float, float]:
    """A rib end's half-width and half-depth along across and deep: half the
    2nd-to-98th percentile spread of its section's surface points, and never
    under 3 mm and MIN_HALF_DEPTH_MM."""
    width = float(np.ptp(np.percentile(section @ across, [2, 98]))) / 2
    depth = float(np.ptp(np.percentile(section @ deep, [2, 98]))) / 2
    return max(width, 3.0), max(depth, MIN_HALF_DEPTH_MM)


def wall_across(tangent: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """The unit direction of a band's width where the band runs along tangent
    in a chest wall that faces outward: at right angles to both, pointing up
    where it can, so the band lies flat in the wall with its broad face
    looking out of the chest (forward, beside the sternum)."""
    across = np.cross(outward, tangent)
    if np.linalg.norm(across) < 1e-6:  # running straight out of the chest: width upward
        across = UP - tangent * (UP @ tangent)
    across = across / np.linalg.norm(across)
    return -across if across[2] < 0 else across


def frames(
    curve: np.ndarray, start_across: np.ndarray, end_across: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frames along a curve: unit tangents, the width direction
    (across) and the depth direction (deep = tangent x across). A
    rotation-minimising frame (double reflection) carries start_across along
    the curve. With end_across, the twist left between the frame's last
    across and it (or its opposite, which is the same band) is spread evenly
    over the arc length, so a band ends with its face turned the way the
    structure it joins is, without a kink."""
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    first = start_across - tangents[0] * (start_across @ tangents[0])
    across = [first / np.linalg.norm(first)]
    for i in range(len(curve) - 1):
        v1 = curve[i + 1] - curve[i]
        c1 = v1 @ v1
        r = across[i] - (2 / c1) * (v1 @ across[i]) * v1
        t = tangents[i] - (2 / c1) * (v1 @ tangents[i]) * v1
        v2 = tangents[i + 1] - t
        c2 = v2 @ v2
        r = r - (2 / c2) * (v2 @ r) * v2 if c2 > 1e-12 else r
        across.append(r / np.linalg.norm(r))
    across = np.array(across)
    if end_across is not None:
        target = end_across - tangents[-1] * (end_across @ tangents[-1])
        target /= np.linalg.norm(target)
        if target @ across[-1] < 0:
            target = -target
        twist = float(np.arctan2(np.cross(across[-1], target) @ tangents[-1], across[-1] @ target))
        length = arc_length(curve)
        angle = twist * length / max(float(length[-1]), 1e-9)
        across = (across * np.cos(angle)[:, None]
                  + np.cross(tangents, across) * np.sin(angle)[:, None])  # fmt: skip
    return tangents, across, np.cross(tangents, across)


def rounded_ends(
    curve: np.ndarray, half_w: np.ndarray, half_d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trims each end of a band's centreline by its half-width there and adds
    a half-disc dome in its place, so the band keeps its length and ends in
    a round tip rather than a cut face. The trim follows the arc length, so
    a dome never folds the curve back on itself."""
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    length = np.concatenate([[0.0], np.cumsum(steps)])
    angle = np.linspace(0, 0.45 * np.pi, 7)[1:]

    def at(distance: float) -> tuple[np.ndarray, float, float]:
        distance = float(np.clip(distance, 0, length[-1]))
        return (
            np.array([np.interp(distance, length, curve[:, d]) for d in range(3)]),
            float(np.interp(distance, length, half_w)),
            float(np.interp(distance, length, half_d)),
        )

    cut0, cut1 = min(half_w[0], length[-1] * 0.3), min(half_w[-1], length[-1] * 0.3)
    inner = (length > cut0) & (length < length[-1] - cut1)
    start, w0, d0 = at(cut0)
    end, w1, d1 = at(length[-1] - cut1)
    tip1 = end - at(length[-1] - cut1 - 1e-3)[0]
    tip1 /= np.linalg.norm(tip1)
    tip0 = start - at(cut0 + 1e-3)[0]
    tip0 /= np.linalg.norm(tip0)
    dome1 = end + np.outer(cut1 * np.sin(angle), tip1)
    dome0 = start + np.outer(cut0 * np.sin(angle), tip0)
    points = np.vstack([dome0[::-1], start[None], curve[inner], end[None], dome1])
    widths = np.concatenate(
        [(w0 * np.cos(angle))[::-1], [w0], half_w[inner], [w1], w1 * np.cos(angle)]
    )
    depths = np.concatenate(
        [(d0 * np.cos(angle))[::-1], [d0], half_d[inner], [d1], d1 * np.cos(angle)]
    )
    return points, widths, depths


def arc_length(curve: np.ndarray) -> np.ndarray:
    """The length along a polyline up to each of its points."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])


def eased(share: np.ndarray) -> np.ndarray:
    """A share clipped to 0..1 and eased in and out (smoothstep)."""
    share = np.clip(share, 0, 1)
    return share**2 * (3 - 2 * share)


def turned(
    tangents: np.ndarray, across: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Width directions turned about their tangents toward target directions,
    by weight (0 to 1) of the angle between them. The target's opposite is
    the same band, so the target is taken the way round that needs less
    turning where the weight is greatest."""
    flat = target - tangents * np.einsum("ij,ij->i", target, tangents)[:, None]
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
    if flat[np.argmax(weight)] @ across[np.argmax(weight)] < 0:
        flat = -flat
    angle = weight * np.arctan2(
        np.einsum("ij,ij->i", np.cross(across, flat), tangents),
        np.einsum("ij,ij->i", across, flat),
    )
    return across * np.cos(angle)[:, None] + np.cross(tangents, across) * np.sin(angle)[:, None]


def arch_path(
    tip: np.ndarray, start: np.ndarray, above: dict, join: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The centreline of a band of ribs 8 to 10, from its rib's end (tip,
    where the rib runs along the unit direction start) to sample join of
    the centreline of the band above (its "curve", "tangent" and "across"),
    with the share (0 to 1) by which each of its samples lies beside that
    band and that band's width direction there. The path is laid out in the
    band above's frame: so far from its centreline along its width, and so
    far out of its plane. The first closes smoothly to nothing at the join,
    with no slope at either end, so the band keeps to its own side all the
    way, bends where the band above bends, and runs into its centreline
    along it. The second closes within ARCH_LEAVE_MM, so the two lie in one
    plane before they overlap and neither lies over the other like a
    shingle. Over that first stretch the path also blends out of the rib's
    own line, so the band leaves its rib without a kink. Behind the band
    above's first sample its line is carried on straight. None if the rib
    ends less than ARCH_MIN_MM before the join."""
    curve, tangent, across = above["curve"][: join + 1], above["tangent"], above["across"]
    along = arc_length(curve)
    nearest = int(np.argmin(np.linalg.norm(curve - tip, axis=1)))
    behind = min(float((tip - curve[0]) @ tangent[0]), 0.0) if nearest == 0 else 0.0
    first, way = along[nearest] + behind, along[-1] - along[nearest] - behind
    if way < ARCH_MIN_MM:
        return None
    u = np.linspace(first, along[-1], int(np.ceil(1.5 * way / CARTILAGE_SAMPLE_MM)) + 1)

    def there(values: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(u, along, values[: join + 1, d]) for d in range(3)], 1)

    centre = there(curve) + np.minimum(u, 0.0)[:, None] * tangent[0]
    # The band above's frame turns unevenly (not at all inside its rib, then
    # at once), and a path held out from it on a long arm would kink with
    # it: the directions it is held along are smoothed along the way.
    faces, spread = there(across), ARCH_SMOOTH_MM / (u[1] - u[0])
    wide = ndimage.gaussian_filter1d(faces, spread, axis=0, mode="nearest")
    wide /= np.linalg.norm(wide, axis=1, keepdims=True)
    out = np.cross(ndimage.gaussian_filter1d(there(tangent), spread, axis=0, mode="nearest"), wide)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    lying = eased((u - first) / min(ARCH_LEAVE_MM, way))
    off = (tip - centre[0]) @ wide[0] * (1 - eased((u - first) / way))
    lift = (tip - centre[0]) @ out[0] * (1 - lying)
    beside = centre + off[:, None] * wide + lift[:, None] * out
    own = tip + np.outer(u - first, start)
    return own + lying[:, None] * (beside - own), lying, faces


def densify(curve: np.ndarray, *profiles: np.ndarray, step: float = CARTILAGE_SAMPLE_MM) -> tuple:
    """The curve, and each per-sample profile with it, resampled (linearly)
    by arc length every step mm."""
    length = arc_length(curve)
    s = np.linspace(0.0, float(length[-1]), int(np.ceil(length[-1] / step)) + 1)
    points = np.stack([np.interp(s, length, curve[:, d]) for d in range(3)], 1)
    return points, *(np.interp(s, length, p) for p in profiles)


def laid_out(band: dict) -> dict:
    """A band ready to be measured against: both ends rounded (rounded_ends),
    the centreline resampled every CARTILAGE_SAMPLE_MM ("curve") with its
    half widths and depths ("halfW", "halfD") and its frames ("tangent",
    "across": the band's own width directions where it has them, or
    frames() from the directions it starts and ends with)."""
    points, half_w, half_d = rounded_ends(band["curve"], band["halfW"], band["halfD"])
    points, half_w, half_d = densify(points, half_w, half_d)
    tangents, across, _ = frames(points, band["across0"], band.get("across1"))
    if "across" in band:  # laid out by band_curves: its own width directions, by arc length
        old, new = arc_length(band["curve"]), arc_length(points)
        own = np.stack([np.interp(new, old, band["across"][:, d]) for d in range(3)], 1)
        across = turned(tangents, across, own, np.ones(len(points)))
    return {"curve": points, "halfW": half_w, "halfD": half_d, "tangent": tangents,
            "across": across}  # fmt: skip


def band_field(band: dict, points: np.ndarray, reach: float) -> np.ndarray:
    """How deep inside a laid-out band (laid_out) each point lies, in mm:
    positive inside, zero on the band's surface, negative outside, and -1e3
    for a point over reach mm from the centreline. In the frame at the
    nearest centreline point (interpolated between samples, so nothing
    steps where the nearest sample changes along a twisting band) a point
    lies u across and v deep, and the surface is where
    (|u| / half_w)^n + (|v| / half_d)^n = 1 with n = CARTILAGE_SQUARE: a
    flat-faced band with rounded edges. The depth is measured along the line
    from the centreline through the point. Past the band's two tips a point
    is outside."""
    from scipy.spatial import cKDTree

    curve, tangents, across = band["curve"], band["tangent"], band["across"]
    half_w, half_d = band["halfW"], band["halfD"]
    depth = np.full(len(points), -1e3)
    distance, k = cKDTree(curve).query(points, distance_upper_bound=reach)
    hit = np.isfinite(distance)
    k = k[hit]
    offset = points[hit] - curve[k]
    w = np.einsum("ij,ij->i", offset, tangents[k])
    j = np.clip(k + np.where(w < 0, -1, 1), 0, len(curve) - 1)
    step = np.linalg.norm(curve[j] - curve[k], axis=1)
    t = np.clip(np.abs(w) / np.maximum(step, 1e-9), 0, 1)[:, None]
    offset = offset - t * (curve[j] - curve[k])
    wide = across[k] + t * (across[j] - across[k])
    wide /= np.linalg.norm(wide, axis=1, keepdims=True)
    thick = np.cross(tangents[k] + t * (tangents[j] - tangents[k]), wide)
    thick /= np.linalg.norm(thick, axis=1, keepdims=True)
    half_wide = half_w[k] + t[:, 0] * (half_w[j] - half_w[k])
    half_thick = half_d[k] + t[:, 0] * (half_d[j] - half_d[k])
    u = np.einsum("ij,ij->i", offset, wide)
    v = np.einsum("ij,ij->i", offset, thick)
    n = CARTILAGE_SQUARE
    scale = (np.abs(u / half_wide) ** n + np.abs(v / half_thick) ** n) ** (1 / n)
    out = np.hypot(u, v)  # the surface lies out / scale from the centreline on this line
    inside = np.where(scale > 1e-6, out / np.maximum(scale, 1e-6) - out, half_thick)
    past = j == k  # beyond either tip: outside by the distance along the curve
    depth[hit] = np.where(past, np.minimum(inside, -np.abs(w)), inside)
    return depth


def rasterise_bands(
    bands: list[dict], voxel_mm: float = CARTILAGE_VOXEL_MM
) -> tuple[np.ndarray, np.ndarray]:
    """The union of cartilage bands as a smooth field sampled every voxel_mm,
    and the RAS mm position of its first sample. Each band has its
    centreline ("curve", RAS mm, sampled finely enough to be smooth as a
    polyline), half widths and depths per sample ("halfW", "halfD") and its
    width direction at the start and, if it joins something, at the end
    ("across0", "across1"). A band's field is how deep inside it a point
    lies (laid_out, band_field). The union is a smooth maximum of the bands'
    fields (UNION_BLEND_MM), so where bands meet they run into each other in
    a fillet and a slit too narrow to show closes; it is cut off at
    FIELD_REACH_MM either way. Its zero level is the bands' surface, with no
    seam where one band joins another and no voxel steps."""
    dense = [laid_out(band) for band in bands]
    blend = UNION_BLEND_MM
    margins = [float(max(d["halfW"].max(), d["halfD"].max())) + blend for d in dense]
    lo = np.min([d["curve"].min(0) - m for d, m in zip(dense, margins, strict=True)], axis=0)
    hi = np.max([d["curve"].max(0) + m for d, m in zip(dense, margins, strict=True)], axis=0)
    lo -= 2 * voxel_mm  # the field ends outside every band, so its surface closes
    shape = np.ceil((hi - lo) / voxel_mm).astype(int) + 3
    field = np.full(shape, -1e3, dtype=np.float32)
    for band, margin in zip(dense, margins, strict=True):
        curve = band["curve"]
        first = np.maximum(np.floor((curve.min(0) - margin - lo) / voxel_mm).astype(int), 0)
        last = np.minimum(np.ceil((curve.max(0) + margin - lo) / voxel_mm).astype(int) + 1, shape)
        ijk = np.indices(last - first).reshape(3, -1).T + first
        own = band_field(band, ijk * voxel_mm + lo, margin).reshape(last - first)
        part = field[first[0] : last[0], first[1] : last[1], first[2] : last[2]]
        # A smooth maximum: where the two are within the blend of each other it
        # lies above both, by as much as their surfaces face each other.
        fillet = np.clip(blend - np.abs(own - part), 0, None) ** 2 / (4 * blend)
        part[:] = np.maximum(own, part) + fillet * facing(own, part, 2 * blend)
    return np.clip(field, -FIELD_REACH_MM, FIELD_REACH_MM), lo


def facing(one: np.ndarray, other: np.ndarray, reach: float) -> np.ndarray:
    """How much two fields' surfaces face each other at each sample, from 0
    to 1: 0 where they slope the same way (two faces lying in one plane), 1
    where they slope at right angles or against each other (a crease, or the
    two sides of a slit), and 1 where either has no slope within reach of
    its surface (far outside it, where it is cut off)."""
    slopes = [np.stack(np.gradient(np.maximum(f, -reach).astype(np.float64))) for f in (one, other)]
    sizes = [np.linalg.norm(s, axis=0) for s in slopes]
    cosine = (slopes[0] * slopes[1]).sum(0) / np.maximum(sizes[0] * sizes[1], 1e-12)
    return np.where(np.minimum(*sizes) > 1e-6, np.clip(1 - cosine, 0, 1), 1.0)


def evened(
    vertices: np.ndarray, triangles: np.ndarray, field: np.ndarray, origin: np.ndarray,
    voxel_mm: float, rounds: int = EVEN_ROUNDS,
) -> np.ndarray:  # fmt: skip
    """The vertices of a surface contoured from a smooth field at its zero
    level, moved so that its triangles are even and it lies on that level
    again. A contour's vertices crowd wherever the surface passes a grid
    point, which shows as a fine ripple in the shading, and smoothing them
    shrinks a thin band. In each round every vertex goes half way to the
    mean of its neighbours (a quarter voxel at most), along the surface only
    (at right angles to the field's gradient), and is then put back on the
    zero level of the field, interpolated by cubic splines: two Newton
    steps, neither longer than half a voxel. Where the field has no clear
    slope (under EVEN_SLOPE: the crotch where two bands part) a vertex stays
    where it is, so nothing folds over there."""
    from scipy.sparse import coo_matrix, diags

    edges = np.unique(
        np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1),
        axis=0,
    ).astype(np.int64)
    n = len(vertices)
    pairs = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
    ring = coo_matrix((np.ones(2 * len(edges)), pairs), shape=(n, n)).tocsr()
    ring = diags(1.0 / np.maximum(np.asarray(ring.sum(1)).ravel(), 1)) @ ring
    spline = ndimage.spline_filter(field.astype(np.float64), order=3)

    def level(points: np.ndarray) -> np.ndarray:
        at = ((points - origin) / voxel_mm).T
        return ndimage.map_coordinates(spline, at, order=3, prefilter=False, mode="nearest")

    def gradient(points: np.ndarray) -> np.ndarray:
        h = voxel_mm / 4
        return np.stack([(level(points + h * axis) - level(points - h * axis)) / (2 * h)
                         for axis in np.eye(3)], 1)  # fmt: skip

    points = vertices.astype(np.float64)
    for _ in range(rounds):
        slope = gradient(points)
        steep = np.linalg.norm(slope, axis=1, keepdims=True)
        normal = slope / np.maximum(steep, 1e-9)
        shift = ring @ points - points
        shift -= np.einsum("ij,ij->i", shift, normal)[:, None] * normal
        length = np.linalg.norm(shift, axis=1, keepdims=True)
        shift *= np.minimum(0.5, voxel_mm / 4 / np.maximum(length, 1e-12))
        points = points + np.where(steep > EVEN_SLOPE, shift, 0.0)
        for _ in range(2):
            slope = gradient(points)
            steep = np.linalg.norm(slope, axis=1, keepdims=True)
            back = -level(points)[:, None] * slope / np.maximum(steep**2, 1e-6)
            length = np.linalg.norm(back, axis=1, keepdims=True)
            back *= np.minimum(1.0, voxel_mm / 2 / np.maximum(length, 1e-12))
            points = points + np.where(steep > EVEN_SLOPE, back, 0.0)
    return points


def cartilage_surface(
    bands: list[dict], voxel_mm: float = CARTILAGE_VOXEL_MM
) -> tuple[np.ndarray, np.ndarray]:
    """The closed surface of the union of cartilage bands, as RAS mm vertices
    and triangles: the zero level of the bands' field (rasterise_bands) on a
    voxel_mm grid, blurred a little (CARTILAGE_BLUR_MM), contoured and
    smoothed as a segmented structure is, then evened out on the field's
    zero level (evened). Bands that touch are one surface, and no faces
    cross each other. It is not decimated: its triangles are all about
    voxel_mm across, and thinning them out leaves slivers along the bands'
    edges that a quality check takes for spikes."""
    field, origin = rasterise_bands(bands, voxel_mm)
    field = ndimage.gaussian_filter(field, CARTILAGE_BLUR_MM / voxel_mm)
    image = vtk.vtkImageData()
    image.SetDimensions(*field.shape)
    image.SetSpacing(voxel_mm, voxel_mm, voxel_mm)
    image.SetOrigin(*origin)
    image.GetPointData().SetScalars(numpy_to_vtk(field.ravel(order="F"), deep=True))
    poly = finished(image, 0.0, pieces=True)
    vertices = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    triangles = triangles_of(poly).astype(np.uint32)
    return evened(vertices, triangles, field, origin, voxel_mm), triangles


def rib_number(rib: str) -> int:
    return int(rib.rsplit("_", 1)[1])


def partner_of(rib: str) -> str:
    return rib.replace("left", "right") if "left" in rib else rib.replace("right", "left")


def sternum_floor(sternum: np.ndarray) -> float:
    """The height (RAS z, mm) of the lower end of the sternum's body: the
    lowest 3 mm level at which the sternum is still XIPHOID_SHARE of its
    usual (median) width. What hangs below is the xiphoid process, which no
    costal cartilage joins."""
    levels = np.round(sternum[:, 2] / 3.0).astype(int)
    widths = {level: float(np.ptp(sternum[levels == level, 0])) for level in np.unique(levels)}
    usual = float(np.median(list(widths.values())))
    return min(level for level, width in widths.items() if width >= XIPHOID_SHARE * usual) * 3.0


def sternum_edge(sternum: np.ndarray, side: float, height: float) -> tuple[np.ndarray, float]:
    """Where a cartilage meets the sternum at a height (RAS z, mm): the point
    on the sternum's border on one side (side +1 is toward larger x, -1
    toward smaller) on the sternum's front face there, and the x of the
    sternum's middle at that height. The border is taken from the sternum
    within 5 mm of the height, or its nearest level if the height is beyond
    its ends; the front face from the border within FACE_SLAB_MM of it (the
    sternum leans, and a thicker slab would read its lean as thickness), as
    the 90th percentile of its points from back to front."""
    off = np.abs(sternum[:, 2] - height)
    slab = sternum[off <= max(5.0, float(off.min()))]
    centre = float(np.median(slab[:, 0]))
    reach = (slab[:, 0] - centre) * side
    off = off[off <= max(5.0, float(off.min()))][reach > reach.max() - 8.0]
    border = slab[reach > reach.max() - 8.0]
    front = float(np.percentile(border[off <= max(FACE_SLAB_MM, float(off.min())), 1], 90))
    return np.array([centre + side * reach.max(), front, height]), centre


def band_curves(
    ends: dict[str, dict], sternum: np.ndarray
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Lays out the costal cartilage bands from the ribs' measured front ends
    (rib_end_of, by rib label) and the sternum's RAS mm points. A band is a
    finely sampled centreline ("curve"), half widths and depths per sample
    ("halfW", "halfD"), its frame along the curve ("tangent", "across"), the
    width directions it starts and ends with ("across0", "across1") and the
    samples at its rib's end and at its target ("tipAt", "endAt").
    The rules:
    - Ribs 1 to 10 in left-right pairs only; ribs 11 and 12 float. A rib 1 to
      7 whose end is over CARTILAGE_MAX_MM from the sternum, or touches it,
      gets none.
    - A band lies flat in the chest wall, its broad face outward
      (wall_across; the wall is taken to face away from an axis
      CHEST_AXIS_MM behind the sternum), and is CARTILAGE_FLAT times wider
      than deep.
    - It begins out of sight inside its rib (JOIN_OVERLAP_MM behind the end,
      at INSIDE_SHARE of the rib's section) and leaves through the rib's end
      in the rib's own line with the rib's own section, so there is no step
      or kink at the join. Within the first third of the way it takes the
      pair's mean width (at most CARTILAGE_HALF_W_MM), so left and right
      match.
    - Ribs 1 to 7 turn medially and rise into the sternum (more for lower
      ribs), widening CARTILAGE_WIDEN on the way, and run into the sternum's
      border on their own side at full width. A pair meets the sternum at
      one height: the mean height of its two rib ends, but within the
      sternum, never on the xiphoid (sternum_floor) and never lower than a
      full band's width above the pair below.
    - Ribs 8 to 10 run up to the band above and merge into it ARCH_JOIN of
      the way along it, taking its line, width, depth and face direction
      there: the two make one smooth costal arch margin with no seam. They
      come in at an angle (a short end tangent, ARCH_REACH) rather than
      alongside, so the slit between the two is short and open.
    Returns the bands by rib and every rib that gets none, with the reason."""
    from scipy.spatial import cKDTree

    tree = cKDTree(sternum)
    midline = float(np.median(sternum[:, 0]))
    axis_y = float(np.median(sternum[:, 1])) - CHEST_AXIS_MM

    def outward(point: np.ndarray) -> np.ndarray:
        out = np.array([point[0] - midline, point[1] - axis_y, 0.0])
        return out / max(float(np.linalg.norm(out)), 1e-9)

    usable, skipped = {}, []
    for rib, end in ends.items():
        number = rib_number(rib)
        if number > 10:
            continue
        if number <= 7 and tree.query(end["tip"])[0] > CARTILAGE_MAX_MM:
            skipped.append((rib, f"its end is over {CARTILAGE_MAX_MM} mm from the sternum"))
        elif number <= 7 and tree.query(end["tip"])[0] < 1.0:
            skipped.append((rib, "its end already touches the sternum"))
        else:
            usable[rib] = end
    paired = [r for r in usable if partner_of(r) in usable]
    skipped += [(r, "its pair has no cartilage") for r in usable if r not in paired]
    starts, sides = {}, {}  # per rib: its width direction and section there; its side of the body
    for rib in paired:
        e = usable[rib]
        across = wall_across(e["direction"], outward(e["tip"]))
        w, d = extents(e["section"], across, np.cross(e["direction"], across))
        starts[rib] = (across, w, d)
        sides[rib] = 1.0 if e["tip"][0] >= usable[partner_of(rib)]["tip"][0] else -1.0
    widths = {
        rib: min((starts[rib][1] + starts[partner_of(rib)][1]) / 2, CARTILAGE_HALF_W_MM)
        for rib in paired
    }
    # The height at which each pair of ribs 1 to 7 meets the sternum, from the lowest pair up.
    heights: dict[int, float] = {}
    lowest, top = sternum_floor(sternum), float(sternum[:, 2].max())
    for number in sorted({rib_number(r) for r in paired if rib_number(r) <= 7}, reverse=True):
        pair = [r for r in paired if rib_number(r) == number]
        reach = (1 + CARTILAGE_WIDEN) * widths[pair[0]]
        level = float(np.mean([usable[r]["tip"][2] for r in pair]))
        heights[number] = max(min(level, top - reach), lowest + (reach if heights else 0.0))
        lowest = heights[number] + reach
    bands: dict[str, dict] = {}
    for rib in sorted(paired, key=lambda r: (rib_number(r), r)):
        e, number = usable[rib], rib_number(rib)
        tip, start = e["tip"], e["direction"]
        across0, w0, d0 = starts[rib]
        w_pair, side = widths[rib], sides[rib]
        if number <= 7:
            edge, centre = sternum_edge(sternum, side, heights[number])
            # Lower cartilages rise more steeply into the sternum, but never
            # much more than the line from rib to sternum does, or the band
            # would dip first and hook up at the end.
            chord = edge - tip
            chord_rise = chord[2] / max(np.hypot(chord[0], chord[1]), 1e-6)
            rise = min(0.15 + 0.2 * (number - 2), max(chord_rise, 0.0) + 0.3)
            # The sternum leans: the band meets it along its front face, not
            # the body's, with its own face SET_BACK_MM behind it, so the
            # sternum does not stand in front of its cartilages like a slab.
            lean = (
                sternum_edge(sternum, side, heights[number] + FACE_SPAN_MM)[0]
                - sternum_edge(sternum, side, heights[number] - FACE_SPAN_MM)[0]
            )
            face = np.array([0.0, lean[2], -lean[1]]) / np.hypot(lean[1], lean[2])
            t1 = np.array([-side, 0.0, rise])
            t1 = t1 - face * (t1 @ face)
            t1 /= np.linalg.norm(t1)
            end_w = w_pair * (1 + CARTILAGE_WIDEN)
            end_d = max(end_w / CARTILAGE_FLAT, MIN_HALF_DEPTH_MM * (1 + CARTILAGE_WIDEN))
            # Into the sternum's border at full width; the rounded end lies
            # inside the sternum, short of its middle.
            p1 = edge - np.array([side * NOTCH_MM, 0.0, 0.0]) - face * (end_d + SET_BACK_MM)
            room = ((p1[0] - centre) * side - 1.0) / abs(t1[0])
            run_out = float(np.clip(room, 0.0, end_w))
            across1 = wall_across(t1, face)
            span = float(np.linalg.norm(p1 - tip))
            if span < 1.0:
                skipped.append((rib, "its end lies on its target"))
                continue
            s = np.linspace(0, 1, max(int(np.ceil(1.5 * span / CARTILAGE_SAMPLE_MM)), 8))[:, None]
            h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
            h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
            # Longer tangents overshoot into a hook on short spans.
            bend = (h00 * tip + h10 * start * TANGENT_SHARE * span
                    + h01 * p1 + h11 * t1 * TANGENT_SHARE * span)  # fmt: skip
        else:
            above = bands.get(rib.replace(f"_{number}", f"_{number - 1}"))
            if above is None:
                skipped.append((rib, "no cartilage above it to join"))
                continue
            # Merge into the band above: beside it, in its plane, and onto
            # its centreline with its section and its face, so the lower
            # edges of the two flow into one arch margin.
            shown = arc_length(above["curve"][above["tipAt"] : above["endAt"] + 1])
            k = above["tipAt"] + int(np.searchsorted(shown, ARCH_JOIN[number] * shown[-1]))
            path = arch_path(tip, start, above, k)
            if path is None:
                skipped.append((rib, "its end lies on its target"))
                continue
            bend, lying, faces = path
            end_w, end_d = above["halfW"][k], above["halfD"][k]
            # Then on inside the band above, thinning out, so it ends nowhere on the surface.
            run = shown[k - above["tipAt"]] + ARCH_RUN_MM
            # But not up to the band above's own end, where the other side's band ends too.
            run = min(run, shown[-1] - 2 * CARTILAGE_HALF_W_MM)
            last = max(above["tipAt"] + int(np.searchsorted(shown, run)), k)
            run_in = slice(k + 1, last + 1)
        # Out of sight inside the rib, in the rib's own line, up to its end.
        lead = JOIN_OVERLAP_MM + INSIDE_SHARE * w0
        count = int(np.ceil(lead / CARTILAGE_SAMPLE_MM))
        inside = tip - np.outer(np.linspace(lead, 0, count, endpoint=False), start)
        shrink = np.linspace(INSIDE_SHARE, 1, count, endpoint=False)
        f = arc_length(bend) / arc_length(bend)[-1]
        blend = np.clip(f / 0.35, 0, 1)  # from the rib's own section to the pair's within a third
        half_w = (1 - blend) * w0 + blend * w_pair
        half_d = (1 - blend) * d0 + blend * max(w_pair / CARTILAGE_FLAT, MIN_HALF_DEPTH_MM)
        if number <= 7:
            half_w, half_d = half_w * (1 + CARTILAGE_WIDEN * f), half_d * (1 + CARTILAGE_WIDEN * f)
            steps = int(np.ceil(run_out / CARTILAGE_SAMPLE_MM))
            beyond = p1 + np.outer(np.linspace(0, run_out, steps + 1)[1:], t1)
            beyond_w, beyond_d = np.full(steps, end_w), np.full(steps, end_d)
        else:
            half_w = half_w * (1 - f**2) + f**2 * end_w
            half_d = half_d * (1 - f**2) + f**2 * end_d
            beyond = above["curve"][run_in]
            thin = 1 - (1 - ARCH_THIN) * eased(np.linspace(0, 1, len(beyond) + 1)[1:])
            beyond_w, beyond_d = above["halfW"][run_in] * thin, above["halfD"][run_in] * thin
        curve = np.vstack([inside, bend, beyond])
        half_w = np.concatenate([shrink * w0, half_w, beyond_w])
        half_d = np.concatenate([shrink * d0, half_d, beyond_d])
        tangent, across, _ = frames(curve, across0, across1 if number <= 7 else None)
        if number > 7:  # its face turns to the band above's as it comes to lie beside it
            faces = np.vstack([np.tile(across0, (count, 1)), faces, above["across"][run_in]])
            lying = np.concatenate([np.zeros(count), lying, np.ones(len(beyond))])
            across = turned(tangent, across, faces, lying)
            across1 = across[-1]
        bands[rib] = {"curve": curve, "halfW": half_w, "halfD": half_d, "tangent": tangent,
                      "across": across, "across0": across0, "across1": across1,
                      "tipAt": count, "endAt": count + len(bend) - 1}  # fmt: skip
    return bands, skipped


def modelled_cartilage(
    sternum_path: Path, ribs: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Costal cartilages modelled, not segmented, for ribs 1 to 10: the bands
    band_curves() lays out from the sternum's mask and the front ends of the
    ribs' meshed surfaces (RAS mm vertices and triangles by rib label),
    surfaced together (cartilage_surface). Bands that meet are one smooth
    closed surface, so ribs 7 to 10 of a side make one costal arch with no
    seam, step or join, and no faces cross each other anywhere in the set.
    Returns RAS mm vertices and triangles, and each rib that gets no
    cartilage with the reason; the manifest marks the node as modelled."""
    sternum_image = nib.load(sternum_path)
    sternum = np.argwhere(np.asanyarray(sternum_image.dataobj) > 0)
    sternum = sternum @ sternum_image.affine[:3, :3].T + sternum_image.affine[:3, 3]
    ends, skipped = {}, []
    for rib, (vertices, triangles) in ribs.items():
        if rib_number(rib) > 10:
            continue  # ribs 11 and 12 float: no cartilage reaches the arch
        end = rib_end_of(vertices, triangles)
        if end is None:
            skipped.append((rib, "its front end could not be measured"))
        else:
            ends[rib] = end
    bands, unplaced = band_curves(ends, sternum)
    skipped += unplaced
    if not bands:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint32), skipped
    ras, triangles = cartilage_surface(list(bands.values()))
    return ras, triangles, skipped


def scan_outside(ct_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The part of a CT volume that was never scanned: the constant fill a
    scanner writes beyond its field of view (a circle in-plane, a slanted edge
    from a tilted gantry, empty end slices). Voxels at or below PADDING_HU
    (never tissue; air is about -1000 HU with noise) in regions that survive a
    small opening and join the volume's border, grown back through the fill
    so the thin end of a gantry wedge stays outside too, then PADDING_BLEND
    voxels further out: a scan resampled onto its grid blends the fill into
    its last scanned slice over that distance (the blended voxels sit well
    above PADDING_HU and measure nothing), so a bone ending against the
    blend was cut by the scan's end just as one against the fill was.
    Returns the mask and the CT's affine, so masks can be checked against
    the same grid."""
    image = nib.load(ct_path)
    fill = np.asanyarray(image.dataobj) <= PADDING_HU
    pad = ndimage.binary_opening(fill, structure=np.ones((3, 3, 3)))
    labels, _ = ndimage.label(pad)
    border = np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])  # fmt: skip
    kept = np.isin(labels, np.setdiff1d(np.unique(border), [0]))
    outside = ndimage.binary_propagation(kept, mask=fill)
    return ndimage.binary_dilation(outside, iterations=PADDING_BLEND), image.affine


def cut_reason(path: Path, outside: tuple[np.ndarray, np.ndarray] | None = None) -> str | None:
    """Why a mask cannot be a whole structure, or None if it is whole.

    Smaller than MIN_ML (measured with the voxel size from the affine, so the
    limit is the same at 0.8 mm and 1.5 mm), touching any face of the volume
    (the first or last slice, or the in-plane border), or touching the part of
    the volume that was never scanned (outside, from scan_outside, blend
    included): the scan cut it. A mask on another grid than the CT's raises
    ValueError.
    """
    if not path.exists():
        return "not segmented"
    image = nib.load(path)
    data = np.asanyarray(image.dataobj) > 0
    if data.sum() * abs(np.linalg.det(image.affine[:3, :3])) / 1000 < MIN_ML:
        return "absent or a stray label"
    for axis in range(3):
        k = np.flatnonzero(data.any(tuple(a for a in range(3) if a != axis)))
        if k[0] == 0 or k[-1] == data.shape[axis] - 1:
            return f"cut by the scan's edge (axis {axis})"
    if outside is not None and (ndimage.binary_dilation(data) & on_grid(image, outside)).any():
        return "cut by the scan's edge (unscanned padding)"
    return None


def on_grid(image: nib.Nifti1Image, outside: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The unscanned padding, after checking the mask lies on the CT's own grid."""
    pad, affine = outside
    if image.shape != pad.shape or not np.allclose(image.affine, affine, atol=1e-3):
        raise ValueError(f"{image.get_filename()} is not on the CT's grid")
    return pad


def cut_faces(path: Path, outside: tuple[np.ndarray, np.ndarray] | None = None) -> set[str]:
    """The sides of the body where the scan cuts a mask: "S" or "I" (its top
    or bottom slice), "L", "R", "A" or "P" (the in-plane border), named by the
    direction each face of the volume points in RAS. Where the mask meets
    unscanned padding (outside), each separate patch of contact names the
    side the padding lies on there: the direction in which most of the patch
    is one step on from the scanned part of the mask. A clavicle the scan
    ends through is cut at "S", however it lies, and a bone cut at both ends
    names both."""
    image = nib.load(path)
    data = np.asanyarray(image.dataobj) > 0
    faces = set()
    sides = []  # (axis, step, side) for the six directions of the grid
    for axis in range(3):
        column = image.affine[:3, axis]
        ras = int(np.argmax(np.abs(column)))
        up, down = "RAS"[ras], "LPI"[ras]  # the RAS axis's positive and negative ends
        high, low = (up, down) if column[ras] > 0 else (down, up)
        sides += [(axis, 1, high), (axis, -1, low)]
        k = np.flatnonzero(data.any(tuple(a for a in range(3) if a != axis)))
        if len(k) == 0:
            continue
        if k[0] == 0:
            faces.add(low)
        if k[-1] == data.shape[axis] - 1:
            faces.add(high)
    if outside is not None:
        pad = on_grid(image, outside)
        met = ndimage.binary_dilation(data) & pad
        scanned = data & ~pad
        patches, count = ndimage.label(met, structure=np.ones((3, 3, 3)))
        contact, patch = np.argwhere(met), patches[met]
        reached = np.zeros((count + 1, len(sides)), dtype=np.int64)
        for k, (axis, step, _) in enumerate(sides):
            source = contact.copy()
            source[:, axis] -= step  # the voxel this one is a step on from
            inside = (source[:, axis] >= 0) & (source[:, axis] < data.shape[axis])
            inside[inside] = scanned[tuple(source[inside].T)]
            reached[:, k] = np.bincount(patch[inside], minlength=count + 1)
        faces |= {sides[int(np.argmax(row))][2] for row in reached[1:] if row.any()}
    return faces


def bridge_gap(mask: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, float | None]:
    """One bone split by the segmentation over thin bone (an osteopenic rib,
    a vertebra's pedicle): when the second-largest piece lies in line with the
    end of the largest, within BRIDGE_GAP_MM, the gap is filled with a tube of
    the bone's local radius. Returns the mask and the gap bridged in mm, or
    None when the pieces are not one bone (too far, or side by side)."""
    from scipy.spatial import cKDTree

    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count < 2:
        return mask, None
    order = np.argsort(np.bincount(labels.ravel())[1:])[::-1]
    a = np.argwhere(labels == order[0] + 1) * spacing
    b = np.argwhere(labels == order[1] + 1) * spacing
    distance, nearest = cKDTree(a).query(b)
    gap = float(distance.min())
    if gap > BRIDGE_GAP_MM:
        return mask, None
    # The facing ends: the points of each piece within a voxel of the gap.
    close = distance <= gap + spacing.max()
    pb = b[close].mean(0)
    pa = a[np.unique(nearest[close])].mean(0)
    end = a[np.linalg.norm(a - pa, axis=1) < BRIDGE_END_MM]
    if len(end) < 10:
        return mask, None
    centred = end - end.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = (pb - pa) / max(float(np.linalg.norm(pb - pa)), 1e-6)
    if abs(float(axis @ along)) < np.cos(np.radians(35)):
        return mask, None  # beside the bone, not along it: another structure's label
    # The end's cross-section, as a covariance across the bone, sets the fill's
    # shape: a flat rib gets a flat bridge.
    perpendicular = centred - np.outer(centred @ axis, axis)
    covariance = perpendicular.T @ perpendicular / len(perpendicular)
    covariance += np.eye(3) * (0.5 * spacing.max()) ** 2  # never thinner than a voxel
    shape = np.linalg.pinv(covariance)
    reach = float(np.sqrt(np.linalg.eigvalsh(covariance).max())) * 2.0
    lo = np.floor((np.minimum(pa, pb) - reach) / spacing).astype(int).clip(0)
    hi = np.minimum(np.ceil((np.maximum(pa, pb) + reach) / spacing).astype(int) + 1, mask.shape)
    grid = np.indices(hi - lo).reshape(3, -1).T + lo
    points = grid * spacing
    t = np.clip((points - pa) @ (pb - pa) / max(gap**2, 1e-12), 0, 1)
    offset = points - (pa + t[:, None] * (pb - pa))
    inside = np.einsum("ij,jk,ik->i", offset, shape, offset) <= 1.9**2
    bridged = mask.copy()
    bridged[tuple(grid[inside].T)] = True
    return bridged, round(gap, 1)


def bone_kind(label: str) -> str:
    return label.split("_")[0]  # vertebrae, rib, clavicula, scapula, ...


def settle_fragments(
    seg_dir: Path,
    labels: list[str],
    left_out: list[str] | None = None,
    given: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Sorts out the detached pieces of the bones in labels before meshing.

    A piece of a bone's mask that is not its largest is, in order: left for
    bridge_gap when it lies in line with a long bone (a gap over thin bone);
    moved to a neighbouring bone of the same kind (a vertebra, a rib) that it
    touches, when it is under REASSIGN_SHARE of the bone, because the
    segmentation gave that neighbour's voxels the wrong label; dropped when
    it is under FRAGMENT_SHARE (a chip off a process, calcified cartilage at
    a rib's end); otherwise left, and the split bone stops the build.
    Pieces are taken largest first, until what is still loose is within
    MAX_DROPPED_SHARE of the bone, the noise the manifest records: several
    chips, each under that limit, can be over it together.
    The neighbour may be a bone in left_out, one the scan cut and the body
    leaves out: its mislabelled piece leaves with it, and the note says so
    ("exported": False). A mask in given (from whole_but_for_lumps) stands
    in for the one on disk. Returns the corrected masks of the bones in
    labels and a note per piece moved or dropped.
    """
    masks, affines = {}, {}
    left_out = [lab for lab in left_out or [] if lab not in labels]
    for label in labels + left_out:
        path = seg_dir / f"{label}.nii.gz"
        if path.exists():
            image = nib.load(path)
            masks[label] = np.asanyarray(image.dataobj) > 0
            affines[label] = image.affine
            if label in (given or {}):
                masks[label] = given[label].copy()
    notes = []
    for label in [lab for lab in labels if lab in masks]:
        pieces, count = ndimage.label(masks[label], structure=np.ones((3, 3, 3)))
        if count < 2:
            continue
        sizes = np.bincount(pieces.ravel())[1:]
        total = int(sizes.sum())
        spacing = np.linalg.norm(affines[label][:3, :3], axis=0)
        voxel_ml = float(np.prod(spacing)) / 1000
        main = int(np.argmax(sizes)) + 1
        loose = 1 - sizes[main - 1] / total  # the share outside the largest piece
        for piece in sorted(set(range(1, count + 1)) - {main}, key=lambda k: -sizes[k - 1]):
            share = sizes[piece - 1] / total
            if loose <= MAX_DROPPED_SHARE:
                break  # noise at the limit the manifest already records
            part = pieces == piece
            if (
                bone_kind(label) in BRIDGEABLE
                and bridge_gap(masks[label] & ((pieces == main) | part), spacing)[1] is not None
            ):
                loose -= share
                continue  # one bone with a gap: bridged when it is meshed
            grown = ndimage.binary_dilation(part)
            contact = {
                other: int((grown & masks[other]).sum())
                for other in masks
                if other != label and bone_kind(other) == bone_kind(label)
            }
            neighbour = max(contact, key=contact.get) if contact else None
            note = {
                "label": label,
                "voxels": int(sizes[piece - 1]),
                "ml": round(sizes[piece - 1] * voxel_ml, 2),
                "share": round(float(share), 4),
            }
            if neighbour and contact[neighbour] >= NEIGHBOUR_CONTACT and share <= REASSIGN_SHARE:
                masks[neighbour] |= part
                masks[label] &= ~part
                gone = {"exported": False} if neighbour in left_out else {}
                notes.append({**note, "action": "moved", "to": neighbour, **gone})
                loose -= share
            elif share < FRAGMENT_SHARE:
                masks[label] &= ~part
                notes.append({**note, "action": "dropped"})
                loose -= share
    return {label: masks[label] for label in labels if label in masks}, notes


def whole_but_for_lumps(
    seg_dir: Path,
    reasons: dict[str, str | None],
    outside: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """The bones the scan seems to cut only because their masks carry a lump
    of a cut neighbour.

    Where a scan ends through the spine, the segmentation may give part of
    the vertebra above the label of the one below, joined to it by a thin
    neck: the lump meets the scan's end, the bone itself ends well short of
    it. For each bone in reasons that the scan cut, an opening (one voxel,
    6-connected) takes the necks away and the largest piece, grown back, is
    the bone. A piece left over is a neighbour's lump when it keeps a core of
    its own through the opening, meets the scan's end and lies on a cut bone
    of the same kind (NEIGHBOUR_CONTACT). When the lumps are within
    REASSIGN_SHARE of the mask and the rest is clear of the scan's end, the
    bone is whole and the lumps leave with the neighbour. A thin tip the scan
    ends through has no core: it is the bone's own, and the bone stays cut;
    so does a bone that is given such a lump. Returns the masks of the bones
    found whole, and a note per lump ("neck": True, "exported": False).
    """
    six = ndimage.generate_binary_structure(3, 1)
    images = {
        label: nib.load(seg_dir / f"{label}.nii.gz")
        for label, reason in reasons.items()
        if (reason or "").startswith("cut by") and (seg_dir / f"{label}.nii.gz").exists()
    }
    masks = {label: np.asanyarray(image.dataobj) > 0 for label, image in images.items()}
    whole, notes = {}, {}
    for label, mask in masks.items():
        others = [o for o in masks if o != label and bone_kind(o) == bone_kind(label)]
        if not others:
            continue
        box = tuple(
            slice(max(s.start - 2, 0), s.stop + 2)
            for s in ndimage.find_objects(mask.astype(np.uint8))[0]
        )  # the mask with two voxels round it, so the opening has room
        bone = mask[box]
        pad = np.zeros(bone.shape, bool)
        if outside is not None:
            pad = on_grid(images[label], outside)[box]
        face = np.zeros(bone.shape, bool)  # the volume's own faces, where the box reaches them
        for axis, s in enumerate(box):
            if s.start == 0:
                face[(slice(None),) * axis + (0,)] = True
            if s.stop >= mask.shape[axis]:
                face[(slice(None),) * axis + (-1,)] = True

        def meets_end(part: np.ndarray, pad: np.ndarray = pad, face: np.ndarray = face) -> bool:
            return bool((part & face).any() or (ndimage.binary_dilation(part) & pad).any())

        opened = ndimage.binary_opening(bone, structure=six)
        cores, count = ndimage.label(opened, structure=np.ones((3, 3, 3)))
        if count < 2:
            continue
        main = cores == int(np.argmax(np.bincount(cores.ravel())[1:])) + 1
        body = ndimage.binary_dilation(main, structure=six) & bone
        rest, pieces = ndimage.label(bone & ~body, structure=np.ones((3, 3, 3)))
        kept, lumps = bone.copy(), []
        for piece in range(1, pieces + 1):
            part = rest == piece
            if not (part & opened).any() or not meets_end(part):
                continue  # a thin tip, or a part the scan's end does not reach: the bone's own
            grown = ndimage.binary_dilation(part)
            contact = {o: int((grown & masks[o][box]).sum()) for o in others}
            neighbour = max(contact, key=contact.get)
            if contact[neighbour] >= NEIGHBOUR_CONTACT:
                kept &= ~part
                lumps.append((int(part.sum()), neighbour))
        total = int(bone.sum())
        if not lumps or sum(v for v, _ in lumps) > REASSIGN_SHARE * total or meets_end(kept):
            continue
        whole[label] = np.zeros_like(mask)
        whole[label][box] = kept
        voxel_ml = abs(float(np.linalg.det(images[label].affine[:3, :3]))) / 1000
        notes[label] = [
            {
                "label": label,
                "voxels": voxels,
                "ml": round(voxels * voxel_ml, 2),
                "share": round(voxels / total, 4),
                "action": "moved",
                "to": neighbour,
                "neck": True,
                "exported": False,
            }
            for voxels, neighbour in lumps
        ]
    while given := {note["to"] for label in whole for note in notes[label]} & set(whole):
        for label in given:  # it takes a lump that meets the scan's end: cut after all
            del whole[label]
    return whole, [note for label in whole for note in notes[label]]


def trim_knobs(mask: np.ndarray, voxel_ml: float) -> tuple[np.ndarray, float]:
    """Takes off lumps joined to a rib only by a thin neck: an opening (one
    voxel, 6-connected) separates them, the largest piece is grown back, and
    what is left over in pieces under KNOB_MAX_ML goes. Larger left-over
    pieces are thin parts of the rib itself and stay. Returns the mask and
    the volume taken off in ml."""
    six = ndimage.generate_binary_structure(3, 1)
    opened = ndimage.binary_opening(mask, structure=six)
    labels, count = ndimage.label(opened, structure=np.ones((3, 3, 3)))
    if count == 0:
        return mask, 0.0
    main = labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    body = ndimage.binary_dilation(main, structure=six) & mask
    rest, _ = ndimage.label(mask & ~body, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(rest.ravel())[1:]
    small = np.flatnonzero(sizes * voxel_ml < KNOB_MAX_ML) + 1
    trimmed = mask & ~np.isin(rest, small)
    return trimmed, float((mask.sum() - trimmed.sum()) * voxel_ml)


def reference_rules(reasons: dict[str, str | None]) -> dict[str, str | None]:
    """The drop rules of a reference body, applied to each label's cut_reason.

    A shoulder girdle with any bone cut is dropped on both sides, arm
    included, so a reference body is symmetric and nothing floats. Paired
    ribs go together: a rib whose partner is cut or missing is dropped too.
    Returns a new mapping; every dropped label has its reason.
    """
    reasons = dict(reasons)
    girdle = [f"{b}_{side}" for b in ("clavicula", "scapula", "humerus")
              for side in ("left", "right")]  # fmt: skip
    cut = [g for g in girdle if reasons.get(g)]
    if cut:
        for g in girdle:
            reasons[g] = reasons.get(g) or f"shoulder girdle dropped: {', '.join(cut)} cut"
    for k in range(1, 13):
        pair = (f"rib_left_{k}", f"rib_right_{k}")
        for a, b in (pair, pair[::-1]):
            if reasons.get(a) and not reasons.get(b):
                reasons[b] = f"its pair {a} is missing"
    return reasons


def landmark_ok(label: str, reason: str, faces: set[str]) -> bool:
    """Whether a dropped femur or humerus can still give its joint centre: it
    is in the scan, and the scan cuts it only at its far (inferior) end, so
    its head is whole."""
    if label not in LANDMARK_BONES or reason.startswith(("absent", "not segmented")):
        return False
    if not reason.startswith("cut by"):
        return True  # dropped by a rule (its girdle), not cut: the bone is whole
    return faces == {"I"}


def drop_small_pieces(poly: vtk.vtkPolyData, min_volume: float) -> vtk.vtkPolyData:
    """Keeps the connected pieces of poly enclosing at least min_volume (mm3)."""
    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputData(poly)
    regions.SetExtractionModeToAllRegions()
    regions.Update()
    keep = vtk.vtkAppendPolyData()
    for region in range(regions.GetNumberOfExtractedRegions()):
        one = vtk.vtkPolyDataConnectivityFilter()
        one.SetInputData(poly)
        one.SetExtractionModeToSpecifiedRegions()
        one.AddSpecifiedRegion(region)
        tidy = vtk.vtkCleanPolyData()
        tidy.SetInputConnection(one.GetOutputPort())
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(tidy.GetOutputPort())
        mass = vtk.vtkMassProperties()
        mass.SetInputConnection(triangulate.GetOutputPort())
        mass.Update()
        if mass.GetVolume() >= min_volume:
            keep.AddInputData(tidy.GetOutput())
    keep.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(keep.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def contour(image: vtk.vtkImageData, level: float) -> vtk.vtkFlyingEdges3D:
    edges = vtk.vtkFlyingEdges3D()
    edges.SetInputData(image)
    edges.SetValue(0, level)
    edges.ComputeNormalsOff()
    edges.Update()
    return edges


def finished(image: vtk.vtkImageData, level: float, pieces: bool) -> vtk.vtkPolyData:
    """Contour at level, smooth (windowed sinc), keep the largest piece (or all, for a set)."""
    edges = contour(image, level)  # kept referenced: VTK must not lose it mid-pipeline
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(edges.GetOutputPort())
    smooth.SetNumberOfIterations(30)
    smooth.SetPassBand(0.02)
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputConnection(smooth.GetOutputPort())
    if pieces:
        regions.SetExtractionModeToAllRegions()
    else:
        regions.SetExtractionModeToLargestRegion()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(regions.GetOutputPort())
    clean.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(clean.GetOutput())
    return result


def enclosed_volume(poly: vtk.vtkPolyData) -> float:
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(poly)
    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(triangulate.GetOutputPort())
    mass.Update()
    return mass.GetVolume()


def rib_axis(
    voxels_mm: np.ndarray, spacing: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """A long, thin bone's centreline as a smooth curve sampled every step mm,
    with a frame that does not twist: the points, and at each point two unit
    directions across the bone (their cross product is the tangent).

    The centreline is found by walking the mask's voxels from one end (graph
    distance over 26 neighbours) and averaging them in RIB_BIN_MM bins, then
    smoothed along its length (Gaussian, RIB_SMOOTH_MM), run on in a straight
    line for RIB_LEAD_MM past both ends, and joined by a cubic spline. The
    frame is carried along it by parallel transport. None when the bone is
    too short to have a centreline.
    """
    from scipy.interpolate import CubicSpline
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import KDTree

    tree = KDTree(voxels_mm)
    reach = float(np.sqrt(3) * spacing.max()) + 1e-3  # the 26-neighbourhood
    pairs = tree.query_pairs(r=reach, output_type="ndarray")
    if len(pairs) == 0:
        return None
    lengths = np.linalg.norm(voxels_mm[pairs[:, 0]] - voxels_mm[pairs[:, 1]], axis=1)
    n = len(voxels_mm)
    graph = coo_matrix((lengths, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    end = int(np.argmax(np.linalg.norm(voxels_mm - voxels_mm.mean(0), axis=1)))
    along = dijkstra(graph, directed=False, indices=end)
    reached = np.isfinite(along)
    bins = (along[reached] // RIB_BIN_MM).astype(int)
    count = np.bincount(bins)
    used = count > 0
    raw = np.stack([np.bincount(bins, voxels_mm[reached][:, d]) for d in range(3)], 1)
    raw = raw[used] / count[used, None]
    if len(raw) < 6:
        return None
    line = ndimage.gaussian_filter1d(raw, RIB_SMOOTH_MM / RIB_BIN_MM, axis=0, mode="nearest")
    lead = np.arange(1, int(RIB_LEAD_MM / RIB_BIN_MM) + 1)[:, None] * RIB_BIN_MM
    first, last = line[0] - line[2], line[-1] - line[-3]
    first, last = first / np.linalg.norm(first), last / np.linalg.norm(last)
    line = np.vstack([line[0] + lead[::-1] * first, line, line[-1] + lead * last])
    chord = np.linalg.norm(np.diff(line, axis=0), axis=1)
    line = line[np.concatenate([[True], chord > 1e-6])]
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    spline = CubicSpline(length, line)
    at = np.arange(0.0, length[-1], step)
    points, tangents = spline(at), spline(at, 1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    across = np.eye(3)[int(np.argmin(np.abs(tangents[0])))]
    normals = np.empty_like(points)
    for i, tangent in enumerate(tangents):
        across = across - (across @ tangent) * tangent
        across /= np.linalg.norm(across)
        normals[i] = across
    return points, normals, np.cross(tangents, normals)


def unroll(
    crop: np.ndarray, spacing: np.ndarray
) -> tuple[np.ndarray, float, Callable[[vtk.vtkPolyData], vtk.vtkPolyData]] | None:
    """A rib's mask resampled along its own centreline: axis 0 runs along the
    rib, axes 1 and 2 across it, all in steps of half a source voxel. Blurring
    that volume along axis 0 smooths the rib along its length, whichever way
    it runs through the scan; a blur along the slice axis cannot, because a
    rib crosses the slices at a slant and is thinner than the blur. Returns
    the volume, the step in mm, and a function that takes a surface of that
    volume back to the crop's millimetres; or None when the bone has no
    centreline to follow.

    Every point of space is sampled only at the place on the centreline
    nearest to it, so nothing is counted twice where the rib curls. If the
    volume then holds less than RIB_KEPT_SHARE of the mask, the centreline
    does not describe the bone and None is returned.
    """
    from scipy.spatial import KDTree

    step = float(spacing.min()) / UPSAMPLE
    voxels_mm = np.argwhere(crop > 0) * spacing
    axis = rib_axis(voxels_mm, spacing, step)
    if axis is None:
        return None
    curve, normals, binormals = axis
    tree = KDTree(curve)
    reach = float(tree.query(voxels_mm)[0].max() + 2 * spacing.max())
    half = int(np.ceil(reach / step))
    offsets = np.arange(-half, half + 1) * step
    points = (
        curve[:, None, None, :]
        + offsets[None, :, None, None] * normals[:, None, None, :]
        + offsets[None, None, :, None] * binormals[:, None, None, :]
    ).reshape(-1, 3)
    values = ndimage.map_coordinates(crop, (points / spacing).T, order=1, mode="constant")
    station = np.repeat(np.arange(len(curve)), len(offsets) ** 2)
    values[np.abs(tree.query(points)[1] - station) > 2] = 0.0
    straight = values.reshape(len(curve), len(offsets), len(offsets)).astype(np.float32)
    if straight.sum() * step**3 < RIB_KEPT_SHARE * (crop > 0).sum() * np.prod(spacing):
        return None  # the bone doubles back more tightly than its centreline can follow

    def back(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        # smoothest() puts sample j of each axis at (j - 0.5) * step.
        index = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64) / step + 0.5
        along = np.clip(index[:, 0], 0, len(curve) - 1)
        u, v = (index[:, 1] - half) * step, (index[:, 2] - half) * step
        stations = np.arange(len(curve))
        frame = [
            np.stack([np.interp(along, stations, a[:, d]) for d in range(3)], 1)
            for a in (curve, normals, binormals)
        ]
        moved = vtk.vtkPolyData()
        moved.DeepCopy(poly)
        mm = frame[0] + u[:, None] * frame[1] + v[:, None] * frame[2]
        moved.GetPoints().SetData(numpy_to_vtk(mm, deep=True))
        return moved

    return straight, step, back


def smoothest(
    fine: np.ndarray, spacing: np.ndarray, blur, target: float, pieces: bool, post=None
) -> vtk.vtkPolyData:
    """The finished surface of the upsampled mask blurred by `blur` source voxels
    (one value, or one per axis), at the contour level whose finished volume is
    the target (bisection)."""
    field = ndimage.gaussian_filter(fine, np.asarray(blur, dtype=float) * UPSAMPLE)
    image = vtk.vtkImageData()
    image.SetDimensions(*field.shape)
    image.SetSpacing(*(spacing / UPSAMPLE))
    image.SetOrigin(*(-spacing * (UPSAMPLE - 1) / (2 * UPSAMPLE)))
    image.GetPointData().SetScalars(numpy_to_vtk(field.ravel(order="F"), deep=True))
    # The level is chosen on the finished surface (contoured, smoothed and cut
    # to the kept pieces), so neither the blur nor the smoothing moves its volume.
    low, high = 0.05, 0.95
    for _ in range(LEVEL_STEPS):
        level = (low + high) / 2
        candidate = finished(image, level, pieces)
        if enclosed_volume(post(candidate) if post else candidate) > target:
            low = level
        else:
            high = level
    result = finished(image, (low + high) / 2, pieces)
    return post(result) if post else result


def surface(
    path: Path,
    max_triangles: int,
    pieces: bool = False,
    along_slices: bool = False,
    centreline: bool = False,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (vertices in RAS mm, triangles, voxel volume in ml, dropped share).

    The dropped share is the part of the mask's voxels outside its largest
    piece, which is the only one kept; for a set of pieces it is 0.

    The mask is upsampled 2x and blurred (Gaussian, one source voxel; half or a
    quarter where a thin, lobed structure would come apart), so the surface
    follows a smooth field instead of voxel steps. The contour level is
    then chosen, by bisection, so the surface encloses the mask's own voxel
    volume: a fixed 0.5 would shrink a thin bone such as a rib by up to 12%.
    A light windowed-sinc pass (Taubin-like, volume-keeping) removes the rest
    of the noise.

    With along_slices the blur is stronger along the slice axis (flat bones
    and vertebrae). With centreline (a rib) the mask is first resampled along
    the bone's own centreline and blurred along that (RIB_BLURS, see
    unroll()), which takes the slice-to-slice ripple off a rib wherever it
    runs and rounds its ends; a bone too short or too curled for that is
    surfaced as with along_slices.
    """
    image = nib.load(path)
    data = mask if mask is not None else np.asanyarray(image.dataobj) > 0
    if not data.any():
        raise ValueError(f"{path.name} is empty")
    # The volume to keep is that of the pieces the mesh keeps: the largest, or,
    # for a set, every piece of at least MIN_PIECE_VOXELS. Stray voxels a
    # segmentation leaves elsewhere are dropped and do not count.
    labels, count = ndimage.label(data, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= MIN_PIECE_VOXELS) + 1 if pieces else [int(sizes.argmax()) + 1]
    data = np.isin(labels, keep)  # the contour level must be tuned on what is kept
    dropped = 0.0 if pieces else 1 - float(sizes.max()) / float(sizes.sum())
    lo = np.maximum(np.argwhere(data).min(0) - 3, 0)
    hi = np.argwhere(data).max(0) + 4
    crop = np.pad(data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], PAD).astype(np.float32)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    target = float(data.sum() * np.prod(spacing))

    # grid_mode keeps voxel centres aligned: fine sample j sits at (j + 0.5) / 2 - 0.5.
    fine = ndimage.zoom(crop, UPSAMPLE, order=1, grid_mode=True, mode="nearest")
    if along_slices:
        # Flat and long bones (sternum, ribs, vertebrae) show the segmentation's
        # slice-to-slice wobble as ripples, so they are blurred more along the
        # slice axis than in-plane.
        axis = int(np.argmax(np.abs(image.affine[2, :3])))
        label = path.name.removesuffix(".nii.gz")
        ladder = {"sternum": STERNUM_BLURS, "costal_cartilages": CARTILAGE_BLURS}.get(
            label, SLICE_BLURS
        )
        blurs = [tuple(z if a == axis else xy for a in range(3)) for xy, z in ladder]
    else:
        blurs = list(BLUR_VOXELS)
    unrolled = unroll(crop, spacing) if centreline else None
    if unrolled is not None:
        # A rib is smoothed along its own length instead: see unroll().
        fine, step, back = unrolled
        samples = np.full(3, step * UPSAMPLE)  # smoothest() halves this again
        tries = [
            (samples, np.array([along, across, across]) / samples, back)
            for across, along in RIB_BLURS
        ]
    else:
        tries = [(spacing, blur, None) for blur in blurs]
    for grid, blur, post in tries:
        poly = smoothest(fine, grid, blur, target, pieces, post)
        if abs(enclosed_volume(poly) / target - 1) <= BLUR_TOLERANCE:
            break  # a thin, lobed organ (the pancreas) can need a lighter blur
    if pieces:
        poly = drop_small_pieces(poly, MIN_PIECE_VOXELS * float(np.prod(spacing)))

    if poly.GetNumberOfCells() > max_triangles:
        poly = decimate(poly, max_triangles)

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    triangles = triangles_of(poly)
    # The padded crop's first voxel sits at the origin.
    ijk = points / spacing + lo - PAD
    ras = ijk @ image.affine[:3, :3].T + image.affine[:3, 3]
    return ras, triangles.astype(np.uint32), target / 1000, dropped


def outward(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orients triangles outward and returns (triangles, vertex normals)."""
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices.astype(np.float32), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.int64)
    cells.SetData(
        numpy_to_vtk(offsets, deep=True),
        numpy_to_vtk(triangles.ravel().astype(np.int64), deep=True),
    )
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.Update()
    out = normals.GetOutput()
    tris = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.uint32)
    return tris, vtk_to_numpy(out.GetPointData().GetNormals()).astype(np.float32)


def open_edges(triangles: np.ndarray) -> int:
    """Edges used by other than exactly two triangles: holes or non-manifold seams."""
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts != 2).sum())


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def credit(case: str) -> dict:
    """The attribution a GLB built from case carries: creator, DOI, licence, case."""
    if case not in SOURCES["cases"]:
        sys.exit(f"{case} is not in sources.json, so its source cannot be credited")
    source = SOURCES["sources"][SOURCES["cases"][case]["source"]]
    return {
        "copyright": (
            f"Derived from case {case} of the {source['title']} by {source['creator']}, "
            f"doi:{source['doi']}, licensed {source['licence']}"
        ),
        "extras": {
            "source": source["title"],
            "creator": source["creator"],
            "doi": source["doi"],
            "licence": source["licence"],
            "case": case,
        },
    }


def write_glb(path: Path, parts: list[dict], attribution: dict | None = None) -> None:
    """Writes one node and mesh per part, each with its own material.

    attribution (from credit()) goes into asset.copyright and asset.extras.
    """
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "open-anatomy mesh.py", **(attribution or {})},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(parts)))}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
    }
    blob = bytearray()

    def add(array: np.ndarray, target: int, component: int, kind: str) -> int:
        while len(blob) % 4:
            blob.append(0)
        gltf["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes, "target": target}
        )
        blob.extend(array.tobytes())
        accessor = {
            "bufferView": len(gltf["bufferViews"]) - 1,
            "componentType": component,
            "count": len(array) if kind != "SCALAR" else array.size,
            "type": kind,
        }
        if kind == "VEC3" and component == 5126:
            accessor["min"] = array.min(0).tolist()
            accessor["max"] = array.max(0).tolist()
        gltf["accessors"].append(accessor)
        return len(gltf["accessors"]) - 1

    for i, part in enumerate(parts):
        s: Structure = part["structure"]
        position = add(part["vertices"], 34962, 5126, "VEC3")
        normal = add(part["normals"], 34962, 5126, "VEC3")
        indices = add(part["triangles"].ravel(), 34963, 5125, "SCALAR")
        gltf["materials"].append(
            {
                "name": s.name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*s.colour, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
            }
        )
        gltf["meshes"].append(
            {
                "name": node_name(s.term),
                "primitives": [
                    {
                        "attributes": {"POSITION": position, "NORMAL": normal},
                        "indices": indices,
                        "material": i,
                    }
                ],
            }
        )
        gltf["nodes"].append(
            {
                "name": node_name(s.term),
                "mesh": i,
                "extras": {
                    "term": s.term,
                    "name": s.name,
                    "system": s.system,
                    "licence": SOURCE_LICENCE,
                    **({"modelled": True} if part.get("modelled") else {}),
                },
            }
        )

    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"].append({"byteLength": len(blob)})
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--seg-dir", type=Path, help="default: data/work/<case>/seg")
    parser.add_argument("--out-dir", type=Path, help="default: data/work/<case>")
    parser.add_argument("--set", choices=SETS, default="organs", help="structures to mesh")
    parser.add_argument("--max-triangles", type=int, default=40000, help="per organ")
    parser.add_argument("--max-bone-triangles", type=int, default=15000, help="per bone")
    parser.add_argument(
        "--synthetic", action="store_true", help="test masks with no source to credit"
    )
    parser.add_argument("--ct", type=Path, help="default: data/cases/<case>/ct.nii.gz")
    args = parser.parse_args()
    attribution = (
        {"copyright": "Synthetic test data", "extras": {"case": args.case}}
        if args.synthetic
        else credit(args.case)
    )

    work = args.out_dir or ROOT / "data" / "work" / args.case
    work.mkdir(parents=True, exist_ok=True)
    seg_dir = args.seg_dir or work / "seg"
    parts, missing, dropped, landmarks = [], [], [], []
    labels = [s.label for s in SETS[args.set]]
    ct_path = args.ct or ROOT / "data" / "cases" / args.case / "ct.nii.gz"
    if not ct_path.exists() and not args.synthetic:
        sys.exit(f"{ct_path} is missing: the CT is needed to see where the scan ends")
    outside = scan_outside(ct_path) if ct_path.exists() else None
    reasons = {label: cut_reason(seg_dir / f"{label}.nii.gz", outside) for label in labels}
    settled, fragments = {}, []
    if args.set in RULED_SETS:
        # A bone that meets the scan's end only with a cut neighbour's lump is whole.
        unmixed, lumps = whole_but_for_lumps(seg_dir, reasons, outside)
        reasons.update(dict.fromkeys(unmixed))
        reasons = reference_rules(reasons)
        bones = [
            s.label
            for s in SETS[args.set]
            if s.system == "skeletal" and s.label not in SETS_OF_PIECES and not reasons[s.label]
        ]  # a set of pieces (the segmented cartilage) is many pieces by nature
        # A bone the scan cut is left out, and takes its mislabelled pieces along.
        cut = [
            s.label
            for s in SETS[args.set]
            if s.system == "skeletal" and s.label not in SETS_OF_PIECES
            and (reasons[s.label] or "").startswith(("cut by", "shoulder girdle", "its pair"))
        ]  # fmt: skip
        settled, fragments = settle_fragments(seg_dir, bones, cut, unmixed)
        fragments = lumps + fragments
        for note in fragments:
            print(f"{note['label']}: {note['ml']} ml {note['action']}"
                  + (f" to {note['to']}" if note.get("to") else "")
                  + (" (left out)" if note.get("exported") is False else ""),
                  file=sys.stderr)  # fmt: skip
    for s in SETS[args.set]:
        bone = s.system == "skeletal"
        cap = args.max_bone_triangles if bone else args.max_triangles
        path = seg_dir / f"{s.label}.nii.gz"
        if s.label == "costal_cartilages" and args.set in RULED_SETS:
            # The segmented cartilage is too lumpy and broken to show, and it
            # may be cut where the ribs are not; the body carries modelled
            # cartilages instead, so its own cut status does not matter.
            # They start from the ribs as meshed above: the ends the body shows.
            to_ras = np.linalg.inv(RAS_TO_GLTF).T
            ribs = {
                p["structure"].label: (p["vertices"] @ to_ras, p["triangles"])
                for p in parts
                if p["structure"].label.startswith("rib_")
            }
            ras, triangles, skipped = modelled_cartilage(seg_dir / "sternum.nii.gz", ribs)
            dropped += [
                {"label": f"costal_cartilages:{rib}", "reason": why} for rib, why in skipped
            ]
            if len(triangles) == 0:
                dropped.append({"label": s.label, "reason": "no rib ends to model from"})
                continue
            parts.append({
                "structure": s, "vertices": ras @ RAS_TO_GLTF.T, "triangles": triangles,
                "voxel_ml": None, "modelled": True,
            })  # fmt: skip
            continue
        reason = reasons[s.label] if args.set in RULED_SETS else None
        if reason and not bone:
            sys.exit(f"{s.label}: {reason} in {args.case}; organs must be whole")
        if reason:
            # Never exported, never capped; the manifest says which side of
            # the body the scan cut it at (S for a shoulder the scan ends
            # through). The femurs and humeri are still meshed into the
            # landmarks file for their joint centres.
            dropped.append({"label": s.label, "reason": reason})
            faces = cut_faces(path, outside) if reason.startswith("cut by") else set()
            if faces:
                dropped[-1]["cut_at"] = "".join(sorted(faces))
            if s.label in LANDMARK_BONES and not reason.startswith(("absent", "not segmented")):
                if landmark_ok(s.label, reason, faces):
                    landmarks.append(s)
                else:
                    # A head cut by the scan would fit a joint to the cut face.
                    dropped[-1]["landmark"] = f"not used: cut at {''.join(sorted(faces))}"
            continue
        knobless, trimmed_ml, bridged_mm = settled.get(s.label), 0.0, None
        if bone and path.exists():
            image = nib.load(path)
            spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
            if knobless is None:
                knobless = np.asanyarray(image.dataobj) > 0
            if bone_kind(s.label) in BRIDGEABLE:
                # One long bone split over thin bone is bridged before anything
                # else, or the knob trim would eat the far piece.
                knobless, bridged_mm = bridge_gap(knobless, spacing)
            if s.label.startswith("rib_"):
                knobless, trimmed_ml = trim_knobs(knobless, float(np.prod(spacing)) / 1000)
        try:
            ras, triangles, voxel_ml, share = surface(
                path, cap, pieces=s.label in SETS_OF_PIECES,
                along_slices=s.label.startswith(SLICE_SMOOTHED),
                centreline=s.label.startswith("rib_"),
                mask=knobless,
            )  # fmt: skip
        except ValueError:
            if not bone:
                sys.exit(f"{s.label}: the mask is empty in {args.case}; organs must be whole")
            missing.append(s.label)
            continue
        if share > MAX_DROPPED_SHARE:
            sys.exit(
                f"{s.label}: {share:.1%} of its volume is outside its largest piece in "
                f"{args.case} (at most {MAX_DROPPED_SHARE:.0%}); a split structure stops the build"
            )
        parts.append({
            "structure": s, "vertices": ras @ RAS_TO_GLTF.T, "triangles": triangles,
            "voxel_ml": voxel_ml, "trimmed_ml": round(trimmed_ml, 2), "dropped_share": share,
            "bridged_mm": bridged_mm,
        })  # fmt: skip

    everything = np.vstack([p["vertices"] for p in parts])
    centre = (everything.min(0) + everything.max(0)) / 2
    for p in parts:
        p["vertices"] = (p["vertices"] - centre).astype(np.float32)
        p["triangles"], p["normals"] = outward(p["vertices"], p["triangles"])
        name = p["structure"].label
        if open_edges(p["triangles"]):
            sys.exit(f"{name}: the surface is not closed and manifold")
        if signed_volume(p["vertices"], p["triangles"]) < 0:
            # VTK's auto-orientation can miss on a ragged fragment; a closed
            # surface with negative volume is simply wound the other way round.
            p["triangles"], p["normals"] = p["triangles"][:, ::-1].copy(), -p["normals"]
        if signed_volume(p["vertices"], p["triangles"]) <= 0:
            sys.exit(f"{name}: the normals point inward")
        if p.get("modelled"):
            p["drift"] = None  # modelled, not segmented: there is no voxel volume
            continue
        p["drift"] = signed_volume(p["vertices"], p["triangles"]) * 1e6 / p["voxel_ml"] - 1
        if abs(p["drift"]) > MAX_VOLUME_DRIFT:
            sys.exit(f"{name}: the mesh's volume is {p['drift']:+.1%} off its voxel volume")

    glb = work / f"{args.case}.{args.set}.glb"
    write_glb(glb, parts, attribution)
    outline_path = seg_dir / "body_trunc.nii.gz"
    if args.set in RULED_SETS and outline_path.exists():
        # The trunk outline, in the same frame, for assemble.py to measure. It is
        # the outer surface of a person's trunk, so it is never released.
        ras, triangles, _, _ = surface(outline_path, args.max_triangles)
        vertices = (ras @ RAS_TO_GLTF.T - centre).astype(np.float32)
        triangles, normals = outward(vertices, triangles)
        outline = {"structure": OUTLINE, "vertices": vertices, "triangles": triangles}
        outline["normals"] = normals
        write_glb(work / f"{args.case}.outline.glb", [outline], attribution)
    manifest = {
        "case": args.case,
        "set": args.set,
        "file": glb.name,
        "sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
        "attribution": attribution["copyright"],
        "units": "metres",
        "axes": "+Y superior, +Z anterior, +X the body's left",
        "centre_offset_m": [round(float(c), 6) for c in centre],
        "seg_dir": str(seg_dir.relative_to(ROOT)) if seg_dir.is_relative_to(ROOT) else str(seg_dir),
        "nodes": [
            {
                "name": node_name(p["structure"].term),
                "term": p["structure"].term,
                "label": p["structure"].name,
                "system": p["structure"].system,
                "licence": SOURCE_LICENCE,
                "triangles": int(len(p["triangles"])),
                "volume_ml": round(signed_volume(p["vertices"], p["triangles"]) * 1e6, 1),
                "voxel_volume_ml": round(p["voxel_ml"], 1) if p["voxel_ml"] else None,
                "modelled": bool(p.get("modelled")),
                "pieces": "several" if p["structure"].label in SETS_OF_PIECES else "one",
                "volume_drift": round(p["drift"], 4) if p["drift"] is not None else None,
                **({"knobs_trimmed_ml": p["trimmed_ml"]} if p.get("trimmed_ml") else {}),
                **({"dropped_share": round(p["dropped_share"], 4)} if "dropped_share" in p else {}),
                **({"bridged_gap_mm": p["bridged_mm"]} if p.get("bridged_mm") else {}),
            }
            for p in parts
        ],
    }
    manifest["missing"] = missing
    manifest["fragments"] = fragments
    manifest["dropped"] = dropped
    if landmarks:
        # Cut bones, measured for their joint centres (femoral and humeral
        # heads) and never released, like the trunk outline.
        marks = []
        for s in landmarks:
            ras, triangles, _, _ = surface(seg_dir / f"{s.label}.nii.gz", args.max_bone_triangles)
            vertices = (ras @ RAS_TO_GLTF.T - centre).astype(np.float32)
            triangles, normals = outward(vertices, triangles)
            marks.append({"structure": s, "vertices": vertices, "triangles": triangles,
                          "normals": normals})  # fmt: skip
        write_glb(work / f"{args.case}.landmarks.glb", marks, attribution)
    (work / f"{args.case}.{args.set}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for node in manifest["nodes"]:
        print(f"{node['name']}  {node['label']:20} {node['triangles']:>6} tris  "
              f"{node['volume_ml']:>8} ml")  # fmt: skip
    print(f"{glb}  sha256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
