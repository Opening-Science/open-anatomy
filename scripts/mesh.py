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
from pathlib import Path

os.environ.setdefault("VTK_SMP_BACKEND_IN_USE", "Sequential")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import vtk  # noqa: E402
from scipy import ndimage  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import SETS, Structure, node_name  # noqa: E402

# RAS millimetres to glTF metres: X = -R (the body's left), Y = S, Z = A.
# Limb bones run out of a trunk scan by design; their joint ends are what count.
MIN_ML = 0.3  # a mask smaller than this is a stray label, not a bone (volume, not voxels)
PADDING_HU = -1020  # at or below: a scanner's fill beyond what it scanned, never tissue
SETS_OF_PIECES = {"costal_cartilages"}  # one node, several separate pieces
MIN_PIECE_VOXELS = 150  # a piece smaller than this (0.5 ml) in a set is a loose blob
LEVEL_STEPS = 12
SLICE_SMOOTHED = ("sternum", "rib_", "vertebrae_", "costal_cartilages", "sacrum")
SLICE_BLURS = ((0.5, 1.5), (0.5, 1.0), (0.25, 0.5))  # (in-plane, along slices), source voxels
STERNUM_BLURS = ((1.0, 3.0), (0.75, 2.0), (0.5, 1.5))  # its ripples span several slices
CARTILAGE_BLURS = ((1.0, 1.5), (0.75, 1.0), (0.5, 0.75))  # thin, lumpy segmentation
CARTILAGE_MAX_MM = 120  # the 6th and 7th costal cartilages run 10-12 cm
JOIN_OVERLAP_MM = 6.0  # a modelled cartilage starts this far inside its rib's end
CARTILAGE_FLAT = 2.5  # a cartilage band is this many times wider than deep
ARCH_JOIN = {8: 0.55, 9: 0.5, 10: 0.45}  # where along the cartilage above ribs 8-10 join it
TANGENT_SHARE = 0.7  # of the rib-to-sternum span, for the cartilage curve's end tangents
KNOB_MAX_ML = 1.0  # a lump the opening separates from a rib is taken off below this volume
RIB_BIN_MM = 3.0
RIB_SMOOTH_MM = 9.0
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


def rib_end(path: Path) -> dict | None:
    """rib_end_of the rib's mask, or of the mask after a small opening when that
    takes a knob off its tip: the opened rib keeps at least 85% of its voxels,
    its tip moves under 20 mm, and its tip is wider than the raw one's."""
    image = nib.load(path)
    mask = np.asanyarray(image.dataobj) > 0
    raw = rib_end_of(mask, image.affine)
    opened = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1))
    labels, count = ndimage.label(opened, structure=np.ones((3, 3, 3)))
    if raw is None or count == 0:
        return raw
    largest = labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    cleaned = rib_end_of(largest, image.affine)
    if (
        cleaned is not None
        and largest.sum() >= 0.85 * mask.sum()
        and np.linalg.norm(cleaned["tip"] - raw["tip"]) < 20
        and cleaned["halfWidth"] > raw["halfWidth"]
    ):
        return cleaned
    return raw


def rib_end_of(mask: np.ndarray, affine: np.ndarray) -> dict | None:
    """A rib's front end in RAS mm: its tip, the unit direction it runs in
    there, and its cross-section just behind the tip (3 to 10 mm back): the
    direction across its width and its half-width and half-thickness. From
    the rib's centreline (the mask walked from its far end in 3 mm bins)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    voxels = np.argwhere(mask)
    if len(voxels) * abs(np.linalg.det(affine[:3, :3])) / 1000 < MIN_ML:
        return None
    ras = voxels @ affine[:3, :3].T + affine[:3, 3]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    pairs = cKDTree(ras).query_pairs(r=float(np.sqrt(3) * spacing.max()) + 1e-3,
                                      output_type="ndarray")  # fmt: skip
    graph = coo_matrix(
        (np.linalg.norm(ras[pairs[:, 0]] - ras[pairs[:, 1]], axis=1), (pairs[:, 0], pairs[:, 1])),
        shape=(len(ras), len(ras)),
    ).tocsr()
    start = int(np.argmax(np.linalg.norm(ras - ras.mean(0), axis=1)))
    along = dijkstra(graph, directed=False, indices=start)
    ok = np.isfinite(along)
    bins = (along[ok] // RIB_BIN_MM).astype(int)
    count = np.bincount(bins)
    used = count > 0
    line = (
        np.stack([np.bincount(bins, ras[ok][:, d]) for d in range(3)], 1)[used] / count[used, None]
    )
    if len(line) < 6:
        return None
    line = ndimage.gaussian_filter1d(line, 1.5, axis=0, mode="nearest")
    if line[0, 1] > line[-1, 1]:  # RAS +A is the front: the front end is the last point
        line = line[::-1]
    tip = line[-1]
    direction = line[-1] - line[-4]
    direction /= np.linalg.norm(direction)
    back = (ras - tip) @ direction
    slab = ras[(back > -10) & (back < -3)]
    if len(slab) < 10:
        return None
    flat = slab - np.outer((slab - tip) @ direction, direction)
    flat -= flat.mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    across = vt[0] - direction * (vt[0] @ direction)
    across /= np.linalg.norm(across)
    deep = np.cross(direction, across)
    half_voxel = spacing.mean() / 2
    width = float(np.ptp(np.percentile(flat @ across, [5, 95]))) / 2 + half_voxel
    depth = float(np.ptp(np.percentile(flat @ deep, [5, 95]))) / 2 + half_voxel
    return {"tip": tip, "direction": direction, "across": across,
            "halfWidth": max(width, 3.0), "halfDepth": max(depth, 2.0)}  # fmt: skip


def sweep(curve: np.ndarray, start_across: np.ndarray, half_w: np.ndarray, half_d: np.ndarray,
          sides: int = 24) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """A closed tube of elliptical section along a curve: a rotation-minimising
    frame (double reflection) carries the section's width direction from the
    start, and both ends are capped."""
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    across = [start_across - tangents[0] * (start_across @ tangents[0])]
    across[0] /= np.linalg.norm(across[0])
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
    deep = np.cross(tangents, across)
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = (curve[:, None, :]
             + (half_w[:, None] * np.cos(angle))[..., None] * across[:, None, :]
             + (half_d[:, None] * np.sin(angle))[..., None] * deep[:, None, :])  # fmt: skip
    vertices = np.vstack([rings.reshape(-1, 3), curve[0], curve[-1]])
    n, triangles = len(curve), []
    for i in range(n - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            c, d = a + sides, b + sides
            triangles += [(a, c, b), (b, c, d)]
    first, last = n * sides, n * sides + 1
    for k in range(sides):
        triangles.append((first, k, (k + 1) % sides))
        a, b = (n - 1) * sides + k, (n - 1) * sides + (k + 1) % sides
        triangles.append((last, b, a))
    return vertices, np.array(triangles, dtype=np.uint32)


def modelled_cartilage(
    seg_dir: Path, kept_ribs: list[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Costal cartilages modelled, not segmented, for ribs 1 to 10 in left-right
    pairs, as flat bands (about CARTILAGE_FLAT times wider than deep). Each
    leaves its rib along the rib's own end direction with the rib's own
    cross-section (no step at the join), then takes the pair's mean section so
    left and right match. Ribs 1 to 7 turn medially and rise into the sternum
    (more for lower ribs), widening about 30% on the way, and end at the
    sternum's edge on their own side. Ribs 8 to 10 rise to join the cartilage
    above them (ARCH_JOIN says where along it), which makes the costal arch.
    Returns RAS mm vertices and triangles, and each kept rib that gets no
    cartilage with the reason; the manifest marks the node as modelled."""
    from scipy.spatial import cKDTree

    sternum_image = nib.load(seg_dir / "sternum.nii.gz")
    sternum = np.argwhere(np.asanyarray(sternum_image.dataobj) > 0)
    sternum = sternum @ sternum_image.affine[:3, :3].T + sternum_image.affine[:3, 3]
    tree = cKDTree(sternum)
    midline = float(np.median(sternum[:, 0]))
    ends, skipped = {}, []
    for rib in kept_ribs:
        number = int(rib.rsplit("_", 1)[1])
        if number > 10:
            continue  # ribs 11 and 12 float: no cartilage reaches the arch
        end = rib_end(seg_dir / f"{rib}.nii.gz")
        if end is None:
            skipped.append((rib, "its front end could not be measured"))
        elif number <= 7 and tree.query(end["tip"])[0] > CARTILAGE_MAX_MM:
            skipped.append((rib, f"its end is over {CARTILAGE_MAX_MM} mm from the sternum"))
        elif number <= 7 and tree.query(end["tip"])[0] < 1.0:
            skipped.append((rib, "its end already touches the sternum"))
        else:
            ends[rib] = end

    def partner_of(r: str) -> str:
        return r.replace("left", "right") if "left" in r else r.replace("right", "left")

    paired = [r for r in ends if partner_of(r) in ends]
    skipped += [(r, "its pair has no cartilage") for r in ends if r not in paired]
    bands: dict[str, dict] = {}  # curve, half widths and depths per rib, for the arch
    vertices, triangles = [], []
    for rib in sorted(paired, key=lambda r: (int(r.rsplit("_", 1)[1]), r)):
        e, partner = ends[rib], ends[partner_of(rib)]
        number = int(rib.rsplit("_", 1)[1])
        side = np.sign(e["tip"][0] - midline)
        p0 = e["tip"] - e["direction"] * JOIN_OVERLAP_MM  # start well inside the rib's end
        w_pair = (e["halfWidth"] + partner["halfWidth"]) / 2
        if number <= 7:
            # Aim at the sternum's border on the rib's own side, so the lowest
            # pairs meet it side by side instead of crossing at the xiphoid.
            own = sternum[(sternum[:, 0] - midline) * side >= 0]
            p1 = own[cKDTree(own).query(e["tip"])[1]]
            rise = 0.15 + 0.2 * (number - 2)  # lower cartilages rise more steeply
            t1 = np.array([-side, 0.0, rise])
            t1 /= np.linalg.norm(t1)
            p1 = p1 + t1 * 2.0  # end just inside the sternum
            # But no further in than the sternum's midline at that height less the
            # band's own half-width, so a left and a right end never overlap.
            level = np.abs(sternum[:, 2] - p1[2]) < 10
            centre = float(np.median(sternum[level, 0])) if level.any() else midline
            if (p1[0] - centre) * side < w_pair * 1.3:
                p1[0] = centre + side * w_pair * 1.3
            end_w, end_d, widen = w_pair * 1.3, w_pair * 1.3 / CARTILAGE_FLAT, 0.3
        else:
            above = bands.get(rib.replace(f"_{number}", f"_{number - 1}"))
            if above is None:
                skipped.append((rib, "no cartilage above it to join"))
                continue
            k = int(round(ARCH_JOIN[number] * (len(above["curve"]) - 1)))
            p1 = above["curve"][k]
            t1 = above["curve"][min(k + 1, len(above["curve"]) - 1)] - above["curve"][k - 1]
            t1 /= np.linalg.norm(t1)
            end_w, end_d, widen = above["halfW"][k], above["halfD"][k], 0.1
        span = np.linalg.norm(p1 - p0)
        if span < 1.0 or abs(float(e["direction"] @ (p1 - p0))) < 1e-6 * span:
            skipped.append((rib, "its end lies on its target"))
            continue
        s = np.linspace(0, 1, 32)[:, None]
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        reach = TANGENT_SHARE * span  # longer tangents overshoot into a hook on short spans
        curve = h00 * p0 + h10 * e["direction"] * reach + h01 * p1 + h11 * t1 * reach
        f = s[:, 0]
        blend = np.clip(f / 0.35, 0, 1)  # from the rib's own section to the pair's within a third
        w_mid, d_mid = w_pair, w_pair / CARTILAGE_FLAT
        half_w = (1 - blend) * e["halfWidth"] + blend * w_mid
        half_d = (1 - blend) * e["halfDepth"] + blend * d_mid
        half_w = half_w * (1 - f**2) + f**2 * end_w * (1 + widen - 0.3)
        half_d = half_d * (1 - f**2) + f**2 * end_d
        bands[rib] = {"curve": curve, "halfW": half_w, "halfD": half_d}
        # Round the far end into a dome, so an end that shows beside the
        # sternum (as at the xiphoid) is not a flat cut face. A band joining
        # the cartilage above ends inside it, so its dome stays short.
        tip = curve[-1] - curve[-2]
        tip /= np.linalg.norm(tip)
        angle = np.linspace(0, 0.45 * np.pi, 7)[1:]
        depth = min(half_w[-1], half_d[-1]) * (1.0 if number <= 7 else 0.6)
        curve = np.vstack([curve, curve[-1] + np.outer(depth * np.sin(angle), tip)])
        half_w = np.concatenate([half_w, half_w[-1] * np.cos(angle)])
        half_d = np.concatenate([half_d, half_d[-1] * np.cos(angle)])
        v, t = sweep(curve, e["across"], half_w, half_d)
        triangles.append(t + sum(len(x) for x in vertices))
        vertices.append(v)
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint32), skipped
    return np.vstack(vertices), np.vstack(triangles).astype(np.uint32), skipped


def scan_outside(ct_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The part of a CT volume that was never scanned: the constant fill a
    scanner writes beyond its field of view (a circle in-plane, a slanted edge
    from a tilted gantry, empty end slices). Voxels at or below PADDING_HU
    (never tissue; air is about -1000 HU with noise) in regions that survive a
    small opening and join the volume's border, grown back through the fill
    so the thin end of a gantry wedge stays outside too. Returns the mask and
    the CT's affine, so masks can be checked against the same grid."""
    image = nib.load(ct_path)
    fill = np.asanyarray(image.dataobj) <= PADDING_HU
    pad = ndimage.binary_opening(fill, structure=np.ones((3, 3, 3)))
    labels, _ = ndimage.label(pad)
    border = np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])  # fmt: skip
    kept = np.isin(labels, np.setdiff1d(np.unique(border), [0]))
    return ndimage.binary_propagation(kept, mask=fill), image.affine


def cut_reason(path: Path, outside: tuple[np.ndarray, np.ndarray] | None = None) -> str | None:
    """Why a mask cannot be a whole structure, or None if it is whole.

    Smaller than MIN_ML (measured with the voxel size from the affine, so the
    limit is the same at 0.8 mm and 1.5 mm), touching any face of the volume
    (the first or last slice, or the in-plane border), or touching the part of
    the volume that was never scanned (outside, from scan_outside): the scan
    cut it. A mask on another grid than the CT's raises ValueError.
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
    unscanned padding (outside), the side is the direction from the mask's
    centre to where it meets it."""
    image = nib.load(path)
    data = np.asanyarray(image.dataobj) > 0
    faces = set()
    for axis in range(3):
        k = np.flatnonzero(data.any(tuple(a for a in range(3) if a != axis)))
        if len(k) == 0:
            continue
        column = image.affine[:3, axis]
        ras = int(np.argmax(np.abs(column)))
        up, down = "RAS"[ras], "LPI"[ras]  # the RAS axis's positive and negative ends
        high, low = (up, down) if column[ras] > 0 else (down, up)
        if k[0] == 0:
            faces.add(low)
        if k[-1] == data.shape[axis] - 1:
            faces.add(high)
    if outside is not None:
        contact = np.argwhere(ndimage.binary_dilation(data) & on_grid(image, outside))
        if len(contact):
            towards = image.affine[:3, :3] @ (contact.mean(0) - np.argwhere(data).mean(0))
            ras = int(np.argmax(np.abs(towards)))
            faces.add("RAS"[ras] if towards[ras] > 0 else "LPI"[ras])
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


def straighten(fine: np.ndarray, voxels_mm: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Removes steps along a long, thin bone (a rib) from its upsampled mask.

    The centreline is found by walking the mask's voxels from one end (graph
    distance over 26 neighbours) and averaging them in RIB_BIN_MM bins, then
    smoothed along its length (Gaussian, RIB_SMOOTH_MM). The mask is warped by
    that smoothing's shift, interpolated along the walk distance of each fine
    sample's nearest voxel, before it is blurred and contoured, so a sideways
    step between scan slices (often breathing) becomes a gentle bend and the
    surface is never sheared.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    tree = cKDTree(voxels_mm)
    reach = float(np.sqrt(3) * spacing.max()) + 1e-3  # the 26-neighbourhood
    pairs = tree.query_pairs(r=reach, output_type="ndarray")
    if len(pairs) == 0:
        return fine
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
    if len(raw) < 4:
        return fine
    shift = ndimage.gaussian_filter1d(raw, RIB_SMOOTH_MM / RIB_BIN_MM, axis=0, mode="nearest") - raw
    stations = (np.flatnonzero(used) + 0.5) * RIB_BIN_MM

    # Fine sample j sits at source index (j + 0.5) / UPSAMPLE - 0.5, i.e. in mm:
    grid = np.indices(fine.shape).reshape(3, -1).T
    mm = ((grid + 0.5) / UPSAMPLE - 0.5) * spacing
    nearest = tree.query(mm)[1]
    t = np.where(reached[nearest], along[nearest], stations[-1])
    move = np.stack([np.interp(t, stations, shift[:, d]) for d in range(3)], 1)
    # A sample at x takes the value from x - move: the shape moves by +move.
    source = (mm - move) / spacing
    source = (source + 0.5) * UPSAMPLE - 0.5
    warped = ndimage.map_coordinates(fine, source.T, order=1, mode="constant")
    return warped.reshape(fine.shape).astype(np.float32)


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
    if centreline:
        fine = straighten(fine, np.argwhere(crop > 0) * spacing, spacing)
    for blur in blurs:
        poly = smoothest(fine, spacing, blur, target, pieces)
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
    if args.set == "reference":
        reasons = reference_rules(reasons)
    for s in SETS[args.set]:
        bone = s.system == "skeletal"
        cap = args.max_bone_triangles if bone else args.max_triangles
        path = seg_dir / f"{s.label}.nii.gz"
        reason = reasons[s.label] if args.set == "reference" else None
        if reason and not bone:
            sys.exit(f"{s.label}: {reason} in {args.case}; organs must be whole")
        if reason:
            # Never exported, never capped. The femurs and humeri are still
            # meshed into the landmarks file for their joint centres.
            dropped.append({"label": s.label, "reason": reason})
            if s.label in LANDMARK_BONES and not reason.startswith(("absent", "not segmented")):
                faces = cut_faces(path, outside) if reason.startswith("cut by") else set()
                if landmark_ok(s.label, reason, faces):
                    landmarks.append(s)
                else:
                    # A head cut by the scan would fit a joint to the cut face.
                    dropped[-1]["landmark"] = f"not used: cut at {''.join(sorted(faces))}"
            continue
        if s.label == "costal_cartilages" and args.set == "reference":
            # The segmented cartilage is too lumpy and broken to show; the
            # reference body carries modelled cartilages instead.
            kept_ribs = [r for r in labels if r.startswith("rib_") and not reasons[r]]
            ras, triangles, skipped = modelled_cartilage(seg_dir, kept_ribs)
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
        knobless, trimmed_ml = None, 0.0
        if s.label.startswith("rib_") and path.exists():
            image = nib.load(path)
            knobless, trimmed_ml = trim_knobs(
                np.asanyarray(image.dataobj) > 0,
                float(abs(np.linalg.det(image.affine[:3, :3]))) / 1000,
            )
        try:
            ras, triangles, voxel_ml, share = surface(
                path, cap, pieces=s.label in SETS_OF_PIECES,
                along_slices=s.label.startswith(SLICE_SMOOTHED),
                centreline=s.label.startswith("rib_"),
                mask=knobless if s.label.startswith("rib_") else None,
            )  # fmt: skip
        except ValueError:
            if not bone:
                sys.exit(f"{s.label}: the mask is empty in {args.case}; organs must be whole")
            missing.append(s.label)
            continue
        bridged_mm = None
        if share > MAX_DROPPED_SHARE and bone and path.exists():
            # A bone split by the segmentation over thin bone is one bone: the
            # gap is bridged and the surface redone. Anything else in pieces
            # (a mislabelled neighbour) stops the build, as a split organ does.
            image = nib.load(path)
            spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
            whole = knobless if knobless is not None else np.asanyarray(image.dataobj) > 0
            whole, bridged_mm = bridge_gap(whole, spacing)
            if bridged_mm is not None:
                ras, triangles, voxel_ml, share = surface(
                    path, cap, along_slices=s.label.startswith(SLICE_SMOOTHED),
                    centreline=s.label.startswith("rib_"), mask=whole,
                )  # fmt: skip
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
    if args.set == "reference" and outline_path.exists():
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
