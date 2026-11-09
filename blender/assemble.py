"""
Builds a reference body: a rigged ANNY body with the organs and trunk skeleton
of one CT (mesh.py --set reference) fitted inside it and attached to its
skeleton. Blender 5.2.x, headless.

Inputs (after `--`):
  --body <glb>        a rigged ANNY body from scripts/bake_anny.py
  --ct <glb>          the CT structures from mesh.py --set reference
  --outline <glb>     the CT's trunk outline from the same run (measured, never exported)
  [--landmarks <glb>] the cut femurs and humeri from the same run, for their heads
                      (measured, never exported)
  --out <glb>         the reference body: ANNY's skin and skeleton plus the CT nodes
  --report <json>     landmarks, the fit, the bone each node follows, containment
  [--render-dir <dir> --render-name <stem>]   front and three-quarter renders
                                              (1440x1080 PNG, the skin see-through)

The fit has two parts. Height comes from four joints that both have: each
hip's centre (a sphere fitted to the femoral head) and each shoulder's (a
sphere fitted to the humeral head), against ANNY's upperleg01 and upperarm01
joints; the vertical scale makes the hip-to-shoulder rise match and the
landmarks' mean height land on ANNY's. Width and depth come from the trunk
outline (TotalSegmentator's body_trunc, --outline) against ANNY's skin
vertices that follow the trunk, measured at four fractions of that rise: the
median ratios set the scales and the outlines' centres the shift. The joints
are not used for width because ANNY's sit wider than anatomical joint centres.
A second step then fits the shoulder girdle (fit_girdle): clavicles, scapulae
and humeri, with the upper thorax following by height, so that a girdle a
supine or stooped scan holds low or forward lands on ANNY's shoulders. The
report gives the inside-skin share of the moved bones before and after, how
many grid settings tied and whether the winner is on the grid's edge, how far
neighbouring bones are pulled apart at their joints, how much of each moved
bone ends up inside an organ, and each landmark's residual both after the trunk
fit alone and after the girdle step.
No rotation:
the CT is axial and ANNY stands upright in the same frame. Each node is then
parented rigidly to one ANNY bone: femur to upperleg01, humerus to upperarm01,
clavicle to clavicle, scapula to shoulder01, and everything else to the spine
or neck bone at its height (root below the spine). The report gives each
landmark's residual and the share of each node's vertices inside the skin.

The CT was taken lying down and ANNY stands, so the organs sit as they do
supine; nothing here models gravity or breathing. No randomness; the same
inputs give the same file.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils.kdtree import KDTree  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    bake_world,
    coords,
    empty_scene,
    import_glb,
    option,
    require_blender,
    rounded,
    script_args,
    set_coords,
    sha256,
    write_json,
)

FEMUR = {"FMA_24475": "L", "FMA_24474": "R"}
HUMERUS = {"FMA_23131": "L", "FMA_23130": "R"}
CLAVICLE = {"FMA_13323": "L", "FMA_13322": "R"}
SCAPULA = {"FMA_13396": "L", "FMA_13395": "R"}
SPINE = ["spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03"]
LIMB_GROUPS = ("upperarm", "lowerarm", "wrist", "metacarpal", "finger", "upperleg", "lowerleg",
               "foot", "toe")  # fmt: skip
LEVELS = (0.25, 0.4, 0.55, 0.7)  # fractions of the hip-to-shoulder rise
SLAB = 0.01  # metres either side of a level
MIN_BAND_POINTS = 20  # fewer outline points than this at a level: the outline misses it
GIRDLE_RAMP = 0.15  # metres below the humeral heads where the girdle's pull fades out
GIRDLE_DY = np.round(np.arange(-0.08, 0.0801, 0.01), 3)  # front-to-back shifts tried, metres
GIRDLE_SX = np.round(np.arange(0.95, 1.2001, 0.025), 3)  # side-to-side scales tried
SHARE_TIE = 0.001  # settings this close to the best score count as tied
OVERLAP_WEIGHT = 2.0  # a share of bone inside an organ costs this many shares outside the skin
HEART = "UBERON_0000948"  # no bone may end up inside it: the limit is 0
NEIGHBOUR_M = 0.005  # bones closer than this before the girdle step meet at a joint
HEAD_DEPTH = 0.045  # metres of bone end used for a joint's sphere fit
SKIN_RGBA = (0.80, 0.62, 0.52, 0.16)
BACKGROUND = (0.82, 0.82, 0.80)


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere through points: (centre, radius)."""
    a = np.hstack([2 * points, np.ones((len(points), 1))])
    b = (points**2).sum(1)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = x[:3]
    return centre, float(math.sqrt(x[3] + centre @ centre))


def femoral_head(points: np.ndarray, midline: float) -> np.ndarray:
    """The femur fragment's medial end is its head: fit a sphere there."""
    medial = np.abs(points[:, 0] - midline)
    return fit_sphere(points[medial <= medial.min() + HEAD_DEPTH])[0]


def humeral_head(points: np.ndarray, midline: float) -> np.ndarray:
    """With the arm down, the humerus's top end is its head."""
    top = points[:, 2]
    return fit_sphere(points[top >= top.max() - HEAD_DEPTH])[0]


def joint(armature: bpy.types.Object, bone: str) -> np.ndarray:
    return np.array(armature.matrix_world @ armature.data.bones[bone].head_local)


def follow_bone(name: str, centroid: np.ndarray, armature: bpy.types.Object) -> str:
    for table, bone in ((FEMUR, "upperleg01"), (HUMERUS, "upperarm01")):
        if name in table:
            return f"{bone}.{table[name]}"
    for table, bone in ((CLAVICLE, "clavicle"), (SCAPULA, "shoulder01")):
        if name in table:
            return f"{bone}.{table[name]}"
    below = [b for b in SPINE if joint(armature, b)[2] <= centroid[2]]
    return below[-1] if below else "root"


def inside_share(points: np.ndarray, skin: BVHTree) -> float:
    """Share of points behind the skin's nearest face (the skin's normals point out)."""
    inside = 0
    for p in points:
        hit = skin.find_nearest(Vector(p))
        if hit[0] is not None and (Vector(p) - hit[0]).dot(hit[1]) < 0:
            inside += 1
    return inside / len(points)


def mesh_tree(obj: bpy.types.Object) -> BVHTree:
    """A closed mesh's surface in world space (baked), for inside tests."""
    return BVHTree.FromPolygons(
        [Vector(p) for p in coords(obj)], [list(f.vertices) for f in obj.data.polygons]
    )


def joint_shifts(points: dict, weights: dict, move: tuple, midline: float) -> list[dict]:
    """How far the girdle step pulls neighbouring bones apart where they meet.

    Each bone moves rigidly by its own weight, so two bones that met before the
    step (closer than NEIGHBOUR_M) are shifted against each other by the weight
    difference times the move at their contact point. Rows are sorted, largest
    first.
    """
    dy, sx, dz = move
    names = sorted(points)
    sub = {n: points[n][::4] for n in names}
    box = {n: (sub[n].min(0) - NEIGHBOUR_M, sub[n].max(0) + NEIGHBOUR_M) for n in names}
    trees: dict[str, KDTree] = {}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dw = weights.get(a, 0.0) - weights.get(b, 0.0)
            if abs(dw) < 1e-6 or np.any(box[a][1] < box[b][0]) or np.any(box[b][1] < box[a][0]):
                continue
            if b not in trees:
                trees[b] = KDTree(len(sub[b]))
                for k, q in enumerate(sub[b]):
                    trees[b].insert(Vector(q), k)
                trees[b].balance()
            gap, contact = math.inf, None
            for q in sub[a]:
                co, _, dist = trees[b].find(Vector(q))
                if dist < gap:
                    gap, contact = dist, (q + np.array(co)) / 2
            if gap > NEIGHBOUR_M:
                continue
            shift = abs(dw) * np.linalg.norm([(contact[0] - midline) * (sx - 1), dy, dz])
            rows.append({
                "bones": [a, b],
                "weights": [round(weights.get(n, 0.0), 2) for n in (a, b)],
                "shiftApartM": round(float(shift), 4),
            })  # fmt: skip
    return sorted(rows, key=lambda r: (-r["shiftApartM"], r["bones"]))


def organ_overlap(points: dict, organs: dict) -> dict:
    """Share of each bone's points inside each organ, where it is at least 1%."""
    found = {}
    for bone, p in sorted(points.items()):
        lo, hi = p.min(0), p.max(0)
        for organ, (tree, olo, ohi) in sorted(organs.items()):
            if np.any(hi < olo) or np.any(ohi < lo):
                continue
            share = inside_share(p[::4], tree)
            if share >= 0.01:
                found[f"{bone} in {organ}"] = round(share, 3)
    return found


def fit_girdle(ct, names, anny: dict, tree: BVHTree, midline: float) -> dict:
    """Second step: moves the shoulder girdle and upper thorax onto ANNY's shoulders.

    Clavicles, scapulae and humeri move fully. The other bones move by a weight
    that ramps from 0 at GIRDLE_RAMP below the humeral heads to 1 at their
    height, by each bone's centroid, so the upper thorax follows the girdle and
    the spine and ribs bend smoothly below it. Organs do not move. The vertical
    shift puts the humeral heads' mean height on ANNY's shoulder joints'; the
    front-to-back shift and a side-to-side scale about the midline are chosen
    on a grid by a score: the share of the moved bones inside the skin, less
    OVERLAP_WEIGHT times their share inside the organs, which do not move (a
    sternum pushed into the heart is worse than a rib tip outside the skin).
    Many settings tie; among those within SHARE_TIE of the best, the smallest
    move wins, and a winner on the grid's edge is reported.

    Also measured, for the report: how far neighbouring bones are pulled apart
    at their joints (joint_shifts), how much of each moved bone ends up inside
    an organ (organ_overlap), and any bone left inside the heart, whose limit
    is 0 (bonesInHeart).
    """
    heads = [humeral_head(coords(names[n]), midline) for n in HUMERUS]
    top = float(np.mean([h[2] for h in heads]))
    dz = float(np.mean([anny[f"upperarm01.{s}"][2] for s in "LR"]) - top)
    weights = {}
    for o in ct:
        if o.get("system") != "skeletal":
            continue
        if o.name in CLAVICLE or o.name in SCAPULA or o.name in HUMERUS:
            weights[o.name] = 1.0
            continue
        height = coords(o).mean(0)[2]
        weights[o.name] = float(np.clip((height - (top - GIRDLE_RAMP)) / GIRDLE_RAMP, 0, 1))
    moving = {n: (coords(names[n]), w) for n, w in weights.items() if w > 0}
    bones = {n: coords(names[n]) for n in weights}
    organs = {}
    for o in ct:
        if o.get("system") != "skeletal":
            points = coords(o)
            organs[o.name] = (mesh_tree(o), points.min(0), points.max(0))

    def place(points, w, dy, sx):
        out = points.copy()
        out[:, 0] = midline + (points[:, 0] - midline) * (1 + w * (sx - 1))
        out[:, 1] += w * dy
        out[:, 2] += w * dz
        return out

    def in_organs(points: np.ndarray) -> float:
        lo, hi = points.min(0), points.max(0)
        near = [t for t, olo, ohi in organs.values() if not (np.any(hi < olo) or np.any(ohi < lo))]
        return max((inside_share(points, t) for t in near), default=0.0)

    scores = []
    for dy in GIRDLE_DY:
        for sx in GIRDLE_SX:
            placed = [place(p[::20], w, dy, sx) for p, w in moving.values()]
            inside = float(np.mean([inside_share(p, tree) for p in placed]))
            overlap = float(np.mean([in_organs(p) for p in placed]))
            scores.append(
                (inside - OVERLAP_WEIGHT * overlap, inside, overlap, float(dy), float(sx))
            )
    top = max(s[0] for s in scores)
    tied = [s for s in scores if s[0] >= top - SHARE_TIE]

    def size(s):
        reach = abs(s[3]) / GIRDLE_DY.max() + abs(s[4] - 1) / (GIRDLE_SX.max() - 1)
        return (reach, abs(s[3]), abs(s[4] - 1), s[3], s[4])

    score, share, overlap, dy, sx = min(tied, key=size)
    edge = [k for k, v, grid in (("dy", dy, GIRDLE_DY), ("sx", sx, GIRDLE_SX))
            if v in (float(grid[0]), float(grid[-1]))]  # fmt: skip
    overlap_before = organ_overlap({n: p for n, (p, _) in moving.items()}, organs)
    for name, (points, w) in moving.items():
        set_coords(names[name], place(points, w, dy, sx))
    overlap_after = organ_overlap({n: coords(names[n]) for n in moving}, organs)
    in_heart = sorted(k.split(" in ")[0] for k in overlap_after if k.endswith(f" in {HEART}"))
    return {
        "dz": dz, "dy": dy, "sx": sx, "weights": weights, "midline": midline,
        "share": share, "score": score, "overlap": overlap, "tied": len(tied), "edge": edge,
        "joints": joint_shifts(bones, weights, (dy, sx, dz), midline),
        "overlapBefore": overlap_before, "overlapAfter": overlap_after, "inHeart": in_heart,
    }  # fmt: skip


def trunk_points(skin: bpy.types.Object) -> np.ndarray:
    """ANNY's skin vertices that mostly follow the trunk, not an arm or a leg."""
    limb = [g.index for g in skin.vertex_groups if g.name.startswith(LIMB_GROUPS)]
    keep = []
    for v in skin.data.vertices:
        on_limbs = sum(g.weight for g in v.groups if g.group in limb)
        keep.append(on_limbs < 0.5)
    return (np.array(skin.matrix_world) @ np.c_[coords(skin), np.ones(len(keep))].T).T[:, :3][keep]


def section(points: np.ndarray, height: float, slab: float) -> tuple[float, float, np.ndarray]:
    """Width (x), depth (y) and centre of the points within slab of height.

    Raises ValueError when fewer than MIN_BAND_POINTS lie there (an outline
    that stops below the level): percentiles of nothing are NaN, and one NaN
    would make the whole scale NaN.
    """
    band = points[np.abs(points[:, 2] - height) <= slab]
    if len(band) < MIN_BAND_POINTS:
        raise ValueError(f"{len(band)} points within {slab * 100:.1f} cm of {height:.3f} m")
    lo, hi = np.percentile(band[:, :2], 1, 0), np.percentile(band[:, :2], 99, 0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1]), (lo + hi) / 2


def glb_asset(path: str) -> dict:
    """The asset object of a GLB's JSON chunk (generator, copyright, extras)."""
    with open(path, "rb") as f:
        f.seek(12)
        length, _ = struct.unpack("<II", f.read(8))
        return json.loads(f.read(length)).get("asset", {})


def credit_line(ct_path: str, body_path: str) -> str:
    """The reference body's copyright: the CT's source (CC BY 4.0) and ANNY's."""
    ct, body = glb_asset(ct_path), glb_asset(body_path)
    if not ct.get("copyright") or not body.get("copyright"):
        sys.stderr.write("The CT or body GLB has no asset copyright to credit; rebuild it.\n")
        sys.exit(1)
    return f"Organs and skeleton: {ct['copyright']}. Body: {body['copyright']}"


def skin_tree(skin: bpy.types.Object) -> BVHTree:
    """The skin in its bind pose, in world space, for containment tests."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = skin.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.transform(skin.matrix_world)
    tree = BVHTree.FromPolygons(
        [v.co.copy() for v in mesh.vertices], [list(p.vertices) for p in mesh.polygons]
    )
    evaluated.to_mesh_clear()
    return tree


def skin_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*SKIN_RGBA[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def see_through(mat: bpy.types.Material) -> None:
    """For renders only: the skin becomes a faint shell, so the inside shows."""
    mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = SKIN_RGBA[3]
    mat.surface_render_method = "BLENDED"


def render(path: Path, lo: Vector, hi: Vector, turn_deg: float) -> None:
    scene = bpy.context.scene
    camera = scene.camera
    centre, extent = (lo + hi) / 2, hi - lo
    width, height = scene.render.resolution_x, scene.render.resolution_y
    camera.data.ortho_scale = 1.12 * max(math.hypot(extent.x, extent.y), extent.z * width / height)
    turn = math.radians(turn_deg)
    distance = 4.0
    camera.location = centre + Vector((distance * math.sin(turn), -distance * math.cos(turn), 0))
    camera.rotation_euler = (math.pi / 2, 0, turn)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def setup_render() -> None:
    scene = bpy.context.scene
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.clip_end = 20
    for name, energy, rotation in (("key", 3.0, (1.0, 0, 0.5)), ("fill", 1.0, (1.3, 0, -0.8))):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.data.angle = 0.3
        light.rotation_euler = rotation
        scene.collection.objects.link(light)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (*BACKGROUND, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    scene.world = world
    scene.view_settings.view_transform = "Standard"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.seed = 0
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 16
    scene.render.resolution_x, scene.render.resolution_y = 1440, 1080
    scene.render.image_settings.file_format = "PNG"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.refresh_devices()
        devices = [d for d in prefs.devices if d.type == backend]
        if devices:
            for d in devices:
                d.use = True
            scene.cycles.device = "GPU"
            break


def main() -> None:
    require_blender()
    args = script_args()
    body_path, ct_path = option(args, "--body"), option(args, "--ct")
    outline_path = option(args, "--outline")
    landmarks_path = option(args, "--landmarks")
    out, report_path = option(args, "--out"), option(args, "--report")
    if not (body_path and ct_path and outline_path and out and report_path):
        sys.stderr.write("assemble.py needs --body, --ct, --outline, --out and --report.\n")
        sys.exit(2)

    credit = credit_line(ct_path, body_path)
    case = glb_asset(ct_path).get("extras", {}).get("case", Path(ct_path).name)
    empty_scene()
    body = import_glb(body_path)
    armature = next(o for o in body if o.type == "ARMATURE")
    skin = next(o for o in body if o.type == "MESH")
    skin.name = "skin"
    ct = [o for o in import_glb(ct_path) if o.type == "MESH"]
    bake_world(ct)
    outline = next(o for o in import_glb(outline_path) if o.type == "MESH")
    bake_world([outline])
    names = {o.name: o for o in ct}
    for o in ct:
        o.name = o.get("term", o.name).replace(":", "_")
        names[o.name] = o
    # Bones the scan cut (the femurs, often the humeri) are not in the body, but
    # their heads still give the joint landmarks: they come from the landmarks
    # file, follow the same transform, and are deleted before export.
    marks = []
    if landmarks_path:
        marks = [o for o in import_glb(landmarks_path) if o.type == "MESH"]
        bake_world(marks)
        for o in marks:
            o.name = "landmark:" + o.get("term", o.name).replace(":", "_")
    bones = dict(names)
    bones.update({o.name.removeprefix("landmark:"): o for o in marks})

    # Landmarks: CT bone ends against ANNY's joints, in Blender's frame (+Z up).
    ct_marks, anny_marks = {}, {}
    midline = float(np.mean([coords(bones[n])[:, 0].mean() for n in FEMUR]))
    for table, finder, bone in (
        (FEMUR, femoral_head, "upperleg01"),
        (HUMERUS, humeral_head, "upperarm01"),
    ):
        for name, side in table.items():
            key = f"{bone}.{side}"
            ct_marks[key] = finder(coords(bones[name]), midline)
            anny_marks[key] = joint(armature, key)
    c = {k: ct_marks[k] for k in sorted(ct_marks)}
    a = {k: anny_marks[k] for k in sorted(anny_marks)}

    def mid(m, bone):
        return (m[f"{bone}.L"] + m[f"{bone}.R"]) / 2

    # Height: the hip-to-shoulder rise. Width and depth: the trunk outlines at
    # the same fractions of that rise (ANNY's joints sit wider than anatomical
    # joint centres, so their spacing would inflate the CT side to side).
    hips_c, shoulders_c = mid(c, "upperleg01")[2], mid(c, "upperarm01")[2]
    hips_a, shoulders_a = mid(a, "upperleg01")[2], mid(a, "upperarm01")[2]
    sz = (shoulders_a - hips_a) / (shoulders_c - hips_c)
    ct_outline = coords(outline)
    anny_trunk = trunk_points(skin)
    widths, depths, centres_c, centres_a = [], [], [], []
    for f in LEVELS:
        try:
            wc, dc, cc = section(ct_outline, hips_c + f * (shoulders_c - hips_c), SLAB / sz)
            wa, da, ca = section(anny_trunk, hips_a + f * (shoulders_a - hips_a), SLAB)
        except ValueError as error:
            sys.stderr.write(
                f"Case {case}: at level {f} of the hip-to-shoulder rise the trunk "
                f"outlines are too sparse to measure ({error}).\n"
            )
            sys.exit(1)
        widths.append(wa / wc)
        depths.append(da / dc)
        centres_c.append(cc)
        centres_a.append(ca)
    scale = np.array([np.median(widths), np.median(depths), sz])
    shift = np.array([
        np.mean(centres_a, 0)[0] - scale[0] * np.mean(centres_c, 0)[0],
        np.mean(centres_a, 0)[1] - scale[1] * np.mean(centres_c, 0)[1],
        np.mean([m[2] for m in a.values()]) - sz * np.mean([m[2] for m in c.values()]),
    ])  # fmt: skip
    for o in [*ct, *marks]:
        set_coords(o, coords(o) * scale + shift)
    tree = skin_tree(skin)
    if inside_share(np.array([joint(armature, "spine03")]), tree) < 1:
        sys.stderr.write("The skin's normals do not point outward; containment is unreliable.\n")
        sys.exit(1)
    residuals = {k: float(np.linalg.norm(c[k] * scale + shift - a[k])) for k in c}
    trunk_residuals = dict(residuals)
    before = {o.name: inside_share(coords(o)[::4], tree) for o in ct}
    has_girdle = all(n in names for n in (*CLAVICLE, *SCAPULA, *HUMERUS))
    if has_girdle:
        girdle = fit_girdle(ct, names, a, tree, midline=shift[0] + scale[0] * midline)
        for key, name in (("upperarm01.L", "FMA_23131"), ("upperarm01.R", "FMA_23130")):
            head = humeral_head(coords(names[name]), girdle["midline"])
            residuals[key] = float(np.linalg.norm(head - a[key]))
    for o in marks:
        bpy.data.objects.remove(o)  # measured only; never exported or rendered

    # Attach each node rigidly to one ANNY bone, keeping where it is.
    followed = {}
    for o in ct:
        bone = follow_bone(o.name, coords(o).mean(0), armature)
        world = o.matrix_world.copy()
        o.parent = armature
        o.parent_type = "BONE"
        o.parent_bone = bone
        o.matrix_world = world
        followed[o.name] = bone

    inside = {o.name: round(inside_share(coords(o)[::4], tree), 3) for o in ct}
    if not has_girdle:
        girdle_report = {"skipped": "no shoulder girdle in this body (the scan cut it)"}
    moved = [n for n, w in girdle["weights"].items() if w > 0] if has_girdle else []
    girdle_report = (
        girdle_report
        if not has_girdle
        else {
            "verticalShiftM": round(girdle["dz"], 4),
            "frontToBackShiftM": round(girdle["dy"], 4),
            "sideToSideScale": round(girdle["sx"], 3),
            "weights": {n: round(w, 2) for n, w in sorted(girdle["weights"].items()) if w > 0},
            "insideSkinBefore": round(float(np.mean([before[n] for n in moved])), 3),
            "insideSkinAfter": round(float(np.mean([inside[n] for n in moved])), 3),
            "allInsideSkinBefore": round(float(np.mean(list(before.values()))), 3),
            "gridSettingsTied": girdle["tied"],
            "onGridEdge": girdle["edge"],
            "largestJointShiftM": (girdle["joints"][0]["shiftApartM"] if girdle["joints"] else 0.0),
            "jointShifts": girdle["joints"][:10],
            "bonesInOrgansBefore": girdle["overlapBefore"],
            "bonesInOrgansAfter": girdle["overlapAfter"],
            "score": round(girdle["score"], 4),
            "overlapWeight": OVERLAP_WEIGHT,
            "meanBoneShareInOrgans": round(girdle["overlap"], 4),
            "bonesInHeart": girdle["inHeart"],
            "residualNote": (
                "trunkFitResidualM is the trunk fit's alone. residualM is after the girdle "
                "step, whose vertical shift puts the humeral heads' mean height on ANNY's "
                "shoulders, so the upperarm01 residuals drop partly by construction."
            ),
        }
    )

    bpy.data.objects.remove(outline)  # measured only; never exported or rendered
    mat = skin_material()
    skin.data.materials.clear()
    skin.data.materials.append(mat)
    for o in ct:
        o.data.shade_smooth()

    bpy.ops.object.select_all(action="DESELECT")
    for o in [armature, skin, *ct]:
        o.select_set(True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out,
        export_format="GLB",
        use_selection=True,
        export_extras=True,
        export_yup=True,
        export_normals=True,
        export_materials="EXPORT",
        export_skins=True,
        export_animations=False,
        export_morph=False,
        export_copyright=credit,
    )

    report = {
        "body": Path(body_path).name,
        "bodySha256": sha256(body_path),
        "ct": Path(ct_path).name,
        "ctSha256": sha256(ct_path),
        "output": Path(out).name,
        "outputSha256": sha256(out),
        "blender": bpy.app.version_string,
        "copyright": credit,
        "scale": rounded(scale),
        "shiftM": rounded(shift),
        "girdle": girdle_report,
        "outlineRatios": {"width": rounded(widths, 3), "depth": rounded(depths, 3)},
        "meanInsideSkin": round(float(np.mean(list(inside.values()))), 3),
        "landmarks": {
            k: {
                "ct": rounded(c[k]),
                "anny": rounded(a[k]),
                "trunkFitResidualM": round(trunk_residuals[k], 4),
                "residualM": round(residuals[k], 4),
            }
            for k in c
        },
        "nodes": {
            o.name: {"bone": followed[o.name], "insideSkin": inside[o.name]}
            for o in sorted(ct, key=lambda o: o.name)
        },
    }
    write_json(report_path, report)
    print(f"scale {rounded(scale, 3)}, landmark residuals (cm), trunk fit -> after girdle: "
          + ", ".join(f"{k} {trunk_residuals[k] * 100:.1f} -> {v * 100:.1f}"
                      for k, v in residuals.items()))  # fmt: skip
    g = girdle_report
    if has_girdle:
        print(f"girdle: inside {g['insideSkinBefore']:.1%} -> {g['insideSkinAfter']:.1%} "
              f"over {len(moved)} bones; all nodes {g['allInsideSkinBefore']:.1%} -> "
              f"{np.mean(list(inside.values())):.1%}")  # fmt: skip
        if g["onGridEdge"]:
            print(f"warning: the girdle's best setting is on the grid's edge ({g['onGridEdge']})")
        if g["bonesInHeart"]:
            print(f"warning: inside the heart after the girdle step (limit 0): {g['bonesInHeart']}")
        print(f"girdle: {g['gridSettingsTied']} grid settings tied; largest joint shift "
              f"{g['largestJointShiftM'] * 100:.1f} cm; bones in organs after: "
              f"{g['bonesInOrgansAfter'] or 'none'}")  # fmt: skip
    else:
        print(f"girdle: {g['skipped']}; all nodes {np.mean(list(inside.values())):.1%} inside")
    outside = {k: v for k, v in inside.items() if v < 0.95}
    print(f"{len(ct)} nodes; under 95% inside the skin: {outside or 'none'}")

    render_dir = option(args, "--render-dir")
    if render_dir:
        stem = option(args, "--render-name", Path(out).stem)
        setup_render()
        see_through(mat)
        everything = [skin, *ct]
        corners = [o.matrix_world @ Vector(b) for o in everything for b in o.bound_box]
        lo = Vector(min(p[i] for p in corners) for i in range(3))
        hi = Vector(max(p[i] for p in corners) for i in range(3))
        render(Path(render_dir) / f"{stem}.front.png", lo, hi, 0)
        render(Path(render_dir) / f"{stem}.three-quarter.png", lo, hi, 35)
        trunk = [Vector(p) for o in ct for p in (o.matrix_world @ Vector(b) for b in o.bound_box)]
        tlo = Vector(min(p[i] for p in trunk) for i in range(3)) - Vector((0.08, 0.08, 0.06))
        thi = Vector(max(p[i] for p in trunk) for i in range(3)) + Vector((0.08, 0.08, 0.06))
        render(Path(render_dir) / f"{stem}.trunk.png", tlo, thi, 35)
        print(f"rendered {render_dir}/{stem}.*.png")


if __name__ == "__main__":
    main()
