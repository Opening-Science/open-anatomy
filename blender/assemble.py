"""
Builds a reference body: a rigged ANNY body with the organs and trunk skeleton
of one CT (mesh.py --set reference) fitted inside it and attached to its
skeleton. Blender 5.2.x, headless.

Inputs (after `--`):
  --body <glb>        a rigged ANNY body from scripts/bake_anny.py
  --ct <glb>          the CT structures from mesh.py --set reference
  --outline <glb>     the CT's trunk outline from the same run (measured, never exported)
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
report gives the inside-skin share of the moved bones before and after.
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

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
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
GIRDLE_RAMP = 0.15  # metres below the humeral heads where the girdle's pull fades out
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


def fit_girdle(ct, names, anny: dict, tree: BVHTree, midline: float) -> dict:
    """Second step: moves the shoulder girdle and upper thorax onto ANNY's shoulders.

    Clavicles, scapulae and humeri move fully. The other bones move by a weight
    that ramps from 0 at GIRDLE_RAMP below the humeral heads to 1 at their
    height, by each bone's centroid, so the upper thorax follows the girdle and
    the spine and ribs bend smoothly below it. Organs do not move. The vertical
    shift puts the humeral heads' mean height on ANNY's shoulder joints'; the
    front-to-back shift and a side-to-side scale about the midline are chosen
    on a grid to keep the most of the moved bones inside the skin.
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

    def place(points, w, dy, sx):
        out = points.copy()
        out[:, 0] = midline + (points[:, 0] - midline) * (1 + w * (sx - 1))
        out[:, 1] += w * dy
        out[:, 2] += w * dz
        return out

    best = (-1.0, 0.0, 1.0)
    for dy in np.round(np.arange(-0.08, 0.0801, 0.01), 3):
        for sx in np.round(np.arange(0.95, 1.2001, 0.025), 3):
            share = np.mean(
                [inside_share(place(p[::20], w, dy, sx), tree) for p, w in moving.values()]
            )
            if share > best[0] + 1e-9:
                best = (share, float(dy), float(sx))
    _, dy, sx = best
    for name, (points, w) in moving.items():
        set_coords(names[name], place(points, w, dy, sx))
    return {"dz": dz, "dy": dy, "sx": sx, "weights": weights, "midline": midline}


def trunk_points(skin: bpy.types.Object) -> np.ndarray:
    """ANNY's skin vertices that mostly follow the trunk, not an arm or a leg."""
    limb = [g.index for g in skin.vertex_groups if g.name.startswith(LIMB_GROUPS)]
    keep = []
    for v in skin.data.vertices:
        on_limbs = sum(g.weight for g in v.groups if g.group in limb)
        keep.append(on_limbs < 0.5)
    return (np.array(skin.matrix_world) @ np.c_[coords(skin), np.ones(len(keep))].T).T[:, :3][keep]


def section(points: np.ndarray, height: float, slab: float) -> tuple[float, float, np.ndarray]:
    """Width (x), depth (y) and centre of the points within slab of height."""
    band = points[np.abs(points[:, 2] - height) <= slab]
    lo, hi = np.percentile(band[:, :2], 1, 0), np.percentile(band[:, :2], 99, 0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1]), (lo + hi) / 2


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
    out, report_path = option(args, "--out"), option(args, "--report")
    if not (body_path and ct_path and outline_path and out and report_path):
        sys.stderr.write("assemble.py needs --body, --ct, --outline, --out and --report.\n")
        sys.exit(2)

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

    # Landmarks: CT bone ends against ANNY's joints, in Blender's frame (+Z up).
    ct_marks, anny_marks = {}, {}
    midline = float(np.mean([coords(names[n])[:, 0].mean() for n in FEMUR]))
    for table, finder, bone in (
        (FEMUR, femoral_head, "upperleg01"),
        (HUMERUS, humeral_head, "upperarm01"),
    ):
        for name, side in table.items():
            key = f"{bone}.{side}"
            ct_marks[key] = finder(coords(names[name]), midline)
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
        wc, dc, cc = section(ct_outline, hips_c + f * (shoulders_c - hips_c), SLAB / sz)
        wa, da, ca = section(anny_trunk, hips_a + f * (shoulders_a - hips_a), SLAB)
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
    for o in ct:
        set_coords(o, coords(o) * scale + shift)
    tree = skin_tree(skin)
    if inside_share(np.array([joint(armature, "spine03")]), tree) < 1:
        sys.stderr.write("The skin's normals do not point outward; containment is unreliable.\n")
        sys.exit(1)
    residuals = {k: float(np.linalg.norm(c[k] * scale + shift - a[k])) for k in c}
    before = {o.name: inside_share(coords(o)[::4], tree) for o in ct}
    girdle = fit_girdle(ct, names, a, tree, midline=shift[0] + scale[0] * midline)
    for key, name in (("upperarm01.L", "FMA_23131"), ("upperarm01.R", "FMA_23130")):
        head = humeral_head(coords(names[name]), girdle["midline"])
        residuals[key] = float(np.linalg.norm(head - a[key]))

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
    moved = [n for n, w in girdle["weights"].items() if w > 0]
    girdle_report = {
        "verticalShiftM": round(girdle["dz"], 4),
        "frontToBackShiftM": round(girdle["dy"], 4),
        "sideToSideScale": round(girdle["sx"], 3),
        "weights": {n: round(w, 2) for n, w in sorted(girdle["weights"].items()) if w > 0},
        "insideSkinBefore": round(float(np.mean([before[n] for n in moved])), 3),
        "insideSkinAfter": round(float(np.mean([inside[n] for n in moved])), 3),
        "allInsideSkinBefore": round(float(np.mean(list(before.values()))), 3),
    }

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
    )

    report = {
        "body": Path(body_path).name,
        "bodySha256": sha256(body_path),
        "ct": Path(ct_path).name,
        "ctSha256": sha256(ct_path),
        "output": Path(out).name,
        "outputSha256": sha256(out),
        "blender": bpy.app.version_string,
        "scale": rounded(scale),
        "shiftM": rounded(shift),
        "girdle": girdle_report,
        "outlineRatios": {"width": rounded(widths, 3), "depth": rounded(depths, 3)},
        "meanInsideSkin": round(float(np.mean(list(inside.values()))), 3),
        "landmarks": {
            k: {"ct": rounded(c[k]), "anny": rounded(a[k]), "residualM": round(residuals[k], 4)}
            for k in c
        },
        "nodes": {
            o.name: {"bone": followed[o.name], "insideSkin": inside[o.name]}
            for o in sorted(ct, key=lambda o: o.name)
        },
    }
    write_json(report_path, report)
    print(f"scale {rounded(scale, 3)}, landmark residuals (cm): "
          + ", ".join(f"{k} {v * 100:.1f}" for k, v in residuals.items()))  # fmt: skip
    g = girdle_report
    print(f"girdle: inside {g['insideSkinBefore']:.1%} -> {g['insideSkinAfter']:.1%} "
          f"over {len(moved)} bones; all nodes {g['allInsideSkinBefore']:.1%} -> "
          f"{np.mean(list(inside.values())):.1%}")  # fmt: skip
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
