"""
Fits ANNY to the anatomy, one to one: the CT's organs and skeleton keep their
true size and shape, and the ANNY body is placed, scaled and posed so that its
joints sit on the CT skeleton's. Blender 5.2.x, headless.

Inputs (after `--`):
  --body <glb>        a rigged ANNY body from scripts/bake_anny.py (sized with
                      scripts/size_body.py)
  --ct <glb>          the CT structures from mesh.py --set reference
  --outline <glb>     the CT's trunk outline from the same run (measured, never exported)
  --out <glb>         ANNY's skin and skeleton, posed, plus the CT nodes
  --report <json>     joint residuals before and after, containment, notes
  [--render-dir <dir> --render-name <stem>]   front, three-quarter and trunk renders

Steps:
1. Scale ANNY uniformly so its hip-to-shoulder rise is the CT's (sphere
   centres fitted to the femoral and humeral heads), turn it about the vertical
   so the hip axes agree, put its hip midpoint at the CT's height, and centre
   its trunk on the CT's trunk outline side to side and front to back. This
   placed, unposed body gives the "before" residuals.
2. Pose the spine: each of ANNY's spine05 to neck01 bones the scan covers takes
   the direction of the CT's vertebral curve (the vertebral centroids from the
   sacrum up) at the same height, as a fraction of the hip-to-shoulder rise.
   Directions, not points: the vertebral bodies lie behind the line ANNY's
   spine bones follow through the trunk.
3. Pose the shoulders: clavicle and shoulder01 are turned (a few rounds of
   cyclic coordinate descent) so the shoulder joint lands on the humeral head.
4. Move the hip joints onto the femoral heads (ANNY's sit wider than
   anatomical joint centres) and pose the upper arms along the humerus where
   at least 15 cm of it is in the scan.
5. The pose becomes the rest pose (the skin is baked to it and the skeleton's
   rest is reset), and each CT structure is parented to the ANNY bone it
   follows, without moving it.

Not posed, and why: elbows, wrists and hands, because the forearm and hand
bones are only in TotalSegmentator's licensed appendicular_bones task, which
this project does not use; knees and ankles, because the trunk scans end at
the hips. The report names every joint it could not place. No randomness;
the same inputs give the same file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from assemble import (  # noqa: E402
    CLAVICLE,
    FEMUR,
    HUMERUS,
    LEVELS,
    SCAPULA,
    SLAB,
    femoral_head,
    humeral_head,
    inside_share,
    render,
    section,
    see_through,
    setup_render,
    skin_material,
    skin_tree,
    trunk_points,
)
from mathutils import Matrix, Vector  # noqa: E402
from structures import SKELETON  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    bake_world,
    coords,
    empty_scene,
    import_glb,
    option,
    require_blender,
    rounded,
    script_args,
    sha256,
    write_json,
)

SPINE_BONES = ["spine05", "spine04", "spine03", "spine02", "spine01", "neck01"]
VERTEBRAE = [s for s in SKELETON if s.label.startswith("vertebrae_")][::-1]  # L5 up to C1
SACRUM = "FMA_16202"
MIN_HUMERUS_FOR_POSE = 0.15  # metres of humerus needed to pose the upper arm
CCD_ROUNDS = 8


def node(term: str) -> str:
    return term.replace(":", "_")


def pose_head(armature: bpy.types.Object, bone: str) -> Vector:
    return armature.matrix_world @ armature.pose.bones[bone].head


def pose_tail(armature: bpy.types.Object, bone: str) -> Vector:
    return armature.matrix_world @ armature.pose.bones[bone].tail


def aim(armature: bpy.types.Object, bone: str, target: Vector) -> None:
    """Turns a pose bone about its head so its tail points at target (world)."""
    pb = armature.pose.bones[bone]
    head, tail = pose_head(armature, bone), pose_tail(armature, bone)
    current, wanted = tail - head, target - head
    if current.length < 1e-9 or wanted.length < 1e-9:
        return
    turn = current.rotation_difference(wanted).to_matrix().to_4x4()
    world = armature.matrix_world @ pb.matrix
    moved = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ world
    pb.matrix = armature.matrix_world.inverted() @ moved
    bpy.context.view_layer.update()


def curve_points(names: dict) -> np.ndarray:
    """The CT spine curve: the sacrum's top, then each vertebra's centroid upwards."""
    points = []
    if SACRUM in names:
        # The sacrum's lower, middle and top thirds extend the curve below L5.
        sacrum = coords(names[SACRUM])
        z0, z1 = sacrum[:, 2].min(), sacrum[:, 2].max()
        for a, b in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
            band = sacrum[
                (sacrum[:, 2] >= z0 + a * (z1 - z0)) & (sacrum[:, 2] <= z0 + b * (z1 - z0))
            ]
            points.append(band.mean(0))
    for v in VERTEBRAE:
        name = node(v.term)
        if name in names:
            points.append(coords(names[name]).mean(0))
    return np.array(points)


def point_at_height(curve: np.ndarray, z: float) -> np.ndarray | None:
    """The CT spine curve's point at height z (linear between vertebrae), or None outside it."""
    order = np.argsort(curve[:, 2])
    c = curve[order]
    if z < c[0, 2] or z > c[-1, 2]:
        return None
    i = min(int(np.searchsorted(c[:, 2], z, side="right")) - 1, len(c) - 2)
    t = (z - c[i, 2]) / max(c[i + 1, 2] - c[i, 2], 1e-9)
    return c[i] + t * (c[i + 1] - c[i])


def residuals(armature, targets: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: round(float(np.linalg.norm(np.array(pose_head(armature, k)) - v)), 4)
            for k, v in targets.items()}  # fmt: skip


def spine_directions(armature, curve: np.ndarray, to_ct) -> dict[str, np.ndarray]:
    """For each ANNY spine bone the scan covers, the CT spine's direction at its height."""
    out = {}
    for bone in SPINE_BONES:
        a = point_at_height(curve, to_ct(pose_head(armature, bone)[2]))
        b = point_at_height(curve, to_ct(pose_tail(armature, bone)[2]))
        if a is not None and b is not None and np.linalg.norm(b - a) > 1e-6:
            out[bone] = (b - a) / np.linalg.norm(b - a)
    return out


def spine_angle(armature, wanted: dict[str, np.ndarray]) -> float:
    """Mean angle, in degrees, between ANNY's spine bones and the CT spine's directions."""
    angles = []
    for bone, d in wanted.items():
        v = np.array(pose_tail(armature, bone) - pose_head(armature, bone))
        cos = np.dot(v, d) / max(np.linalg.norm(v), 1e-12)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
    return round(float(np.mean(angles)), 2) if angles else float("nan")


def follow(name: str, centroid: np.ndarray, armature) -> str:
    for table, bone in ((FEMUR, "upperleg01"), (HUMERUS, "upperarm01"),
                        (CLAVICLE, "clavicle"), (SCAPULA, "shoulder01")):  # fmt: skip
        if name in table:
            return f"{bone}.{table[name]}"
    below = [
        b for b in SPINE_BONES + ["neck02", "neck03"] if pose_head(armature, b)[2] <= centroid[2]
    ]
    return below[-1] if below else "root"


def main() -> None:
    require_blender()
    args = script_args()
    body_path, ct_path = option(args, "--body"), option(args, "--ct")
    outline_path = option(args, "--outline")
    out, report_path = option(args, "--out"), option(args, "--report")
    if not (body_path and ct_path and outline_path and out and report_path):
        sys.stderr.write("fit_body.py needs --body, --ct, --outline, --out and --report.\n")
        sys.exit(2)

    empty_scene()
    body = import_glb(body_path)
    armature = next(o for o in body if o.type == "ARMATURE")
    skin = next(o for o in body if o.type == "MESH")
    skin.name = "skin"
    ct = [o for o in import_glb(ct_path) if o.type == "MESH"]
    bake_world(ct)
    for o in ct:
        o.name = o.get("term", o.name).replace(":", "_")
    names = {o.name: o for o in ct}
    outline = next(o for o in import_glb(outline_path) if o.type == "MESH")
    bake_world([outline])

    # CT joints (true size; the anatomy never moves).
    midline = float(np.mean([coords(names[n])[:, 0].mean() for n in FEMUR]))
    targets = {}
    for name, side in FEMUR.items():
        targets[f"upperleg01.{side}"] = femoral_head(coords(names[name]), midline)
    for name, side in HUMERUS.items():
        targets[f"upperarm01.{side}"] = humeral_head(coords(names[name]), midline)
    curve = curve_points(names)

    # 1. Scale, place and turn ANNY.
    def mid(prefix, source):
        return (np.array(source(f"{prefix}.L")) + np.array(source(f"{prefix}.R"))) / 2

    bpy.context.view_layer.update()
    rest_rise = (
        mid("upperarm01", lambda b: pose_head(armature, b))[2]
        - mid("upperleg01", lambda b: pose_head(armature, b))[2]
    )
    ct_rise = mid("upperarm01", targets.get)[2] - mid("upperleg01", targets.get)[2]
    scale = float(ct_rise / rest_rise)
    armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    hip_ct = targets["upperleg01.L"] - targets["upperleg01.R"]
    hip_anny = np.array(pose_head(armature, "upperleg01.L") - pose_head(armature, "upperleg01.R"))
    turn = math.atan2(hip_ct[1], hip_ct[0]) - math.atan2(hip_anny[1], hip_anny[0])
    armature.rotation_euler = (0, 0, turn)
    bpy.context.view_layer.update()
    shift = mid("upperleg01", targets.get) - mid("upperleg01", lambda b: pose_head(armature, b))
    armature.location = Vector(shift)
    bpy.context.view_layer.update()
    # Side to side and front to back, the trunk outlines' centres agree (the CT
    # outline is measured only): a supine scan's trunk sits off the hip joints'
    # line differently from a standing body's.
    hips = mid("upperleg01", targets.get)[2]
    ct_outline, anny_trunk = coords(outline), trunk_points(skin)
    offsets = []
    for f in LEVELS:
        z = hips + f * ct_rise
        offsets.append(section(ct_outline, z, SLAB)[2] - section(anny_trunk, z, SLAB)[2])
    offset = np.mean(offsets, 0)
    armature.location = Vector((shift[0] + offset[0], shift[1] + offset[1], shift[2]))
    bpy.context.view_layer.update()
    before = residuals(armature, targets)

    # 2. Spine: each bone takes the CT spine's direction at the same height (as a
    # fraction of the hip-to-shoulder rise), from the pelvis up. Directions, not
    # points: the vertebral bodies lie behind the line ANNY's spine bones follow.
    hips_a = mid("upperleg01", lambda b: pose_head(armature, b))[2]
    shoulders_a = mid("upperarm01", lambda b: pose_head(armature, b))[2]
    hips_c, shoulders_c = (
        mid("upperleg01", targets.get)[2],
        ct_rise + mid("upperleg01", targets.get)[2],
    )

    def to_ct(z: float) -> float:
        return hips_c + (z - hips_a) / (shoulders_a - hips_a) * (shoulders_c - hips_c)

    wanted = spine_directions(armature, curve, to_ct)
    spine_before = spine_angle(armature, wanted)
    for bone in SPINE_BONES:
        if bone in wanted:
            head = pose_head(armature, bone)
            length = (pose_tail(armature, bone) - head).length
            aim(armature, bone, head + Vector(wanted[bone]) * length)
    notes = [f"spine bones the scan does not cover were not posed: "
             f"{', '.join(b for b in SPINE_BONES if b not in wanted) or 'none'}"]  # fmt: skip

    # 3. Shoulders by cyclic coordinate descent over clavicle and shoulder01.
    for side in "LR":
        target = Vector(targets[f"upperarm01.{side}"])
        for _ in range(CCD_ROUNDS):
            for bone in (f"shoulder01.{side}", f"clavicle.{side}"):
                pb_head = pose_head(armature, bone)
                end = pose_head(armature, f"upperarm01.{side}")
                if (end - target).length < 1e-4:
                    break
                to_end, to_target = end - pb_head, target - pb_head
                if to_end.length < 1e-9 or to_target.length < 1e-9:
                    continue
                spin = to_end.rotation_difference(to_target).to_matrix().to_4x4()
                pb = armature.pose.bones[bone]
                world = armature.matrix_world @ pb.matrix
                moved = Matrix.Translation(pb_head) @ spin @ Matrix.Translation(-pb_head) @ world
                pb.matrix = armature.matrix_world.inverted() @ moved
                bpy.context.view_layer.update()

    # 3b. Hips: ANNY's hip joints sit wider than anatomical joint centres, and
    # upperleg01 is not connected to its pelvis bone, so each moves onto the
    # femoral head; the leg follows, and the skin blends through its weights.
    for side in "LR":
        bone = f"upperleg01.{side}"
        pb = armature.pose.bones[bone]
        delta = Vector(targets[bone]) - pose_head(armature, bone)
        pb.matrix = armature.matrix_world.inverted() @ (
            Matrix.Translation(delta) @ (armature.matrix_world @ pb.matrix)
        )
        bpy.context.view_layer.update()

    # 4. Upper arms along the humerus, where enough of it is in the scan.
    posed_arms = []
    for name, side in HUMERUS.items():
        h = coords(names[name])
        _, _, vt = np.linalg.svd(h - h.mean(0), full_matrices=False)
        axis = vt[0] if vt[0][2] < 0 else -vt[0]  # pointing down, shoulder to elbow
        length = float(np.ptp((h - h.mean(0)) @ vt[0]))
        if length < MIN_HUMERUS_FOR_POSE:
            notes.append(f"upperarm01.{side} not posed: only {length * 100:.0f} cm of humerus")
            continue
        head = pose_head(armature, f"upperarm01.{side}")
        aim(armature, f"upperarm01.{side}", head + Vector(axis))
        posed_arms.append(side)
    notes.append("elbows, wrists and hands not posed: the forearm and hand bones are only in "
                 "TotalSegmentator's licensed appendicular_bones task")  # fmt: skip
    notes.append("knees and ankles not posed: the scan ends at the hips")
    after = residuals(armature, targets)
    spine_after = spine_angle(armature, wanted)

    # 5. The pose becomes the rest pose; the skin is baked to it first.
    bpy.context.view_layer.objects.active = skin
    for mod in skin.modifiers:
        if mod.type == "ARMATURE":
            bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    modifier = skin.modifiers.new("armature", "ARMATURE")
    modifier.object = armature
    bpy.context.view_layer.update()

    bpy.data.objects.remove(outline)  # measured only; never exported or rendered
    tree = skin_tree(skin)
    followed = {}
    for o in ct:
        bone = follow(o.name, coords(o).mean(0), armature)
        world = o.matrix_world.copy()
        o.parent = armature
        o.parent_type = "BONE"
        o.parent_bone = bone
        o.matrix_world = world
        followed[o.name] = bone
    inside = {o.name: round(inside_share(coords(o)[::4], tree), 3) for o in ct}

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
        filepath=out, export_format="GLB", use_selection=True, export_extras=True,
        export_yup=True, export_normals=True, export_materials="EXPORT",
        export_skins=True, export_animations=False, export_morph=False,
    )  # fmt: skip

    report = {
        "method": "one-to-one: the anatomy keeps its true size; ANNY is scaled, placed and posed",
        "body": Path(body_path).name, "bodySha256": sha256(body_path),
        "ct": Path(ct_path).name, "ctSha256": sha256(ct_path),
        "output": Path(out).name, "outputSha256": sha256(out),
        "blender": bpy.app.version_string,
        "annyScale": round(scale, 4), "annyTurnDeg": round(math.degrees(turn), 2),
        "outlineOffsetM": rounded(offset),
        "jointResidualM": {
            k: {"before": before[k], "after": after[k]} for k in sorted(targets)
        },
        "spineAngleDeg": {"before": spine_before, "after": spine_after},
        "posedArms": posed_arms,
        "notes": notes,
        "meanInsideSkin": round(float(np.mean(list(inside.values()))), 3),
        "nodes": {n: {"bone": followed[n], "insideSkin": inside[n]} for n in sorted(followed)},
    }  # fmt: skip
    write_json(report_path, report)
    print(f"ANNY scale {scale:.3f}, turn {math.degrees(turn):.1f} deg")
    for k in sorted(targets):
        print(f"  {k}: {before[k] * 100:.1f} -> {after[k] * 100:.1f} cm")
    print(f"  spine direction: {spine_before:.1f} -> {spine_after:.1f} deg (mean per bone)")
    print(f"  inside skin: {report['meanInsideSkin']:.1%}")
    for n in notes:
        print(f"  {n}")

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
        trunk = [o.matrix_world @ Vector(b) for o in ct for b in o.bound_box]
        tlo = Vector(min(p[i] for p in trunk) for i in range(3)) - Vector((0.08, 0.08, 0.06))
        thi = Vector(max(p[i] for p in trunk) for i in range(3)) + Vector((0.08, 0.08, 0.06))
        render(Path(render_dir) / f"{stem}.trunk.png", tlo, thi, 35)
        print(f"rendered {render_dir}/{stem}.*.png")


if __name__ == "__main__":
    main()
