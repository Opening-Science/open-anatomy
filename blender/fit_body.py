"""
Fits ANNY to the anatomy, one to one: the CT's organs and skeleton keep their
true size and shape, and the ANNY body is placed, scaled and posed so that its
joints sit on the CT skeleton's. Blender 5.2.x, headless. Run it through
scripts/blender.py, which checks the Blender version and refuses personal data:

  python scripts/blender.py fit-body --body <glb> --ct <glb> --outline <glb> \
      --out <glb> --report <json> --case <id>

Inputs (after `--`):
  --body <glb>        a rigged ANNY body from scripts/bake_anny.py (sized with
                      scripts/size_body.py)
  --ct <glb>          the CT structures from mesh.py --set reference
  --outline <glb>     the CT's trunk outline from the same run (measured, never exported)
  [--landmarks <glb>] the femurs and humeri mesh.py kept only for their joint centres
                      (cut by the scan at their far end, or dropped with their
                      girdle); measured, never exported
  --out <glb>         ANNY's skin and skeleton, posed, plus the CT nodes
  --report <json>     joint residuals before and after, containment, notes
  --case <id>         the CT's case in sources.json; its record gives the sex for the
                      ANSUR II comparison
  [--sex male|female] the sex, for a CT with no case record; must agree with the record
                      if both are given. Without either, the script stops.
  [--render-dir <dir> --render-name <stem>]   front, three-quarter and trunk renders
  --self-check        instead of a fit: checks the spine curve, the joint targets
                      (a missing side is mirrored), the sex rule, the spine pose and
                      the exported credit on a small synthetic scene

Steps:
1. Scale ANNY uniformly by the geometric mean of three ratios to the CT: the
   hip-to-shoulder rise (sphere centres fitted to the femoral and humeral
   heads) and the trunk's width and depth halfway up it (size_body.uniform_scale,
   the scale size_body.py --scale-free assumes); turn it about the vertical
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
the hips. The report names every joint it could not place. The output's asset
copyright credits the CT's source (the TotalSegmentator dataset, CC BY 4.0:
creator, DOI, case) and ANNY, as assemble.py's does. No randomness; the same
inputs give the same file.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
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
    credit_line,
    femoral_head,
    glb_asset,
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

from size_body import uniform_scale  # noqa: E402
from structures import SKELETON  # noqa: E402

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


def pose_spine(armature, wanted: dict[str, np.ndarray]) -> None:
    """Turns each spine bone the scan covers to its CT direction, from the pelvis up."""
    for bone in SPINE_BONES:
        if bone in wanted:
            head = pose_head(armature, bone)
            length = (pose_tail(armature, bone) - head).length
            aim(armature, bone, head + Vector(wanted[bone]) * length)


def joint_targets(bones: dict, midline: float) -> tuple[dict[str, np.ndarray], list[str]]:
    """The CT's hip and shoulder joint centres, and a note for each one mirrored.

    A femur or humerus the scan left out gives no joint; its side then takes
    the other side's joint mirrored across the midline (CT x is left-right).
    With neither side there is nothing to fit to, and the script stops.
    """
    targets, notes = {}, []
    for table, bone, centre in ((FEMUR, "upperleg01", femoral_head),
                                (HUMERUS, "upperarm01", humeral_head)):  # fmt: skip
        for name, side in table.items():
            if name in bones:
                targets[f"{bone}.{side}"] = centre(coords(bones[name]), midline)
        for name, side in table.items():
            other = f"{bone}.{'R' if side == 'L' else 'L'}"
            if f"{bone}.{side}" in targets:
                continue
            if other not in targets:
                sys.exit(f"fit_body.py: no {name} on either side for the {bone} joints")
            mirrored = targets[other].copy()
            mirrored[0] = 2 * midline - mirrored[0]
            targets[f"{bone}.{side}"] = mirrored
            notes.append(f"{bone}.{side}: {name} not usable, joint mirrored from {other}")
    return targets, notes


def export_fitted(objects: list, out: str, credit: str) -> None:
    """Writes the objects as one GLB whose asset copyright is credit."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format="GLB", use_selection=True, export_extras=True,
        export_yup=True, export_normals=True, export_materials="EXPORT",
        export_skins=True, export_animations=False, export_morph=False,
        export_copyright=credit,
    )  # fmt: skip


def follow(name: str, centroid: np.ndarray, armature) -> str:
    for table, bone in ((FEMUR, "upperleg01"), (HUMERUS, "upperarm01"),
                        (CLAVICLE, "clavicle"), (SCAPULA, "shoulder01")):  # fmt: skip
        if name in table:
            return f"{bone}.{table[name]}"
    below = [
        b for b in SPINE_BONES + ["neck02", "neck03"] if pose_head(armature, b)[2] <= centroid[2]
    ]
    return below[-1] if below else "root"


ROOT = Path(__file__).resolve().parent.parent
ANSUR = {"male": "ANSUR_II_MALE_Public.csv", "female": "ANSUR_II_FEMALE_Public.csv"}
ANSUR_MEASURES = ("stature", "biacromialbreadth", "bicristalbreadth", "chestbreadth",
                  "chestdepth", "hipbreadth")  # fmt: skip
# ANSUR measures these over the skin above the bony landmarks; here they are
# taken bone to bone on the CT, so they sit systematically low and are given
# for orientation only, never flagged against ANSUR's range.
BONE_MEASURES = ("biacromialbreadth", "bicristalbreadth")
SCAPULA_RIBS = (2, 7)  # a scapula lies over the back of ribs 2 to 7
SEX = {"m": "male", "f": "female", "male": "male", "female": "female"}


def case_sex(case: str | None, given: str | None, cases: dict | None = None) -> str:
    """The sex for the ANSUR II comparison: the case record's, else --sex; never a guess.

    cases is sources.json's "cases" unless given (the self-check passes its own).
    """
    recorded = None
    if case:
        if cases is None:
            cases = json.loads((ROOT / "sources.json").read_text())["cases"]
        if case not in cases:
            sys.exit(f"fit_body.py: case {case} is not in sources.json")
        entry = cases[case]
        raw = entry.get("record", {}).get("sex", entry.get("sex"))
        recorded = SEX.get(str(raw).lower()) if raw else None
    if given is not None and given not in ("male", "female"):
        sys.exit(f"fit_body.py: --sex must be male or female, not {given}")
    if recorded and given and recorded != given:
        sys.exit(f"fit_body.py: --sex {given}, but case {case}'s record says {recorded}")
    if not (recorded or given):
        sys.exit(
            "fit_body.py needs the sex for the ANSUR II comparison: pass --case with a "
            "case whose record in sources.json gives it, or --sex male|female"
        )
    return recorded or given


def ansur_ranges(sex: str) -> dict[str, list[float]]:
    """1st, 50th and 99th percentiles (mm) of the ANSUR II measures for one sex."""
    import csv

    path = ROOT / "data" / "sources" / ANSUR[sex]
    with path.open(encoding="latin-1") as handle:
        rows = list(csv.DictReader(handle))
    return {
        k: [float(np.percentile([float(r[k]) for r in rows], q)) for q in (1, 50, 99)]
        for k in ANSUR_MEASURES
    }


def proportions(skin, names: dict, sex: str) -> dict:
    """Body measures (mm) against ANSUR II, and where the shoulder girdle sits.

    Skin (ANNY, fitted): stature; chest breadth and depth at the sternum's
    middle height; hip breadth at the hip joints' height (trunk vertices only).
    These are placed among the ANSUR II percentiles for the sex and flagged
    outside the 1st to 99th.
    Bone (the CT, true size): biacromial breadth between the acromions (each
    scapula's most lateral point in its top fifth) and bicristal breadth across
    the iliac crests (the hip bones' top 3 cm). ANSUR takes both over the skin,
    so a bone-to-bone value sits below it by the soft tissue on each side; they
    are listed with ANSUR's percentiles for orientation but not flagged.
    """
    skin_pts = trunk_points(skin)
    body_pts = coords_world(skin)
    out: dict = {}
    out["stature"] = float(np.ptp(body_pts[:, 2]))
    if "FMA_7485" in names:
        z = coords(names["FMA_7485"])[:, 2].mean()
        width, depth, _ = section(skin_pts, z, 0.01)
        out["chestbreadth"], out["chestdepth"] = width, depth
    hips = [coords(names[n]) for n in FEMUR if n in names]
    if hips:
        z = float(np.mean([femoral_head(h, 0.0)[2] for h in hips]))
        out["hipbreadth"] = section(skin_pts, z, 0.02)[0]
    if all(n in names for n in SCAPULA):
        tips = []
        for n, side in SCAPULA.items():  # Blender +X is the body's left
            sc = coords(names[n])
            top = sc[sc[:, 2] >= sc[:, 2].max() - 0.2 * np.ptp(sc[:, 2])]
            tips.append(top[np.argmax(top[:, 0]) if side == "L" else np.argmin(top[:, 0])])
        out["biacromialbreadth"] = float(abs(tips[0][0] - tips[1][0]))
    hip_bones = [coords(names[n]) for n in ("FMA_16587", "FMA_16586") if n in names]
    if len(hip_bones) == 2:
        crest = np.vstack([h[h[:, 2] >= h[:, 2].max() - 0.03] for h in hip_bones])
        out["bicristalbreadth"] = float(np.ptp(crest[:, 0]))
    ranges = ansur_ranges(sex)
    table = {}
    for k, metres in out.items():
        mm = metres * 1000
        lo, mid, hi = ranges[k]
        table[k] = {"mm": round(mm), "ansur1": round(lo), "ansur50": round(mid),
                    "ansur99": round(hi)}  # fmt: skip
        if k in BONE_MEASURES:
            table[k]["basis"] = "bone to bone on the CT; ANSUR's is over the skin, not comparable"
        else:
            table[k]["basis"] = "skin of the fitted ANNY body, as ANSUR"
            table[k]["inRange"] = bool(lo <= mm <= hi)
    girdle = {}
    for side, (scap, clav) in {
        "L": ("FMA_13396", "FMA_13323"),
        "R": ("FMA_13395", "FMA_13322"),
    }.items():
        if scap in names:
            sc = coords(names[scap])
            ribs = {}
            for k in SCAPULA_RIBS:
                rib = next(
                    (
                        s
                        for s in SKELETON
                        if s.label == f"rib_{'left' if side == 'L' else 'right'}_{k}"
                    ),
                    None,
                )
                if rib and node(rib.term) in names:
                    r = coords(names[node(rib.term)])
                    back = r[
                        r[:, 1] >= np.percentile(r[:, 1], 80)
                    ]  # posterior part (Blender +Y is back)
                    ribs[k] = float(back[:, 2].mean())
            entry = {"scapulaTopM": round(float(sc[:, 2].max()), 3),
                     "scapulaBottomM": round(float(sc[:, 2].min()), 3)}  # fmt: skip
            if len(ribs) == 2:
                entry["rib2BackM"], entry["rib7BackM"] = round(ribs[2], 3), round(ribs[7], 3)
                entry["spansRibs2To7"] = bool(sc[:, 2].max() >= ribs[2] - 0.03
                                              and sc[:, 2].min() <= ribs[7] + 0.03)  # fmt: skip
            girdle[f"scapula.{side}"] = entry
        if clav in names:
            c = coords(names[clav])
            medial = c[np.argmin(np.abs(c[:, 0]))]
            lateral = c[np.argmax(np.abs(c[:, 0]))]
            rise = lateral[2] - medial[2]
            run = abs(lateral[0] - medial[0])
            girdle[f"clavicle.{side}"] = {
                "angleFromHorizontalDeg": round(math.degrees(math.atan2(rise, run)), 1)
            }
    return {"vsAnsurII": table, "girdle": girdle}


def coords_world(obj) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    pts = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    evaluated.to_mesh_clear()
    return pts


def main() -> None:
    require_blender()
    args = script_args()
    body_path, ct_path = option(args, "--body"), option(args, "--ct")
    outline_path = option(args, "--outline")
    out, report_path = option(args, "--out"), option(args, "--report")
    if not (body_path and ct_path and outline_path and out and report_path):
        sys.stderr.write("fit_body.py needs --body, --ct, --outline, --out and --report.\n")
        sys.exit(2)
    case = option(args, "--case")
    sex = case_sex(case, option(args, "--sex"))
    credit = credit_line(ct_path, body_path)

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
    # Femurs and humeri left out of the body still give their joint centres.
    marks = []
    landmarks_path = option(args, "--landmarks")
    if landmarks_path:
        marks = [o for o in import_glb(landmarks_path) if o.type == "MESH"]
        bake_world(marks)
        for o in marks:
            o.name = "landmark:" + o.get("term", o.name).replace(":", "_")
    bones = dict(names)
    bones.update({o.name.removeprefix("landmark:"): o for o in marks})

    # CT joints (true size; the anatomy never moves).
    femurs = [n for n in FEMUR if n in bones]
    if not femurs:
        sys.exit("fit_body.py: neither femur is in the CT or its landmarks file")
    midline = float(np.mean([coords(bones[n])[:, 0].mean() for n in femurs]))
    targets, notes = joint_targets(bones, midline)
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
    # One uniform scale for ANNY: the geometric mean of the rise, trunk width and
    # trunk depth ratios (the trunk measured halfway up the rise). Scaling by the
    # rise alone let an unusually long trunk make ANNY over two metres tall; the
    # pose takes up what the scale leaves at the shoulders.
    hips_rest = mid("upperleg01", lambda b: pose_head(armature, b))[2]
    hips_ct = mid("upperleg01", targets.get)[2]
    cw, cd, _ = section(coords(outline), hips_ct + ct_rise / 2, SLAB)
    aw, ad, _ = section(trunk_points(skin), hips_rest + rest_rise / 2, SLAB)
    scale, ratios = uniform_scale(
        {"rise": ct_rise, "width": cw, "depth": cd}, {"rise": rest_rise, "width": aw, "depth": ad}
    )
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
    pose_spine(armature, wanted)
    notes.append(f"spine bones the scan does not cover were not posed: "
                 f"{', '.join(b for b in SPINE_BONES if b not in wanted) or 'none'}")  # fmt: skip

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
        if name not in bones:
            notes.append(f"upperarm01.{side} not posed: no {name} in the scan")
            continue
        h = coords(bones[name])
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
    for o in marks:
        bpy.data.objects.remove(o)  # landmarks only; never exported or rendered
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
    measured = proportions(skin, names, sex)

    mat = skin_material()
    skin.data.materials.clear()
    skin.data.materials.append(mat)
    for o in ct:
        o.data.shade_smooth()
    export_fitted([armature, skin, *ct], out, credit)

    report = {
        "method": "one-to-one: the anatomy keeps its true size; ANNY is scaled, placed and posed",
        "body": Path(body_path).name, "bodySha256": sha256(body_path),
        "ct": Path(ct_path).name, "ctSha256": sha256(ct_path), "case": case, "sex": sex,
        "output": Path(out).name, "outputSha256": sha256(out),
        "blender": bpy.app.version_string, "copyright": credit,
        "annyScale": round(scale, 4), "annyTurnDeg": round(math.degrees(turn), 2),
        "scaleRatios": {k: round(float(v), 4) for k, v in ratios.items()},
        "outlineOffsetM": rounded(offset),
        "jointResidualM": {
            k: {"before": before[k], "after": after[k]} for k in sorted(targets)
        },
        "spineAngleDeg": {"before": spine_before, "after": spine_after},
        "posedArms": posed_arms,
        "notes": notes,
        "meanInsideSkin": round(float(np.mean(list(inside.values()))), 3),
        "proportions": measured,
        "nodes": {n: {"bone": followed[n], "insideSkin": inside[n]} for n in sorted(followed)},
    }  # fmt: skip
    write_json(report_path, report)
    print(f"ANNY scale {scale:.3f}, turn {math.degrees(turn):.1f} deg")
    for k in sorted(targets):
        print(f"  {k}: {before[k] * 100:.1f} -> {after[k] * 100:.1f} cm")
    print(f"  spine direction: {spine_before:.1f} -> {spine_after:.1f} deg (mean per bone)")
    print(f"  inside skin: {report['meanInsideSkin']:.1%}")
    for k, v in measured["vsAnsurII"].items():
        mark = "bone" if "inRange" not in v else "ok  " if v["inRange"] else "OUT "
        spread = f"{v['ansur1']}/{v['ansur50']}/{v['ansur99']}"
        caveat = "; bone to bone, ANSUR's is over the skin" if "inRange" not in v else ""
        print(f"  {mark} {k}: {v['mm']} mm (ANSUR II 1st/50th/99th {spread}{caveat})")
    for k, v in measured["girdle"].items():
        print(f"  {k}: {v}")
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


def self_check() -> int:
    """A small synthetic scene: spheres for the bone heads, a straight spine rig, a bent curve."""
    failures = []

    # The spine curve: linear between its points in any order, nothing outside them.
    curve = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.2], [0.1, 0.1, 1.4]])
    at = point_at_height(curve[::-1], 1.3)
    if at is None or np.abs(at - [0.05, 0.1, 1.3]).max() > 1e-9:
        failures.append(f"the spine curve at 1.3 m is {at}, not halfway between its points")
    if point_at_height(curve, 0.99) is not None or point_at_height(curve, 1.41) is not None:
        failures.append("a height outside the spine curve gave a point")

    # The sex rule: the record's, else --sex; they must agree; never a guess.
    cases = {"f1": {"record": {"sex": "f"}}, "m1": {"sex": "m"}, "blank": {"record": {}}}

    def stops(run) -> bool:
        try:
            run()
        except SystemExit:
            return True
        return False

    sexes = [case_sex(*a, cases) for a in (("f1", None), ("m1", "male"), (None, "female"),
                                           ("blank", "male"))]  # fmt: skip
    if sexes != ["female", "male", "female", "male"]:
        failures.append(f"the sex rule gave {sexes}")
    for label, a in (("a record and a --sex that disagree", ("f1", "male")),
                     ("a case with no sex on record and no --sex", ("blank", None)),
                     ("neither a case nor --sex", (None, None)),
                     ("a case that is not in sources.json", ("x1", "male")),
                     ("a --sex that is neither male nor female", (None, "other"))):  # fmt: skip
        if not stops(lambda a=a: case_sex(*a, cases)):
            failures.append(f"the sex rule let through {label}")

    # Joint targets: both femurs and the left humerus are spheres; the right
    # humerus is missing and must come out mirrored across the midline.
    empty_scene()

    def ball(name: str, centre: tuple[float, float, float]) -> bpy.types.Object:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.024, location=centre, segments=16, ring_count=10
        )
        obj = bpy.context.active_object
        obj.name = name
        bake_world([obj])
        return obj

    centres = {"FMA_24475": (0.10, 0.0, 0.90), "FMA_24474": (-0.08, 0.0, 0.90),
               "FMA_23131": (0.20, 0.02, 1.40)}  # fmt: skip
    bones = {name: ball(name, centre) for name, centre in centres.items()}
    midline = float(np.mean([coords(bones[n])[:, 0].mean() for n in FEMUR]))
    targets, notes = joint_targets(bones, midline)
    expected = {
        "upperleg01.L": centres["FMA_24475"],
        "upperleg01.R": centres["FMA_24474"],
        "upperarm01.L": centres["FMA_23131"],
        "upperarm01.R": (-0.18, 0.02, 1.40),
    }
    for key, centre in expected.items():
        if key not in targets or np.abs(targets[key] - centre).max() > 1e-4:
            failures.append(f"joint {key} is at {targets.get(key)}, not {centre}")
    if len(notes) != 1 or "upperarm01.R" not in notes[0] or "mirrored" not in notes[0]:
        failures.append(f"the mirrored joint is not noted: {notes}")
    if not stops(lambda: joint_targets({n: bones[n] for n in FEMUR}, midline)):
        failures.append("a scan with neither humerus still gave shoulder joints")

    # The spine pose: a straight rig, bone on bone from 1.0 m up, against a
    # curve that leans and ends at 1.45 m, so the top bones are not covered.
    data = bpy.data.armatures.new("rig")
    armature = bpy.data.objects.new("rig", data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    parent = None
    for i, bone in enumerate([*SPINE_BONES, "neck02", "neck03"]):
        edit = data.edit_bones.new(bone)
        edit.head, edit.tail = (0, 0, 1.0 + 0.1 * i), (0, 0, 1.1 + 0.1 * i)
        edit.parent, edit.use_connect = parent, parent is not None
        parent = edit
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    for name, height, bone in (("FMA_24475", 0.9, "upperleg01.L"), ("FMA_13322", 1.5, "clavicle.R"),
                               ("UBERON_0002107", 1.25, "spine03"),
                               ("UBERON_0001255", 0.8, "root")):  # fmt: skip
        got = follow(name, np.array([0.0, 0.0, height]), armature)
        if got != bone:
            failures.append(f"{name} at {height} m follows {got}, not {bone}")
    off = residuals(armature, {"spine05": np.array([0.0, 0.03, 1.04])})["spine05"]
    if abs(off - 0.05) > 1e-4:
        failures.append(f"a joint 5 cm from its target has a residual of {off} m")
    heights = np.linspace(0.9, 1.45, 12)
    lean = np.c_[0.1 * (heights - 0.9), 0.5 * (heights - 0.9) ** 2, heights]
    wanted = spine_directions(armature, lean, lambda z: z)
    if sorted(wanted) != ["spine02", "spine03", "spine04", "spine05"]:
        failures.append(f"the curve covers spine05 to spine02, but directions came for {wanted}")
    before = spine_angle(armature, wanted)
    pose_spine(armature, wanted)
    after = spine_angle(armature, wanted)
    if not (before > 5 and after < 0.05):
        failures.append(f"the spine pose took the angle from {before} to {after} degrees")
    foot = np.array(pose_head(armature, "spine05"))
    gap = (pose_head(armature, "spine04") - pose_tail(armature, "spine05")).length
    if np.abs(foot - [0, 0, 1.0]).max() > 1e-6 or gap > 1e-6:
        failures.append("the spine pose moved the lowest bone's head or pulled the chain apart")

    # The credit: the CT's and the body's copyright, written into the output's asset.
    with tempfile.TemporaryDirectory() as tmp:
        ct_path, body_path, out = (str(Path(tmp) / f"{n}.glb") for n in ("ct", "body", "out"))
        ct_credit = "Derived from case t1 of a synthetic set, doi:none, licensed CC BY 4.0"
        export_fitted([bones["FMA_24475"]], ct_path, ct_credit)
        export_fitted([bones["FMA_24474"]], body_path, "A synthetic body")
        credit = credit_line(ct_path, body_path)
        export_fitted([armature, *bones.values()], out, credit)
        written = glb_asset(out).get("copyright")
    if ct_credit not in credit or "A synthetic body" not in credit:
        failures.append(f"the credit leaves out the CT's or the body's: {credit}")
    if written != credit:
        failures.append(f"the output's asset copyright is {written!r}, not the credit")

    for line in failures:
        print(f"SELF-CHECK FAIL {line}")
    print("fit_body self-check: " + ("ok" if not failures else f"{len(failures)} failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    require_blender()
    if "--self-check" in script_args():
        sys.exit(self_check())
    main()
