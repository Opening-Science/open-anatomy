"""
Retopologises every bone of a mesh.py GLB into clean quads, guided by the CT
surface. Blender 5.2.x, headless.

  blender --background --factory-startup --python blender/retopo.py -- \
      --in <reference glb> --out <glb> --report <json>
  blender --background --factory-startup --python blender/retopo.py -- --self-check

Each bone (system "skeletal", not modelled) is remeshed with QuadriFlow to a
face budget set by its class (FACES), then wrapped back onto its own original
surface so crests, foramina and processes stay where the CT put them. Organs
and the modelled cartilage are left as they are. The report gives, per bone,
the face count before and after, the largest distance from the new surface
to the old one, and whether the result is still closed and manifold; a bone
that comes out open or further than MAX_DRIFT_MM from its CT surface keeps
its original mesh and is listed under "kept". The drift is measured from the
CT surface to the new one, so a smoothed-over foramen or crest shows. Node extras (term, name,
system, licence, modelled) travel through unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
    write_json,
)

# Quad faces per bone class: enough for the vertebrae's processes and the
# sacral foramina, fewer for the long smooth bones.
FACES = {
    "vertebra": 6000,
    "sacrum": 9000,
    "hip bone": 9000,
    "rib": 2500,
    "sternum": 3000,
    "clavicle": 2000,
    "scapula": 6000,
    "humerus": 3000,
    "femur": 3000,
}
MIN_FACES = 800
MAX_DRIFT_MM = 1.0  # a retopologised surface may sit this far from the CT surface at most


def bone_class(name: str) -> str:
    lower = name.lower()
    for key in FACES:
        if key in lower or (key == "vertebra" and "vertebra" in lower):
            return key
    return "rib" if "rib" in lower else "vertebra"


def is_bone(obj: bpy.types.Object) -> bool:
    return obj.type == "MESH" and obj.get("system") == "skeletal" and not obj.get("modelled")


def measure(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    report = {
        "faces": len(bm.faces),
        "quads": sum(1 for f in bm.faces if len(f.verts) == 4),
        "open_edges": sum(1 for e in bm.edges if e.is_boundary),
        "non_manifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
    }
    bm.free()
    return report


def lost_detail(original: bpy.types.Object, obj: bpy.types.Object) -> float:
    """The largest distance (metres) from the original surface's vertices to
    the new surface: what the retopology smoothed over (a foramen, a crest).
    The other way round is always near zero after the shrinkwrap."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    worst = 0.0
    for v in original.data.vertices:
        point = original.matrix_world @ v.co
        hit = tree.find_nearest(point)
        if hit[0] is not None:
            worst = max(worst, (hit[0] - point).length)
    return worst


def remesh(obj: bpy.types.Object, faces: int) -> set[str]:
    with bpy.context.temp_override(
        object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]
    ):
        return bpy.ops.object.quadriflow_remesh(
            target_faces=faces, use_preserve_sharp=False, use_preserve_boundary=False,
            use_mesh_symmetry=False, smooth_normals=True, seed=0,
        )  # fmt: skip


def wrap_to(obj: bpy.types.Object, target: bpy.types.Object) -> None:
    """Puts obj's vertices back onto target's surface, then relaxes the quads."""
    wrap = obj.modifiers.new("wrap", "SHRINKWRAP")
    wrap.target = target
    wrap.wrap_method = "NEAREST_SURFACEPOINT"
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier="wrap")
    smooth = obj.modifiers.new("relax", "CORRECTIVE_SMOOTH")
    smooth.factor = 0.5
    smooth.iterations = 5
    smooth.smooth_type = "LENGTH_WEIGHTED"
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier="relax")
    wrap = obj.modifiers.new("wrap2", "SHRINKWRAP")
    wrap.target = target
    wrap.wrap_method = "NEAREST_SURFACEPOINT"
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier="wrap2")


def self_check() -> int:
    """A dented sphere, retopologised: it must come back closed, in quads,
    and within MAX_DRIFT_MM of where it was, dent included."""
    empty_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.03)
    obj = bpy.context.active_object
    obj["system"], obj["name"] = "skeletal", "test vertebra"
    for v in obj.data.vertices:
        if v.co.z > 0.02:
            v.co.z -= 0.006 * (v.co.z - 0.02) / 0.01  # a dent in the top, like a foramen's rim
    original = obj.copy()
    original.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(original)
    before = measure(obj)
    result = remesh(obj, 1500)
    wrap_to(obj, original)
    after = measure(obj)
    drift = lost_detail(original, obj) * 1000
    checks = {
        "remesh finished": "FINISHED" in result,
        "all quads": after["quads"] == after["faces"] > 0,
        "closed and manifold": after["open_edges"] == 0 and after["non_manifold_edges"] == 0,
        f"within {MAX_DRIFT_MM} mm of the surface ({drift:.3f} mm)": drift <= MAX_DRIFT_MM,
        "fewer faces than before": after["faces"] < before["faces"],
    }
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("retopo self-check " + ("passed" if all(checks.values()) else "failed"))
    return 0 if all(checks.values()) else 1


def main() -> None:
    require_blender()
    args = script_args()
    if "--self-check" in args:
        sys.exit(self_check())
    source, out, report_path = option(args, "--in"), option(args, "--out"), option(args, "--report")
    if not (source and out and report_path):
        sys.stderr.write("retopo.py needs --in, --out and --report.\n")
        sys.exit(2)
    empty_scene()
    objects = import_glb(source)
    report = {"bones": {}, "kept": []}
    for obj in [o for o in objects if is_bone(o)]:
        before = measure(obj)
        original = obj.copy()
        original.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(original)
        budget = max(FACES[bone_class(obj.get("name", obj.name))], MIN_FACES)
        try:
            result = remesh(obj, budget)
        except RuntimeError as error:
            result = {str(error)}
        if "FINISHED" not in result or measure(obj)["quads"] == 0:
            report["kept"].append({"node": obj.name, "why": f"QuadriFlow gave up: {result}"})
            obj.data = original.data
            bpy.data.objects.remove(original)
            continue
        wrap_to(obj, original)
        after = measure(obj)
        drift = lost_detail(original, obj) * 1000
        after["driftMm"] = round(drift, 3)
        ok = after["open_edges"] == 0 and after["non_manifold_edges"] == 0 and drift <= MAX_DRIFT_MM
        if not ok:
            report["kept"].append(
                {"node": obj.name, "why": "open, non-manifold or drifted", **after}
            )
            obj.data = original.data
        bpy.data.objects.remove(original)
        obj.data.shade_smooth()
        report["bones"][obj.name] = {
            "class": bone_class(obj.get("name", obj.name)),
            "before": before,
            "after": after,
            "retopologised": ok,
        }
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        if o.name in bpy.data.objects:
            o.select_set(True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format="GLB", use_selection=True, export_extras=True,
        export_yup=True, export_normals=True, export_materials="EXPORT",
        export_animations=False, export_morph=False,
    )  # fmt: skip
    done = sum(1 for b in report["bones"].values() if b["retopologised"])
    report["summary"] = {"retopologised": done, "kept": len(report["kept"])}
    write_json(Path(report_path), report)
    print(f"retopo: {done} bones retopologised, {len(report['kept'])} kept as they were -> {out}")


if __name__ == "__main__":
    main()
