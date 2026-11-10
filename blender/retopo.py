"""
Retopologises every bone of a mesh.py GLB into clean quads, guided by the CT
surface. Blender 5.2.x, headless.

  blender --background --factory-startup --python blender/retopo.py -- \
      --in <reference glb> --out <glb> --report <json>
  blender --background --factory-startup --python blender/retopo.py -- --self-check
  [--only <node>] retopologises that node alone, for looking at one bone

Each bone (system "skeletal", not modelled) is remeshed into quads to a face
budget set by its class (FACES): QuadriFlow first, then the voxel remesher
at rising density, until the result is closed, manifold, free of faces that
cross each other and of spikes, within MAX_DRIFT_MM of the CT surface
everywhere (so crests, foramina and processes stay), no rougher than the QA
allows (MAX_BEND95) and of the volume it had (MAX_VOLUME_CHANGE). No
shrinkwrap: wrapping a quad mesh back onto a thin process folded faces
through each other. Every result is moved out along its normals by the
hair's breadth that gives it its volume back, since coarse quads and
smoothing both pull a thin bone's borders in; the few places where faces
cross or a vertex stands out as a spike (QuadriFlow pinches a handful of
tiny quads together on some bones) are smoothed where they are. Crossings
and spikes are counted as qa.py counts them, on the triangles the GLB
carries (shipped), not on the quads: a warped quad can clear its
neighbours as a quad and cut through one as two triangles. Organs and the
modelled cartilage are left as they are. The report gives, per bone, the
face count before and after, every try with its method, the largest
distance from the CT surface to the new one, the change in volume, and
whether it passed; a bone that never passes keeps its original mesh and is
listed under "kept". Node extras (term, name, system, licence, modelled)
travel through unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from qa import MAX_BEND95, SPIKE_FLOOR, SPIKE_MAD  # noqa: E402
from qa import measure as qa_measure  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
    write_json,
)

# Quad faces per bone class: enough for the vertebrae's processes and the
# sacral foramina, fewer for the long smooth bones. A rib is long but thin:
# its borders turn through 180 degrees in a few millimetres, and with fewer
# faces the quads across them are too coarse to pass MAX_BEND95.
FACES = {
    "vertebra": 8000,
    "sacrum": 12000,
    "hip bone": 12000,
    "rib": 8000,
    "sternum": 6000,
    "clavicle": 4000,
    "scapula": 10000,
    "humerus": 4000,
    "femur": 4000,
}
# Tried in turn until the result is sound: QuadriFlow first, then the voxel
# remesher at rising density. QuadriFlow's flow-aligned quads fold through
# each other on the thin processes of a 1.5 mm segmentation more often than
# not; the voxel remesher's quads are uniform and never cross.
RETRIES = (("quadriflow", 1.0), ("voxel", 1.0), ("voxel", 1.6), ("voxel", 2.5))
MIN_FACES = 800
MAX_DRIFT_MM = 1.0  # a retopologised surface may sit this far from the CT surface at most
MAX_VOLUME_CHANGE = 0.005  # the share of its volume a bone may gain or lose here
TINY_FACE = 0.1  # of the median face's area: a face under it is part of a pinch
UNTANGLE_ROUNDS = 6
UNTANGLE_RINGS = 3
UNTANGLE_PASSES = 10


def bone_class(name: str) -> str:
    lower = name.lower()
    for key in FACES:
        if key in lower or (key == "vertebra" and "vertebra" in lower):
            return key
    return "rib" if "rib" in lower else "vertebra"


def is_bone(obj: bpy.types.Object) -> bool:
    return obj.type == "MESH" and obj.get("system") == "skeletal" and not obj.get("modelled")


def shipped(mesh: bpy.types.Mesh) -> bmesh.types.BMesh:
    """The mesh as the GLB carries it: the exporter writes every quad as the
    two triangles Blender draws it with (its loop triangles), and qa.py
    measures those. The vertices keep their numbers."""
    mesh.calc_loop_triangles()
    corners = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", corners)
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    triangles = bpy.data.meshes.new("shipped")
    triangles.from_pydata(co.reshape(-1, 3).tolist(), [], corners.reshape(-1, 3).tolist())
    bm = bmesh.new()
    bm.from_mesh(triangles)
    bpy.data.meshes.remove(triangles)
    return bm


def crossing_faces(bm: bmesh.types.BMesh) -> list[tuple[int, int]]:
    """Pairs of faces that pass through each other and share no vertex."""
    tree = BVHTree.FromBMesh(bm)
    bm.faces.ensure_lookup_table()
    return [
        (a, b)
        for a, b in tree.overlap(tree)
        if a < b and not {v.index for v in bm.faces[a].verts} & {v.index for v in bm.faces[b].verts}
    ]


def spikes(bm: bmesh.types.BMesh) -> list[int]:
    """The vertices qa.py counts as spikes: those standing off the mean of
    their neighbours, in local edge lengths, by more than SPIKE_FLOOR and by
    more than SPIKE_MAD median deviations above the mesh's median."""
    score = np.zeros(len(bm.verts))
    for v in bm.verts:
        ring = [e.other_vert(v).co for e in v.link_edges]
        if len(ring) < 3:
            continue
        mean = sum(ring, Vector()) / len(ring)
        edge = sum((r - v.co).length for r in ring) / len(ring)
        score[v.index] = (v.co - mean).length / max(edge, 1e-9)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) or 1e-9
    return np.flatnonzero(score > max(median + SPIKE_MAD * mad, SPIKE_FLOOR)).tolist()


def pinched(mesh: bpy.types.Mesh) -> list[int]:
    """The faces under TINY_FACE of the median face's area. Where QuadriFlow's
    flow lines meet badly it presses a handful of quads together into a
    crease a fraction of a millimetre long; an even mesh has none."""
    area = np.empty(len(mesh.polygons), dtype=np.float32)
    mesh.polygons.foreach_get("area", area)
    return np.flatnonzero(area < TINY_FACE * np.median(area)).tolist()


def faults(mesh: bpy.types.Mesh) -> tuple[dict[str, int], set[int]]:
    """Crossing face pairs and spikes of the mesh as shipped, and its pinched
    faces: how many of each, and the vertices they are at."""
    bm = shipped(mesh)
    pairs, spiky, tiny = crossing_faces(bm), spikes(bm), pinched(mesh)
    at = {v.index for pair in pairs for i in pair for v in bm.faces[i].verts} | set(spiky)
    at |= {v for i in tiny for v in mesh.polygons[i].vertices}
    bm.free()
    return {"self_intersections": len(pairs), "spikes": len(spiky), "pinched": len(tiny)}, at


def measure(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bends = np.degrees([e.calc_face_angle(0.0) for e in bm.edges if e.is_manifold])
    report = {
        "faces": len(bm.faces),
        "quads": sum(1 for f in bm.faces if len(f.verts) == 4),
        "open_edges": sum(1 for e in bm.edges if e.is_boundary),
        "non_manifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
        **faults(obj.data)[0],
        "bend95Deg": round(float(np.percentile(bends, 95)), 2) if bends.size else 0.0,
    }
    bm.free()
    return report


def untangle(obj: bpy.types.Object) -> int:
    """Evens out the few places where faces cross, a vertex stands out as a
    spike or quads are pinched together (see pinched(); the voxel remesher
    leaves a folded quad or two instead): the vertices there and those up to
    UNTANGLE_RINGS edges away are moved towards the mean of their
    neighbours, which spreads the pressed quads over the patch; nothing else
    moves. Returns the number of faults left after UNTANGLE_ROUNDS."""
    for _ in range(UNTANGLE_ROUNDS):
        counts, at = faults(obj.data)
        if not at:
            return 0
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        verts = {bm.verts[i] for i in at}
        for _ in range(UNTANGLE_RINGS):
            verts |= {e.other_vert(v) for v in verts for e in v.link_edges}
        for _ in range(UNTANGLE_PASSES):
            bmesh.ops.smooth_vert(
                bm, verts=list(verts), factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True
            )
        bm.to_mesh(obj.data)
        bm.free()
    return sum(faults(obj.data)[0].values())


def sound(after: dict) -> bool:
    """Whether a remeshed bone, measured, may replace the surface it came from."""
    return (
        after["open_edges"] == 0
        and after["non_manifold_edges"] == 0
        and after["self_intersections"] == 0
        and after["spikes"] == 0
        and after["pinched"] == 0
        and after["driftMm"] <= MAX_DRIFT_MM
        and after["bend95Deg"] <= MAX_BEND95
        and abs(after["volumeChange"]) <= MAX_VOLUME_CHANGE
    )


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


def remesh(
    obj: bpy.types.Object, faces: int, seed: int = 0, method: str = "quadriflow"
) -> set[str]:
    """QuadriFlow on obj, or the voxel remesher. QuadriFlow refuses a mesh
    with doubled vertices or a non-manifold edge, so those are mended first.
    If it still refuses (it does on some long thin bones that are manifold by
    every other test), or when method is "voxel", the mesh is voxel-remeshed
    at a size that gives about the same face count: that is an all-quad
    manifold mesh too, only less evenly flowing."""
    mend(obj)
    target = volume_of(obj.data)
    result = quadriflow(obj, faces, seed) if method == "quadriflow" else set()
    if "FINISHED" not in result:
        result = voxel(obj, faces)
        if "FINISHED" in result:
            relax(obj)  # QuadriFlow's quads lie on the surface and need none
    if "FINISHED" in result:
        untangle(obj)
        keep_volume(obj, target)
        untangle(obj)  # what the move outward pressed together again, or left standing
    return result


def relax(obj: bpy.types.Object) -> None:
    """Two light smoothing passes: the voxel remesher's quads step along a
    thin rib at the voxel size, and this takes the steps out (and most of
    its folded quads with them). The drift check afterwards says whether it
    moved the surface too far; keep_volume gives back what it shrank."""
    smooth = obj.modifiers.new("relax", "SMOOTH")
    smooth.factor = 0.5
    smooth.iterations = 2
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier="relax")


def volume_of(mesh: bpy.types.Mesh) -> float:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = abs(bm.calc_volume(signed=True))
    bm.free()
    return volume


def keep_volume(obj: bpy.types.Object, target: float) -> None:
    """Moves every vertex along its normal by one small distance, the same
    everywhere, so the mesh encloses target (m3) again: smoothing and coarse
    quads both pull a thin bone's borders in, by 1 to 4% of a rib's volume."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for _ in range(3):  # vertex normals under-reach on a sharp border, so repeat
        bm.normal_update()
        signed = bm.calc_volume(signed=True)
        area = sum(f.calc_area() for f in bm.faces)
        offset = (target - abs(signed)) / area * (1 if signed >= 0 else -1)
        for v in bm.verts:
            v.co += v.normal * offset
    bm.to_mesh(obj.data)
    bm.free()


def voxel(obj: bpy.types.Object, faces: int) -> set[str]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    obj.data.remesh_voxel_size = max(math.sqrt(area / faces), 0.0005)
    obj.data.remesh_voxel_adaptivity = 0.0
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        return bpy.ops.object.voxel_remesh()


def mend(obj: bpy.types.Object) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)  # one stray inward face stops QuadriFlow
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def quadriflow(obj: bpy.types.Object, faces: int, seed: int) -> set[str]:
    with bpy.context.temp_override(
        object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]
    ):
        return bpy.ops.object.quadriflow_remesh(
            target_faces=faces, use_preserve_sharp=False, use_preserve_boundary=False,
            use_mesh_symmetry=False, smooth_normals=True, seed=seed,
        )  # fmt: skip


def as_qa_sees(obj: bpy.types.Object) -> dict:
    """qa.py's own measurement of obj as the GLB would carry it."""
    bm = shipped(obj.data)
    triangles = bpy.data.objects.new("shipped", bpy.data.meshes.new("shipped"))
    bm.to_mesh(triangles.data)
    bm.free()
    return qa_measure(triangles)


def quad_sphere(radius: float) -> bpy.types.Object:
    """A sphere of even quads (a cube, subdivided and blown up), as a remesher
    leaves them: a UV sphere's poles are fans of thin triangles, which count
    as pinched before anything is done to them."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=15, use_grid_fill=True)
    for v in bm.verts:
        v.co = v.co.normalized() * radius
    mesh = bpy.data.meshes.new("quad sphere")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("quad sphere", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def folded_checks() -> dict[str, bool]:
    """A sphere with its cap twisted through itself is found to cross itself
    and is smoothed apart; one with a vertex pulled out is found to have a
    spike and is smoothed; both are counted as qa.py counts them; a smoothed
    sphere gets its volume back; a coarse sphere is too rough to be sound."""
    folded = quad_sphere(0.02)
    for v in folded.data.vertices:
        if 0.0185 < v.co.z < 0.02 - 1e-6:  # the ring around the top vertex: a twisted cap
            v.co.x, v.co.y, v.co.z = -v.co.x * 0.5, -v.co.y * 0.5, v.co.z - 0.002
    crossed = measure(folded)["self_intersections"]
    crossed_qa = as_qa_sees(folded)["selfIntersections"]
    left = untangle(folded)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.02)
    spiked = bpy.context.active_object
    side = min(spiked.data.vertices, key=lambda v: abs(v.co.z))
    side.co *= 1.4  # one vertex of the equator, 8 mm out
    spiky = measure(spiked)["spikes"]
    spiky_qa = as_qa_sees(spiked)["spikes"]
    flattened = untangle(spiked)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.02)
    shrunk = bpy.context.active_object
    full = volume_of(shrunk.data)
    relax(shrunk)
    lost = volume_of(shrunk.data) / full - 1
    keep_volume(shrunk, full)
    kept = volume_of(shrunk.data) / full - 1

    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=0.02)
    coarse = measure(bpy.context.active_object)
    coarse.update(driftMm=0.0, volumeChange=0.0, self_intersections=0, spikes=0, pinched=0)
    return {
        f"a twisted cap is seen to cross itself ({crossed} pairs)": crossed > 0,
        f"as qa.py sees it ({crossed_qa} pairs)": crossed == crossed_qa,
        f"and is smoothed apart ({left} left)": left == 0,
        f"a pulled-out vertex is seen as a spike ({spiky})": spiky > 0,
        f"as qa.py sees it ({spiky_qa})": spiky == spiky_qa,
        f"and is smoothed down ({flattened} left)": flattened == 0,
        f"smoothing shrinks a sphere ({lost:+.3%})": lost < 0,
        f"and keep_volume gives it back ({kept:+.5%})": abs(kept) < 1e-4,
        f"a coarse sphere is too rough (bend95 {coarse['bend95Deg']} deg)": not sound(coarse),
    }


def self_check() -> int:
    """A dented sphere, retopologised by QuadriFlow and by the voxel
    remesher: it must come back closed, in quads, within MAX_DRIFT_MM of
    where it was, dent included, smooth and of the same volume."""
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
    checks = {}
    for method in ("quadriflow", "voxel"):
        obj.data = original.data.copy()
        result = remesh(obj, 1500, method=method)
        after = measure(obj)
        drift = lost_detail(original, obj) * 1000
        change = volume_of(obj.data) / volume_of(original.data) - 1
        after.update(driftMm=drift, volumeChange=change)
        checks |= {
            f"{method}: remesh finished": "FINISHED" in result,
            f"{method}: all quads": after["quads"] == after["faces"] > 0,
            f"{method}: closed and manifold": after["open_edges"] == 0
            and after["non_manifold_edges"] == 0,
            f"{method}: no faces cross": after["self_intersections"] == 0,
            f"{method}: no spikes": after["spikes"] == 0,
            f"{method}: within {MAX_DRIFT_MM} mm of the surface ({drift:.3f} mm)": drift
            <= MAX_DRIFT_MM,
            f"{method}: keeps its volume ({change:+.3%})": abs(change) <= MAX_VOLUME_CHANGE,
            f"{method}: smooth (bend95 {after['bend95Deg']} deg)": after["bend95Deg"] <= MAX_BEND95,
            f"{method}: sound": sound(after),
        }
        if method == "quadriflow":
            checks["quadriflow: fewer faces than before"] = after["faces"] < before["faces"]
    checks |= folded_checks()
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
    only = option(args, "--only")  # one node, to look at a bone that gives trouble
    report = {"bones": {}, "kept": []}
    for obj in [o for o in objects if is_bone(o) and (not only or o.name == only)]:
        before = measure(obj)
        original = obj.copy()
        original.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(original)
        budget = max(FACES[bone_class(obj.get("name", obj.name))], MIN_FACES)
        tries, ok, after = [], False, {}
        for k, (method, factor) in enumerate(RETRIES):
            # QuadriFlow can fold faces through each other on a thin process;
            # a denser try with another seed often settles it, and the voxel
            # remesh (all quads, never self-crossing) is the last resort.
            obj.data = original.data.copy()
            try:
                result = remesh(obj, int(budget * factor), seed=k, method=method)
            except RuntimeError as error:
                result = {str(error)}
            asked = {"method": method, "asked": int(budget * factor)}
            if "FINISHED" not in result or measure(obj)["quads"] == 0:
                tries.append({**asked, "why": f"the remesher gave up: {result}"})
                continue
            after = measure(obj)
            drift = lost_detail(original, obj) * 1000
            after["driftMm"] = round(drift, 3)
            after["volumeChange"] = round(volume_of(obj.data) / volume_of(original.data) - 1, 5)
            tries.append({**asked, **after})
            ok = sound(after)
            if ok:
                break
        if not ok:
            report["kept"].append(
                {
                    "node": obj.name,
                    "why": "no try was sound: see each try and sound()",
                    "tries": tries,
                }
            )
            obj.data = original.data
        bpy.data.objects.remove(original)
        obj.data.shade_smooth()
        report["bones"][obj.name] = {
            "class": bone_class(obj.get("name", obj.name)),
            "before": before,
            "after": after if ok else {},
            "method": tries[-1]["method"] if ok else None,
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
