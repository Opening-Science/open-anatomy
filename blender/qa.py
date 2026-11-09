"""
Close-up quality check of every structure in a mesh.py GLB. Blender 5.2.x, headless.

Inputs (after `--`):
  --in <glb>           a GLB from scripts/mesh.py (one node per structure)
  --manifest <json>    its manifest, for each node's voxel volume
  --out-dir <dir>      tiles/<node>.<view>.png and qa.json
  [--size 320]         tile size in pixels
  [--only <node,...>]  check only these nodes

For each structure it measures:
- holes: edges used by one face (the surface must be closed);
- fragments: separate connected pieces, and the smallest one's vertex count;
- self-intersections: pairs of faces that cross and share no vertex;
- spikes: vertices whose distance from their neighbours' mean, relative to
  the local edge length, is a robust outlier (more than SPIKE_MAD median
  absolute deviations above the mesh's median) and above SPIKE_FLOOR;
- roughness: the angle between neighbouring faces (mean and 95th
  percentile), which voxel terraces push up and an outlier test misses;
- volume drift: the mesh's volume against the mask's voxel volume.

It renders six tiles with Workbench (studio light, cavity on, so small
surface faults stand out): four turntable views at 0, 90, 180 and 270
degrees, a close-up on the worst spike (or the sharpest vertex, if none)
and one on the vertex farthest from the centre, where a rib or process
ends. scripts/qa_sheet.py lays the tiles out as contact sheets.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
    write_json,
)

SPIKE_MAD = 8.0
SPIKE_FLOOR = 0.85  # in local edge lengths; the old jagged rib tips reached 0.95
MAX_DRIFT = 0.03
MIN_FRAGMENT_VERTICES = 50
MAX_BEND95 = 20.0  # degrees between neighbouring faces; voxel terraces reach 23-29
VIEWS = (("view0", 0), ("view90", 90), ("view180", 180), ("view270", 270))


def measure(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.verts.ensure_lookup_table()
    holes = sum(1 for e in bm.edges if e.is_boundary)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)

    # Connected pieces.
    seen, pieces = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, size = [v], 0
        seen.add(v.index)
        while stack:
            u = stack.pop()
            size += 1
            for e in u.link_edges:
                w = e.other_vert(u)
                if w.index not in seen:
                    seen.add(w.index)
                    stack.append(w)
        pieces.append(size)

    # Self-intersections: overlapping faces that share no vertex.
    tree = BVHTree.FromBMesh(bm)
    faces = bm.faces
    faces.ensure_lookup_table()
    crossings = sum(
        1
        for a, b in tree.overlap(tree)
        if a < b and not set(v.index for v in faces[a].verts) & set(v.index for v in faces[b].verts)
    )

    # Spikes: umbrella displacement relative to local edge length.
    co = np.array([v.co[:] for v in bm.verts])
    score = np.zeros(len(co))
    for v in bm.verts:
        ring = [e.other_vert(v).co for e in v.link_edges]
        if len(ring) < 3:
            continue
        mean = sum(ring, Vector()) / len(ring)
        edge = sum((r - v.co).length for r in ring) / len(ring)
        score[v.index] = (v.co - mean).length / max(edge, 1e-9)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) or 1e-9
    # Relative to the mesh's own roughness, but never below an absolute floor:
    # on a very smooth mesh the median deviation is tiny and would flag bumps.
    threshold = max(median + SPIKE_MAD * mad, SPIKE_FLOOR)
    spikes = np.flatnonzero(score > threshold)
    worst = int(np.argmax(score))
    centre = co.mean(0)
    tip = int(np.argmax(np.linalg.norm(co - centre, axis=1)))
    # Roughness: the angle between neighbouring faces. Voxel terraces fold the
    # surface at every step, which a spike test (an outlier test) cannot see.
    bends = np.degrees([e.calc_face_angle(0.0) for e in bm.edges if e.is_manifold])
    volume = bm.calc_volume(signed=True)
    bm.free()
    return {
        "vertices": len(co),
        "holes": holes,
        "nonManifoldEdges": non_manifold,
        "fragments": len(pieces),
        "smallestFragment": min(pieces),
        "selfIntersections": crossings,
        "spikes": len(spikes),
        "spikeScoreMax": round(float(score.max()), 3),
        "spikeThreshold": round(threshold, 3),
        "volumeMl": round(volume * 1e6, 2),
        "bendMeanDeg": round(float(bends.mean()), 2),
        "bend95Deg": round(float(np.percentile(bends, 95)), 2),
        "_worst": co[worst].tolist(),
        "_tip": co[tip].tolist(),
        "_centre": centre.tolist(),
        "_size": float(np.linalg.norm(co.max(0) - co.min(0))),
    }


def flags(m: dict) -> list[str]:
    out = []
    if m["holes"] or m["nonManifoldEdges"]:
        out.append(f"holes {m['holes']}")
    small = m["smallestFragment"] < MIN_FRAGMENT_VERTICES
    if m["fragments"] > 1 and (small or m.get("pieces") != "several"):
        out.append(f"{m['fragments']} pieces" + (" (tiny)" if small else ""))
    if m["selfIntersections"]:
        out.append(f"self-intersections {m['selfIntersections']}")
    if m["spikes"]:
        out.append(f"spikes {m['spikes']}")
    if m["bend95Deg"] > MAX_BEND95:
        out.append(f"rough (bend95 {m['bend95Deg']:.0f} deg)")
    if m.get("volumeDrift") is not None and abs(m["volumeDrift"]) > MAX_DRIFT:
        out.append(f"volume {m['volumeDrift']:+.1%}")
    return out


def setup(size: int) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.82, 0.80, 0.76)
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.curvature_ridge_factor = 1.5
    shading.curvature_valley_factor = 1.5
    shading.show_specular_highlight = True
    scene.display.render_aa = "16"
    world = bpy.data.worlds.new("qa")
    world.color = (0.28, 0.29, 0.31)
    scene.world = world
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.image_settings.file_format = "PNG"
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.clip_start = 0.0005
    camera.data.clip_end = 10
    return camera


def shoot(camera, target: Vector, span: float, azimuth: float, elevation: float, path: Path):
    camera.data.ortho_scale = span
    a, e = math.radians(azimuth), math.radians(elevation)
    direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    camera.location = target + direction * 2.0
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    require_blender()
    args = script_args()
    glb, manifest_path, out = (
        option(args, "--in"),
        option(args, "--manifest"),
        option(args, "--out-dir"),
    )
    if not (glb and manifest_path and out):
        sys.stderr.write("qa.py needs --in, --manifest and --out-dir.\n")
        sys.exit(2)
    size = int(option(args, "--size", "320"))
    only = set(option(args, "--only", "").split(",")) - {""}
    manifest = json.loads(Path(manifest_path).read_text())
    nodes = {n["name"]: n for n in manifest["nodes"]}
    out_dir = Path(out)
    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)

    empty_scene()
    objects = [o for o in import_glb(glb) if o.type == "MESH"]
    for o in objects:
        o.data.shade_smooth()
    camera = setup(size)
    report = {"input": Path(glb).name, "case": manifest.get("case"), "structures": []}
    for obj in sorted(objects, key=lambda o: [n["name"] for n in manifest["nodes"]].index(o.name)
                      if o.name in nodes else 1e9):  # fmt: skip
        if only and obj.name not in only:
            continue
        node = nodes.get(obj.name, {})
        m = measure(obj)
        m["pieces"] = node.get("pieces", "one")
        if node.get("voxel_volume_ml"):
            m["volumeDrift"] = round(m["volumeMl"] / node["voxel_volume_ml"] - 1, 4)
        for other in objects:
            other.hide_render = other is not obj
        centre, span = Vector(m["_centre"]), m["_size"] * 1.1
        for view, azimuth in VIEWS:
            shoot(camera, centre, span, azimuth, 20, tiles / f"{obj.name}.{view}.png")
        close = max(span * 0.18, 0.012)
        for view, point in (("spike", m["_worst"]), ("tip", m["_tip"])):
            p = Vector(point)
            look = (p - centre).normalized() if (p - centre).length > 1e-6 else Vector((0, -1, 0))
            azimuth = math.degrees(math.atan2(look.x, -look.y))
            elevation = math.degrees(math.asin(max(-1, min(1, look.z)))) * 0.6
            shoot(camera, p, close, azimuth, elevation, tiles / f"{obj.name}.{view}.png")
        entry = {k: v for k, v in m.items() if not k.startswith("_")}
        entry.update({"name": obj.name, "label": node.get("label", obj.name),
                      "system": node.get("system"), "flags": flags(m)})  # fmt: skip
        report["structures"].append(entry)
        print(f"{obj.name:16} {entry['label'][:28]:28} {', '.join(entry['flags']) or 'ok'}")
    flagged = [s["name"] for s in report["structures"] if s["flags"]]
    report["flagged"] = flagged
    write_json(str(out_dir / "qa.json"), report)
    print(f"{len(report['structures'])} structures, {len(flagged)} flagged")


if __name__ == "__main__":
    main()
