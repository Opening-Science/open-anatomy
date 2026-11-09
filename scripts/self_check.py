"""
Self-check for the Blender stage: builds a tiny synthetic mesh, exports it as
glTF, reads it back and checks that it survived. Blender 5.2.x, headless.

Run it through scripts/self-check.sh, or directly:
  blender --background --factory-startup --python scripts/self_check.py

The mesh is an icosphere named by an UBERON ID, as every released mesh is. The
check fails if the mesh is not closed and manifold, if its volume is not
positive (normals pointing out), or if the name, vertex count or volume change
on the glTF round trip. No real data is read or written; the GLB goes to a
temporary folder.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import bmesh
import bpy

NAME = "UBERON_0002113"  # kidney; any UBERON ID would do
VOLUME_TOLERANCE = 1e-4


def build() -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new(NAME)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.05)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def measure(obj: bpy.types.Object) -> dict[str, float]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    report = {
        "vertices": len(bm.verts),
        "open_edges": sum(1 for e in bm.edges if e.is_boundary),
        "non_manifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
        "volume": bm.calc_volume(signed=True),
    }
    bm.free()
    return report


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(("ok   " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []
    before = measure(build())
    check(before["open_edges"] == 0, "the mesh is closed", failures)
    check(before["non_manifold_edges"] == 0, "the mesh is manifold", failures)
    check(before["volume"] > 0, "the normals point outward", failures)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "self-check.glb"
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
        check(path.stat().st_size > 0, "the GLB was written", failures)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(path))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    check(len(meshes) == 1, "one mesh came back", failures)
    if meshes:
        after = measure(meshes[0])
        check(meshes[0].name == NAME, f"it is still named {NAME}", failures)
        # glTF stores vertices per corner normal, so compare after merging.
        bm = bmesh.new()
        bm.from_mesh(meshes[0].data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        merged = len(bm.verts)
        bm.free()
        check(merged == before["vertices"], "the vertex count is unchanged", failures)
        check(
            abs(after["volume"] - before["volume"]) < VOLUME_TOLERANCE * before["volume"],
            "the volume is unchanged",
            failures,
        )

    print("self-check passed" if not failures else f"self-check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
