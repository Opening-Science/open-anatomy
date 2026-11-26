"""
Makes a synthetic body scan in the shape of the scanner's export, from a
rigged ANNY body, so scan enhancement is tested on the same kind of file it
will meet. Blender 5.2.x, headless. Never touches a real scan.

  blender --background --factory-startup --python blender/make_demo_scan.py -- \
      --in <rigged ANNY glb> --out <glb>

The output is a skinned glTF binary: one skin with ANNY's joints, one mesh
with POSITION, NORMAL, JOINTS_0, WEIGHTS_0, a cylindrical TEXCOORD_0, and
three float attributes standing in for the scanner's own: _DepthMap (how
squarely the nearest of eight cameras around the body sees each vertex, 0 to
1), _GaussianCurvature and _MeanCurvature (angle-deficit and cotangent
estimates). No texture image: ANNY has none.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
)

CAMERAS = 8


def add_attributes(obj: bpy.types.Object) -> None:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    co = np.array([v.co[:] for v in bm.verts])
    normals = np.array([v.normal[:] for v in bm.verts])
    centre = co.mean(0)
    # Eight cameras in a ring at chest height, looking in at the body.
    angles = np.linspace(0, 2 * math.pi, CAMERAS, endpoint=False)
    cams = centre + np.stack([np.cos(angles), np.sin(angles), np.zeros(CAMERAS)], 1) * 1.5
    view = cams[None, :, :] - co[:, None, :]
    view /= np.linalg.norm(view, axis=2, keepdims=True)
    depth = np.clip((view * normals[:, None, :]).sum(2).max(1), 0, 1)
    gauss = np.zeros(len(co))
    mean = np.zeros(len(co))
    for v in bm.verts:
        total = sum(f.calc_area() for f in v.link_faces) / 3 or 1e-12
        corner = sum(loop.calc_angle() for loop in v.link_loops)
        gauss[v.index] = (2 * math.pi - corner) / total
        bend = sum(
            e.calc_face_angle_signed(0.0) * e.calc_length() for e in v.link_edges if e.is_manifold
        )
        mean[v.index] = bend / (4 * total)
    bm.free()
    for name, values in (("_DepthMap", depth), ("_GaussianCurvature", gauss),
                         ("_MeanCurvature", mean)):  # fmt: skip
        attribute = mesh.attributes.new(name, "FLOAT", "POINT")
        attribute.data.foreach_set("value", values.astype(np.float32))


def add_uv(obj: bpy.types.Object) -> None:
    """A cylindrical UV: u around the body, v up it."""
    mesh = obj.data
    uv = mesh.uv_layers.new(name="UVMap")
    co = np.array([v.co[:] for v in mesh.vertices])
    centre = co.mean(0)
    low, high = co[:, 2].min(), co[:, 2].max()
    for loop in mesh.loops:
        p = co[loop.vertex_index] - centre
        u = (math.atan2(p[1], p[0]) / (2 * math.pi)) % 1.0
        uv.data[loop.index].uv = (u, (co[loop.vertex_index][2] - low) / (high - low))


def main() -> None:
    require_blender()
    args = script_args()
    source, out = option(args, "--in"), option(args, "--out")
    if not source or not out:
        sys.stderr.write("make_demo_scan.py needs --in and --out.\n")
        sys.exit(2)
    empty_scene()
    objects = import_glb(source)
    armature = next(o for o in objects if o.type == "ARMATURE")
    meshes = [o for o in objects if o.type == "MESH"]
    skin = max(meshes, key=lambda o: len(o.data.vertices))
    for o in meshes:
        if o is not skin:
            bpy.data.objects.remove(o)  # ANNY's small helper mesh is not part of a scan
    skin.data.shade_smooth()  # a scan has smooth normals; flat ones would split every face
    add_uv(skin)
    add_attributes(skin)
    bpy.ops.object.select_all(action="DESELECT")
    for o in (armature, skin):
        o.select_set(True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format="GLB", use_selection=True, export_skins=True,
        export_normals=True, export_texcoords=True, export_attributes=True,
        export_materials="NONE", export_animations=False, export_morph=False,
    )  # fmt: skip
    print(f"make_demo_scan: wrote {out}")


if __name__ == "__main__":
    main()
