"""
Renders a GLB from mesh.py from the front, on a neutral background, with
headless Blender 5.2.x.

  blender --background --factory-startup --python scripts/render.py -- \
      --in <glb> --out <png> [--size 1440x1080] [--samples 64]

The camera is orthographic, looks at the body's front (glTF +Z) and frames every
mesh with a margin. Materials come from the GLB. Cycles uses the GPU when one
is available and a fixed seed, so a rerun on the same machine gives the same
picture.
"""

from __future__ import annotations

import argparse
import math
import sys

import bpy
from mathutils import Vector

BACKGROUND = (0.82, 0.82, 0.80)


def parse() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="1440x1080")
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def use_gpu(scene: bpy.types.Scene) -> None:
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
            return


def main() -> None:
    args = parse()
    width, height = (int(v) for v in args.size.split("x"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for obj in meshes:
        obj.data.shade_smooth()

    corners = [obj.matrix_world @ Vector(c) for obj in meshes for c in obj.bound_box]
    lo = Vector(min(c[i] for c in corners) for i in range(3))
    hi = Vector(max(c[i] for c in corners) for i in range(3))
    centre, extent = (lo + hi) / 2, hi - lo

    scene = bpy.context.scene
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    # Blender imports glTF +Y up as +Z and the body's front (glTF +Z) as -Y.
    camera.data.ortho_scale = 1.15 * max(extent.x, extent.z * width / height)
    camera.location = centre - Vector((0, extent.y + 1.0, 0))
    camera.rotation_euler = (math.pi / 2, 0, 0)  # looks along +Y, at the body's front
    camera.data.clip_end = extent.y + 3.0

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
    # Show the background as the flat colour, unaffected by the view transform.
    scene.view_settings.view_transform = "Standard"

    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.seed = 0
    scene.cycles.use_denoising = True
    use_gpu(scene)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
    print(f"rendered {args.out}")


if __name__ == "__main__":
    main()
