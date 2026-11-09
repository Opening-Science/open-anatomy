"""
Studio renders of a skeleton GLB, framed like a textbook plate. Blender
5.2.x, headless, Cycles.

  blender --background --factory-startup --python blender/studio.py -- \
      --in <glb> --hdri <hdr> --out-dir <dir> --name <stem> [--views front,three-quarter]
      [--samples 256]

Lighting is a soft CC0 studio HDRI (lighting only; the camera sees a light
neutral backdrop). Bone is a warm material with slight subsurface
scattering and ambient occlusion in its crevices; nodes marked modelled
(the costal cartilages) are pale blue-white. The front view is orthographic,
framed from the top of the shoulders to the top of the pelvis when the hip
bones are in the file, else to the lowest vertebra; the three-quarter view
turns 35 degrees and looks down 8 degrees. Writes <name>.<view>.png.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
)

BONE = (0.93, 0.86, 0.76)
CARTILAGE = (0.82, 0.88, 0.92)
BACKDROP = (0.94, 0.94, 0.93)
VIEWS = {"front": (0.0, 0.0), "three-quarter": (35.0, 8.0)}  # turn about Z, look down, degrees
RESOLUTION = (1800, 2400)


def material(
    name: str, colour: tuple[float, float, float], subsurface: float
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.025
    ao.inputs["Color"].default_value = (*colour, 1.0)
    ao.samples = 8
    links.new(ao.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Subsurface Weight"].default_value = subsurface
    bsdf.inputs["Subsurface Radius"].default_value = (0.004, 0.002, 0.0012)
    bsdf.inputs["Subsurface Scale"].default_value = 0.5
    return mat


def world(hdri: str) -> None:
    scene = bpy.context.scene
    w = bpy.data.worlds.new("studio")
    w.use_nodes = True
    nodes, links = w.node_tree.nodes, w.node_tree.links
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri)
    light = nodes["Background"]
    light.inputs["Strength"].default_value = 1.2
    links.new(env.outputs["Color"], light.inputs["Color"])
    backdrop = nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (*BACKDROP, 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    path = nodes.new("ShaderNodeLightPath")
    links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(light.outputs["Background"], mix.inputs[1])
    links.new(backdrop.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], nodes["World Output"].inputs["Surface"])
    scene.world = w


def gpu() -> str:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = kind
            preferences.get_devices()
            if any(d.use for d in preferences.devices if d.type == kind):
                bpy.context.scene.cycles.device = "GPU"
                return kind
        except TypeError:
            continue
    return "CPU"


def frame(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """The box to frame: the whole skeleton, from the top of the shoulders
    down to the top of the hip bones (or the lowest vertebra)."""
    points = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    hips = [o for o in objects if "hip bone" in o.get("name", "")]
    if hips:
        tops = [max((o.matrix_world @ Vector(c)).z for c in o.bound_box) for o in hips]
        lo.z = min(tops) - 0.02
    return lo, hi


def render(objects: list[bpy.types.Object], view: str, path: Path) -> None:
    scene = bpy.context.scene
    turn, tilt = (math.radians(v) for v in VIEWS[view])
    lo, hi = frame(objects)
    centre = (lo + hi) / 2
    extent = hi - lo
    camera = scene.camera
    width, height = RESOLUTION
    camera.data.type = "ORTHO"
    span = max(math.hypot(extent.x, extent.y), extent.z * width / height)
    camera.data.ortho_scale = 1.08 * span
    distance = 3.0
    camera.location = centre + Vector((
        distance * math.sin(turn) * math.cos(tilt),
        -distance * math.cos(turn) * math.cos(tilt),
        distance * math.sin(tilt),
    ))  # fmt: skip
    camera.rotation_euler = (math.pi / 2 - tilt, 0, turn)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    require_blender()
    args = script_args()
    source, hdri = option(args, "--in"), option(args, "--hdri")
    out_dir, name = option(args, "--out-dir"), option(args, "--name")
    if not (source and hdri and out_dir and name):
        sys.stderr.write("studio.py needs --in, --hdri, --out-dir and --name.\n")
        sys.exit(2)
    views = option(args, "--views", "front,three-quarter").split(",")
    samples = int(option(args, "--samples", "256"))
    empty_scene()
    objects = [o for o in import_glb(source) if o.type == "MESH"]
    skeleton = [o for o in objects if o.get("system") == "skeletal"]
    for o in objects:
        o.hide_render = o not in skeleton
    bone, cartilage = material("bone", BONE, 0.12), material("cartilage", CARTILAGE, 0.3)
    for o in skeleton:
        o.data.materials.clear()
        o.data.materials.append(cartilage if o.get("modelled") else bone)
        o.data.shade_smooth()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "Filmic"  # gentle highlights on the bone
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    world(hdri)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.clip_end = 50
    device = gpu()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        render(skeleton, view, out / f"{name}.{view}.png")
    print(f"studio: {len(views)} view(s) of {len(skeleton)} nodes on {device} -> {out}")


if __name__ == "__main__":
    main()
