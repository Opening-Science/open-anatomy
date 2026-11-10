"""
Studio renders of a skeleton GLB, framed like a textbook plate. Blender
5.2.x, headless, Cycles.

  blender --background --factory-startup --python blender/studio.py -- \
      --in <glb> --hdri <hdr> --out-dir <dir> --name <stem> [--views front,three-quarter]
      [--samples 256]
  blender --background --factory-startup --python blender/studio.py -- --self-check

Lighting is a CC0 studio HDRI, turned and tilted so that its key lamp
stands to the upper left of the camera, with the rest of the studio turned
down to a fill; the camera sees a plain light neutral backdrop, not the
HDRI. Bone is a warm ivory with slight subsurface scattering, going to a
warm brown in its crevices (ambient occlusion) and on edges that turn away;
nodes marked modelled (the costal cartilages) are pale blue-white. The view
transform is neutral, so the picture shows those colours. As on a drawn
plate, the far half of the skeleton pales toward the backdrop with
distance, and hip bones, which leave the picture at its bottom edge, fade
out there instead of being cut.

The front view is orthographic, framed from the top of the shoulders to the
top of the pelvis when the hip bones are in the file, else to the lowest
vertebra; the three-quarter view turns 40 degrees to the body's right, so
the sternum is on the right as in Gray's plate 114, and looks down 8
degrees. Writes <name>.<view>.png. --self-check renders blocks for bones
and checks the colours, the haze and the fade in the picture.
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

# Colours are scene-linear. Each material has a lit colour and the colour its
# crevices and turning edges go to, so shade on bone stays warm, not grey.
BONE, BONE_SHADE = (0.87, 0.80, 0.66), (0.38, 0.29, 0.20)
CARTILAGE, CARTILAGE_SHADE = (0.76, 0.86, 0.96), (0.40, 0.54, 0.72)
BACKDROP = (0.93, 0.93, 0.92)
HDRI_TURN = 310.0  # degrees about Z: the key lamp to the picture's left of the camera
HDRI_TILT = 25.0  # degrees about X, after the turn: lifts the lamps in front of the bones
HDRI_STRENGTH = 2.0
FILL = 0.2  # how much of the HDRI's light is kept away from its lamps
FLOOR = 0.6  # and how much of that from below the horizon (the studio's white floor)
EXPOSURE = 0.0
VIEW_TRANSFORMS = ("Khronos PBR Neutral", "Standard")  # the first one this Blender has
AO_DISTANCE = 0.03  # metres
AO_POWER = 1.5
EDGE = 0.5  # how far a surface turning away from the camera goes to its shade colour
HAZE = 0.25  # how far the bones farthest from the camera fade into the backdrop
FADE = 0.08  # share of the picture's height, from the bottom, over which bones fade out
VIEWS = {"front": (0.0, 0.0), "three-quarter": (-40.0, 8.0)}  # turn about Z, look down, degrees
RESOLUTION = (1800, 2400)


def material(
    name: str,
    colour: tuple[float, float, float],
    shade: tuple[float, float, float],
    roughness: float,
    subsurface: float,
    middle: Vector,
    reach: float,
    fade: float,
) -> bpy.types.Material:
    """A Principled surface whose colour runs from `shade` in crevices
    (ambient occlusion) and on edges turning away from the camera to
    `colour` in the open, with subsurface scattering. Two things let it
    dissolve into the backdrop, as on a drawn plate: distance (from
    `middle` back, as the camera looks, it pales, by HAZE at `reach` metres
    and beyond; the near half stays whole) and, with `fade`, the picture's
    bottom edge (over that share of its height), so a bone that leaves the
    picture there fades out instead of being cut."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    def math_node(operation: str, *values: float) -> bpy.types.Node:
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        for i, value in enumerate(values, start=1):
            node.inputs[i].default_value = value
        return node

    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = AO_DISTANCE
    ao.samples = 16
    open_ = math_node("POWER", AO_POWER)
    links.new(ao.outputs["AO"], open_.inputs[0])
    facing = nodes.new("ShaderNodeLayerWeight")
    facing.inputs["Blend"].default_value = 0.35
    edge = math_node("MULTIPLY_ADD", -EDGE, 1.0)  # 1 - EDGE * facing
    links.new(facing.outputs["Facing"], edge.inputs[0])
    lit = math_node("MULTIPLY")
    lit.use_clamp = True
    links.new(open_.outputs[0], lit.inputs[0])
    links.new(edge.outputs[0], lit.inputs[1])
    tone = nodes.new("ShaderNodeMix")
    tone.data_type = "RGBA"
    tone.inputs["A"].default_value = (*shade, 1.0)
    tone.inputs["B"].default_value = (*colour, 1.0)
    links.new(lit.outputs[0], tone.inputs["Factor"])
    links.new(tone.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    bsdf.inputs["Subsurface Weight"].default_value = subsurface
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.8, 0.6)
    bsdf.inputs["Subsurface Scale"].default_value = 0.003

    # How much of the surface is left: 1 - HAZE at the far side, and nothing
    # at the picture's bottom edge.
    place = nodes.new("ShaderNodeNewGeometry")
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    offset.inputs[1].default_value = middle
    links.new(place.outputs["Position"], offset.inputs[0])
    seen = nodes.new("ShaderNodeVectorTransform")
    seen.vector_type, seen.convert_from, seen.convert_to = "VECTOR", "WORLD", "CAMERA"
    links.new(offset.outputs["Vector"], seen.inputs["Vector"])
    depth = nodes.new("ShaderNodeSeparateXYZ")
    links.new(seen.outputs["Vector"], depth.inputs[0])
    left = nodes.new("ShaderNodeMapRange")
    left.inputs["From Min"].default_value = 0.0
    left.inputs["From Max"].default_value = reach
    left.inputs["To Min"].default_value = 1.0
    left.inputs["To Max"].default_value = 1.0 - HAZE
    links.new(depth.outputs["Z"], left.inputs["Value"])
    amount = left.outputs["Result"]
    if fade > 0:
        window = nodes.new("ShaderNodeTexCoord")
        height = nodes.new("ShaderNodeSeparateXYZ")
        links.new(window.outputs["Window"], height.inputs[0])
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.interpolation_type = "SMOOTHSTEP"
        ramp.inputs["From Min"].default_value = 0.0
        ramp.inputs["From Max"].default_value = fade
        links.new(height.outputs["Y"], ramp.inputs["Value"])
        both = math_node("MULTIPLY")
        links.new(amount, both.inputs[0])
        links.new(ramp.outputs["Result"], both.inputs[1])
        amount = both.outputs[0]
    paper = nodes.new("ShaderNodeEmission")
    paper.inputs["Color"].default_value = (*BACKDROP, 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    links.new(amount, mix.inputs["Fac"])
    links.new(paper.outputs["Emission"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], nodes["Material Output"].inputs["Surface"])
    return mat


def world(hdri: str | None) -> None:
    """The HDRI lights the bones, turned and tilted so that its key lamp
    stands to the upper left of the camera; the camera sees the plain
    backdrop. Without an HDRI (the self-check) the light is plain white."""
    scene = bpy.context.scene
    w = bpy.data.worlds.new("studio")
    w.use_nodes = True
    nodes, links = w.node_tree.nodes, w.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    turn = nodes.new("ShaderNodeMapping")
    turn.inputs["Rotation"].default_value = (math.radians(HDRI_TILT), 0.0, math.radians(HDRI_TURN))
    links.new(coords.outputs["Generated"], turn.inputs["Vector"])
    if hdri:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri)
        links.new(turn.outputs["Vector"], env.inputs["Vector"])
    else:
        env = nodes.new("ShaderNodeRGB")
        env.outputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    # The lamps keep their light; the rest of the studio (its white wall, and
    # more so its white floor) is turned down, so the bones are modelled by a
    # key light and keep some shade on their undersides.
    brightness = nodes.new("ShaderNodeRGBToBW")
    links.new(env.outputs["Color"], brightness.inputs["Color"])
    lamps = nodes.new("ShaderNodeMapRange")
    lamps.inputs["From Min"].default_value = 1.5
    lamps.inputs["From Max"].default_value = 6.0
    lamps.inputs["To Min"].default_value = FILL
    lamps.inputs["To Max"].default_value = 1.0
    links.new(brightness.outputs["Val"], lamps.inputs["Value"])
    up = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], up.inputs[0])
    floor = nodes.new("ShaderNodeMapRange")
    floor.interpolation_type = "SMOOTHSTEP"
    floor.inputs["From Min"].default_value = -0.25
    floor.inputs["From Max"].default_value = 0.1
    floor.inputs["To Min"].default_value = HDRI_STRENGTH * FLOOR
    floor.inputs["To Max"].default_value = HDRI_STRENGTH
    links.new(up.outputs["Z"], floor.inputs["Value"])
    strength = nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    links.new(lamps.outputs["Result"], strength.inputs[0])
    links.new(floor.outputs["Result"], strength.inputs[1])
    light = nodes["Background"]
    links.new(env.outputs["Color"], light.inputs["Color"])
    links.new(strength.outputs[0], light.inputs["Strength"])
    backdrop = nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (*BACKDROP, 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    path = nodes.new("ShaderNodeLightPath")
    links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(light.outputs["Background"], mix.inputs[1])
    links.new(backdrop.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], nodes["World Output"].inputs["Surface"])
    scene.world = w


def stage(skeleton: list[bpy.types.Object], hdri: str | None, samples: int) -> None:
    """Dresses the bones and sets the scene for them: materials, light and
    backdrop, the camera, Cycles and the view transform."""
    # The haze runs across the skeleton's depth, whichever way the camera looks.
    corners = [o.matrix_world @ Vector(c) for o in skeleton for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    middle, reach = (low + high) / 2, min(high.x - low.x, high.y - low.y) / 2
    # Only a file with hip bones runs out of the picture's bottom edge.
    fade = FADE if any("hip bone" in o.get("name", "") for o in skeleton) else 0.0
    bone = material("bone", BONE, BONE_SHADE, 0.5, 0.15, middle, reach, fade)
    cartilage = material("cartilage", CARTILAGE, CARTILAGE_SHADE, 0.5, 0.3, middle, reach, fade)
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
    # A neutral view transform: the picture shows the colours set above, with
    # only the highlights eased, where Filmic and AgX wash the bone out.
    for transform in VIEW_TRANSFORMS:
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    scene.view_settings.look = "None"
    scene.view_settings.exposure = EXPOSURE
    world(hdri)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.clip_end = 50


def look_check() -> int:
    """Blocks for bones, rendered small from the front under a plain white
    light: the backdrop is light and neutral, bone is warm, cartilage is
    blue-white, a far bone is paler than a near one, and a hip bone shows
    above the picture's bottom edge and has faded into the backdrop on it."""
    import tempfile

    import numpy as np
    from bpy_extras.object_utils import world_to_camera_view

    empty_scene()
    scene = bpy.context.scene

    def block(name: str, at: tuple[float, float, float], size: float) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cube_add(size=size, location=at)
        obj = bpy.context.active_object
        obj["system"], obj["name"] = "skeletal", name
        return obj

    bones = [
        block("sternum", (-0.10, -0.12, 0.40), 0.10),
        block("left scapula", (0.10, 0.12, 0.40), 0.10),
        block("costal cartilage", (-0.10, -0.12, 0.22), 0.08),
        block("left hip bone", (0.10, 0.0, 0.0), 0.16),
        block("right hip bone", (-0.10, 0.0, 0.0), 0.16),
    ]
    bones[2]["modelled"] = True
    stage(bones, None, 16)
    scene.render.resolution_percentage = 5
    with tempfile.TemporaryDirectory() as tmp:
        render(bones, "front", Path(tmp) / "front.png")
        image = bpy.data.images.load(str(Path(tmp) / "front.png"))
        width, height = image.size
        picture = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, :3]  # rows from below

    def seen(x: float, y: float, z: float) -> np.ndarray:
        """The picture's colour where a point of the scene lands."""
        at = world_to_camera_view(scene, scene.camera, Vector((x, y, z)))
        return picture[min(int(at.y * height), height - 1), min(int(at.x * width), width - 1)]

    def apart(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max())

    paper = picture[height - 1, 0]
    near, far = seen(-0.10, -0.17, 0.40), seen(0.10, 0.07, 0.40)  # the blocks' front faces
    cartilage = seen(-0.10, -0.16, 0.22)
    hip = seen(0.10, -0.08, 0.07)
    edge = picture[
        0,
        min(int(world_to_camera_view(scene, scene.camera, bones[3].location).x * width), width - 1),
    ]
    checks = {
        "the backdrop is light": paper.min() > 0.85,
        "the backdrop is neutral": paper.max() - paper.min() < 0.03,
        "bone is warm": near[0] > near[1] > near[2] + 0.03,
        "cartilage is blue-white": cartilage[2] > cartilage[1] > cartilage[0]
        and cartilage.min() > 0.4,
        "a far bone is paler than a near one": apart(far, paper) < apart(near, paper) - 0.02,
        "a hip bone shows above the picture's bottom edge": apart(hip, paper) > 0.1,
        "a hip bone has faded into the backdrop on the edge": apart(edge, paper) < 0.05,
    }
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("studio self-check " + ("passed" if all(checks.values()) else "failed"))
    return 0 if all(checks.values()) else 1


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
    # ortho_scale spans the render's longer side; fit both the skeleton's
    # width as seen from this view and its height.
    across = extent.x if view == "front" else math.hypot(extent.x, extent.y)
    tall = extent.z
    aspect = width / height
    span = max(tall, across / aspect) if aspect < 1 else max(across, tall * aspect)
    camera.data.ortho_scale = 1.04 * span
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
    if "--self-check" in args:
        sys.exit(look_check())
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
    stage(skeleton, hdri, samples)
    device = gpu()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        render(skeleton, view, out / f"{name}.{view}.png")
    print(f"studio: {len(views)} view(s) of {len(skeleton)} nodes on {device} -> {out}")


if __name__ == "__main__":
    main()
