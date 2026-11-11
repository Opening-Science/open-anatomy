"""
Studio renders of a skeleton GLB, framed like a textbook plate. Blender
5.2.x, headless, Cycles.

  blender --background --factory-startup --python blender/studio.py -- \
      --in <glb> --hdri <hdr> --out-dir <dir> --name <stem> [--views front,three-quarter]
      [--samples 256] [--pitch <degrees>|upright] [--look NAME=value,...]
  blender --background --factory-startup --python blender/studio.py -- --self-check

Lighting is a CC0 studio HDRI, turned and tilted so that its key lamp
stands to the upper left of the camera, with the rest of the studio turned
down to a fill; the camera sees a plain light neutral backdrop, not the
HDRI. Bone is a warm ivory with slight subsurface scattering, going to a
warm brown in its crevices (ambient occlusion) and on edges that turn away;
nodes marked modelled (the costal cartilages) are pale blue-white. The view
transform is neutral, so the picture shows those colours. As on a drawn
plate, the far half of the skeleton pales with distance, bone to a warm
grey and cartilage to a grey-blue that still stand clear of the backdrop
and of each other, and hip bones, which leave the picture at its bottom
edge, fade out there instead of being cut.

Both views are orthographic and framed from the top of the shoulders to the
top of the pelvis when the hip bones are in the file, else to the lowest
vertebra: the picture is fitted to those bones as the view sees them, with
a small margin, so none is cut by its edge. The three-quarter view turns 45
degrees to the body's right, so the sternum is on the right as in Gray's
plate 114, and looks down 8 degrees.

A CT is taken lying down and the thorax in it leans back. --pitch turns the
camera, with its light, about the body's left-right axis through the
skeleton's middle, for both views: by so many degrees (positive looks down on
the thorax, which tips its top toward the viewer), or, with `upright`, by
the angle that brings the jugular notch level with the lower border of the
second thoracic vertebra's body, where Gray has it in the erect body. No
mesh is moved or changed. Without --pitch the view is the scan's own.
--look tries other values of the look's constants for one run.

Writes <name>.<view>.png and <name>.view.json (the pitch used, how it was
asked for, and the measure behind `upright`). --self-check renders blocks
for bones and checks the colours, the haze, the fade, the framing of both
views, and that a scene leaning back, taken at its upright pitch, gives the
picture of the same scene standing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from twinblend.blender_env import (  # noqa: E402
    coords,
    empty_scene,
    import_glb,
    option,
    require_blender,
    script_args,
    write_json,
)

# Colours are scene-linear. Each material has a lit colour and the colour its
# crevices and turning edges go to, so shade on bone stays warm, not grey.
BONE, BONE_SHADE = (0.87, 0.80, 0.66), (0.38, 0.29, 0.20)
# Cartilage holds its red back: the warm light and the scattering add some,
# and with more of it the blue-white reads as grey-lilac.
CARTILAGE, CARTILAGE_SHADE = (0.76, 0.88, 0.98), (0.42, 0.58, 0.76)
BACKDROP = (0.93, 0.93, 0.92)
HDRI_TURN = 310.0  # degrees about Z: the key lamp to the picture's left of the camera
HDRI_TILT = 25.0  # degrees about X, after the turn: lifts the lamps in front of the bones
HDRI_STRENGTH = 2.0
FILL = 0.2  # how much of the HDRI's light is kept away from its lamps
FLOOR = 0.6  # and how much of that from below the horizon (the studio's white floor)
EXPOSURE = 0.0
VIEW_TRANSFORMS = ("Khronos PBR Neutral", "Standard")  # the first one this Blender has
AO_DISTANCE = 0.04  # metres
AO_POWER = 1.75
EDGE = 0.7  # how far a surface turning away from the camera goes to its shade colour
HAZE = 0.5  # how far the bones farthest from the camera pale to the far tone
# The far tone is a warm grey a clear step darker than the backdrop: mixed
# toward the backdrop itself, lit ivory, which is nearly as light, dissolves
# into it and the far ribs seem to end in mid-air.
FAR = (0.66, 0.62, 0.55)
# Cartilage has a far tone of its own, a grey-blue: blue-white mixed toward the
# warm grey loses its hue and comes out a light grey next to the backdrop's.
CARTILAGE_FAR = (0.55, 0.64, 0.77)
HAZE_DEPTH = 0.6  # share of the far half's depth over which the haze comes on
FADE = 0.08  # share of the picture's height, from the bottom, over which bones fade out
MARGIN = 0.04  # share of the picture left free between the bones and each of its other edges
VIEWS = {"front": (0.0, 0.0), "three-quarter": (-45.0, 8.0)}  # turn about Z, look down, degrees
RESOLUTION = (1800, 2400)
# Scattering in bone and in cartilage: how much of the light goes under the
# surface, and how far (metres, times the radius red travels farthest by).
SCATTER, CARTILAGE_SCATTER = (0.2, 0.003), (0.35, 0.003)
SCATTER_RADIUS = (1.0, 0.8, 0.6)
ROUGHNESS, CARTILAGE_ROUGHNESS = 0.55, 0.6
CARTILAGE_COAT = 0.0  # a clear coat on cartilage, for a soft sheen
LOOK = "None"  # the view transform's look
RULE = "the jugular notch level with the lower border of the second thoracic vertebra's body"
LEAN = 15.0  # degrees the self-check's second scene leans back
NOTCH = 0.002  # metres: the top of the sternum is what lies this close under its highest point
BODY = 0.3  # share of a vertebra's depth, from its front, that is its body
BRIGHT = 7.5  # the self-check's strong light, as a strength of its plain white world
# How much darker than the backdrop a bone must be to stand clear of it, and
# how far a far cartilage's colour must be from the backdrop's and a far bone's.
CLEAR = 0.07


def trial(settings: str) -> None:
    """Sets constants of the look for this run only, to try a variation
    without adopting it: NAME=value pairs joined by commas, the values of
    a tuple joined by colons (`--look AO_DISTANCE=0.06,BONE=0.9:0.8:0.6`)."""
    for pair in filter(None, settings.split(",")):
        name, _, value = pair.partition("=")
        old = globals().get(name)
        if not name.isupper() or not isinstance(old, (float, int, str, tuple)):
            sys.exit(f"--look: {name} is not a constant of the look")
        if isinstance(old, tuple):
            globals()[name] = tuple(type(old[0])(v) for v in value.split(":"))
        else:
            globals()[name] = type(old)(value)


def material(
    name: str,
    colour: tuple[float, float, float],
    shade: tuple[float, float, float],
    roughness: float,
    scatter: tuple[float, float],
    coat: float,
    middle: Vector,
    reach: float,
    fade: float,
    far: tuple[float, float, float],
) -> bpy.types.Material:
    """A Principled surface whose colour runs from `shade` in crevices
    (ambient occlusion) and on edges turning away from the camera to
    `colour` in the open, with subsurface scattering. Two things soften
    it, as on a drawn plate. Distance: from `middle` back, as the camera
    looks, it pales and loses contrast, going by HAZE to the tone `far` at
    `reach` metres and beyond; the near half stays whole, and the far half
    stays opaque and apart from the backdrop. `far` is of the material's
    own hue, so far cartilage is still told from far bone. And, with
    `fade`, the picture's bottom edge (over that share of its height),
    where it dissolves into the backdrop, so a bone that leaves the
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
    bsdf.inputs["Subsurface Weight"].default_value = scatter[0]
    bsdf.inputs["Subsurface Radius"].default_value = SCATTER_RADIUS
    bsdf.inputs["Subsurface Scale"].default_value = scatter[1]
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.3

    # How much of the surface is left: 1 - HAZE of it at the far side, the
    # rest the far tone.
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
    distance = nodes.new("ShaderNodeEmission")
    distance.inputs["Color"].default_value = (*far, 1.0)
    hazed = nodes.new("ShaderNodeMixShader")
    links.new(left.outputs["Result"], hazed.inputs["Fac"])
    links.new(distance.outputs["Emission"], hazed.inputs[1])
    links.new(bsdf.outputs["BSDF"], hazed.inputs[2])
    surface = hazed.outputs["Shader"]
    if fade > 0:
        # And nothing of it is left at the picture's bottom edge, only the backdrop.
        window = nodes.new("ShaderNodeTexCoord")
        height = nodes.new("ShaderNodeSeparateXYZ")
        links.new(window.outputs["Window"], height.inputs[0])
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.interpolation_type = "SMOOTHSTEP"
        ramp.inputs["From Min"].default_value = 0.0
        ramp.inputs["From Max"].default_value = fade
        links.new(height.outputs["Y"], ramp.inputs["Value"])
        paper = nodes.new("ShaderNodeEmission")
        paper.inputs["Color"].default_value = (*BACKDROP, 1.0)
        faded = nodes.new("ShaderNodeMixShader")
        links.new(ramp.outputs["Result"], faded.inputs["Fac"])
        links.new(paper.outputs["Emission"], faded.inputs[1])
        links.new(surface, faded.inputs[2])
        surface = faded.outputs["Shader"]
    links.new(surface, nodes["Material Output"].inputs["Surface"])
    return mat


def world(hdri: str | None, strength: float | None = None) -> None:
    """The HDRI lights the bones, turned and tilted so that its key lamp
    stands to the upper left of the camera; the camera sees the plain
    backdrop. Without an HDRI (the self-check) the light is plain white.
    Needs the scene's camera: the light stands in its room."""
    scene = bpy.context.scene
    strength = HDRI_STRENGTH if strength is None else strength
    w = bpy.data.worlds.new("studio")
    w.use_nodes = True
    nodes, links = w.node_tree.nodes, w.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    turn = nodes.new("ShaderNodeMapping")
    turn.inputs["Rotation"].default_value = (math.radians(HDRI_TILT), 0.0, math.radians(HDRI_TURN))
    around = coords.outputs["Generated"]
    if scene.camera.parent:
        # The light is the room's and turns with it under a view pitch.
        pitched = nodes.new("ShaderNodeMapping")
        pitched.inputs["Rotation"].default_value = room().inverted().to_euler()
        links.new(around, pitched.inputs["Vector"])
        around = pitched.outputs["Vector"]
    links.new(around, turn.inputs["Vector"])
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
    links.new(around, up.inputs[0])
    floor = nodes.new("ShaderNodeMapRange")
    floor.interpolation_type = "SMOOTHSTEP"
    floor.inputs["From Min"].default_value = -0.25
    floor.inputs["From Max"].default_value = 0.1
    floor.inputs["To Min"].default_value = strength * FLOOR
    floor.inputs["To Max"].default_value = strength
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


def stage(
    skeleton: list[bpy.types.Object], hdri: str | None, samples: int, pitch: float = 0.0
) -> None:
    """Dresses the bones and sets the scene for them: the camera, pitched
    by `pitch` degrees, materials, light and backdrop, Cycles and the view
    transform."""
    scene = bpy.context.scene
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.clip_end = 50
    if pitch:
        # The camera hangs on a rig at the skeleton's middle, turned about the
        # body's left-right axis. No bone moves.
        rig = bpy.data.objects.new("rig", None)
        scene.collection.objects.link(rig)
        points = np.concatenate([in_room(o) for o in skeleton])
        rig.location = (points.min(axis=0) + points.max(axis=0)) / 2
        rig.rotation_euler = (-math.radians(pitch), 0.0, 0.0)
        camera.parent = rig
    # The haze runs across the skeleton's depth, whichever way the camera looks,
    # and is whole by HAZE_DEPTH of the way back, so the ribs' far halves are of
    # one tone up to the spine.
    points = np.concatenate([in_room(o) for o in skeleton])
    low, high = points.min(axis=0), points.max(axis=0)
    middle = room() @ Vector((low + high) / 2)
    reach = HAZE_DEPTH * min(high[0] - low[0], high[1] - low[1]) / 2
    fade = fading(skeleton)
    bone = material("bone", BONE, BONE_SHADE, ROUGHNESS, SCATTER, 0.0, middle, reach, fade, FAR)
    cartilage = material(
        "cartilage",
        CARTILAGE,
        CARTILAGE_SHADE,
        CARTILAGE_ROUGHNESS,
        CARTILAGE_SCATTER,
        CARTILAGE_COAT,
        middle,
        reach,
        fade,
        CARTILAGE_FAR,
    )
    for o in skeleton:
        o.data.materials.clear()
        o.data.materials.append(cartilage if o.get("modelled") else bone)
        o.data.shade_smooth()
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
    scene.view_settings.look = LOOK
    scene.view_settings.exposure = EXPOSURE
    world(hdri)


def look_check() -> int:
    """Blocks for bones, rendered small from the front under a plain white
    light: the backdrop is light and neutral, bone is warm, cartilage is
    blue-white, a far bone is paler than a near one, a far cartilage is
    paler than a near one, still blue and apart from a far bone, and a hip
    bone shows above the picture's bottom edge and has faded into the
    backdrop on it. Then once more under a light strong enough to make
    bone as light as the backdrop: the farthest bone still stands clear of
    it, and the far cartilage of it and of a far bone."""
    import tempfile

    from bpy_extras.object_utils import world_to_camera_view

    def block(name: str, at: tuple[float, float, float], size: float) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cube_add(size=size, location=at)
        obj = bpy.context.active_object
        obj["system"], obj["name"] = "skeletal", name
        return obj

    def blocks(lean: float = 0.0) -> list[bpy.types.Object]:
        """A fresh scene of blocks, standing, or leaning back by `lean`
        degrees as one. The two vertebrae hide behind the sternum, whose
        top is level with the disc between them."""
        empty_scene()
        made = [
            block("sternum", (-0.10, -0.12, 0.40), 0.10),
            block("left scapula", (0.10, 0.12, 0.40), 0.10),
            block("costal cartilage", (-0.10, -0.12, 0.22), 0.08),
            block("left hip bone", (0.10, 0.0, 0.0), 0.16),
            block("right hip bone", (-0.10, 0.0, 0.0), 0.16),
            block("left twelfth rib", (0.10, 0.20, 0.22), 0.04),
            block("far costal cartilage", (0.0, 0.11, 0.31), 0.08),
            block("second thoracic vertebra", (-0.10, 0.0, 0.47), 0.03),
            block("third thoracic vertebra", (-0.10, 0.0, 0.43), 0.03),
        ]
        made[2]["modelled"] = made[6]["modelled"] = True
        for o in made:
            o.matrix_world = Matrix.Rotation(-math.radians(lean), 4, "X") @ o.matrix_world
        bpy.context.view_layer.update()
        return made

    bones = blocks()
    scene = bpy.context.scene
    standing = upright(bones)
    stage(bones, None, 16)
    scene.render.resolution_percentage = 5

    def shot() -> np.ndarray:
        """The front view as rows of colours, from the bottom row up."""
        with tempfile.TemporaryDirectory() as tmp:
            render(bones, "front", Path(tmp) / "front.png")
            image = bpy.data.images.load(str(Path(tmp) / "front.png"))
            return np.array(image.pixels[:]).reshape(image.size[1], image.size[0], 4)[:, :, :3]

    picture = shot()
    height, width = picture.shape[:2]

    def seen(x: float, y: float, z: float, picture: np.ndarray = picture) -> np.ndarray:
        """The picture's colour where a point of the scene lands."""
        at = world_to_camera_view(scene, scene.camera, Vector((x, y, z)))
        return picture[min(int(at.y * height), height - 1), min(int(at.x * width), width - 1)]

    def apart(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max())

    paper = picture[height - 1, 0]
    near, far = seen(-0.10, -0.17, 0.40), seen(0.10, 0.07, 0.40)  # the blocks' front faces
    cartilage, far_cartilage = seen(-0.10, -0.16, 0.22), seen(0.0, 0.07, 0.31)
    hip = seen(0.10, -0.08, 0.07)
    edge = picture[
        0,
        min(int(world_to_camera_view(scene, scene.camera, bones[3].location).x * width), width - 1),
    ]
    world(None, BRIGHT)
    bright = shot()
    lit, farthest = seen(-0.10, -0.17, 0.40, bright), seen(0.10, 0.18, 0.22, bright)
    lit_cartilage, lit_far = seen(0.0, 0.07, 0.31, bright), seen(0.10, 0.07, 0.40, bright)

    def placed(view: str) -> np.ndarray:
        """Where the blocks' corners inside the box to frame land in the picture."""
        aim(bones, view)
        bpy.context.view_layer.update()
        floor = frame(bones)[0][2]
        corners = [o.matrix_world @ Vector(c) for o in bones for c in o.bound_box]
        at = [world_to_camera_view(scene, scene.camera, c) for c in corners if c.z > floor]
        return np.array([(a.x, a.y) for a in at])

    def held(at: np.ndarray) -> bool:
        """Margins at the sides and the top, the fade's band free below."""
        slack = 0.005
        return bool(
            at[:, 0].min() > MARGIN - slack
            and at.max() < 1 - MARGIN + slack
            and at[:, 1].min() > FADE - slack
        )

    def filled(at: np.ndarray) -> bool:
        """The bones reach the margins across the picture or up and down it."""
        across, upward = np.ptp(at, axis=0)
        return bool(across > 1 - 2 * MARGIN - 0.01 or upward > 1 - MARGIN - FADE - 0.01)

    views = {view: placed(view) for view in VIEWS}
    # The same blocks leaning back, as a thorax lies in a CT, at the pitch the
    # rule finds for them.
    bones = blocks(LEAN)
    leaning = upright(bones)
    stage(bones, None, 16, leaning["pitch"])
    bpy.context.scene.render.resolution_percentage = 5
    stood = shot()
    checks = {
        "the backdrop is light": paper.min() > 0.85,
        "the backdrop is neutral": paper.max() - paper.min() < 0.03,
        "bone is warm": near[0] > near[1] > near[2] + 0.03,
        "cartilage is blue-white": cartilage[2] > cartilage[1] > cartilage[0]
        and cartilage.min() > 0.4,
        "a far bone is paler than a near one": apart(far, paper) < apart(near, paper) - 0.02,
        "a far cartilage is paler than a near one": apart(far_cartilage, paper)
        < apart(cartilage, paper) - 0.02,
        "and still blue": far_cartilage[2] > far_cartilage[1] > far_cartilage[0] + 0.03,
        "and stands clear of a far bone": apart(far_cartilage, far) > CLEAR,
        "the light of the second picture makes bone as light as the backdrop": lit.max()
        > paper.max() - 0.03,
        "and the farthest bone still stands clear of the backdrop": paper.mean() - farthest.mean()
        > CLEAR,
        "as the far cartilage does, and of a far bone": apart(lit_cartilage, paper) > CLEAR
        and apart(lit_cartilage, lit_far) > CLEAR,
        "a hip bone shows above the picture's bottom edge": apart(hip, paper) > 0.1,
        "a hip bone has faded into the backdrop on the edge": apart(edge, paper) < 0.05,
        "each view holds the bones above the hips with a margin": all(map(held, views.values())),
        "and is filled by them": all(map(filled, views.values())),
        "blocks standing upright need no pitch": abs(standing["pitch"]) < 0.5,
        "blocks leaning back need the pitch they lean by": abs(leaning["pitch"] - LEAN) < 0.5,
        "and at that pitch give the picture of the blocks standing": np.abs(stood - picture).mean()
        < 0.01
        and np.percentile(np.abs(stood - picture), 99) < 0.08,
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


def upright(objects: list[bpy.types.Object]) -> dict[str, float] | None:
    """The view pitch that stands a lying thorax upright, by Gray's level
    (1918, surface anatomy of the thorax): in the erect body the jugular
    notch is in the same horizontal plane as the lower border of the body
    of the second thoracic vertebra. Measured in the scan, in the body's
    sagittal plane: how many degrees the top of the sternum stands above
    the level of the disc between the second and third thoracic vertebrae,
    as seen from that disc. That angle is the pitch. None when the sternum
    or one of the two vertebrae is not in the file."""
    named = {o.get("name", ""): o for o in objects}
    parts = [
        named.get(n) for n in ("sternum", "second thoracic vertebra", "third thoracic vertebra")
    ]
    if not all(parts):
        return None
    sternum, *vertebrae = (coords(o) @ np.array(o.matrix_world.to_3x3()).T
                           + np.array(o.matrix_world.translation) for o in parts)  # fmt: skip
    notch = sternum[sternum[:, 2] > sternum[:, 2].max() - NOTCH].mean(axis=0)
    # A vertebra's body is its front part; the disc lies midway between two bodies.
    bodies = [v[v[:, 1] < v[:, 1].min() + BODY * np.ptp(v[:, 1])].mean(axis=0) for v in vertebrae]
    disc = (bodies[0] + bodies[1]) / 2
    above, ahead = notch[2] - disc[2], disc[1] - notch[1]
    return {
        "pitch": round(math.degrees(math.atan2(above, ahead)), 1),
        "notch_above_t2_t3_disc_mm": round(1000 * above, 1),
        "notch_ahead_of_t2_t3_disc_mm": round(1000 * ahead, 1),
    }


def room() -> Matrix:
    """The studio as it stands around the bones: the camera, the light,
    the haze and the box to frame are all set in it. A view pitch turns
    it, on the camera's rig, about the body's left-right axis; without one
    it is the scene itself."""
    rig = bpy.context.scene.camera.parent
    return rig.matrix_basis.copy() if rig else Matrix()


def in_room(o: bpy.types.Object) -> np.ndarray:
    """An object's vertices as they stand in the room."""
    world = np.array(room().inverted() @ o.matrix_world)
    return coords(o) @ world[:3, :3].T + world[:3, 3]


def frame(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    """The box to frame, in the room: the whole skeleton, from the top of
    the shoulders down to the top of the hip bones (or the lowest
    vertebra)."""
    points = np.concatenate([in_room(o) for o in objects])
    lo, hi = points.min(axis=0), points.max(axis=0)
    hips = [o for o in objects if "hip bone" in o.get("name", "")]
    if hips:
        lo[2] = min(in_room(o)[:, 2].max() for o in hips) - 0.02
    return lo, hi


def fading(objects: list[bpy.types.Object]) -> float:
    """The share of the picture's height over which bones fade out at its
    bottom edge: FADE when some reach below the box to frame, else none."""
    floor = frame(objects)[0][2]
    lowest = min(in_room(o)[:, 2].min() for o in objects)
    return FADE if lowest < floor - 1e-4 else 0.0


def aim(objects: list[bpy.types.Object], view: str) -> None:
    """Turns the camera to the view and fits the picture to the bones as it
    sees them: everything inside the box to frame, with MARGIN free on each
    side and above. Below, bones that go on under the box fade out over a
    band of their own, so nothing inside the box is faded or cut."""
    turn, tilt = (math.radians(v) for v in VIEWS[view])
    camera = bpy.context.scene.camera
    camera.data.type = "ORTHO"
    camera.rotation_euler = (math.pi / 2 - tilt, 0, turn)
    right, up, back = (np.array(axis) for axis in camera.rotation_euler.to_matrix().transposed())
    lo, hi = frame(objects)
    points = np.concatenate([in_room(o) for o in objects])
    points = points[np.all((points >= lo - 1e-4) & (points <= hi + 1e-4), axis=1)]
    across, upward, toward = points @ right, points @ up, points @ back
    wide, tall = np.ptp(across), np.ptp(upward)
    below = fading(objects) or MARGIN
    aspect = RESOLUTION[0] / RESOLUTION[1]
    height = max(tall / (1 - MARGIN - below), wide / (1 - 2 * MARGIN) / aspect)
    spare = height * (1 - MARGIN - below) - tall  # when the width decides: shared above and below
    middle = upward.min() - below * height - spare / 2 + height / 2
    # ortho_scale spans the picture's longer side.
    camera.data.ortho_scale = height * max(1.0, aspect)
    camera.location = Vector(
        (across.min() + across.max()) / 2 * right + middle * up + (toward.max() + 1.0) * back
    )


def render(objects: list[bpy.types.Object], view: str, path: Path) -> None:
    scene = bpy.context.scene
    aim(objects, view)
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
    asked = option(args, "--pitch", "0")
    trial(option(args, "--look", ""))
    empty_scene()
    objects = [o for o in import_glb(source) if o.type == "MESH"]
    skeleton = [o for o in objects if o.get("system") == "skeletal"]
    for o in objects:
        o.hide_render = o not in skeleton
    measured = upright(skeleton)
    if asked == "upright" and measured is None:
        sys.exit("studio.py: --pitch upright needs the sternum and the second and third vertebrae")
    pitch = measured["pitch"] if asked == "upright" else float(asked)
    stage(skeleton, hdri, samples, pitch)
    device = gpu()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        render(skeleton, view, out / f"{name}.{view}.png")
    view = {"pitch": pitch, "asked": asked, "rule": RULE, "upright": measured}
    write_json(str(out / f"{name}.view.json"), view)
    print(f"studio: {len(views)} view(s) of {len(skeleton)} nodes on {device} -> {out}")


if __name__ == "__main__":
    main()
