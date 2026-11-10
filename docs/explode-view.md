# Explode view

**Status:** `scripts/mesh.py` writes each set's GLB and its manifest under
this contract, version 1, and `tests/test_mesh.py` and `scripts/mesh_check.py`
check it. Two things do not follow it yet and say so below: a retopologised
GLB has no manifest of its own, and rigged reference bodies still carry the
values of the scan they were meshed from.

A viewer reads the contract to spread a body's structures apart with a slider
and to isolate one structure at a time.

## What the viewer does

- **Explode:** a slider t from 0 to 1 moves every structure outward from the
  body's centre, along its own direction, so overlapping organs and bones come
  apart without changing shape. At t = 0 the body is as built; at t = 1 no two
  structures' bounds overlap.
- **Isolate:** picking a structure hides the others (or fades them), frames the
  camera on it, shows its plain name and the one line about it, and says when
  the shape is modelled rather than segmented from a scan. The anatomical
  name, ID, system and licence are there for a clinician's view.
- **Filter by system:** show or hide whole systems (skeleton, digestive and so
  on), as layers do today.

## Body frame

Everything below is in the body frame: metres, +Y superior, +Z anterior, +X
the body's left, with the origin at the centre of the bounding box of all the
structures in the file. The manifest states the units and axes, and gives the
origin's place in the scan as `centre_offset_m`.

## Node contract (version 1)

Every structure is **one glTF node with one mesh**, never merged with another.

| Field | Where | Content |
| --- | --- | --- |
| name | `node.name` | the ontology ID with `_` for `:`, unique in the file: `UBERON_0002107`, `FMA_7485` |
| mesh | `node.mesh` | the structure's surface, its vertices relative to its **centre** |
| home | `node.translation` | the centre in the parent node's space; equal to `extras.centre` when the parent is the scene root, as in every GLB mesh.py writes |
| `extras.term` | string | `UBERON:0002107`, `FMA:7485` |
| `extras.label` | string | the anatomical name in English: `sternum`, `left seventh rib` |
| `extras.plain` | string | the plain-language name: `Breastbone`, `Left 7th rib` |
| `extras.about` | string | one line on what it is or does, in plain words |
| `extras.system` | string | one of the systems below |
| `extras.layer` | string | `skin`, `muscle`, `skeleton`, `organs` or `imaging`; `skin` is ANNY's synthetic skin only |
| `extras.licence` | string | SPDX ID of the asset's licence, e.g. `CC-BY-4.0` |
| `extras.modelled` | bool | true when the shape is modelled, not segmented from the CT (the costal cartilages) |
| `extras.typical` | bool | false when the source case is pathological (its record in sources.json says `"typical": false`) |
| `extras.pieces` | `one` or `several` | `several` for sets such as the costal cartilages |
| `extras.centre` | `[x,y,z]` | the centroid of the volume the surface encloses, to the micrometre |
| `extras.bounds` | `[[x,y,z],[x,y,z]]` | axis-aligned bounds at t = 0, lowest corner then highest, rounded outward to the micrometre |
| `extras.volume_ml` | number | enclosed volume in millilitres, to one decimal |
| `extras.explode` | object | `{"direction": [x,y,z], "distance": d}` (see below) |

Field names are lower case with underscores. An earlier draft of this page
spelled four of them `bbox`, `volumeMl`, `centreM` and `bodyCentreM`; no file
was ever written with those.

With the **centre as the node's origin**, `translation + vertex` is a point's
position in the body frame, moving a structure is changing one translation,
and rotating it for inspection turns it about its own middle.

### Systems

`skeleton`, `cardiovascular`, `respiratory`, `digestive`, `urinary`,
`nervous`, `endocrine`, `muscular` (`SYSTEMS` in scripts/structures.py). The
set is closed: a new system is a contract change. The costal cartilages are
part of `skeleton`. The spleen is a lymphoid organ; with no lymphoid system
in the list it goes with `digestive`, the organs it lies among.

The layer follows from the system: `skeleton` is the `skeleton` layer,
`muscular` the `muscle` layer and every other system the `organs` layer.
mesh.py writes no `skin` or `imaging` node.

### Explode direction and distance

mesh.py computes them when it builds the GLB, so every viewer explodes a body
the same way, and the same masks always give the same values:

- **direction:** from the body centre (the centroid of all structures'
  centres, weighted by volume) to the structure's centre, unit length, to six
  decimals. A structure whose centre is less than 1 cm from the body centre
  takes the mean direction of its system's other structures instead. If that
  mean is shorter than 0.1 (left and right cancel, as in the skeleton), or
  the system has no other structure, the direction is +Z, anterior. mesh.py
  never normalises a vector shorter than these limits.
- **distance:** how far it moves at t = 1, in metres, on a millimetre grid.
  It is at least the structure's bounding radius (the farthest corner of its
  bounds from its centre) plus 2 cm. That alone leaves neighbours overlapping,
  so the structures are then placed one by one, nearest the body centre first
  (ties by their segmentation label): each takes the smallest distance at
  which its moved bounds stay 5 mm clear of the moved bounds of every
  structure placed before it. A left and right pair is placed together and
  shares the larger distance the two need, so the pair moves symmetrically and
  the two clear each other as well. Two of a pair that have the same direction
  (both at the body centre) could never part that way and are placed one after
  the other.

So at t = 1 any two structures' bounds are at least 5 mm apart, on the numbers
the files carry: the distances are worked out from the rounded centres, bounds
and directions. Between 0 and 1 bounds may still overlap; only t = 1 is
promised.

At slider value t the viewer sets `translation = home + t × distance ×
direction` when the node's parent is the scene root, where home is the same
as the centre.

### After mesh.py: retopology

blender/retopo.py remeshes the bones of a GLB. Every node stays a child of
the scene root and keeps its name, its translation and its extras as mesh.py
wrote them. The fields are not worked out again: `centre`, `bounds` and
`volume_ml` describe mesh.py's surface, and retopo.py keeps a remeshed
surface only when every point of mesh.py's is within 1 mm of it. The 5 mm gap
at t = 1 leaves room for that. The retopologised GLB has no manifest of its
own yet: the manifest's `file` and `sha256` name the GLB mesh.py wrote.

### Rigged reference bodies

In a reference body each structure node is a child of an ANNY bone, so it
moves with the skeleton. The explode offset is then applied in the parent
bone's space: `node.translation` (home) is relative to the bone, not the body
frame, so the viewer converts `t × distance × direction` from body space into
the bone's space at bind pose and adds it to home. `extras.centre` stays the
body-frame value for menus, framing and the direction rule. With
the body posed, the offset turns with the bone, which keeps structures beside
their own part of the body.

This part is design only. blender/assemble.py copies each node's extras as
mesh.py wrote them, in the frame of the scan, so a reference body's `centre`,
`bounds` and `explode` are not yet those of the body it is fitted to.

## Manifest

Next to every GLB, a JSON manifest lists the same fields per node, so a viewer
can build menus, search and filters without parsing the GLB. Each entry of
`nodes` is the node's name and its extras, field for field, followed by the
build's own record of that mesh:

```json
{
  "contract": 1,
  "case": "s0086",
  "set": "skeleton",
  "file": "s0086.skeleton.glb",
  "sha256": "…",
  "attribution": "Derived from case s0086 of the TotalSegmentator dataset …",
  "units": "metres",
  "axes": "+Y superior, +Z anterior, +X the body's left",
  "centre_offset_m": [0.003016, 0.321954, 0.15582],
  "body_centre_m": [-0.001865, -0.010577, -0.028886],
  "seg_dir": "data/work/s0086/seg",
  "nodes": [
    {
      "name": "FMA_7485", "term": "FMA:7485", "label": "sternum",
      "plain": "Breastbone",
      "about": "The flat bone in the middle of the chest that the ribs join at the front",
      "system": "skeleton", "layer": "skeleton", "licence": "CC-BY-4.0",
      "modelled": false, "typical": true, "pieces": "one",
      "centre": [-0.000968, 0.200118, 0.074632],
      "bounds": [[-0.034984, 0.082678, 0.040446], [0.033603, 0.266077, 0.116292]],
      "volume_ml": 90.0,
      "explode": {"direction": [0.00382, 0.897516, 0.440966], "distance": 0.15},
      "triangles": 15000, "voxel_volume_ml": 90.0, "volume_drift": -0.0001,
      "dropped_share": 0.0
    }
  ],
  "missing": [],
  "fragments": [],
  "dropped": []
}
```

`contract`, `file`, `sha256`, `body_centre_m` (the centre the directions run
from) and the nodes' contract fields are what a viewer reads. The rest is the
build's record: the case and set, the attribution, where the masks were, and
per node the triangle count, the mask's voxel volume and the mesh's drift from
it (both null for a modelled shape), and, where they apply, `dropped_share`,
`knobs_trimmed_ml` and `bridged_gap_mm`. `missing`, `fragments` and `dropped`
list what the build left out or moved, with the reason.

## Rules for producers

1. One node per structure; no merged meshes, no shared meshes between nodes.
2. Node names are unique and stable across releases: the ontology ID.
3. Vertices relative to the centre; the centre in `extras.centre` (body
   frame) and home in `translation` (parent's space).
4. Every `extras` field in the table present on every node, in the GLB and
   the manifest alike, with the same values.
5. A pathological case marks every node `typical: false`.
6. Only anatomy nodes. The CT trunk outline (outline.glb) and the landmarks
   GLB are fitting inputs and never become nodes under this contract.
7. At t = 1 no two nodes' bounds overlap, and the same input gives the same
   bytes.

## Open questions

1. Should the explode run per system first (systems move apart, then
   structures within each), or all structures at once as above?
2. For imaging slices (the `imaging` layer), should explode move the planes
   apart along their normals?
3. Does the viewer need a per-structure "home" rotation for isolation, or is
   the bounding box enough to frame the camera?
4. Clearing axis-aligned bounds makes the distances long on a whole skeleton:
   up to 0.6 m for the scapulae of s0086, on a skeleton 0.63 m tall. Is a
   tighter rule wanted, such as staging by system or clearing the surfaces
   rather than their boxes?
