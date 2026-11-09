# Explode view

**Status: design only.** This is the contract the twin-llm viewer will read to
spread a body's structures apart with a slider and to isolate one structure at
a time. Nothing here is built yet; the GLBs today carry part of it (one node
per structure, its ID, name, system and licence in `extras`).

## What the viewer does

- **Explode:** a slider t from 0 to 1 moves every structure outward from the
  body's centre, along its own direction, so overlapping organs and bones come
  apart without changing shape. At t = 0 the body is as built.
- **Isolate:** picking a structure hides the others (or fades them), frames the
  camera on it, and shows its name, ID, system and licence.
- **Filter by system:** show or hide whole systems (skeletal, digestive and so
  on), as layers do today.

## Node contract (version 1)

Every structure is **one glTF node with one mesh**, never merged with another.

| Field | Where | Content |
| --- | --- | --- |
| name | `node.name` | the ontology ID with `_` for `:`, unique in the file: `UBERON_0002107`, `FMA_7485` |
| mesh | `node.mesh` | the structure's surface, its vertices relative to its **centre** |
| centre | `node.translation` (and `extras.centre`) | the structure's volume centroid in the body frame, metres |
| `extras.term` | string | `UBERON:0002107`, `FMA:7485` |
| `extras.label` | string | English name, for display |
| `extras.system` | string | one of the systems below |
| `extras.layer` | string | `skin`, `muscle`, `skeleton`, `organs` or `imaging` |
| `extras.licence` | string | SPDX ID of the asset's licence, e.g. `CC-BY-4.0` |
| `extras.bbox` | `[[x,y,z],[x,y,z]]` | axis-aligned bounds in the body frame, metres, at t = 0 |
| `extras.volumeMl` | number | enclosed volume |
| `extras.pieces` | `one` or `several` | `several` for sets such as the costal cartilages |
| `extras.explode` | object | `{"direction": [x,y,z], "distance": d}` (see below) |
| `extras.typical` | bool | false when the source case is pathological (see sources.json) |

The **centre as the node's origin** is the one change from today's GLBs, where
every mesh sits in body coordinates under an identity node. With the centre as
origin, moving a structure is changing one translation, and rotating it for
inspection turns it about its own middle.

### Systems

`skeletal`, `cardiovascular`, `digestive`, `urinary`, `lymphoid`, and later
`respiratory`, `nervous`, `muscular`, `integumentary`. The set is closed: a new
system is a contract change.

### Explode direction and distance

mesh.py computes them when it builds the GLB, so every viewer explodes a body
the same way:

- **direction:** from the body centre (the centroid of all structures' centres,
  weighted by volume) to the structure's centre, unit length. A structure at
  the body centre gets its system's mean direction instead.
- **distance:** how far it moves at t = 1, in metres: the structure's bounding
  radius plus 2 cm, so neighbours clear each other. Paired structures (left and
  right) move symmetrically.

At slider value t the viewer sets `translation = centre + t × distance × direction`.

### Rigged reference bodies

In a reference body each structure node is a child of an ANNY bone, so it
moves with the skeleton. The explode offset is then applied in the parent
bone's space: the viewer converts `t × distance × direction` from body space
into the bone's space at bind pose and adds it to the node's translation. With
the body posed, the offset turns with the bone, which keeps structures beside
their own part of the body.

## Manifest

Next to every GLB, a JSON manifest lists the same fields per node, so a viewer
can build menus, search and filters without parsing the GLB:

```json
{
  "contract": 1,
  "file": "anny-adult-f.reference.glb",
  "sha256": "…",
  "bodyCentreM": [0.0, 1.21, 0.01],
  "nodes": [
    {
      "name": "UBERON_0002107", "term": "UBERON:0002107", "label": "liver",
      "system": "digestive", "layer": "organs", "licence": "CC-BY-4.0",
      "centreM": [-0.06, 1.18, 0.03], "volumeMl": 1497.2, "pieces": "one",
      "explode": {"direction": [-0.82, -0.31, 0.48], "distance": 0.13},
      "typical": true
    }
  ]
}
```

## Rules for producers

1. One node per structure; no merged meshes, no shared meshes between nodes.
2. Node names are unique and stable across releases: the ontology ID.
3. Vertices relative to the centre; the centre in `translation`.
4. Every field in the table present on every node, in the GLB and the
   manifest alike.
5. A pathological case marks every node `typical: false`.

## Open questions

1. Should the explode run per system first (systems move apart, then
   structures within each), or all structures at once as above?
2. For imaging slices (the `imaging` layer), should explode move the planes
   apart along their normals?
3. Does the viewer need a per-structure "home" rotation for isolation, or is
   the bounding box enough to frame the camera?
