# open-anatomy

Open, imaging-based human anatomy models that anyone can rebuild from scripts.

The models are built from openly licensed CT scans and open body models, not
from a single donor. Every step, from segmentation to the released glTF, is a
script in this repository, so the same inputs give the same files. TwinLLM and
OpenTwin load the releases as their inner anatomy layers.

**These models are for research, education and software. They are not
clinically validated and must not guide care.**

## Layers

| Layer    | What it shows                                                                    |
| -------- | -------------------------------------------------------------------------------- |
| skin     | the body surface, from an open parametric body (ANNY or MakeHuman)               |
| muscle   | the muscles                                                                      |
| skeleton | bones and joints, rigged to the ANNY skeleton                                    |
| organs   | organs and vessels, segmented from CT                                            |
| imaging  | the "MRI layer": the same segmented structures shown as slice planes or a volume |

Every mesh is named by its [UBERON](https://obophenotype.github.io/uberon/) or
[FMA](https://bioportal.bioontology.org/ontologies/FMA) ID.

## Pipeline

1. **Inputs:** only CC BY or CC0 data (see [Sources](#sources)).
2. **Segmentation:** [TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
   using its Apache-2.0 tasks only (`total`, `body` and similar).
3. **Meshing:** per label, smooth, then surface nets or marching cubes (VTK).
4. **Blender, headless:** clean up the meshes, build a reference body for each
   sex and several body types, rig them, and export glTF levels of detail.
5. **Release:** versioned GLBs pinned by SHA-256, with a manifest listing each
   node's ID, body system and licence.

Large inputs and outputs (CT volumes, meshes) never go into git. They live in
the gitignored `data/` folder and are pinned by SHA-256.

## Sources

Only sources under CC BY or CC0 are used:

- TotalSegmentator CT dataset v2.0.1 (CC BY 4.0,
  [Zenodo 10047292](https://zenodo.org/records/10047292))
- AMOS (CC BY 4.0)
- CT-ORG (CC BY 3.0)
- HRA organs (CC BY 4.0)
- ANNY and MakeHuman body assets (CC0)

Non-commercial datasets and models, and scans of real people outside these
datasets, are never used. [NOTICE](NOTICE) carries the credit line for every
source.

## Licences

- **Code** is under the [Apache License 2.0](LICENSE).
- **Assets** derived from CC BY sources are in [`assets/cc-by/`](assets/cc-by/),
  under CC BY.
- **Assets** derived from CC BY-SA sources are in
  [`assets/cc-by-sa/`](assets/cc-by-sa/), under CC BY-SA. Share-alike material
  never mixes into the CC BY folder.

## Personalisation (planned)

A person's clinician-verified facts will adjust the layers at load time. For
example, after a nephrectomy the model shows one kidney. A small rule map from
FHIR Condition and Procedure codes to anatomy edits will drive this, applied
on the person's own machine. The released base models stay generic. This is a
design only for now: see [docs/personalisation.md](docs/personalisation.md).

## Build one case

Needs Python 3.11+ (3.12 recommended), Blender 5.2+ for the render, and a GPU
for a quick segmentation (a CPU works, slowly).

```bash
uv venv .venv -p 3.12 && uv pip install --python .venv/bin/python -e '.[pipeline]'
.venv/bin/python scripts/fetch.py               # 3.2 GB subset, checked, then case s1086
.venv/bin/python scripts/segment.py --case s1086
.venv/bin/python scripts/mesh.py --case s1086   # data/work/s1086/s1086.organs.glb
blender --background --factory-startup --python scripts/render.py -- \
  --in data/work/s1086/s1086.organs.glb --out data/renders/s1086-organs-front.png
```

`fetch.py` downloads the pinned source from its official record, checks its
MD5 and SHA-256 against [sources.json](sources.json), and unpacks one case.
`segment.py` runs TotalSegmentator's `total` task for ten structures and scores
each against the dataset's own reference segmentation. `mesh.py` writes one GLB
with a closed mesh per structure, named by UBERON ID, and a manifest; the same
masks always give the same bytes. The GLB credits the dataset's creator, DOI
and case in its asset copyright, so the attribution travels with the file.

## Reference bodies

The bodies are ANNY (CC0 shapes): an adult male and an adult female, each
rigged with ANNY's own 104-bone skeleton. They are synthetic: no scan of any
person is used.

```bash
uv run --script scripts/bake_anny.py male data/bodies/anny-adult-m.rigged.glb
uv run --script scripts/bake_anny.py female data/bodies/anny-adult-f.rigged.glb
```

The bake is pinned by SHA-256 in [sources.json](sources.json);
`python3 scripts/bake_check.py --bake` bakes both again and compares. Headless
Blender 5.2.x scripts in [`blender/`](blender/) clean meshes up and fit an
atlas to a body limb by limb; run them through `scripts/blender.py`.

## Self-checks

```bash
.venv/bin/python scripts/mesh_check.py   # meshes synthetic ellipsoids; no download
scripts/self-check.sh                     # every Blender self-check; skips without Blender
python3 scripts/bake_check.py             # the bake's lock, and any baked body
```

See [CONTRIBUTING](CONTRIBUTING.md) to take part.
