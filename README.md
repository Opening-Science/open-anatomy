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
design only for now.

## Self-check

```bash
scripts/self-check.sh   # builds and checks a tiny synthetic mesh; skips without Blender
```

See [CONTRIBUTING](CONTRIBUTING.md) to take part.
