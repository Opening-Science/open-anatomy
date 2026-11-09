# open-anatomy

Human anatomy models built from openly licensed CT scans by scripts anyone can
run. Each structure (organ, bone, cartilage) is segmented from a real CT,
meshed, named by its [UBERON](https://obophenotype.github.io/uberon/) or
[FMA](https://bioportal.bioontology.org/ontologies/FMA) ID, and placed inside
an open, rigged ANNY body. The same inputs give the same files.

**These models are for research, education and software. They are not
clinically validated and must not guide care.**

## What a build gives you

For each reference body, in the gitignored `data/` folder:

- `data/reference/<body>.reference.glb`: the ANNY body (skin and its 104-bone
  skeleton) with 59 to 65 CT structures, each its own node, parented to the
  ANNY bone it follows;
- `data/reference/<body>.report.json`: the fit (joint residuals, the share of
  each structure inside the skin) and the case's record (sex, age, recorded
  pathology, whether it shows typical anatomy);
- `data/work/<case>/<case>.reference.manifest.json`: every structure's ID,
  name, system, licence, triangles, volume and its drift from the voxel volume.

| Reference | CT case | Recorded pathology | Typical anatomy |
| --- | --- | --- | --- |
| anny-adult-m | s0903, male, 88 | none | yes |
| anny-adult-f | s0859, female, 74 | inflammation (thorax) | yes |
| anny-adult-m-w0.9-h0.5.s0667 | s0667, male, 65 | none | yes |
| anny-adult-m-w0.9-h0.9.s1086 | s1086, male, 45 | tumour (thorax, abdomen) | **no: a pathological reference** |

The structures are the liver, gallbladder, stomach, pancreas, spleen, both
kidneys, heart, aorta and inferior vena cava, and the trunk skeleton: the
vertebrae, sacrum, ribs, sternum, clavicles, scapulae and hip bones, where the
scan covers them whole. A structure the scan cuts is left out, and a shoulder
girdle cut on either side is left out on both, so every body is symmetric; each
manifest lists what was left out and why.

**Modelled, not from the scan:** the costal cartilages. Their segmentation is
too broken to show, so each is a smooth bridge from a rib's front end to the
sternum (ribs 2 to 7, in left-right pairs). Their node and manifest entry say
`modelled: true`. Everything else is segmented from the CT. The CTs
were taken lying down and the ANNY bodies stand, so organs sit as they do
supine.

Large inputs and outputs never go into git; [sources.json](sources.json) pins
every download and every build by SHA-256.

## How to build

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), Blender 5.2.x LTS, and a
GPU for segmentation (a CPU works, slowly).

```bash
uv venv .venv -p 3.12 && uv pip install --python .venv/bin/python -e '.[pipeline]'
.venv/bin/python scripts/reference.py --render
```

`reference.py` runs, for each reference body:

1. `scripts/fetch.py`: downloads the TotalSegmentator dataset from Zenodo,
   checks it against its pins, and unpacks the case;
2. `scripts/segment.py`: TotalSegmentator's Apache-2.0 tasks only, scored
   against the dataset's own reference masks;
3. `scripts/mesh.py`: smooth, closed surfaces that keep each mask's volume;
4. `scripts/bake_anny.py`: the ANNY body, sized to the case;
5. `blender/assemble.py`: the CT structures placed in the body and attached to
   its skeleton; with `--render`, front, three-quarter and trunk views.

Self-checks, also run in CI:

```bash
.venv/bin/python scripts/mesh_check.py
scripts/self-check.sh   # Blender's; skips without Blender
```

## Licences

- **Code** is under the [Apache License 2.0](LICENSE).
- **Assets** derived from CC BY sources go in [`assets/cc-by/`](assets/cc-by/),
  under CC BY; from CC BY-SA sources in [`assets/cc-by-sa/`](assets/cc-by-sa/).
  Share-alike material never mixes into the CC BY folder.

## Credits

[NOTICE](NOTICE) has every credit line in full. In short:

- **TotalSegmentator dataset (CC BY 4.0): credit required.** Every mesh,
  slice and reference body here is derived from its CT scans, and attribution
  is the licence's condition, not a courtesy. Anything you share that is
  built from them must carry the credit: *TotalSegmentator dataset, Jakob
  Wasserthal, Zenodo, CC BY 4.0* (DOI
  [10.5281/zenodo.10047292](https://doi.org/10.5281/zenodo.10047292), small
  subset [10.5281/zenodo.10047263](https://doi.org/10.5281/zenodo.10047263)),
  and please cite Wasserthal et al., *Radiology: Artificial Intelligence*
  2023 ([10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)).
- **TotalSegmentator (Apache-2.0):** the segmentation code and the weights of
  the Apache-2.0 tasks we run. It is installed from PyPI, not redistributed
  here, and ships no NOTICE file of its own; NOTICE credits it, and if you
  redistribute its code, keep its licence with it.
- **ANNY (CC0 shapes, Apache-2.0 code):** CC0 asks for no credit; we credit
  NAVER LABS Europe anyway (arXiv:2511.03589), as good practice. The bake uses
  only the CC0 shapes, never ANNY's non-commercial SMPL/SMPL-X data.
- **BodyParts3D and HRA:** nothing from them is used in this repository's
  outputs, so no credit is needed. `blender/fit_atlas.py` can fit them; if
  you use it, credit them, and put BodyParts3D results (CC BY-SA) under
  `assets/cc-by-sa/`.

See [CONTRIBUTING](CONTRIBUTING.md) to take part.
