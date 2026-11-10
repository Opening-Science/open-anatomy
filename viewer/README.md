# Viewer: take the body apart

A static page that loads a body's GLB and its manifest and lets anyone take
the body apart with one slider, look at one part at a time and find parts by
their everyday names. Plain HTML and ES modules; three.js comes from a pinned
CDN import map (`three@0.187.0` on jsdelivr, MIT licence). No build step, no
npm dependency at run time.

## Run it

The page fetches the GLB and the manifest, so serve the repository root over
HTTP (`file://` will not do):

```bash
python3 -m http.server 8000
# the synthetic test body
open 'http://localhost:8000/viewer/index.html?glb=/tests/fixtures/viewer/body.glb&manifest=/tests/fixtures/viewer/body.manifest.json'
# a built case, through a worktree's data symlink
open 'http://localhost:8000/viewer/index.html?glb=/data/atlas/s0086/s0086.skeleton.glb&manifest=/data/work/s0086/s0086.skeleton.manifest.json'
```

Without `glb` and `manifest` the page looks for `./body.glb` and
`./body.manifest.json` next to `index.html`, and offers a file picker (choose
both files) when it finds neither or cannot read them.

## What it does

- **Take apart.** One slider, 0 (as built) to 1 (fully apart). Every structure
  moves out along its explode direction by `t × distance`; when the body has
  several systems they also move apart as blocks. At 1 no two structures'
  bounds overlap. The camera keeps the whole body in view as it spreads.
- **One part at a time.** Click or tap a part, or pick it from the list: the
  others fade to a faint ghost, the camera frames it, and a card shows its
  plain name, one line about it and whether it was *segmented from CT* or
  *modelled*. Escape, "Show all", a tap on empty space or a second tap on the
  part bring the whole body back. Drag to turn, scroll or pinch to zoom. With
  a mouse, the part under the pointer is named before you click.
- **Parts list.** Every part by plain name, grouped by system (Skeleton, Heart
  and vessels, Lungs, Digestive, Urinary, Nervous, Endocrine, Muscles, as
  present), with a checkbox per group to show or hide it, and a search box
  (`/` focuses it, Escape clears it) that matches plain names, anatomical
  names and IDs, and a few everyday words ("spine", "pelvis", "chest", "T1").
- **Clinician.** A toggle in the header switches the names to the anatomical
  terms and adds the ID, system, licence, volume, piece count and triangle
  count to the card, and says which words were derived in the viewer. The main
  view has no Latin, no letter codes and no IDs.
- **Mouse, touch, keyboard.** Tab reaches every control; the list items are
  buttons; the slider takes the arrow keys, Home and End; with the 3D view
  focused the arrow keys turn the body and `+` and `-` zoom. The layout fits a
  390 px phone: the slider spans the width, the list becomes a sheet behind the
  "Parts" button, and an isolated part's card sits under the 3D view, which
  shrinks to make room. People who ask their system for less motion get the
  camera moves at once instead of glides.
- **Links.** The URL carries the view: `t`, `isolate=<node id>`,
  `hide=<systems>`, `clinician=1`, `q=<search>`, plus `glb` and `manifest`.
  The page keeps the address up to date, so copying it shares the view.

## How the explode is computed

`explode.js` follows docs/explode-view.md. A manifest with `explode`
fields is used as it is. Where a field is missing the viewer derives it and
marks it derived: the direction runs from the body centre (the manifest's, or
the volume-weighted centroid of the structures' centres) to the structure's
centre; a structure within 1 cm of the body centre takes its system's mean
direction, and a mean shorter than 0.1 falls back to +Z (anterior); the
distance is the bounding radius plus 2 cm; left and right partners mirror each
other. The centre and bounds come from the manifest or, failing that, from the
loaded mesh. The plain name and the one line come from the manifest's `plain`
and `about` or, failing that, from the anatomical name (`manifest.js`).

Then, nearest to the body centre first, each structure keeps its direction
and moves only as far along it as it needs to clear the structures already
placed, so at 1 no two axis-aligned bounds overlap. The console says how many
structures were derived and how many distances had to grow (and by how much).

## Files

| File | What |
| --- | --- |
| `index.html` | the page, its styles and the three.js import map |
| `main.js` | three.js scene, loading, picking, camera, panel and controls |
| `explode.js` | pure geometry: directions, distances, block offsets, overlap resolution |
| `state.js` | pure state: slider, isolation, hidden systems, clinician, search, URL parameters |
| `manifest.js` | pure manifest reading: validation, plain names, about lines, search words, derived fields |
| `package.json` | marks the folder's `.js` files as ES modules for `node --test`; not a package |
| `test/` | `node --test` suites, the browser test and the fixture writer |

## Tests

```bash
node --test 'viewer/test/*.test.js'                                  # pure modules, no browser
node --test --test-timeout=300000 viewer/test/screenshots.test.mjs   # headless Chromium
```

The browser test serves the repository on a local port and drives Chromium
over its DevTools protocol, with software rendering so it runs the same
everywhere.

- From URL parameters alone it captures the body at t = 0, 0.5 and 1 and one
  isolated part, and asserts that the PNGs exist, are drawn and shaded, differ
  in the 3D view, that the page reported no errors and that no bounds overlap
  at 1. One more capture shows the clinician view.
- With the mouse and the keyboard at 1280 px it hovers, clicks a part (the
  picture must match the linked one), presses Escape, uses "Show all", drags,
  scrolls, turns and zooms with the keys, moves the slider with the keys,
  searches, hides and shows a system, flips the clinician toggle and picks a
  modelled part from the list.
- With no body next to the page it checks the file picker, then chooses the
  two files.
- With touch at 390 px it taps a part, checks the card sits under the 3D view,
  pinches, taps "Show all", opens the parts sheet and picks from it, and
  checks that nothing scrolls sideways and that the controls are finger-sized.

Output goes to `out/viewer-screenshots/` (`t0`, `t05`, `t1`, `isolate`,
`clinician`, `hover`, `isolate-click`, `picker`, `picked`, `phone-whole`,
`phone`, `phone-parts`, and a `report.json`). Environment: `CHROMIUM`
(binary), `VIEWER_GLB` and `VIEWER_MANIFEST` (another body), `VIEWER_ISOLATE`,
`VIEWER_SHOTS`. A built case in software rendering draws a few frames a
second, so its run takes about a minute:

```bash
VIEWER_GLB=data/atlas/s0086/s0086.skeleton.glb \
VIEWER_MANIFEST=data/work/s0086/s0086.skeleton.manifest.json \
VIEWER_SHOTS=out/viewer-screenshots-s0086 \
node --test --test-timeout=600000 viewer/test/screenshots.test.mjs
```

The default body is the synthetic fixture in `tests/fixtures/viewer/`, written
by `node viewer/test/make-fixture.mjs` (ten simple shapes in two systems, no
scan data; `fixture.test.js` checks the committed files match). CI runs both
suites in the job "Viewer tests" (`.github/workflows/viewer.yml`) and keeps
the screenshots as an artifact.

## Changing three.js

The import map in `index.html` names one release and gives the SHA-384 of
each file the page loads from it, so the browser refuses anything else. To
move to another release, change the version in every URL and replace each
hash:

```bash
for f in build/three.module.js build/three.core.js \
         examples/jsm/loaders/GLTFLoader.js examples/jsm/controls/OrbitControls.js \
         examples/jsm/utils/BufferGeometryUtils.js examples/jsm/utils/SkeletonUtils.js; do
  printf '%s sha384-%s\n' "$f" \
    "$(curl -fsSL "https://cdn.jsdelivr.net/npm/three@0.187.0/$f" | openssl dgst -sha384 -binary | openssl base64 -A)"
done
```

Then run the browser test. A wrong hash stops the page with "Could not load
the 3D library"; a file the page fetches from outside that the map does not
list fails the test, because the browser would load it unchecked.

## Manifest fields the viewer reads

Node id from `name` (when it is an id such as `FMA_7485`), `node` or `term`;
anatomical name from `label` or a non-id `name`; `plain`; `about`; `system`
(`skeletal` maps to `skeleton`, `lymphatic` to `lymphoid`); `layer`; `licence`;
`modelled`; `pieces`; `volume_ml` or `volumeMl`; `triangles`; `centre`,
`centreM` or `centre_m`; `bounds` or `bbox`; `explode` `{direction, distance}`;
and the body centre from `body_centre_m` or `bodyCentreM` at the top level.
Errors (no nodes, duplicate or missing ids, a term that does not match its id,
a missing system, malformed geometry or explode fields) stop the load with a
message; unknown systems and non-unit directions are warnings in the console.
