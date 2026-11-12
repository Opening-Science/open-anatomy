# Scan enhancement

How a person's 3D body scan is cleaned up into a sound surface, and how we
know it got better. Today this runs on synthetic ANNY bodies only, damaged on
purpose; nothing here has touched a real scan.

## Where it runs, and what never leaves the machine

Peter decided this on 9 Nov 2026: real scans are processed only on his own
Mac, and the repository holds code only. "Never a real person's scan" means
never in the repository, the tests, a render or any shared place: a real scan,
anything derived from it, and its measures stay on that machine.

- **GitHub holds the code** (the Python and Blender scripts) and nothing else.
  A real scan never goes into the repository, CI, a release, an issue, a log
  or any server.
- **It runs on the person's own Mac**, one scan at a time.
- **Credentials** for a scanner account come from the macOS Keychain (or an
  environment variable) at run time. They are never written to a file in the
  repository and never logged.
- **Outputs stay local**, in a folder the person chooses. The measures below
  stay local too: they are never pasted into an issue, a pull request, a chat
  or any other shared place, because body measures identify a person.
- **No face render ever leaves the machine**, and a protected face region (no
  smoothing, no filling) is required before any real scan is processed;
  `scripts/scan_quality.py --real` refuses to run without it, and every run
  fails when the face changed beyond its limit (below).

## Disk budget

One scan is in flight at a time, in a temporary folder that is deleted when it
finishes or fails.

| Item | Size |
| --- | --- |
| input scan (a body mesh, about 15k–300k vertices) | up to 200 MB with photos |
| cleaned mesh, full detail and two levels of detail | under 30 MB |
| measures (JSON) | under 1 MB |
| peak, while one scan is processed | under 500 MB |

The synthetic demo (13.7k vertices, no photos) uses 0.6 MB per run.

## Input

A scan arrives as two local files, given at run time:

- **The body scan:** a glTF binary (`.glb`, 25–60 MB), synced to the Mac by
  open-twin's `@open-twin/provider-vitronic`. It is skinned: one skin with
  34–36 joints and 2–4 meshes, with `POSITION`, `NORMAL`, `JOINTS_0`,
  `WEIGHTS_0`, usually `TEXCOORD_0` and a texture, and three per-vertex
  attributes of the scanner's own: `_DepthMap`, `_GaussianCurvature` and
  `_MeanCurvature`. The scan settings record the mesh detail (High, Normal or
  Preview), the pose (A-Pose or FreeScan) and the clothing (Tight or Loose).
- **The scanner's measurements:** a JSON file in the shape the scanner's API
  returns, with its markers (each a position and a normal) and its measured
  heights, distances, cross-sections, angles and axes.

The pipeline writes no scanner client: open-twin's provider already handles
the API, its TLS pinning and the credentials, which stay in TwinLLM's
existing connection. The export has no camera photos and no calibration
(neither an endpoint nor files), so reprojection cannot be measured.

The pipeline never reads the folder where real scans are synced. Everything in
this repository is tested on a synthetic scan in the same shape, made by
`blender/make_demo_scan.py` from a rigged ANNY body: one skin, its joints and
weights, smooth normals, a cylindrical UV, and synthetic `_DepthMap`,
`_GaussianCurvature` and `_MeanCurvature` attributes.

## What the cleanup does

`blender/cleanup_mesh.py`: merge duplicate vertices, fill holes, make the
normals consistent and outward, smooth lightly (Taubin, which keeps volume),
and export the cleaned mesh plus levels of detail. A skinned scan keeps its
rig: every vertex of the cleaned mesh, including those that fill a hole, takes
the joint weights and the per-vertex attributes of the nearest point on the
original surface (interpolated across that face), keeps its four strongest
joints and is normalised, and the mesh is exported with its skeleton.

## How we know a scan got better, without ground truth

A real scan has no perfect original to compare against, so every measure
below works from the scan itself.

| Measure | What it shows | Passes when |
| --- | --- | --- |
| **Mesh validity** | holes, open and non-manifold edges, edges whose faces disagree on their winding | none after, and fewer than before |
| **Shape change** | distance from the cleaned surface to the input surface, per region | mean / 95th percentile under: trunk 2 / 5 mm, head 1 / 3 mm, hands 1 / 3 mm, feet 1.5 / 4 mm: the person stays the person |
| **Face unchanged** | the same distance on the face alone (the front of the head below the brow) | mean / 95th percentile under 0.5 / 1 mm: a protected face is neither smoothed nor filled |
| **Repeatability** | stature and hip breadth measured on repeat scans of the same body | spread under 10 mm |
| **Plausibility** | the same measures against ANSUR II for the sex | between the 1st and 99th percentiles. Outside is a flag to look at, never a reason to reject or reshape the scan: real people fall outside a military survey |
| **Scanner landmarks** | the scanner's own markers and measurements (heights, distances, cross-sections) re-measured on the cleaned mesh | each marker moves under 2 mm along its normal, each measurement drifts under 1% |
| **Rig preserved** | the skin's joints, every vertex's weights, the UVs and texture | same joints, every vertex weighted, weights summing to 1 |

`scripts/scan_quality.py` computes these from cleanup_mesh.py's report and
output, and exits with status 1 when any region of any run, the face included,
is over its mean or 95th-percentile limit. The face is checked on its own
because it is small: a crumpled face moves the head's numbers too little to
fail them. With `--real` a run without its input surface is refused, since
then the face cannot be checked. The scanner-landmark measure needs a measurements file; on the
synthetic demo it is skipped, since ANNY has no scanner markers.

Reprojection against camera photos is not available from this scanner's
export.

## Using the scanner's depth map

`_DepthMap` may say, per vertex, how well the cameras saw that part of the
body. If it does, smoothing should be lighter where the data is good and
stronger where it is poor. That is to be tested on the synthetic demo (where
the "depth" can be simulated) before it is relied on.

## A damage model, to train on synthetic bodies

A model that repairs scans has to learn from pairs: a scan as the scanner
gives it, and the body as it is. No real scan may be used for that, so
`scripts/scan_damage.py` makes the pairs from clean synthetic ANNY bodies. It
reads no scan and was not tuned against one: the scanner is modelled from
its published specification, and what the specification leaves open is
written down as an assumption.

    python scripts/scan_damage.py --in data/bodies/anny-adult-f.rigged.glb \
        --out data/work/demo/anny-adult-f.seed7 --seed 7

It writes the clean body at the scanner's triangle density (`.clean.glb`, the
training target), the damaged body (`.damaged.glb`), the labels
(`.labels.npz`) and a record of every parameter, its basis and the counts
(`.json`). One seed gives the same bytes every time. It needs only numpy and
scipy, takes about 25 s for one body, and refuses a path under
`data/vitronic/`.

**What is published.** The flyer for the VITUS bodyscan (Humanetics Digital
Europe, who now sell it; read 12 Nov 2026) gives: optical triangulation with
eye-safe laser light, 8 sensors, a measuring range 2100 mm high and 1200 mm
by 1200 mm, an average maximum girth error under 1 mm on a tube of 110 mm, a
scan in 6 to 10 s, about 300 points per cm², colour as an option, and a
scanner 2400 mm square and 2950 mm high. Maurer's paper on the VITUS scanners
(Vitronic; 3rd Int. Conf. on 3D Body Scanning Technologies, Lugano 2012,
pp. 99–105) describes how they work: each pillar holds a head that travels
from top to bottom; the head's laser throws a horizontal line, and one camera
above and one below the laser plane look at it slightly downward and upward;
whatever no camera sees is missing from the scan; very dark or shiny surfaces
leave gaps. Its table for the older VITUS smart XXL (4 heads of two cameras,
12 s, 27 points per cm², 1000 mm deep) is the second model, `--model
smart-xxl`.

**The artefacts**, each with its own switch (`--no-holes` and so on):

| Artefact | What it does | Rests on |
| --- | --- | --- |
| resample | halves edges longer than 4/3 of `--edge-mm` (5 mm) and joins linked vertices closer than half of it | assumption: the export's density is not published. An edge below the published point spacing (0.58 mm) is refused |
| holes | a vertex is measured when a head's laser lights it and a camera of that head sees it, both within `--max-incidence-deg` (75°) of the surface normal; faces with an unmeasured corner go, and patches under `--min-island-faces` (30) with them | published: 4 heads of 2 cameras, horizontal laser line, cameras above and below, range. Assumptions: heads `--pillar-inset-m` (0.2 m) in from the corners of the footprint, so 1.5 m from the middle; body facing between two pillars; cameras `--baseline-m` (0.3 m) from the laser; the incidence limit; the island size |
| fingers | a finger is cut short with chance `--finger-missing` (0.25), losing at least `--finger-lost-least` (0.3) of its length; neighbours closer than `--fuse-gap-mm` (10 mm) are joined into one lump with chance `--finger-fused` (0.5) | assumptions throughout. Fingers come from the joints of ANNY's rig |
| banding | an offset along the line of sight, the same along each horizontal line and changing up the body, with standard deviation `--band-mm` (0.3 mm) | published: the head travels 2.1 m in 6 to 10 s. Assumptions: it shakes at `--ripple-hz` (25 Hz), which makes a band 10.5 mm high; the size of the offset |
| noise | each point moves along the line to its camera by `--noise-mm` (0.5 mm standard deviation), and each of the 8 sensors by its own `--sensor-bias-mm` (0.15 mm) | noise: assumption, since the girth error says nothing about single points. Bias: derived, a girth error of 1 mm is a radius error of 0.16 mm |
| seams | each sensor's vertex colours are scaled per channel by `--seam-gain` (5%), so the colour steps where views meet | assumption; only for a mesh with vertex colours, which ANNY has not |

**The labels** say, for every vertex and face of the clean body, whether it
is in the damaged one and if not why (not seen, finger cut, finger joined,
stray fragment, outside the range), which clean vertex and face each damaged
one is, which sensor it was taken from, and how far it was moved.

**On the demo body** (ANNY adult female, seed 7): the clean body has 135 281
vertices and no open edge. The damaged one has lost 13 235 vertices and
29 683 faces (26 546 not seen, 653 of a cut finger, 3 718 of joined fingers,
40 in stray fragments) and has 50 holes: the crown, the soles, the crotch,
under the arms, and the thighs where the hands hang in front of them. Its
points have moved 0.47 mm on average, 1.15 mm at the 95th percentile.
Cleaned by `blender/cleanup_mesh.py` and measured by `scripts/scan_quality.py`,
the holes go from 50 to 39 and the shape changes by 0.25 mm on average, inside
every region's limit: the measures move the right way, and the present
cleanup closes only the small holes. Filling the large ones is the model's
job.

**Not modelled**: the body swaying during the scan, gaps from dark or shiny
surfaces, noise that grows as the surface turns away, clothing and hair, the
scanner software's own hole filling and smoothing, and textures (only vertex
colours get seams). The assumptions above are guesses at a sensible size, not
measurements: before a model trained on these pairs is trusted, they need
checking against what the maker or an owner of the scanner can state.

## Demo results (synthetic ANNY female, three damaged repeats)

Each repeat damages the same skinned synthetic scan differently (8 holes
punched, about 65 faces flipped, 2 mm of noise), then cleans it. In every
repeat the rig check passes: the same 104 joints, no vertex without weights,
every vertex's weights summing to 1, at most 4 joints per vertex.

| Repeat | Holes | Flipped edges | Error against undamaged body (mean) | Shape change (mean / p95) | Stature | Hip breadth |
| --- | --- | --- | --- | --- | --- | --- |
| seed 7 | 8 → 1 | 66 → 0 | 1.53 → 0.77 mm | 0.69 / 2.02 mm | 1776 mm | 302 mm |
| seed 8 | 8 → 0 | 68 → 0 | 1.52 → 0.77 mm | 0.68 / 2.01 mm | 1774 mm | 301 mm |
| seed 9 | 8 → 1 | 62 → 0 | 1.54 → 0.78 mm | 0.68 / 2.03 mm | 1771 mm | 302 mm |

Shape change by region (mean / 95th percentile, all three repeats): trunk
0.82–0.83 / 2.3 mm, head 0.52–0.53 / 1.6 mm, hands 0.65–0.73 / 1.6–1.9 mm,
feet 0.59–0.62 / 1.7–1.8 mm, all inside their limits. The head passes on
distance while the close-up shows the inner eye and mouth surfaces crumpled.
The face alone does not: 0.39–0.40 / 1.14–1.16 mm, over its 0.5 / 1 mm limit,
so `scan_quality.py` fails all three repeats. That is the intended result:
the demo cleanup does not protect the face yet.

Repeatability spread: stature 4.2 mm, hip breadth 1.0 mm. Both measures are
inside ANSUR II's range for women. These repeats damage one synthetic body
three ways; they test that the cleanup and the measures are stable, not how
repeatable a real scanner is. Real repeat scans of one person differ in pose,
breathing and clothing, none of which is in these repeats, so 4.2 mm is not a
scanner's repeatability; that needs repeat scans of one person.

## What is not good enough yet

- **One hole in three repeats stays open.** The hole filler leaves it when the
  punched hole meets the noise; the next fix tries a second pass.
- **The face is damaged by the cleanup.** Around the eyes and mouth, smoothing
  crumples ANNY's inner surfaces, and `scan_quality.py` fails the demo on the
  face's change. A real scan's face must be protected (no smoothing in a face
  region) before any real scan is processed.
- **Chest breadth is not measured.** Where the arms rest against the chest
  there is no gap to tell them from the trunk.
- **Attribute names change case.** Blender's glTF exporter writes custom
  attributes upper-cased (`_DEPTHMAP`), where the scanner writes `_DepthMap`;
  the viewer must match them without regard to case, or the names are put back
  after export.
- **The demo has ANNY's 104 joints**, where the scanner has 34–36; the rig
  check does not depend on the count.
