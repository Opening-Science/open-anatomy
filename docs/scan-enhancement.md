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
  `scripts/scan_quality.py --real` refuses to run without it.

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
| **Repeatability** | stature and hip breadth measured on repeat scans of the same body | spread under 10 mm |
| **Plausibility** | the same measures against ANSUR II for the sex | between the 1st and 99th percentiles. Outside is a flag to look at, never a reason to reject or reshape the scan: real people fall outside a military survey |
| **Scanner landmarks** | the scanner's own markers and measurements (heights, distances, cross-sections) re-measured on the cleaned mesh | each marker moves under 2 mm along its normal, each measurement drifts under 1% |
| **Rig preserved** | the skin's joints, every vertex's weights, the UVs and texture | same joints, every vertex weighted, weights summing to 1 |

`scripts/scan_quality.py` computes these from cleanup_mesh.py's report and
output. The scanner-landmark measure needs a measurements file; on the
synthetic demo it is skipped, since ANNY has no scanner markers.

Reprojection against camera photos is not available from this scanner's
export.

## Using the scanner's depth map

`_DepthMap` may say, per vertex, how well the cameras saw that part of the
body. If it does, smoothing should be lighter where the data is good and
stronger where it is poor. That is to be tested on the synthetic demo (where
the "depth" can be simulated) before it is relied on.

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
distance while the close-up shows the inner eye and mouth surfaces crumpled:
a mean distance does not see damage to fine detail, which is why face
protection is required rather than measured.

Repeatability spread: stature 4.2 mm, hip breadth 1.0 mm. Both measures are
inside ANSUR II's range for women. These repeats damage one synthetic body
three ways; they test that the cleanup and the measures are stable, not how
repeatable a real scanner is, which needs repeat scans of one person.

## What is not good enough yet

- **One hole in three repeats stays open.** The hole filler leaves it when the
  punched hole meets the noise; the next fix tries a second pass.
- **The face is damaged by the cleanup.** Around the eyes and mouth, smoothing
  crumples ANNY's inner surfaces. A real scan's face must be protected (no
  smoothing in a face region) before any real scan is processed.
- **Chest breadth is not measured.** Where the arms rest against the chest
  there is no gap to tell them from the trunk.
- **Attribute names change case.** Blender's glTF exporter writes custom
  attributes upper-cased (`_DEPTHMAP`), where the scanner writes `_DepthMap`;
  the viewer must match them without regard to case, or the names are put back
  after export.
- **The demo has ANNY's 104 joints**, where the scanner has 34–36; the rig
  check does not depend on the count.
