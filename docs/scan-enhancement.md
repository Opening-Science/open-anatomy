# Scan enhancement

How a person's 3D body scan is cleaned up into a sound surface, and how we
know it got better. Today this runs on synthetic ANNY bodies only, damaged on
purpose; nothing here has touched a real scan.

## Where it runs, and what never leaves the machine

- **GitHub holds the code** (the Python and Blender scripts) and nothing else.
  A real scan never goes into the repository, CI, a release, an issue, a log
  or any server.
- **It runs on the person's own Mac**, one scan at a time.
- **Credentials** for a scanner account come from the macOS Keychain (or an
  environment variable) at run time. They are never written to a file in the
  repository and never logged.
- **Outputs stay local**, in a folder the person chooses. The pipeline prints
  only the measures below, never the scan's contents.

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

## Input: an adapter, not a guess

A scan reaches the pipeline through an adapter that returns one thing: a
surface (vertices and triangles, in metres, +Y up), plus, if the export has
them, the camera photos and their calibration.

- **Local file adapter:** a mesh file the person already has.
- **Scanner API adapter:** downloads one scan with the person's credentials
  from the Keychain, into the temporary folder.

The scanner's export format is not known yet: which mesh format, which JSON
keys, whether the roughly 27 camera photos and their calibration are included.
The adapters are written once that is known, and no field name is guessed
before then.

## What the cleanup does

`blender/cleanup_mesh.py`: merge duplicate vertices, fill holes, make the
normals consistent and outward, smooth lightly (Taubin, which keeps volume),
and export the cleaned mesh plus levels of detail.

## How we know a scan got better, without ground truth

A real scan has no perfect original to compare against, so every measure
below works from the scan itself.

| Measure | What it shows | Passes when |
| --- | --- | --- |
| **Mesh validity** | holes, open and non-manifold edges, edges whose faces disagree on their winding | none after, and fewer than before |
| **Shape change** | distance from the cleaned surface to the input surface | mean under 2 mm, 95th percentile under 5 mm: the person stays the person |
| **Repeatability** | stature and hip breadth measured on repeat scans of the same body | spread under 10 mm |
| **Plausibility** | the same measures against ANSUR II for the sex | between the 1st and 99th percentiles |
| **Reprojection** | the cleaned surface against the scanner's own camera photos | not measured yet: needs the photos and calibration from the export |

`scripts/scan_quality.py` computes all but the last from cleanup_mesh.py's
report and output.

## Demo results (synthetic ANNY female, three damaged repeats)

Each repeat damages the same body differently (8 holes punched, about 65 faces
flipped, 2 mm of noise), then cleans it.

| Repeat | Holes | Flipped edges | Error against undamaged body (mean) | Shape change (mean / p95) | Stature | Hip breadth |
| --- | --- | --- | --- | --- | --- | --- |
| seed 7 | 8 → 1 | 66 → 0 | 1.53 → 0.77 mm | 0.69 / 2.02 mm | 1776 mm | 302 mm |
| seed 8 | 8 → 0 | 68 → 0 | 1.52 → 0.77 mm | 0.68 / 2.01 mm | 1774 mm | 301 mm |
| seed 9 | 8 → 1 | 62 → 0 | 1.54 → 0.78 mm | 0.68 / 2.03 mm | 1771 mm | 302 mm |

Repeatability spread: stature 4.2 mm, hip breadth 1.0 mm. Both measures are
inside ANSUR II's range for women.

## What is not good enough yet

- **One hole in three repeats stays open.** The hole filler leaves it when the
  punched hole meets the noise; the next fix tries a second pass.
- **The face is damaged by the cleanup.** Around the eyes and mouth, smoothing
  crumples ANNY's inner surfaces. A real scan's face must be protected (no
  smoothing in a face region) before any real scan is processed.
- **Chest breadth is not measured.** Where the arms rest against the chest
  there is no gap to tell them from the trunk.
- **Reprojection waits for the export format.**
