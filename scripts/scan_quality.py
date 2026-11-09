"""
Measures whether body scans got better, without ground truth, on the outputs
of blender/cleanup_mesh.py: one scan, or repeat scans of the same body.

  python scripts/scan_quality.py --sex female --runs <dir> [<dir> ...] [--out <json>]

Each run directory holds cleanup_mesh.py's report.json and demo.lod0.glb. For
each run it reports:
- validity before and after: holes, open and non-manifold edges, edges whose
  faces disagree on their winding (from the report);
- shape change: the cleaned surface against the input, mean and 95th
  percentile in mm, per region (trunk, head, hands, feet), each under its
  REGION_TOLERANCE_MM (the person must stay the person); on a demo body, also
  the error against the undamaged body;
- body measures: stature and hip breadth on the cleaned surface (chest breadth
  is not taken where the arms touch the chest),
  each placed among the ANSUR II percentiles for the sex.

Across runs of the same body it reports repeatability: the spread (max - min)
of each body measure, which must stay under REPEAT_TOLERANCE_MM. Reprojection
error against the scanner's camera images needs the export format (see
docs/scan-enhancement.md) and is reported as not measured until then.

It reads only local files and writes only --out: no network, no logging of
anything from the scan beyond these numbers, which stay on the machine too.
With --real it refuses any run whose cleanup did not protect a face region.
ANSUR II outside the 1st to 99th percentile is a flag, never a reject.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from size_body import accessor, read_glb  # noqa: E402

# Per region: (mean, 95th percentile) shape change allowed, mm.
REGION_TOLERANCE_MM = {"trunk": (2.0, 5.0), "head": (1.0, 3.0), "hands": (1.0, 3.0),
                       "feet": (1.5, 4.0)}  # fmt: skip
HEAD_ABOVE, FEET_BELOW = 0.87, 0.05  # fractions of stature
NEAREST = 8  # triangles checked exactly per vertex
REPEAT_TOLERANCE_MM = 10.0
HIP_AT = 0.52  # fraction of stature where hip breadth is taken (ANSUR-like)
GAP_MM = 15  # a side-to-side gap this wide separates an arm from the trunk
ANSUR = {"male": "ANSUR_II_MALE_Public.csv", "female": "ANSUR_II_FEMALE_Public.csv"}


def surface(glb: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (mm) and triangles of the first mesh in a plain GLB."""
    gltf, blob = read_glb(glb)
    prim = gltf["meshes"][0]["primitives"][0]
    points = accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(float) * 1000
    triangles = accessor(gltf, blob, prim["indices"]).astype(int).reshape(-1, 3)
    return points, triangles


def cross_section(points: np.ndarray, triangles: np.ndarray, height: float) -> np.ndarray:
    """Points where the surface crosses the plane y = height: the slice's outline."""
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = points[triangles[:, a]], points[triangles[:, b]]
        crossing = (p[:, 1] - height) * (q[:, 1] - height) < 0
        t = (height - p[crossing, 1]) / (q[crossing, 1] - p[crossing, 1])
        out.append(p[crossing] + t[:, None] * (q[crossing] - p[crossing]))
    return np.vstack(out)


def measures(points: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    """Stature and breadths on a standing body (glTF: +Y up), in mm."""
    y = points[:, 1]
    floor, stature = y.min(), float(np.ptp(y))

    midline = float(np.median(points[:, 0]))

    def breadth(fraction: float) -> float:
        """Width of the trunk alone: the slice split where points leave a gap of
        GAP_MM side to side (arms and hands hang apart from the trunk), keeping
        the run that holds the body's midline."""
        x = np.sort(cross_section(points, triangles, floor + fraction * stature)[:, 0])
        runs = np.split(x, np.flatnonzero(np.diff(x) > GAP_MM) + 1)
        trunk = min(runs, key=lambda r: 0 if r[0] <= midline <= r[-1] else abs(r.mean() - midline))
        return float(trunk[-1] - trunk[0])

    # Chest breadth is not taken: where the arms rest against the chest there is
    # no gap to tell them from the trunk, and a breadth with the arms in it
    # would look implausible for the wrong reason.
    return {"stature": stature, "hipbreadth": breadth(HIP_AT)}


def ansur(sex: str) -> dict[str, list[float]]:
    path = ROOT / "data" / "sources" / ANSUR[sex]
    with path.open(encoding="latin-1") as handle:
        rows = list(csv.DictReader(handle))
    return {
        k: [float(np.percentile([float(r[k]) for r in rows], q)) for q in (1, 50, 99)]
        for k in ("stature", "hipbreadth")
    }


def regions(points: np.ndarray) -> dict[str, np.ndarray]:
    """Masks for head, feet, hands and trunk on a standing body (+Y up), by height
    and by distance from the midline (hands hang beside the hips)."""
    y = points[:, 1]
    floor, stature = y.min(), np.ptp(y)
    h = (y - floor) / stature
    head, feet = h > HEAD_ABOVE, h < FEET_BELOW
    side = np.abs(points[:, 0] - np.median(points[:, 0]))
    hip_half = np.percentile(side[(h > 0.48) & (h < 0.56)], 60)
    hands = (h > 0.3) & (h < 0.5) & (side > hip_half + 30)
    trunk = ~(head | feet | hands)
    return {"trunk": trunk, "head": head, "hands": hands, "feet": feet}


def closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest points on triangles (a, b, c) to points p, all (n, 3) (Ericson, RTCD 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = np.where(va + vb + vc == 0, 1e-30, va + vb + vc)
    v, w = vb / denom, vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]  # inside the face
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[edge_ab] = (a + ab * (d1 / (d1 - d3))[:, None])[edge_ab]
        edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[edge_ac] = (a + ac * (d2 / (d2 - d6))[:, None])[edge_ac]
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        edge_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        out[edge_bc] = (b + (c - b) * t[:, None])[edge_bc]
    out[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = b[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = c[(d6 >= 0) & (d5 <= d6)]
    return out


def shape_change(output: tuple, source: tuple) -> dict[str, dict[str, float]]:
    """Exact distance from each output vertex to the input surface, per region (mm):
    the nearest NEAREST triangles by centroid are checked point to triangle."""
    from scipy.spatial import cKDTree

    points, triangles = source
    corners = points[triangles]
    _, near = cKDTree(corners.mean(1)).query(output[0], k=NEAREST)
    best = np.full(len(output[0]), np.inf)
    for k in range(NEAREST):
        tri = corners[near[:, k]]
        q = closest_on_triangles(output[0], tri[:, 0], tri[:, 1], tri[:, 2])
        best = np.minimum(best, np.linalg.norm(output[0] - q, axis=1))
    out = {}
    for name, mask in regions(output[0]).items():
        d = best[mask]
        mean_lim, p95_lim = REGION_TOLERANCE_MM[name]
        out[name] = {
            "meanMm": round(float(d.mean()), 3), "p95Mm": round(float(np.percentile(d, 95)), 3),
            "within": bool(d.mean() <= mean_lim and np.percentile(d, 95) <= p95_lim),
        }  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sex", choices=ANSUR, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--real", action="store_true",
        help="a real person's scan: refused unless the cleanup protected the face",
    )  # fmt: skip
    args = parser.parse_args()
    ranges = ansur(args.sex)
    runs, per_measure = [], {}
    for run in args.runs:
        report = json.loads((run / "report.json").read_text())
        if args.real and not report.get("faceProtected"):
            sys.exit(f"{run.name}: refused: the cleanup did not protect a face region, which a "
                     "real scan requires (docs/scan-enhancement.md)")  # fmt: skip
        output = surface(run / "demo.lod0.glb")
        source = surface(run / "demo.input.glb") if (run / "demo.input.glb").exists() else None
        body = measures(*output)
        keys = ("holes", "openEdges", "nonManifoldEdges", "inconsistentEdges")
        entry = {
            "run": run.name,
            "validity": {
                "before": {k: report["before"][k] for k in keys},
                "after": {k: report["after"][k] for k in keys},
            },
            "shapeChangeMm": report.get("inputToOutputMm"),
            "shapeChangeByRegionMm": shape_change(output, source) if source else None,
            "errorAgainstUndamagedMm": {
                "before": report.get("surfaceErrorBefore"),
                "after": report.get("surfaceErrorAfter"),
            },
            "bodyMeasuresMm": {
                k: {
                    "mm": round(v),
                    "ansur1": round(ranges[k][0]),
                    "ansur99": round(ranges[k][2]),
                    "plausible": bool(ranges[k][0] <= v <= ranges[k][2]),
                }
                for k, v in body.items()
            },  # fmt: skip
            "reprojection": "not available from this scanner's export (no photos)",
        }
        for k, v in body.items():
            per_measure.setdefault(k, []).append(v)
        runs.append(entry)
    repeat = {k: round(float(np.ptp(v)), 1) for k, v in per_measure.items()}
    summary = {
        "runs": runs,
        "repeatabilitySpreadMm": repeat,
        "repeatable": bool(all(v <= REPEAT_TOLERANCE_MM for v in repeat.values())),
        "tolerances": {"shapeChangeMm": REGION_TOLERANCE_MM, "repeatSpreadMm": REPEAT_TOLERANCE_MM},
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    for r in runs:
        v, e = r["validity"], r["errorAgainstUndamagedMm"]
        holes = f"holes {v['before']['holes']} -> {v['after']['holes']}"
        flips = f"flipped {v['before']['inconsistentEdges']} -> {v['after']['inconsistentEdges']}"
        error = f"error vs undamaged {e['before']['meanMm']} -> {e['after']['meanMm']} mm"
        shape = r["shapeChangeMm"]
        error += f", input to output mean {shape['meanMm']} / p95 {shape['p95Mm']} mm"
        body = ", ".join(
            f"{k} {m['mm']}{'' if m['plausible'] else ' (OUT)'}"
            for k, m in r["bodyMeasuresMm"].items()
        )
        print(f"{r['run']}: {holes}, {flips}, {error}; {body}")
        for region, m in (r["shapeChangeByRegionMm"] or {}).items():
            ok = "ok " if m["within"] else "OVER"
            print(f"    {ok} {region}: mean {m['meanMm']} / p95 {m['p95Mm']} mm")
    ok = summary["repeatable"]
    print(f"repeatability spread (mm): {repeat}; within {REPEAT_TOLERANCE_MM} mm: {ok}")


if __name__ == "__main__":
    main()
