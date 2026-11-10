"""
Picks the ANNY body that matches a CT case's size, for a reference body.

  python scripts/size_body.py --case s0982 --sex male

Measures the CT, from segment.py's masks:
- the rise from the hip joints to the shoulder joints (sphere centres fitted to
  the femoral and humeral heads);
- the trunk's width and depth (body_trunc) halfway up that rise.

It then bakes ANNY on a grid of weight and height phenotypes (0.1 to 1.0,
ANNY's top), caching each body in data/bodies/grid/, measures each the same
way (its upperleg01 and upperarm01 joints; skin vertices that follow the
trunk), and takes the weight and height whose rise, width and depth differ
least from the CT's, summed as squared relative errors. With --scale-free the
same errors are taken after ANNY is scaled uniformly by uniform_scale(), the
geometric mean of the rise, width and depth ratios, which is the scale
blender/fit_body.py applies; so the body chosen is the one whose shape fits
best at that scale. It then refines around that best in steps of 0.05, and
warns when the best lies on the edge of what was searched: at ANNY's limit (0
or 1), the CT may be bigger or smaller than ANNY can be. sources.json then
records the choice.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
LIMITS = (0.0, 1.0)  # ANNY's phenotype range
FINE_STEP = 0.05  # the refinement's step around the grid's best
LIMBS = ("upperarm", "lowerarm", "wrist", "metacarpal", "finger", "upperleg", "lowerleg",
         "foot", "toe")  # fmt: skip
SLAB_MM = 10
SCALE_MEASURES = ("rise", "width", "depth")


def uniform_scale(ct: dict, anny: dict) -> tuple[float, dict[str, float]]:
    """The one scale fit_body.py gives ANNY: the geometric mean of the CT-to-ANNY
    ratios of the rise, trunk width and trunk depth, and the ratios themselves.

    Imported by blender/fit_body.py, so --scale-free here and the fit there agree.
    """
    ratios = {k: float(ct[k] / anny[k]) for k in SCALE_MEASURES}
    return float(np.exp(np.mean(np.log(list(ratios.values()))))), ratios


def fit_sphere(points: np.ndarray) -> np.ndarray:
    a = np.hstack([2 * points, np.ones((len(points), 1))])
    x, *_ = np.linalg.lstsq(a, (points**2).sum(1), rcond=None)
    return x[:3]


def surface_points(path: Path) -> np.ndarray:
    """RAS millimetres of a mask's boundary voxels (canonical orientation)."""
    # Imported here so blender/fit_body.py can import uniform_scale in Blender's
    # Python, which has no nibabel.
    import nibabel as nib

    image = nib.as_closest_canonical(nib.Nifti1Image.from_filename(path))
    mask = np.asanyarray(image.dataobj) > 0
    inner = mask.copy()
    for axis in range(3):
        inner &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
    edge = np.argwhere(mask & ~inner)
    return edge @ image.affine[:3, :3].T + image.affine[:3, 3]


def measure_ct(case: str) -> dict:
    seg = ROOT / "data" / "work" / case / "seg"
    femurs = [surface_points(seg / f"femur_{s}.nii.gz") for s in ("left", "right")]
    midline = np.mean([f[:, 0].mean() for f in femurs])
    hips = []
    for f in femurs:
        medial = np.abs(f[:, 0] - midline)
        hips.append(fit_sphere(f[medial <= medial.min() + 45]))
    shoulders = []
    for s in ("left", "right"):
        h = surface_points(seg / f"humerus_{s}.nii.gz")
        shoulders.append(fit_sphere(h[h[:, 2] >= h[:, 2].max() - 45]))
    hip_z, shoulder_z = np.mean([p[2] for p in hips]), np.mean([p[2] for p in shoulders])
    trunk = surface_points(seg / "body_trunc.nii.gz")
    band = trunk[np.abs(trunk[:, 2] - (hip_z + shoulder_z) / 2) <= SLAB_MM]
    lo, hi = np.percentile(band[:, :2], 1, 0), np.percentile(band[:, :2], 99, 0)
    return {"rise": float(shoulder_z - hip_z), "width": float(hi[0] - lo[0]),
            "depth": float(hi[1] - lo[1])}  # fmt: skip


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20 : 20 + length]), data[20 + length + 8 :]


def accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[
        acc["componentType"]
    ]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    array = np.frombuffer(blob, dtype, acc["count"] * width, start).reshape(acc["count"], width)
    return array / 255.0 if acc.get("normalized") and dtype is np.uint8 else array


def world_positions(gltf: dict) -> dict[str, np.ndarray]:
    def local(node):
        m = np.eye(4)
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]) * node.get("scale", [1, 1, 1])  # fmt: skip
        m[:3, 3] = node.get("translation", [0, 0, 0])
        return m

    parent = {c: i for i, n in enumerate(gltf["nodes"]) for c in n.get("children", [])}
    out = {}
    for i, node in enumerate(gltf["nodes"]):
        m, j = local(node), i
        while j in parent:
            j = parent[j]
            m = local(gltf["nodes"][j]) @ m
        out[node.get("name", str(i))] = m[:3, 3]
    return out


def measure_anny(path: Path) -> dict:
    """The same measures on a baked ANNY body (glTF: +Y up, +Z front)."""
    gltf, blob = read_glb(path)
    joints = world_positions(gltf)
    rise = np.mean([joints[f"upperarm01.{s}"][1] for s in "LR"]) - np.mean(
        [joints[f"upperleg01.{s}"][1] for s in "LR"]
    )
    mid = (np.mean([joints[f"upperarm01.{s}"][1] for s in "LR"])
           + np.mean([joints[f"upperleg01.{s}"][1] for s in "LR"])) / 2  # fmt: skip
    prim = next(m for m in gltf["meshes"])["primitives"][0]
    positions = accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(float)
    which = accessor(gltf, blob, prim["attributes"]["JOINTS_0"]).astype(int)
    weights = accessor(gltf, blob, prim["attributes"]["WEIGHTS_0"]).astype(float)
    names = [gltf["nodes"][j]["name"] for j in gltf["skins"][0]["joints"]]
    limb = np.array([n.startswith(LIMBS) for n in names])
    trunk = positions[(weights * limb[which]).sum(1) < 0.5]
    band = trunk[np.abs(trunk[:, 1] - mid) <= SLAB_MM / 1000]
    lo, hi = np.percentile(band, 1, 0), np.percentile(band, 99, 0)
    return {"rise": float(rise * 1000), "width": float((hi[0] - lo[0]) * 1000),
            "depth": float((hi[2] - lo[2]) * 1000)}  # fmt: skip


def around(value: float) -> list[float]:
    """The refinement's values: two fine steps either side of value, within ANNY's range."""
    return sorted({round(min(max(value + k * FINE_STEP, LIMITS[0]), LIMITS[1]), 2)
                   for k in range(-2, 3)})  # fmt: skip


def edge_warnings(weight: float, height: float, searched: dict[str, list[float]]) -> list[str]:
    """Where the best weight or height sits on the edge of what was searched."""
    warnings = []
    for name, value in (("weight", weight), ("height", height)):
        if value in LIMITS:
            warnings.append(f"{name} {value} is at ANNY's limit; the CT may lie beyond it")
        elif value in (min(searched[name]), max(searched[name])):
            warnings.append(f"{name} {value} is on the edge of the search; widen it")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--sex", choices=("male", "female"), required=True)
    parser.add_argument(
        "--scale-free", action="store_true",
        help="match shape only, after the uniform scale fit_body.py applies "
        "(the geometric mean of the rise, width and depth ratios)",
    )  # fmt: skip
    args = parser.parse_args()
    target = measure_ct(args.case)
    print(f"{args.case}: rise {target['rise']:.0f} mm, trunk {target['width']:.0f} x "
          f"{target['depth']:.0f} mm")  # fmt: skip
    grid_dir = ROOT / "data" / "bodies" / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    scored: dict[tuple[float, float], tuple] = {}

    def score(weights: list[float], heights: list[float]) -> tuple:
        for weight in weights:
            for height in heights:
                if (weight, height) in scored:
                    continue
                path = grid_dir / f"anny-{args.sex}-w{weight}-h{height}.rigged.glb"
                if not path.exists():
                    subprocess.run(
                        ["uv", "run", "-q", "--script", str(ROOT / "scripts" / "bake_anny.py"),
                         args.sex, str(path), "--weight", str(weight), "--height", str(height)],
                        check=True, capture_output=True,
                    )  # fmt: skip
                got = measure_anny(path)
                k = uniform_scale(target, got)[0] if args.scale_free else 1.0
                error = sum(((got[m] * k - target[m]) / target[m]) ** 2 for m in SCALE_MEASURES)
                scored[(weight, height)] = (error, weight, height, got)
        return min(scored.values(), key=lambda r: r[0])

    coarse = score(list(GRID), list(GRID))
    fine = {"weight": around(coarse[1]), "height": around(coarse[2])}
    score(fine["weight"], fine["height"])
    results = sorted(scored.values(), key=lambda r: r[0])
    searched = {
        "weight": sorted({w for w, _ in scored}),
        "height": sorted({h for _, h in scored}),
    }
    for error, weight, height, got in results[:5]:
        print(f"  weight {weight} height {height}: rise {got['rise']:.0f}, trunk "
              f"{got['width']:.0f} x {got['depth']:.0f} mm, error {error:.4f}")  # fmt: skip
    best = results[0]
    warnings = edge_warnings(best[1], best[2], searched)
    for warning in warnings:
        print(f"warning: {warning}")
    print(json.dumps({"case": args.case, "sex": args.sex, "weight": best[1], "height": best[2],
                      "ct": {k: round(v) for k, v in target.items()},
                      "anny": {k: round(v) for k, v in best[3].items()},
                      "edge": warnings}))  # fmt: skip


if __name__ == "__main__":
    main()
