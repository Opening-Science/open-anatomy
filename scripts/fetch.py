"""
Downloads a pinned source from its official record into data/sources/, checks
it against sources.json, and unpacks one CT case into data/cases/<case>/.

  python scripts/fetch.py [--source totalsegmentator-small-v201] [--case s1086]
  python scripts/fetch.py --pick   # list cases that contain every organ

A source is only used when its MD5 (from the record) and SHA-256 (ours) both
match sources.json. A partial download is resumed. The case's ct.nii.gz and its
reference segmentations of the reference set (organs and trunk skeleton)
are extracted; nothing else is.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import ORGANS, REFERENCE  # noqa: E402

SOURCES = json.loads((ROOT / "sources.json").read_text())


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch_source(name: str) -> Path:
    source = SOURCES["sources"][name]
    path = ROOT / "data" / "sources" / source["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size < source["bytes"]:
        print(f"downloading {source['url']}")
        subprocess.run(
            ["curl", "-fL", "-C", "-", "--retry", "10", "-o", str(path), source["url"]],
            check=True,
        )
    for algorithm in ("md5", "sha256"):
        found = digest(path, algorithm)
        if found != source[algorithm]:
            sys.exit(f"{path.name}: {algorithm} {found} does not match {source[algorithm]}")
    print(f"{path.name}: md5 and sha256 match sources.json")
    return path


def members(archive: zipfile.ZipFile, case: str) -> list[str]:
    wanted = {f"{case}/ct.nii.gz"} | {f"{case}/segmentations/{s.label}.nii.gz" for s in REFERENCE}
    names = [n for n in archive.namelist() if n.split("/", 1)[-1] in wanted or n in wanted]
    if len(names) != len(wanted):
        sys.exit(f"{case}: expected {len(wanted)} files in the archive, found {len(names)}")
    return names


def extract_case(zip_path: Path, case: str) -> Path:
    out = ROOT / "data" / "cases" / case
    with zipfile.ZipFile(zip_path) as archive:
        for name in members(archive, case):
            target = out / name.split(f"{case}/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    pinned = SOURCES["cases"].get(case, {}).get("ct_sha256")
    found = digest(out / "ct.nii.gz", "sha256")
    if pinned and found != pinned:
        sys.exit(f"{case}/ct.nii.gz: sha256 {found} does not match {pinned}")
    print(f"{case}: extracted to {out.relative_to(ROOT)} (ct sha256 {found})")
    return out


def pick(zip_path: Path) -> None:
    """Prints the cases whose reference segmentations contain every organ."""
    import nibabel as nib
    import numpy as np

    with zipfile.ZipFile(zip_path) as archive:
        cases = sorted({n.split("/")[-3] for n in archive.namelist() if "/segmentations/" in n})
        for case in cases:
            counts = []
            for s in ORGANS:
                data = gzip.decompress(archive.read(_member(archive, case, s.label)))
                image = nib.Nifti1Image.from_bytes(data)
                counts.append(int(np.count_nonzero(np.asanyarray(image.dataobj))))
            if all(counts):
                print(case, min(counts))


def _member(archive: zipfile.ZipFile, case: str, label: str) -> str:
    suffix = f"{case}/segmentations/{label}.nii.gz"
    return next(n for n in archive.namelist() if n.endswith(suffix))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=SOURCES["default"]["source"])
    parser.add_argument("--case", default=SOURCES["default"]["case"])
    parser.add_argument("--pick", action="store_true")
    args = parser.parse_args()
    zip_path = fetch_source(args.source)
    if args.pick:
        pick(zip_path)
    else:
        extract_case(zip_path, args.case)


if __name__ == "__main__":
    main()
