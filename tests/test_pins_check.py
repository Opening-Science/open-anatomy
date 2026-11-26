from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pins_check import MATCHES, NOT_BUILT, main, pins, status


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(root: Path, path: str, data: bytes) -> str:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return digest(data)


def make_repo(root: Path) -> dict[str, Any]:
    """A sources.json with one pin of each kind, and every pinned file on disk."""
    sources: dict[str, Any] = {
        "sources": {
            "set": {"file": "set.zip", "sha256": write(root, "data/sources/set.zip", b"zip")}
        },
        "cases": {
            "s0001": {"ct_sha256": write(root, "data/cases/s0001/ct.nii.gz", b"ct")},
            "s0002": {"why": "listed, not pinned yet"},
        },
        "bodies": {
            "body": {"file": "body.rigged.glb",
                     "sha256": write(root, "data/bodies/body.rigged.glb", b"body")}
        },
        "references": {
            "body": {
                "body": "body",
                "case": "s0001",
                "built": {
                    "file": "body.reference.glb",
                    "sha256": write(root, "data/reference/body.reference.glb", b"built"),
                    "ctSha256": write(root, "data/work/s0001/s0001.reference.glb", b"meshes"),
                },
            },
            "planned": {"body": "body", "case": "s0002"},
        },
    }  # fmt: skip
    (root / "sources.json").write_text(json.dumps(sources))
    return sources


def test_every_kind_of_pin_is_listed(tmp_path: Path) -> None:
    sources = make_repo(tmp_path)
    assert pins(sources) == [
        ("data/sources/set.zip", digest(b"zip")),
        ("data/cases/s0001/ct.nii.gz", digest(b"ct")),
        ("data/bodies/body.rigged.glb", digest(b"body")),
        ("data/reference/body.reference.glb", digest(b"built")),
        ("data/work/s0001/s0001.reference.glb", digest(b"meshes")),
    ]


def test_skip_sources_leaves_the_archives_out(tmp_path: Path) -> None:
    paths = [path for path, _ in pins(make_repo(tmp_path), skip_sources=True)]
    assert "data/sources/set.zip" not in paths
    assert len(paths) == 4


def test_a_file_that_matches_its_pin_passes(tmp_path: Path) -> None:
    make_repo(tmp_path)
    assert status(tmp_path, "data/bodies/body.rigged.glb", digest(b"body")) == MATCHES
    assert main(["pins_check.py", str(tmp_path)]) == 0


def test_a_file_that_differs_from_its_pin_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_repo(tmp_path)
    write(tmp_path, "data/reference/body.reference.glb", b"rebuilt")
    found = status(tmp_path, "data/reference/body.reference.glb", digest(b"built"))
    assert found == f"sha256 {digest(b'rebuilt')}, sources.json pins {digest(b'built')}"
    assert main(["pins_check.py", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"data/reference/body.reference.glb: {found}" in out
    assert "pins check failed: 1 differ from sources.json (4 match, 0 not built here)" in out


def test_an_absent_file_is_not_built_here_and_does_not_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_repo(tmp_path)
    (tmp_path / "data/reference/body.reference.glb").unlink()
    (tmp_path / "data/sources/set.zip").unlink()
    assert status(tmp_path, "data/reference/body.reference.glb", digest(b"built")) == NOT_BUILT
    assert main(["pins_check.py", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "data/reference/body.reference.glb: not built here" in out
    assert "pins check passed: 3 match, 2 not built here" in out


def test_a_clone_without_data_passes(tmp_path: Path) -> None:
    sources = make_repo(tmp_path)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "sources.json").write_text(json.dumps(sources))
    assert main(["pins_check.py", "--skip-sources", str(clone)]) == 0


def test_the_repository_pins_are_well_formed() -> None:
    # Every pin in the real sources.json is a path under data/ with a SHA-256.
    root = Path(__file__).resolve().parent.parent
    found = pins(json.loads((root / "sources.json").read_text()))
    assert found
    for path, pinned in found:
        assert path.startswith("data/")
        assert len(pinned) == 64 and set(pinned) <= set("0123456789abcdef")
