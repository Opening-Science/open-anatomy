"""Tests for atlas.py's pure parts: the side-by-side sheet, the scope and the case check."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

import atlas


def test_side_by_side_scales_both_to_one_height(tmp_path: Path) -> None:
    render = tmp_path / "render.png"
    plate = tmp_path / "plate.png"
    Image.new("RGB", (300, 400), (200, 100, 50)).save(render)
    Image.new("RGB", (436, 600), (255, 255, 255)).save(plate)
    out = tmp_path / "sheet.png"
    atlas.side_by_side(render, plate, out, height=600)
    sheet = Image.open(out)
    assert sheet.size == (450 + 436 + 24, 600)  # 300 * 600 / 400 = 450 wide render
    assert sheet.getpixel((10, 10)) == (200, 100, 50)  # the render on the left
    assert sheet.getpixel((460, 10)) == (255, 255, 255)  # the gap, then the plate
    assert sheet.getpixel((700, 300)) == (255, 255, 255)


def test_an_unpinned_case_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["atlas.py", "s9999"])
    with pytest.raises(SystemExit, match="s9999 is not pinned"):
        atlas.main()


def test_stage_1_says_it_is_the_thorax_without_the_girdle() -> None:
    assert "thorax only" in atlas.SCOPE
    assert "girdle is not part of Stage 1" in atlas.SCOPE


def test_girdle_left_out_lists_only_the_girdle_bones_the_build_dropped() -> None:
    cut = "cut by the scan's edge (unscanned padding)"
    manifest = {
        "dropped": [
            {"label": "vertebrae_C7", "reason": cut},
            {"label": "clavicula_left", "reason": cut, "cut_at": "S"},
            {"label": "scapula_right", "reason": cut},
            {"label": "humerus_left", "reason": "its pair is cut"},
            {"label": "rib_left_12", "reason": cut},
        ]
    }
    left_out = atlas.girdle_left_out(manifest)
    assert [d["label"] for d in left_out] == ["clavicula_left", "scapula_right", "humerus_left"]
    assert left_out[0]["cut_at"] == "S"
    assert atlas.girdle_left_out({}) == []
