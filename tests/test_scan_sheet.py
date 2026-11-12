"""Tests for scan_sheet.py on a capsule figure: the head is never drawn."""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

import scan_sheet
from scan_damage import Settings, damage, write_glb
from scan_sheet import BAND, INK, NECK_AT, NOTE, PAPER, neck_height, panel, sheet, write_text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scan_damage import capsule, figure, rigged_glb  # noqa: E402

SIZE = (200, 300)


def test_nothing_is_drawn_above_the_neck_in_any_panel() -> None:
    body = figure()
    neck = NECK_AT * 1.72
    cut = int(round(SIZE[1] - 10 - neck * (SIZE[1] - 20) / 2.1))
    for back in (False, True):
        image = panel(body, neck, back, SIZE)
        assert (image[:cut] == BAND).all()  # the band, and only the band
        assert (image[cut:] != BAND).all()
        drawn = np.flatnonzero((image[cut:] < PAPER).any(1))
        assert drawn.min() <= 3 and drawn.max() >= SIZE[1] - cut - 15  # neck to feet


def test_a_face_sized_mark_on_the_head_does_not_reach_the_picture() -> None:
    # The same figure with and without its head gives the same picture.
    whole = figure()
    headless_faces = whole.triangles[(whole.points[whole.triangles][:, :, 1] < 1.45).all(1)]
    headless = scan_sheet.Mesh(whole.points, headless_faces)
    neck = NECK_AT * 1.72
    assert np.array_equal(panel(whole, neck, False, SIZE), panel(headless, neck, False, SIZE))


def test_front_shows_the_outside_and_a_hole_shows_the_dark_inside() -> None:
    body = capsule(0.0, 0.0, 0.3, 1.0, 0.2, around=48)
    front_gone = body.triangles[
        ~((body.points[body.triangles][:, :, 2] > 0.15).all(1)
          & (np.abs(body.points[body.triangles][:, :, 1] - 0.65) < 0.1).all(1))
    ]  # fmt: skip
    holed = scan_sheet.Mesh(body.points, front_gone)
    scale = (SIZE[1] - 20) / 2.1
    row, column = int(SIZE[1] - 10 - 0.55 * scale), SIZE[0] // 2  # the capsule starts at 0.1 m
    spot = (slice(row - 2, row + 3), slice(column - 2, column + 3))
    assert np.median(panel(body, 1.5, False, SIZE)[spot]) > 200  # lit, facing the viewer
    assert np.median(panel(holed, 1.5, False, SIZE)[spot]) < 60  # the inside of the back
    assert np.median(panel(holed, 1.5, True, SIZE)[spot]) > 200  # the back is whole


def test_neck_comes_from_the_rig_or_from_the_height(tmp_path: Path) -> None:
    body = figure()
    assert neck_height(None, body) == pytest.approx(NECK_AT * 1.72, abs=1e-6)
    joint_of = np.where(body.points[:, 1] > 1.5, 1, 0).astype(np.int64)
    rigged_glb(tmp_path / "rig.glb", body, ["spine01", "neck01"], joint_of)
    lowest = body.points[joint_of == 1, 1].min()
    assert neck_height(tmp_path / "rig.glb", body) == pytest.approx(lowest, abs=1e-6)
    write_glb(tmp_path / "plain.glb", body)
    assert neck_height(tmp_path / "plain.glb", body) == pytest.approx(NECK_AT * 1.72, abs=1e-6)


def test_sheet_has_four_named_panels_and_says_the_head_is_blanked() -> None:
    body = figure()
    damaged = damage(body, 7, Settings(resample=False))[0]
    image = sheet(body, damaged, NECK_AT * 1.72, SIZE)
    assert image.shape == (30 + SIZE[1], 4 * SIZE[0])
    line = np.full((30, 4 * SIZE[0]), PAPER, dtype=np.uint8)
    write_text(line, NOTE, 8, 8)
    assert "NO FACE IS RENDERED" in NOTE and (line == INK).any()
    assert np.array_equal(image[:30], line)
    names = ["CLEAN FRONT", "CLEAN BACK", "DAMAGED FRONT", "DAMAGED BACK"]
    for k, name in enumerate(names):
        label = np.full((14, SIZE[0]), BAND, dtype=np.uint8)
        write_text(label, name, 8, 0)
        assert np.array_equal(image[38:52, k * SIZE[0] : (k + 1) * SIZE[0]], label)
    # The damaged panels differ from the clean ones: holes show.
    assert not np.array_equal(image[:, : SIZE[0]], image[:, 2 * SIZE[0] : 3 * SIZE[0]])


def test_every_letter_of_the_sheet_is_in_the_font() -> None:
    for text in (NOTE, "CLEAN FRONT BACK DAMAGED"):
        assert set(text) <= set(scan_sheet.FONT)
    assert all(len(rows.split()) == 7 and all(len(r) == 5 for r in rows.split())
               for rows in scan_sheet.FONT.values())  # fmt: skip


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_sheet.py", *args])
    scan_sheet.main()


def test_command_line_writes_a_grey_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = figure()
    write_glb(tmp_path / "clean.glb", body)
    write_glb(tmp_path / "damaged.glb", damage(body, 7, Settings(resample=False))[0])
    out = tmp_path / "sheets" / "sheet.png"
    run_cli(monkeypatch, "--clean", str(tmp_path / "clean.glb"),
            "--damaged", str(tmp_path / "damaged.glb"), "--out", str(out))  # fmt: skip
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    assert (width, height) == (4 * scan_sheet.PANEL[0], 30 + scan_sheet.PANEL[1])
    assert data[24:26] == b"\x08\x00"  # eight bits, grey
    at = data.index(b"IDAT")
    rows = zlib.decompress(data[at + 4 : at + 4 + int.from_bytes(data[at - 4 : at], "big")])
    assert len(rows) == height * (width + 1)


@pytest.mark.parametrize("which", ["--clean", "--damaged", "--body", "--out"])
def test_command_line_refuses_the_folder_of_real_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    write_glb(tmp_path / "body.glb", figure())
    paths = dict.fromkeys(["--clean", "--damaged", "--body"], str(tmp_path / "body.glb"))
    paths["--out"] = str(tmp_path / "sheet.png")
    paths[which] = str(tmp_path / "data" / "vitronic" / "x")
    with pytest.raises(SystemExit, match="refused: data/vitronic/ holds real people's scans"):
        run_cli(monkeypatch, *[part for pair in paths.items() for part in pair])
    assert not (tmp_path / "sheet.png").exists()
