"""Tests for scan_sheet.py on a capsule figure: the head is never drawn, a
clean body shows no inside and no gaps, and a hole shows red."""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

import scan_sheet
from scan_damage import Settings, damage, write_glb
from scan_sheet import (
    BAND,
    BELOW,
    FORWARD,
    FRONT,
    HOLE,
    LABEL,
    LEGEND,
    NAMES,
    NECK_AT,
    NOTE,
    PAPER,
    UP,
    body_box,
    from_below,
    hand_boxes,
    hands,
    neck_height,
    panel,
    render,
    sheet,
    whole_body,
    write_text,
)  # fmt: skip

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


def red(image: np.ndarray) -> np.ndarray:
    """Where an RGB picture shows the inside of the body."""
    return (image == np.array(HOLE, dtype=np.uint8)).all(2)


def test_sheet_has_eight_named_panels_and_says_the_head_is_blanked() -> None:
    body = figure()
    damaged = damage(body, 7, Settings(resample=False))[0]
    image = sheet(body, damaged, NECK_AT * 1.72, body_height=200, lower_height=100)
    head = np.full((34, image.shape[1], 3), PAPER, dtype=np.uint8)
    write_text(head, NOTE, 6, 2)
    write_text(head, LEGEND, 6, 18)
    assert "NO FACE IS RENDERED" in NOTE and np.array_equal(image[:18], head[:18])
    width = (image.shape[1] - 3 * 4) // 4
    for k, name in enumerate(NAMES):
        top = 34 if k < 4 else 34 + LABEL + 200 + 6
        left = (k % 4) * (width + 4)
        label = np.full((LABEL, width, 3), BAND, dtype=np.uint8)
        write_text(label, name, 6, 4)
        assert np.array_equal(image[top : top + LABEL, left : left + width], label), name
    clean_panels = image[34 + LABEL :, : 2 * width + 4]
    damaged_panels = image[34 + LABEL :, 2 * width + 8 :]
    assert not red(clean_panels).any()
    assert not np.array_equal(clean_panels[:, :width], damaged_panels[:, :width])


def test_the_top_row_stops_at_the_neck_and_is_cropped_to_the_body() -> None:
    body = figure()
    neck = NECK_AT * 1.72
    low, high = body_box(body, neck)
    assert high[1] - low[1] == pytest.approx(neck + scan_sheet.MARGIN_M)
    x = body.points[:, 0]
    assert high[0] - low[0] == pytest.approx(x.max() - x.min() + 2 * scan_sheet.MARGIN_M)
    width = int(np.ceil(200 * (high[0] - low[0]) / (high[1] - low[1])))
    image = whole_body(body, (low, high), FRONT, (width, 200))
    drawn = (image != PAPER).any(2)
    rows, columns = np.flatnonzero(drawn.any(1)), np.flatnonzero(drawn.any(0))
    room = scan_sheet.MARGIN_M * 200 / (high[1] - low[1])  # the margin in pixels
    assert rows.min() <= 1 and rows.max() >= 200 - room - 2  # neck to feet fill the panel
    assert columns.min() <= room + 2 and columns.max() >= width - room - 3  # and arm to arm


def test_a_clean_body_shows_no_inside_and_no_gaps() -> None:
    # Each pixel inside the outline shows the outside of the body: no speck of
    # the far side, and no paper, comes through between the points.
    body = capsule(0.0, 0.0, 0.3, 1.0, 0.2, around=48)
    keep = np.ones(len(body.triangles), dtype=bool)
    image = render(body, keep, FRONT, UP, (0.0, 0.65, 0.5, 1.0), (100, 200))
    assert not red(image).any()
    middle = image[30:170, 30:70]  # well inside the outline
    assert (middle != PAPER).any(2).all() and not red(middle).any()
    figure_body = figure()
    box = body_box(figure_body, NECK_AT * 1.72)
    for picture in (whole_body(figure_body, box, FRONT, (120, 200)),
                    whole_body(figure_body, box, scan_sheet.BACK, (120, 200)),
                    from_below(figure_body, box, (120, 80))):  # fmt: skip
        assert not red(picture).any()


def test_a_hole_shows_the_inside_in_red() -> None:
    body = capsule(0.0, 0.0, 0.3, 1.0, 0.2, around=48)
    corners = body.points[body.triangles]
    front_gone = ~(
        (corners[:, :, 2] > 0.15).all(1) & (np.abs(corners[:, :, 1] - 0.65) < 0.1).all(1)
    )
    image = render(body, front_gone, FRONT, UP, (0.0, 0.65, 0.5, 1.0), (100, 200))
    assert red(image[95:106, 45:56]).all()  # through the hole, the inside of the back
    whole = render(body, np.ones(len(body.triangles), dtype=bool), FRONT, UP,
                   (0.0, 0.65, 0.5, 1.0), (100, 200))  # fmt: skip
    assert not red(whole).any()


def test_from_below_shows_the_soles_with_the_front_at_the_top() -> None:
    body = figure()
    box = body_box(body, NECK_AT * 1.72)
    image = from_below(body, box, (120, 80))
    assert (image != PAPER).any()
    # The view is up from the floor, so +Z (the front) is up the picture.
    low, high = box
    tip = render(body, np.ones(len(body.triangles), dtype=bool), BELOW, FORWARD,
                 ((low[0] + high[0]) / 2, float(high[2]), float(high[0] - low[0]), 0.02),
                 (120, 4))  # fmt: skip
    assert (tip == PAPER).all()  # nothing beyond the front edge


def test_hands_come_from_the_rig_or_from_the_ends_of_the_arms(tmp_path: Path) -> None:
    body = figure()
    boxes = hand_boxes(None, body)
    assert len(boxes) == 2 and boxes[0][0][0] < 0 < boxes[1][0][0]
    for low, high in boxes:  # each the lowest part of an arm
        assert low[1] == pytest.approx(0.85 - 0.04 - scan_sheet.PAD_M, abs=0.005)
        assert high[1] - low[1] == pytest.approx(scan_sheet.HAND_M + 2 * scan_sheet.PAD_M, abs=0.01)
    names = ["spine01", "wrist.L", "wrist.R"]
    joint_of = np.where(body.points[:, 1] > 0.95, 0, np.where(body.points[:, 0] > 0.1, 1,
                        np.where(body.points[:, 0] < -0.1, 2, 0))).astype(np.int64)  # fmt: skip
    rigged_glb(tmp_path / "rig.glb", body, names, joint_of)
    rigged = hand_boxes(tmp_path / "rig.glb", body)
    assert len(rigged) == 2 and rigged[0][1][0] < 0 < rigged[1][0][0]
    image = hands(body, rigged, NECK_AT * 1.72, (120, 80))
    assert (image[:, :60] != PAPER).any() and (image[:, 60:] != PAPER).any()


def test_every_letter_of_the_sheet_is_in_the_font() -> None:
    for text in (NOTE, LEGEND, *NAMES):
        assert set(text) <= set(scan_sheet.FONT)
    assert all(len(rows.split()) == 7 and all(len(r) == 5 for r in rows.split())
               for rows in scan_sheet.FONT.values())  # fmt: skip


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_sheet.py", *args])
    scan_sheet.main()


def test_command_line_writes_a_colour_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = figure()
    write_glb(tmp_path / "clean.glb", body)
    write_glb(tmp_path / "damaged.glb", damage(body, 7, Settings(resample=False))[0])
    out = tmp_path / "sheets" / "sheet.png"
    run_cli(monkeypatch, "--clean", str(tmp_path / "clean.glb"),
            "--damaged", str(tmp_path / "damaged.glb"), "--out", str(out))  # fmt: skip
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    expected = sheet(body, damage(body, 7, Settings(resample=False))[0], NECK_AT * 1.72)
    assert (height, width) == expected.shape[:2]
    assert data[24:26] == b"\x08\x02"  # eight bits, RGB
    at = data.index(b"IDAT")
    rows = zlib.decompress(data[at + 4 : at + 4 + int.from_bytes(data[at - 4 : at], "big")])
    assert len(rows) == height * (3 * width + 1)


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


# The same on a real ANNY body
#
# The ANNY bodies are baked locally into data/bodies and are not in git, so
# this test is skipped where there is none (CI has none). Run it locally after
# any change to the sheet or the damage model.

ANNY_BODY = scan_sheet.ROOT / "data" / "bodies" / "anny-adult-f.rigged.glb"


@pytest.mark.skipif(not ANNY_BODY.exists(), reason=f"no baked ANNY body at {ANNY_BODY}")
def test_an_anny_body_is_clean_until_damaged_and_its_holes_show_red() -> None:
    from scan_damage import read_mesh

    damaged, clean, _, _ = damage(read_mesh(ANNY_BODY), 7)
    neck = neck_height(ANNY_BODY, clean)
    box = body_box(clean, neck)
    low, high = box
    size = (int(np.ceil(400 * (high[0] - low[0]) / (high[1] - low[1]))), 400)
    for view in (FRONT, scan_sheet.BACK):
        assert not red(whole_body(clean, box, view, size)).any()
    assert not red(from_below(clean, box, (size[0], 200))).any()
    # The armpits show from the front and the soles from below.
    assert red(whole_body(damaged, box, FRONT, size)).sum() > 50
    assert red(from_below(damaged, box, (size[0], 200))).sum() > 200
    assert len(hand_boxes(ANNY_BODY, clean)) == 2
