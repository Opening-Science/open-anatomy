"""
Lays out blender/qa.py's tiles as contact sheets: one row per structure, six
tiles per row (four turntable views, the spike close-up, the tip close-up),
with the structure's name, ID and measures on the left, and any flag in red.

  python scripts/qa_sheet.py --qa-dir <dir> [--rows 12] [--title <text>]

Writes <dir>/sheet-01.png, sheet-02.png and so on, and prints their paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VIEWS = ("view0", "view90", "view180", "view270", "spike", "tip")
HEADERS = ("front", "left side", "back", "right side", "worst spike, close", "tip, close")
LABEL_WIDTH = 300
BACKGROUND = (40, 41, 44)
TEXT = (225, 225, 220)
FLAG = (240, 90, 80)
OK = (120, 200, 120)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    qa = json.loads((args.qa_dir / "qa.json").read_text())
    structures = qa["structures"]
    tiles = args.qa_dir / "tiles"
    drawn = [s for s in structures if not s.get("empty")]
    size = Image.open(tiles / f"{drawn[0]['name']}.view0.png").size[0] if drawn else 320
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    header = 34
    pages = [structures[i : i + args.rows] for i in range(0, len(structures), args.rows)]
    for number, page in enumerate(pages, 1):
        sheet = Image.new("RGB", (LABEL_WIDTH + size * len(VIEWS), header + size * len(page)),
                          BACKGROUND)  # fmt: skip
        draw = ImageDraw.Draw(sheet)
        title = f"{args.title}  page {number}/{len(pages)}".strip()
        draw.text((10, 8), title, fill=TEXT, font=font)
        for i, name in enumerate(HEADERS):
            draw.text((LABEL_WIDTH + i * size + 8, 10), name, fill=TEXT, font=small)
        for row, s in enumerate(page):
            y = header + row * size
            if s.get("empty"):
                # An empty mesh has no tiles; its row shows only the flag.
                draw.text((10, y + 10), s["label"], fill=TEXT, font=font)
                draw.text((10, y + 30), "empty mesh", fill=FLAG, font=small)
                draw.line([(0, y), (sheet.width, y)], fill=(70, 70, 74))
                continue
            for col, view in enumerate(VIEWS):
                tile = Image.open(tiles / f"{s['name']}.{view}.png").convert("RGB")
                sheet.paste(tile, (LABEL_WIDTH + col * size, y))
            lines = [
                (s["label"], TEXT, font),
                (s["name"], TEXT, small),
                (f"{s['vertices']} vertices, {s['fragments']} piece(s)", TEXT, small),
                (f"open edges {s['holes']}, non-manifold {s['nonManifoldEdges']}", TEXT, small),
                (f"crossings {s['selfIntersections']}", TEXT, small),
                (f"spikes {s['spikes']} (max {s['spikeScoreMax']})", TEXT, small),
                (f"bend mean {s['bendMeanDeg']} / 95% {s['bend95Deg']} deg", TEXT, small),
            ]
            if "volumeDrift" in s:
                lines.append((f"volume {s['volumeMl']} ml ({s['volumeDrift']:+.1%})", TEXT, small))
            lines.append(("; ".join(s["flags"]) or "no flags", FLAG if s["flags"] else OK, small))
            for k, (text, colour, face) in enumerate(lines):
                draw.text((10, y + 10 + k * 20), text, fill=colour, font=face)
            draw.line([(0, y), (sheet.width, y)], fill=(70, 70, 74))
        path = args.qa_dir / f"sheet-{number:02d}.png"
        sheet.save(path, optimize=True)
        print(path)


if __name__ == "__main__":
    main()
