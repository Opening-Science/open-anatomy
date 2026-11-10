from __future__ import annotations

from pathlib import Path

from licence_check import main, missing_licences


def make_repo(root: Path, *files: str) -> None:
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("licence text\n")


def test_a_complete_repo_passes(tmp_path: Path) -> None:
    make_repo(tmp_path, "LICENSE", "NOTICE", "assets/cc-by/LICENSE", "assets/cc-by-sa/LICENSE")
    assert missing_licences(tmp_path) == []
    assert main(["licence_check.py", str(tmp_path)]) == 0


def test_root_files_are_required(tmp_path: Path) -> None:
    assert missing_licences(tmp_path) == ["LICENSE", "NOTICE"]


def test_an_asset_folder_without_licence_fails(tmp_path: Path) -> None:
    make_repo(tmp_path, "LICENSE", "NOTICE", "assets/cc-by/LICENSE", "assets/cc0/model.glb")
    assert missing_licences(tmp_path) == ["assets/cc0/LICENSE"]
    assert main(["licence_check.py", str(tmp_path)]) == 1


def test_only_immediate_subfolders_need_a_licence(tmp_path: Path) -> None:
    make_repo(tmp_path, "LICENSE", "NOTICE", "assets/cc-by/LICENSE", "assets/cc-by/kidney/a.glb")
    assert missing_licences(tmp_path) == []


def test_files_directly_in_assets_are_ignored(tmp_path: Path) -> None:
    make_repo(tmp_path, "LICENSE", "NOTICE", "assets/README.md")
    assert missing_licences(tmp_path) == []


def test_this_repository_passes() -> None:
    root = Path(__file__).resolve().parent.parent
    assert missing_licences(root) == []
