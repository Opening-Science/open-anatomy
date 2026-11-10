"""Tests for blender.py's task table and personal-data guard; none of them needs Blender."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import blender

ROOT = Path(__file__).resolve().parent.parent


def test_the_guard_passes_its_own_cases() -> None:
    assert blender.guard_check() == []


def test_every_task_runs_a_script_that_exists() -> None:
    missing = [s for s in blender.SCRIPTS.values() if not (ROOT / "blender" / s).is_file()]
    assert missing == []


def test_fit_body_is_a_task_so_it_goes_through_the_guard() -> None:
    assert blender.SCRIPTS["fit-body"] == "fit_body.py"


@pytest.mark.parametrize(
    "args",
    [
        ["--body", "data/bodies/anny-adult-f.rigged.glb", "--ct", "data/vitronic/p1/ct.glb"],
        ["--ct", "ct.glb", "--outline", "/home/someone/twin-llm/data/vitronic/p1/outline.glb"],
        ["--landmarks=data/vitronic/p1/landmarks.glb", "--case", "s0859"],
        ["--ct", "ct.glb", "--out", "data/vitronic/fit.glb", "--report", "fit.json"],
        ["--ct", "ct.glb", "--render-dir", "data/vitronic/renders"],
    ],
)
def test_a_fit_body_argument_under_data_vitronic_is_refused(args: list[str]) -> None:
    assert blender.personal_paths(args) != []


def test_fit_body_arguments_on_open_data_are_let_through() -> None:
    args = [
        "--body", "data/bodies/anny-adult-f.rigged.glb", "--ct", "data/work/s0859/s0859.glb",
        "--outline", "data/work/s0859/s0859.outline.glb", "--case", "s0859", "--sex", "female",
        "--out", "data/onetoone/s0859.onetoone.glb", "--report=data/onetoone/s0859.json",
    ]  # fmt: skip
    assert blender.personal_paths(args) == []


def test_an_unknown_task_shows_the_usage_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["blender.py", "fit_body"])
    assert blender.main() == 2
    assert "blender.py fit-body" in capsys.readouterr().err


def test_the_self_check_runs_every_script_that_has_one() -> None:
    # The scripts run only inside Blender, so check the sources: each script
    # in blender/ that takes --self-check is on blender.py's self-check list.
    runner = (ROOT / "scripts" / "blender.py").read_text()
    for script in sorted((ROOT / "blender").glob("*.py")):
        if '"--self-check" in' in script.read_text():
            assert f'"{script.name}", ["--self-check"]' in runner, script.name
