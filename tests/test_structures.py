"""Tests for the structure tables in structures.py: one list per body system,
the sets built from them, and the plain-language fields."""

from __future__ import annotations

import re

import pytest

from structures import (
    ALLOWED_TASKS,
    BODY,
    ORGANS,
    REFERENCE,
    RULED_SETS,
    SETS,
    SKELETON,
    SYSTEMS,
    Structure,
    node_name,
)

TERM = re.compile(r"^(UBERON:\d{7}|FMA:\d+)$")
SYSTEM_NAMES = ("skeleton", "cardiovascular", "respiratory", "digestive", "urinary", "nervous",
                "endocrine", "muscular")  # fmt: skip
TEN_ORGANS = {
    "liver": "digestive",
    "gallbladder": "digestive",
    "stomach": "digestive",
    "pancreas": "digestive",
    "spleen": "digestive",
    "kidney_left": "urinary",
    "kidney_right": "urinary",
    "heart": "cardiovascular",
    "aorta": "cardiovascular",
    "inferior_vena_cava": "cardiovascular",
}


def test_the_sets_are_the_organs_the_reference_each_system_and_the_body() -> None:
    assert tuple(SYSTEMS) == SYSTEM_NAMES
    assert SYSTEMS["skeleton"] is SKELETON
    assert BODY == tuple(s for system in SYSTEMS.values() for s in system)
    assert REFERENCE == ORGANS + SKELETON
    assert SETS == {"organs": ORGANS, "reference": REFERENCE, **SYSTEMS, "body": BODY}
    assert RULED_SETS == ("reference", *SYSTEM_NAMES, "body")
    assert "organs" not in RULED_SETS


def test_the_organ_demo_keeps_its_ten_organs_in_order() -> None:
    assert [s.label for s in ORGANS] == list(TEN_ORGANS)
    assert set(ORGANS) <= set(BODY)


def test_each_of_the_ten_organs_sits_in_its_system() -> None:
    for label, system in TEN_ORGANS.items():
        assert label in {s.label for s in SYSTEMS[system]}, label


def test_each_list_holds_its_own_system() -> None:
    for system, structures in SYSTEMS.items():
        field = "skeletal" if system == "skeleton" else system
        assert all(s.system == field for s in structures), system


def test_no_structure_is_in_two_systems() -> None:
    for field in ("label", "term", "name", "plain"):
        values = [getattr(s, field) for s in BODY]
        assert len(values) == len(set(values)), field


@pytest.mark.parametrize("structure", BODY, ids=lambda s: str(s.label))
def test_each_term_is_well_formed(structure: Structure) -> None:
    assert TERM.match(structure.term)
    assert re.fullmatch(r"[a-z][a-z0-9_]*|vertebrae_[CTL]\d+", structure.label)


@pytest.mark.parametrize("structure", BODY, ids=lambda s: str(s.label))
def test_each_structure_has_a_plain_name_and_one_line_about_it(structure: Structure) -> None:
    assert structure.plain and structure.plain == structure.plain.strip()
    assert structure.plain[0].islower(), "plain names read like the anatomical ones, in lower case"
    assert structure.about and "\n" not in structure.about
    assert structure.about[0].isupper() and structure.about.endswith(".")
    assert len(structure.about) <= 120, "one line"


def test_the_plain_fields_default_to_empty() -> None:
    s = Structure("x", "UBERON:0000001", "x", "digestive", (0.0, 0.0, 0.0))
    assert (s.plain, s.about) == ("", "")


def test_organs_use_uberon_and_bones_use_fma() -> None:
    assert all(s.term.startswith("UBERON:") for s in ORGANS)
    assert all(s.term.startswith("FMA:") for s in SKELETON)
    assert all(s.system == "skeletal" for s in SKELETON)


def test_every_sided_structure_has_its_partner_in_the_same_system() -> None:
    by_label = {s.label: s for s in BODY}
    for label, s in by_label.items():
        if "left" in label:
            partner = by_label.get(label.replace("left", "right"))
            assert partner is not None, label
            assert partner.system == s.system
            assert partner.plain == s.plain.replace("left", "right")


def test_there_are_twelve_ribs_a_side_and_twenty_four_vertebrae() -> None:
    labels = {s.label for s in SKELETON}
    for side in ("left", "right"):
        assert {f"rib_{side}_{k}" for k in range(1, 13)} <= labels
    spine = [s for s in SKELETON if s.label.startswith("vertebrae_")]
    assert [s.label[10:] for s in spine] == (
        [f"C{k}" for k in range(1, 8)]
        + [f"T{k}" for k in range(1, 13)]
        + [f"L{k}" for k in range(1, 6)]
    )
    assert spine[0].name == "atlas" and spine[1].name == "axis"
    assert spine[2].name == "third cervical vertebra"
    assert spine[-1].name == "fifth lumbar vertebra"


def test_colours_are_linear_rgb_in_range() -> None:
    for s in BODY:
        assert len(s.colour) == 3
        assert all(0.0 <= c <= 1.0 for c in s.colour), s.label


def test_node_name_replaces_the_colon_for_both_ontologies() -> None:
    assert node_name("UBERON:0002107") == "UBERON_0002107"
    assert node_name("FMA:7987") == "FMA_7987"
    names = [node_name(s.term) for s in BODY]
    assert len(names) == len(set(names))
    assert all(":" not in n for n in names)


def test_only_the_apache_tasks_are_allowed() -> None:
    assert ALLOWED_TASKS == {"total", "body", "vertebrae_body", "head_glands_cavities"}
