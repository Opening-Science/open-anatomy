"""Tests for the structure tables in structures.py."""

from __future__ import annotations

import re

import pytest

from structures import ALLOWED_TASKS, ORGANS, REFERENCE, SETS, SKELETON, Structure, node_name

TERM = re.compile(r"^(UBERON:\d{7}|FMA:\d+)$")


def test_the_reference_set_is_the_organs_then_the_skeleton() -> None:
    assert REFERENCE == ORGANS + SKELETON
    assert SETS == {"organs": ORGANS, "reference": REFERENCE}


def test_labels_terms_and_names_are_unique() -> None:
    for field in ("label", "term", "name"):
        values = [getattr(s, field) for s in REFERENCE]
        assert len(values) == len(set(values)), field


@pytest.mark.parametrize("structure", REFERENCE, ids=lambda s: str(s.label))
def test_each_term_is_well_formed(structure: Structure) -> None:
    assert TERM.match(structure.term)
    assert re.fullmatch(r"[a-z][a-z0-9_]*|vertebrae_[CTL]\d+", structure.label)


def test_organs_use_uberon_and_bones_use_fma() -> None:
    assert all(s.term.startswith("UBERON:") for s in ORGANS)
    assert all(s.term.startswith("FMA:") for s in SKELETON)
    assert all(s.system == "skeletal" for s in SKELETON)


def test_every_sided_structure_has_its_partner_in_the_same_system() -> None:
    by_label = {s.label: s for s in REFERENCE}
    for label, s in by_label.items():
        if "left" in label:
            partner = by_label.get(label.replace("left", "right"))
            assert partner is not None, label
            assert partner.system == s.system


def test_there_are_twelve_ribs_a_side() -> None:
    labels = {s.label for s in SKELETON}
    for side in ("left", "right"):
        assert {f"rib_{side}_{k}" for k in range(1, 13)} <= labels


def test_colours_are_linear_rgb_in_range() -> None:
    for s in REFERENCE:
        assert len(s.colour) == 3
        assert all(0.0 <= c <= 1.0 for c in s.colour), s.label


def test_node_name_replaces_the_colon_for_both_ontologies() -> None:
    assert node_name("UBERON:0002107") == "UBERON_0002107"
    assert node_name("FMA:7987") == "FMA_7987"
    names = [node_name(s.term) for s in REFERENCE]
    assert len(names) == len(set(names))
    assert all(":" not in n for n in names)


def test_only_the_apache_tasks_are_allowed() -> None:
    assert "total" in ALLOWED_TASKS
