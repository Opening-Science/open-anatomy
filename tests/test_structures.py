"""Tests for the structure tables in structures.py."""

from __future__ import annotations

import re

import pytest

from structures import (
    ALLOWED_TASKS,
    LAYERS,
    ORGANS,
    PLAIN,
    REFERENCE,
    SETS,
    SKELETON,
    SYSTEMS,
    Structure,
    layer_of,
    node_name,
    ordinal,
    partner_of,
)

TERM = re.compile(r"^(UBERON:\d{7}|FMA:\d+)$")


def test_the_reference_set_is_the_organs_then_the_skeleton() -> None:
    assert REFERENCE == ORGANS + SKELETON
    assert SETS == {"organs": ORGANS, "reference": REFERENCE, "skeleton": SKELETON}


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
    assert all(s.system == "skeleton" for s in SKELETON)


def test_every_system_is_in_the_contracts_closed_list() -> None:
    assert SYSTEMS == (
        "skeleton", "cardiovascular", "respiratory", "digestive", "urinary", "nervous",
        "endocrine", "muscular",
    )  # fmt: skip
    for s in REFERENCE:
        assert s.system in SYSTEMS, s.label


def test_layer_of_puts_bones_in_the_skeleton_layer_and_the_rest_in_organs() -> None:
    assert layer_of("skeleton") == "skeleton"
    assert layer_of("muscular") == "muscle"
    for system in set(SYSTEMS) - {"skeleton", "muscular"}:
        assert layer_of(system) == "organs"
    assert {layer_of(system) for system in SYSTEMS} <= set(LAYERS)
    with pytest.raises(ValueError, match="skeletal"):
        layer_of("skeletal")


def test_every_structure_has_a_plain_name_and_one_line_about_it() -> None:
    assert set(PLAIN) == {s.label for s in REFERENCE}
    for s in REFERENCE:
        assert s.plain and (s.plain[0].isupper() or s.plain[0].isdigit()), s.label
        assert s.about and "\n" not in s.about and len(s.about) <= 120, s.label
        assert not s.about.endswith("."), s.label


def test_sided_structures_have_sided_plain_names() -> None:
    for s in REFERENCE:
        if "_left" in s.label:
            assert s.plain.startswith("Left "), s.label
            partner = next(p for p in REFERENCE if p.label == partner_of(s.label))
            assert partner.plain == s.plain.replace("Left ", "Right ")


def test_every_sided_structure_has_its_partner_in_the_same_system() -> None:
    by_label = {s.label: s for s in REFERENCE}
    for label, s in by_label.items():
        if "left" in label:
            partner = by_label.get(label.replace("left", "right"))
            assert partner is not None, label
            assert partner.system == s.system


def test_partner_of_swaps_the_side_and_leaves_the_midline_alone() -> None:
    assert partner_of("kidney_left") == "kidney_right"
    assert partner_of("rib_right_3") == "rib_left_3"
    assert partner_of("sternum") is None
    assert partner_of("costal_cartilages") is None


def test_ordinal() -> None:
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
    ]  # fmt: skip


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
