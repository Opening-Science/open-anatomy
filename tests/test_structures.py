"""Tests for the structure tables in structures.py: one list per body system,
the sets built from them, and the plain-language fields."""

from __future__ import annotations

import re

import pytest

from structures import (
    ALLOWED_TASKS,
    BODY,
    CARDIOVASCULAR,
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


UNPAIRED = {"atrial_appendage_left"}  # the "total" task has no label for the right one


def test_every_sided_structure_has_its_partner_in_the_same_system() -> None:
    by_label = {s.label: s for s in BODY}
    for label, s in by_label.items():
        if "left" in label and label not in UNPAIRED:
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


# The heart and vessels: every label of the "total" task that is one, with the
# term each was matched to on OLS (UBERON, or FMA where UBERON has no sided term).
HEART_AND_VESSELS = {
    "heart": "UBERON:0000948",
    "atrial_appendage_left": "UBERON:0006630",
    "aorta": "UBERON:0000947",
    "pulmonary_vein": "UBERON:0002016",
    "superior_vena_cava": "UBERON:0001585",
    "inferior_vena_cava": "UBERON:0001072",
    "brachiocephalic_trunk": "UBERON:0001529",
    "brachiocephalic_vein_left": "FMA:4761",
    "brachiocephalic_vein_right": "FMA:4751",
    "subclavian_artery_left": "UBERON:0001584",
    "subclavian_artery_right": "UBERON:0001534",
    "common_carotid_artery_left": "UBERON:0001536",
    "common_carotid_artery_right": "UBERON:0001531",
    "portal_vein_and_splenic_vein": "UBERON:0001639",
    "iliac_artery_left": "UBERON:0035529",
    "iliac_artery_right": "UBERON:0035322",
    "iliac_vena_left": "FMA:21388",
    "iliac_vena_right": "FMA:21387",
}


def test_the_cardiovascular_list_is_the_heart_and_every_vessel_of_the_total_task() -> None:
    assert {s.label: s.term for s in CARDIOVASCULAR} == HEART_AND_VESSELS
    assert len(CARDIOVASCULAR) == len(HEART_AND_VESSELS)  # no label twice
    assert SETS["cardiovascular"] is CARDIOVASCULAR
    assert set(CARDIOVASCULAR) <= set(SETS["body"])
    assert all(s.system == "cardiovascular" for s in CARDIOVASCULAR)
    assert CARDIOVASCULAR[0].label == "heart"


def test_each_cardiovascular_term_is_used_once_and_names_its_side() -> None:
    terms = [s.term for s in CARDIOVASCULAR]
    assert len(terms) == len(set(terms))
    for s in CARDIOVASCULAR:
        assert TERM.match(s.term)
        for side in ("left", "right"):
            if s.label.endswith(f"_{side}"):
                assert s.name.startswith(f"{side} ") and side in s.plain, s.label
    # FMA only where UBERON has no term for the side: the brachiocephalic and iliac veins.
    fma = {s.label for s in CARDIOVASCULAR if s.term.startswith("FMA:")}
    assert fma == {f"{vein}_{side}" for vein in ("brachiocephalic_vein", "iliac_vena")
                   for side in ("left", "right")}  # fmt: skip


def test_a_name_says_so_when_a_mask_holds_more_than_its_term() -> None:
    by_label = {s.label: s for s in CARDIOVASCULAR}
    assert by_label["portal_vein_and_splenic_vein"].name == "hepatic portal vein and splenic vein"
    for label in ("iliac_artery_left", "iliac_vena_right"):
        assert "common and external iliac" in by_label[label].name


def test_arteries_are_red_and_veins_blue() -> None:
    for s in CARDIOVASCULAR:
        red, _, blue = s.colour
        if s.label in ("pulmonary_vein", "heart", "atrial_appendage_left"):
            assert red > blue, s.label  # oxygen-rich blood, and the heart muscle
        elif "vein" in s.label or "vena" in s.label:
            assert blue > red, s.label  # the portal vein's violet is still more blue than red
        else:
            assert red > 3 * blue, s.label
    left = {s.label: s.colour for s in CARDIOVASCULAR if s.label.endswith("_left")}
    for label, colour in left.items():
        if label not in UNPAIRED:
            partner = label.removesuffix("left") + "right"
            assert colour == next(s.colour for s in CARDIOVASCULAR if s.label == partner)


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
