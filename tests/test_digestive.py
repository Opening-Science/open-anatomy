"""Tests for the digestive list in structures.py: its labels and terms, the
sets it belongs to, and its plain-language fields."""

from __future__ import annotations

import re

from structures import BODY, DIGESTIVE, ORGANS, RULED_SETS, SETS, SYSTEMS, node_name

# The "total" task's label and the UBERON term whose label on the Ontology
# Lookup Service is the structure's name, in the list's order.
TERMS = {
    "esophagus": ("UBERON:0001043", "esophagus"),
    "stomach": ("UBERON:0000945", "stomach"),
    "duodenum": ("UBERON:0002114", "duodenum"),
    "small_bowel": ("UBERON:0002108", "small intestine"),
    "colon": ("UBERON:0001155", "colon"),
    "liver": ("UBERON:0002107", "liver"),
    "gallbladder": ("UBERON:0002110", "gallbladder"),
    "pancreas": ("UBERON:0001264", "pancreas"),
    "spleen": ("UBERON:0002106", "spleen"),
}


def test_the_list_is_the_gut_in_order_then_the_organs_beside_it() -> None:
    assert [s.label for s in DIGESTIVE] == list(TERMS)


def test_each_structure_has_the_term_of_its_name() -> None:
    assert {s.label: (s.term, s.name) for s in DIGESTIVE} == TERMS
    assert all(re.fullmatch(r"UBERON:\d{7}", s.term) for s in DIGESTIVE)


def test_no_term_name_or_node_is_used_twice_in_the_body() -> None:
    for s in DIGESTIVE:
        same = [o for o in BODY if o.term == s.term or o.name == s.name or o.plain == s.plain]
        assert same == [s], s.label
    nodes = [node_name(s.term) for s in DIGESTIVE]
    assert len(set(nodes)) == len(nodes) == 9


def test_the_digestive_set_is_the_list_and_part_of_the_body() -> None:
    assert SETS["digestive"] is DIGESTIVE
    assert SYSTEMS["digestive"] is DIGESTIVE
    assert "digestive" in RULED_SETS, "built under the cut rules"
    assert set(DIGESTIVE) <= set(BODY)
    assert all(s.system == "digestive" for s in DIGESTIVE)
    assert [s.label for s in BODY if s.system == "digestive"] == list(TERMS)


def test_the_first_render_keeps_its_five_digestive_organs() -> None:
    labels = [s.label for s in ORGANS if s.system == "digestive"]
    assert labels == ["liver", "gallbladder", "stomach", "pancreas", "spleen"]


def test_each_colour_is_its_own() -> None:
    colours = [s.colour for s in DIGESTIVE]
    assert len(set(colours)) == len(colours)
    assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in colours)


def test_the_plain_names_are_everyday_words_with_one_line_each() -> None:
    plain = {s.label: s.plain for s in DIGESTIVE}
    assert plain["esophagus"] == "food pipe"
    assert plain["small_bowel"] == "small bowel"
    assert plain["colon"] == "large bowel"
    for s in DIGESTIVE:
        assert s.plain == s.plain.strip().lower()
        assert s.about[0].isupper() and s.about.endswith(".") and len(s.about) <= 120
