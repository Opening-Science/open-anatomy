"""
The structures v0 builds: TotalSegmentator "total" label name, UBERON ID, body
system and display colour. mesh.py names each mesh by its UBERON ID, and
render.py colours it by system.

The IDs were checked against UBERON on the EBI Ontology Lookup Service.
"""

from __future__ import annotations

from typing import NamedTuple


class Structure(NamedTuple):
    label: str  # TotalSegmentator "total" task label
    uberon: str
    name: str
    system: str
    colour: tuple[float, float, float]  # linear RGB, 0..1


STRUCTURES: tuple[Structure, ...] = (
    Structure("liver", "UBERON:0002107", "liver", "digestive", (0.45, 0.12, 0.08)),
    Structure("gallbladder", "UBERON:0002110", "gallbladder", "digestive", (0.18, 0.40, 0.12)),
    Structure("stomach", "UBERON:0000945", "stomach", "digestive", (0.80, 0.45, 0.40)),
    Structure("pancreas", "UBERON:0001264", "pancreas", "digestive", (0.85, 0.65, 0.35)),
    Structure("spleen", "UBERON:0002106", "spleen", "lymphoid", (0.35, 0.08, 0.20)),
    Structure("kidney_left", "UBERON:0004538", "left kidney", "urinary", (0.55, 0.20, 0.15)),
    Structure("kidney_right", "UBERON:0004539", "right kidney", "urinary", (0.55, 0.20, 0.15)),
    Structure("heart", "UBERON:0000948", "heart", "cardiovascular", (0.42, 0.14, 0.13)),
    Structure("aorta", "UBERON:0000947", "aorta", "cardiovascular", (0.80, 0.08, 0.08)),
    Structure(
        "inferior_vena_cava", "UBERON:0001072", "inferior vena cava", "cardiovascular",
        (0.15, 0.20, 0.65),
    ),
)  # fmt: skip

# TotalSegmentator tasks whose weights are Apache-2.0. Licensed tasks are never run.
ALLOWED_TASKS = frozenset({"total", "body", "vertebrae_body"})


def node_name(uberon: str) -> str:
    """glTF node name for an UBERON ID: UBERON:0002107 -> UBERON_0002107."""
    return uberon.replace(":", "_")
