"""
The structures v0 builds: TotalSegmentator "total" label name, ontology term,
body system and display colour. Organs are named by UBERON, bones by FMA, which
has a term for every numbered and sided bone where UBERON does not. mesh.py
names each mesh by its term (UBERON:0002107 -> UBERON_0002107).

ORGANS is the first render's set; REFERENCE adds the trunk skeleton for the
reference bodies. Every term was matched exactly, label for label, on the EBI
Ontology Lookup Service.
"""

from __future__ import annotations

from typing import NamedTuple


class Structure(NamedTuple):
    label: str  # TotalSegmentator "total" task label
    term: str  # UBERON or FMA ID
    name: str
    system: str
    colour: tuple[float, float, float]  # linear RGB, 0..1


ORGANS: tuple[Structure, ...] = (
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

BONE = (0.86, 0.82, 0.70)
CARTILAGE = (0.78, 0.84, 0.86)
SKELETON: tuple[Structure, ...] = (
    Structure("vertebrae_C1", "FMA:12519", "atlas", "skeletal", BONE),
    Structure("vertebrae_C2", "FMA:12520", "axis", "skeletal", BONE),
    Structure("vertebrae_C3", "FMA:12521", "third cervical vertebra", "skeletal", BONE),
    Structure("vertebrae_C4", "FMA:12522", "fourth cervical vertebra", "skeletal", BONE),
    Structure("vertebrae_C5", "FMA:12523", "fifth cervical vertebra", "skeletal", BONE),
    Structure("vertebrae_C6", "FMA:12524", "sixth cervical vertebra", "skeletal", BONE),
    Structure("vertebrae_C7", "FMA:12525", "seventh cervical vertebra", "skeletal", BONE),
    Structure("vertebrae_T1", "FMA:9165", "first thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T2", "FMA:9187", "second thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T3", "FMA:9209", "third thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T4", "FMA:9248", "fourth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T5", "FMA:9922", "fifth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T6", "FMA:9945", "sixth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T7", "FMA:9968", "seventh thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T8", "FMA:9991", "eighth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T9", "FMA:10014", "ninth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T10", "FMA:10037", "tenth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T11", "FMA:10059", "eleventh thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_T12", "FMA:10081", "twelfth thoracic vertebra", "skeletal", BONE),
    Structure("vertebrae_L1", "FMA:13072", "first lumbar vertebra", "skeletal", BONE),
    Structure("vertebrae_L2", "FMA:13073", "second lumbar vertebra", "skeletal", BONE),
    Structure("vertebrae_L3", "FMA:13074", "third lumbar vertebra", "skeletal", BONE),
    Structure("vertebrae_L4", "FMA:13075", "fourth lumbar vertebra", "skeletal", BONE),
    Structure("vertebrae_L5", "FMA:13076", "fifth lumbar vertebra", "skeletal", BONE),
    Structure("sacrum", "FMA:16202", "sacrum", "skeletal", BONE),
    Structure("rib_left_1", "FMA:7987", "left first rib", "skeletal", BONE),
    Structure("rib_left_2", "FMA:8012", "left second rib", "skeletal", BONE),
    Structure("rib_left_3", "FMA:8039", "left third rib", "skeletal", BONE),
    Structure("rib_left_4", "FMA:8148", "left fourth rib", "skeletal", BONE),
    Structure("rib_left_5", "FMA:8093", "left fifth rib", "skeletal", BONE),
    Structure("rib_left_6", "FMA:8202", "left sixth rib", "skeletal", BONE),
    Structure("rib_left_7", "FMA:8256", "left seventh rib", "skeletal", BONE),
    Structure("rib_left_8", "FMA:8310", "left eighth rib", "skeletal", BONE),
    Structure("rib_left_9", "FMA:8391", "left ninth rib", "skeletal", BONE),
    Structure("rib_left_10", "FMA:8472", "left tenth rib", "skeletal", BONE),
    Structure("rib_left_11", "FMA:8532", "left eleventh rib", "skeletal", BONE),
    Structure("rib_left_12", "FMA:8534", "left twelfth rib", "skeletal", BONE),
    Structure("rib_right_1", "FMA:7857", "right first rib", "skeletal", BONE),
    Structure("rib_right_2", "FMA:7882", "right second rib", "skeletal", BONE),
    Structure("rib_right_3", "FMA:7909", "right third rib", "skeletal", BONE),
    Structure("rib_right_4", "FMA:7957", "right fourth rib", "skeletal", BONE),
    Structure("rib_right_5", "FMA:8066", "right fifth rib", "skeletal", BONE),
    Structure("rib_right_6", "FMA:8175", "right sixth rib", "skeletal", BONE),
    Structure("rib_right_7", "FMA:8229", "right seventh rib", "skeletal", BONE),
    Structure("rib_right_8", "FMA:8283", "right eighth rib", "skeletal", BONE),
    Structure("rib_right_9", "FMA:8364", "right ninth rib", "skeletal", BONE),
    Structure("rib_right_10", "FMA:8445", "right tenth rib", "skeletal", BONE),
    Structure("rib_right_11", "FMA:8531", "right eleventh rib", "skeletal", BONE),
    Structure("rib_right_12", "FMA:8533", "right twelfth rib", "skeletal", BONE),
    Structure("sternum", "FMA:7485", "sternum", "skeletal", BONE),
    Structure("costal_cartilages", "FMA:85213", "set of costal cartilages", "skeletal", CARTILAGE),
    Structure("clavicula_left", "FMA:13323", "left clavicle", "skeletal", BONE),
    Structure("clavicula_right", "FMA:13322", "right clavicle", "skeletal", BONE),
    Structure("scapula_left", "FMA:13396", "left scapula", "skeletal", BONE),
    Structure("scapula_right", "FMA:13395", "right scapula", "skeletal", BONE),
    Structure("humerus_left", "FMA:23131", "left humerus", "skeletal", BONE),
    Structure("humerus_right", "FMA:23130", "right humerus", "skeletal", BONE),
    Structure("hip_left", "FMA:16587", "left hip bone", "skeletal", BONE),
    Structure("hip_right", "FMA:16586", "right hip bone", "skeletal", BONE),
    Structure("femur_left", "FMA:24475", "left femur", "skeletal", BONE),
    Structure("femur_right", "FMA:24474", "right femur", "skeletal", BONE),
)  # fmt: skip

# Segmented only to keep faces out of every output (imaging.py's head guard);
# never meshed or released.
HEAD_GUARD = ("skull", "brain")

STRUCTURES = ORGANS
REFERENCE = ORGANS + SKELETON
SETS = {"organs": ORGANS, "reference": REFERENCE}

# TotalSegmentator tasks whose weights are Apache-2.0. Licensed tasks are never run.
# The README lists them under "Openly available for any usage (Apache-2.0 license)":
# https://github.com/wasserth/TotalSegmentator#subtasks
ALLOWED_TASKS = frozenset({"total", "body", "vertebrae_body"})


def node_name(term: str) -> str:
    """glTF node name for a term: UBERON:0002107 -> UBERON_0002107, FMA:7987 -> FMA_7987."""
    return term.replace(":", "_")
