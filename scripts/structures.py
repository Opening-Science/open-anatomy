"""
The structures v0 builds: TotalSegmentator "total" label name, ontology term,
body system and display colour, with a plain-language name and one line on
what each one does for the explode viewer. Organs are named by UBERON, bones
by FMA, which has a term for every numbered and sided bone where UBERON does
not. mesh.py names each mesh by its term (UBERON:0002107 -> UBERON_0002107).

ORGANS is the first render's set; REFERENCE adds the trunk skeleton for the
reference bodies. Every term was matched exactly, label for label, on the EBI
Ontology Lookup Service.

SYSTEMS is the closed list a node's `system` may take under the GLB contract
(a new system is a contract change), and layer_of() maps a system to the
contract's layer. The spleen is a lymphoid organ; with no lymphoid system in
the list it goes with the digestive organs it lies among.
"""

from __future__ import annotations

from typing import NamedTuple

SYSTEMS = (
    "skeleton",
    "cardiovascular",
    "respiratory",
    "digestive",
    "urinary",
    "nervous",
    "endocrine",
    "muscular",
)
LAYERS = ("skin", "muscle", "skeleton", "organs", "imaging")


def layer_of(system: str) -> str:
    """The contract's layer for a system: bones and cartilage are the skeleton
    layer, muscles the muscle layer and every other organ the organs layer."""
    if system not in SYSTEMS:
        raise ValueError(f"{system!r} is not one of the contract's systems {SYSTEMS}")
    return {"skeleton": "skeleton", "muscular": "muscle"}.get(system, "organs")


class Structure(NamedTuple):
    label: str  # TotalSegmentator "total" task label
    term: str  # UBERON or FMA ID
    name: str
    system: str
    colour: tuple[float, float, float]  # linear RGB, 0..1

    @property
    def plain(self) -> str:
        """The plain-language name the viewer shows: "Left kidney", "Breastbone"."""
        return PLAIN[self.label][0]

    @property
    def about(self) -> str:
        """One line on what the structure is or does, in plain words."""
        return PLAIN[self.label][1]


ORGANS: tuple[Structure, ...] = (
    Structure("liver", "UBERON:0002107", "liver", "digestive", (0.45, 0.12, 0.08)),
    Structure("gallbladder", "UBERON:0002110", "gallbladder", "digestive", (0.18, 0.40, 0.12)),
    Structure("stomach", "UBERON:0000945", "stomach", "digestive", (0.80, 0.45, 0.40)),
    Structure("pancreas", "UBERON:0001264", "pancreas", "digestive", (0.85, 0.65, 0.35)),
    Structure("spleen", "UBERON:0002106", "spleen", "digestive", (0.35, 0.08, 0.20)),
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
    Structure("vertebrae_C1", "FMA:12519", "atlas", "skeleton", BONE),
    Structure("vertebrae_C2", "FMA:12520", "axis", "skeleton", BONE),
    Structure("vertebrae_C3", "FMA:12521", "third cervical vertebra", "skeleton", BONE),
    Structure("vertebrae_C4", "FMA:12522", "fourth cervical vertebra", "skeleton", BONE),
    Structure("vertebrae_C5", "FMA:12523", "fifth cervical vertebra", "skeleton", BONE),
    Structure("vertebrae_C6", "FMA:12524", "sixth cervical vertebra", "skeleton", BONE),
    Structure("vertebrae_C7", "FMA:12525", "seventh cervical vertebra", "skeleton", BONE),
    Structure("vertebrae_T1", "FMA:9165", "first thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T2", "FMA:9187", "second thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T3", "FMA:9209", "third thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T4", "FMA:9248", "fourth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T5", "FMA:9922", "fifth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T6", "FMA:9945", "sixth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T7", "FMA:9968", "seventh thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T8", "FMA:9991", "eighth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T9", "FMA:10014", "ninth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T10", "FMA:10037", "tenth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T11", "FMA:10059", "eleventh thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_T12", "FMA:10081", "twelfth thoracic vertebra", "skeleton", BONE),
    Structure("vertebrae_L1", "FMA:13072", "first lumbar vertebra", "skeleton", BONE),
    Structure("vertebrae_L2", "FMA:13073", "second lumbar vertebra", "skeleton", BONE),
    Structure("vertebrae_L3", "FMA:13074", "third lumbar vertebra", "skeleton", BONE),
    Structure("vertebrae_L4", "FMA:13075", "fourth lumbar vertebra", "skeleton", BONE),
    Structure("vertebrae_L5", "FMA:13076", "fifth lumbar vertebra", "skeleton", BONE),
    Structure("sacrum", "FMA:16202", "sacrum", "skeleton", BONE),
    Structure("rib_left_1", "FMA:7987", "left first rib", "skeleton", BONE),
    Structure("rib_left_2", "FMA:8012", "left second rib", "skeleton", BONE),
    Structure("rib_left_3", "FMA:8039", "left third rib", "skeleton", BONE),
    Structure("rib_left_4", "FMA:8148", "left fourth rib", "skeleton", BONE),
    Structure("rib_left_5", "FMA:8093", "left fifth rib", "skeleton", BONE),
    Structure("rib_left_6", "FMA:8202", "left sixth rib", "skeleton", BONE),
    Structure("rib_left_7", "FMA:8256", "left seventh rib", "skeleton", BONE),
    Structure("rib_left_8", "FMA:8310", "left eighth rib", "skeleton", BONE),
    Structure("rib_left_9", "FMA:8391", "left ninth rib", "skeleton", BONE),
    Structure("rib_left_10", "FMA:8472", "left tenth rib", "skeleton", BONE),
    Structure("rib_left_11", "FMA:8532", "left eleventh rib", "skeleton", BONE),
    Structure("rib_left_12", "FMA:8534", "left twelfth rib", "skeleton", BONE),
    Structure("rib_right_1", "FMA:7857", "right first rib", "skeleton", BONE),
    Structure("rib_right_2", "FMA:7882", "right second rib", "skeleton", BONE),
    Structure("rib_right_3", "FMA:7909", "right third rib", "skeleton", BONE),
    Structure("rib_right_4", "FMA:7957", "right fourth rib", "skeleton", BONE),
    Structure("rib_right_5", "FMA:8066", "right fifth rib", "skeleton", BONE),
    Structure("rib_right_6", "FMA:8175", "right sixth rib", "skeleton", BONE),
    Structure("rib_right_7", "FMA:8229", "right seventh rib", "skeleton", BONE),
    Structure("rib_right_8", "FMA:8283", "right eighth rib", "skeleton", BONE),
    Structure("rib_right_9", "FMA:8364", "right ninth rib", "skeleton", BONE),
    Structure("rib_right_10", "FMA:8445", "right tenth rib", "skeleton", BONE),
    Structure("rib_right_11", "FMA:8531", "right eleventh rib", "skeleton", BONE),
    Structure("rib_right_12", "FMA:8533", "right twelfth rib", "skeleton", BONE),
    Structure("sternum", "FMA:7485", "sternum", "skeleton", BONE),
    Structure("costal_cartilages", "FMA:85213", "set of costal cartilages", "skeleton", CARTILAGE),
    Structure("clavicula_left", "FMA:13323", "left clavicle", "skeleton", BONE),
    Structure("clavicula_right", "FMA:13322", "right clavicle", "skeleton", BONE),
    Structure("scapula_left", "FMA:13396", "left scapula", "skeleton", BONE),
    Structure("scapula_right", "FMA:13395", "right scapula", "skeleton", BONE),
    Structure("humerus_left", "FMA:23131", "left humerus", "skeleton", BONE),
    Structure("humerus_right", "FMA:23130", "right humerus", "skeleton", BONE),
    Structure("hip_left", "FMA:16587", "left hip bone", "skeleton", BONE),
    Structure("hip_right", "FMA:16586", "right hip bone", "skeleton", BONE),
    Structure("femur_left", "FMA:24475", "left femur", "skeleton", BONE),
    Structure("femur_right", "FMA:24474", "right femur", "skeleton", BONE),
)  # fmt: skip

STRUCTURES = ORGANS
REFERENCE = ORGANS + SKELETON
SETS = {"organs": ORGANS, "reference": REFERENCE, "skeleton": SKELETON}
RULED_SETS = ("reference", "skeleton")  # the sets built under the cut and symmetry rules

# TotalSegmentator tasks whose weights are Apache-2.0. Licensed tasks are never run.
# The README lists them under "Openly available for any usage (Apache-2.0 license)":
# https://github.com/wasserth/TotalSegmentator#subtasks
ALLOWED_TASKS = frozenset({"total", "body", "vertebrae_body"})


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, ... 11th, 12th, 13th, ... 21st."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rib_about(number: int) -> str:
    if number == 1:
        return "The shortest rib, tucked under the collarbone"
    if number <= 7:
        return (
            "One of twelve pairs of curved bones that shield the heart and lungs; "
            "its cartilage joins the breastbone"
        )
    if number <= 10:
        return "Its cartilage joins the rib above it, which forms the lower edge of the ribcage"
    return "A floating rib: it ends free in the muscles of the back"


# Plain-language name and one line on what it is, per label, for the viewer's
# main view (the anatomical names stay behind its clinician toggle).
PLAIN: dict[str, tuple[str, str]] = {
    "liver": (
        "Liver", "Processes nutrients from food, makes bile and clears toxins from the blood",
    ),
    "gallbladder": ("Gallbladder", "Stores bile from the liver and releases it to help digest fat"),
    "stomach": ("Stomach", "Mixes food with acid and starts to break it down"),
    "pancreas": ("Pancreas", "Makes digestive juices and the hormone insulin"),
    "spleen": ("Spleen", "Filters the blood and helps fight infection"),
    "kidney_left": ("Left kidney", "Filters the blood and makes urine"),
    "kidney_right": ("Right kidney", "Filters the blood and makes urine"),
    "heart": ("Heart", "Pumps blood around the body"),
    "aorta": ("Main artery", "Carries blood from the heart to the rest of the body"),
    "inferior_vena_cava": (
        "Main vein of the lower body", "Returns blood from the legs and the belly to the heart",
    ),
    "vertebrae_C1": (
        "1st neck bone", "The top bone of the neck: the skull rests on it and nods on it",
    ),
    "vertebrae_C2": (
        "2nd neck bone", "The second neck bone: the head turns from side to side on its peg",
    ),
    **{
        f"vertebrae_C{k}": (
            f"{ordinal(k)} neck bone",
            "One of the seven bones of the neck; the spinal cord runs through it",
        )
        for k in range(3, 8)
    },
    **{
        f"vertebrae_T{k}": (
            f"{ordinal(k)} upper back bone",
            "One of the twelve bones of the upper back; a pair of ribs joins it",
        )
        for k in range(1, 13)
    },
    **{
        f"vertebrae_L{k}": (
            f"{ordinal(k)} lower back bone",
            "One of the five bones of the lower back, the largest in the spine",
        )
        for k in range(1, 6)
    },
    "sacrum": (
        "Base of the spine",
        "Five bones fused into one wedge at the base of the spine, between the hip bones",
    ),
    **{
        f"rib_{side}_{k}": (f"{side.capitalize()} {ordinal(k)} rib", _rib_about(k))
        for side in ("left", "right")
        for k in range(1, 13)
    },
    "sternum": (
        "Breastbone", "The flat bone in the middle of the chest that the ribs join at the front",
    ),
    "costal_cartilages": (
        "Rib cartilage",
        "Flexible cartilage that joins the ribs to the breastbone and lets the chest expand",
    ),
    **{
        f"clavicula_{side}": (
            f"{side.capitalize()} collarbone",
            "The strut from the breastbone to the shoulder blade that holds the shoulder out",
        )
        for side in ("left", "right")
    },
    **{
        f"scapula_{side}": (
            f"{side.capitalize()} shoulder blade",
            "The flat triangular bone on the back of the shoulder; "
            "the arm's socket is on its outer corner",
        )
        for side in ("left", "right")
    },
    **{
        f"humerus_{side}": (
            f"{side.capitalize()} upper arm bone",
            "The bone of the upper arm, from the shoulder socket to the elbow",
        )
        for side in ("left", "right")
    },
    **{
        f"hip_{side}": (
            f"{side.capitalize()} hip bone",
            "One of the two large bones of the pelvis; the thigh bone sits in its socket",
        )
        for side in ("left", "right")
    },
    **{
        f"femur_{side}": (
            f"{side.capitalize()} thigh bone",
            "The thigh bone, the longest and strongest bone in the body",
        )
        for side in ("left", "right")
    },
}  # fmt: skip


def node_name(term: str) -> str:
    """glTF node name for a term: UBERON:0002107 -> UBERON_0002107, FMA:7987 -> FMA_7987."""
    return term.replace(":", "_")


def partner_of(label: str) -> str | None:
    """The label of a sided structure's other side (kidney_left -> kidney_right),
    or None for an unpaired one."""
    if "_left" in label:
        return label.replace("_left", "_right")
    if "_right" in label:
        return label.replace("_right", "_left")
    return None
