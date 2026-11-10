"""
The structures the bodies are built from: TotalSegmentator "total" label name,
ontology term, anatomical name, body system, display colour, plain-language
name and one line on what it is. Organs are named by UBERON, bones by FMA,
which has a term for every numbered and sided bone where UBERON does not.
mesh.py names each mesh by its term (UBERON:0002107 -> UBERON_0002107).

One list per body system, each in its own block: SKELETON, then
CARDIOVASCULAR, RESPIRATORY, DIGESTIVE, URINARY, NERVOUS, ENDOCRINE and
MUSCULAR. SYSTEMS maps each system's name to its list, and SETS the --set
names of segment.py and mesh.py to what they build: "organs" (the first
render's ten), "reference" (the organs, then the trunk skeleton), "skeleton",
one set per system, and "body" (the skeleton and every system). A system's
builder fills its own list. Every term is matched exactly, label for label,
on the EBI Ontology Lookup Service before it goes in.
"""

from __future__ import annotations

from typing import NamedTuple


class Structure(NamedTuple):
    label: str  # TotalSegmentator "total" task label
    term: str  # UBERON or FMA ID
    name: str  # anatomical name
    system: str
    colour: tuple[float, float, float]  # linear RGB, 0..1
    plain: str = ""  # plain-language name, for the viewer's main view
    about: str = ""  # one line on what it is and does, for the same view


# Skeleton: the trunk skeleton, from the neck to the hips. Every bone says
# "skeletal", which mesh.py, retopo.py and studio.py test for. The vertebrae
# and ribs are built from their FMA tables below, in the order C1 to L5,
# then the left ribs 1 to 12, then the right.

BONE = (0.86, 0.82, 0.70)
CARTILAGE = (0.78, 0.84, 0.86)
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth", "tenth", "eleventh", "twelfth")  # fmt: skip
VERTEBRA_TERMS = {  # FMA, top to bottom
    "C": (12519, 12520, 12521, 12522, 12523, 12524, 12525),
    "T": (9165, 9187, 9209, 9248, 9922, 9945, 9968, 9991, 10014, 10037, 10059, 10081),
    "L": (13072, 13073, 13074, 13075, 13076),
}
RIB_TERMS = {  # FMA, ribs 1 to 12
    "left": (7987, 8012, 8039, 8148, 8093, 8202, 8256, 8310, 8391, 8472, 8532, 8534),
    "right": (7857, 7882, 7909, 7957, 8066, 8175, 8229, 8283, 8364, 8445, 8531, 8533),
}


def bone(label: str, term: str, name: str, plain: str, about: str) -> Structure:
    return Structure(label, term, name, "skeletal", BONE, plain, about)


def vertebra(region: str, k: int) -> Structure:
    """Vertebra k of region C, T or L, counted from the top."""
    label, term = f"vertebrae_{region}{k}", f"FMA:{VERTEBRA_TERMS[region][k - 1]}"
    nth = ORDINALS[k - 1]
    if region == "C":
        if k == 1:
            about = "The top neck bone; it carries the skull and lets the head nod."
            return bone(label, term, "atlas", "top neck bone (atlas)", about)
        if k == 2:
            about = "The second neck bone; the head turns from side to side on its peg."
            return bone(label, term, "axis", "second neck bone (axis)", about)
        about = f"Neck bone {k} of 7; the neck bones carry the head and guard the spinal cord."
        return bone(label, term, f"{nth} cervical vertebra", f"neck bone {k}", about)
    if region == "T":
        about = f"Back bone {k} of 12 in the chest; the {nth} pair of ribs is anchored to it."
        return bone(label, term, f"{nth} thoracic vertebra", f"upper back bone {k}", about)
    about = f"Lower back bone {k} of 5, the spine's largest; they carry the upper body's weight."
    return bone(label, term, f"{nth} lumbar vertebra", f"lower back bone {k}", about)


RIB_KINDS = {
    "true": "a true rib: its own strip of cartilage joins it to the breastbone",
    "false": "a false rib: its cartilage joins the one above, forming the costal arch",
    "floating": "a floating rib: short, with a free front end",
}


def rib(side: str, k: int) -> Structure:
    """Rib k of the side, counted from the top."""
    if k == 1:
        about = "The topmost rib: short, flat and curved, tucked under the collarbone."
    else:
        kind = "true" if k <= 7 else "false" if k <= 10 else "floating"
        about = f"Rib {k} of 12, {RIB_KINDS[kind]}."
    term = f"FMA:{RIB_TERMS[side][k - 1]}"
    return bone(f"rib_{side}_{k}", term, f"{side} {ORDINALS[k - 1]} rib", f"{side} rib {k}", about)


SKELETON: tuple[Structure, ...] = (
    *(vertebra(region, k) for region, terms in VERTEBRA_TERMS.items()
      for k in range(1, len(terms) + 1)),
    bone("sacrum", "FMA:16202", "sacrum", "base of the spine",
         "Five fused bones at the base of the spine that join it to the pelvis."),
    *(rib(side, k) for side in ("left", "right") for k in range(1, 13)),
    bone("sternum", "FMA:7485", "sternum", "breastbone",
         "The flat bone in the middle of the chest that the collarbones and rib cartilages join."),
    Structure("costal_cartilages", "FMA:85213", "set of costal cartilages", "skeletal", CARTILAGE,
              "rib cartilage",
              "Flexible strips that join the front ends of ribs 1 to 10 to the breastbone and let "
              "the chest expand."),
    bone("clavicula_left", "FMA:13323", "left clavicle", "left collarbone",
         "The strut from the breastbone to the shoulder blade that holds the shoulder out."),
    bone("clavicula_right", "FMA:13322", "right clavicle", "right collarbone",
         "The strut from the breastbone to the shoulder blade that holds the shoulder out."),
    bone("scapula_left", "FMA:13396", "left scapula", "left shoulder blade",
         "The flat triangular bone over the back of the ribcage; its socket holds the upper arm."),
    bone("scapula_right", "FMA:13395", "right scapula", "right shoulder blade",
         "The flat triangular bone over the back of the ribcage; its socket holds the upper arm."),
    bone("humerus_left", "FMA:23131", "left humerus", "left upper arm bone",
         "The long bone of the upper arm, from the shoulder socket to the elbow."),
    bone("humerus_right", "FMA:23130", "right humerus", "right upper arm bone",
         "The long bone of the upper arm, from the shoulder socket to the elbow."),
    bone("hip_left", "FMA:16587", "left hip bone", "left hip bone",
         "The large bone forming one side of the pelvis; its socket holds the thigh bone."),
    bone("hip_right", "FMA:16586", "right hip bone", "right hip bone",
         "The large bone forming one side of the pelvis; its socket holds the thigh bone."),
    bone("femur_left", "FMA:24475", "left femur", "left thigh bone",
         "The longest bone in the body, from the hip socket to the knee."),
    bone("femur_right", "FMA:24474", "right femur", "right thigh bone",
         "The longest bone in the body, from the hip socket to the knee."),
)  # fmt: skip

# Cardiovascular: the heart and the vessels the "total" task names (aorta,
# venae cavae, pulmonary vein, brachiocephalic trunk and veins, subclavian and
# carotid arteries, portal and splenic vein, iliac arteries and veins).

CARDIOVASCULAR: tuple[Structure, ...] = (
    Structure("heart", "UBERON:0000948", "heart", "cardiovascular", (0.42, 0.14, 0.13),
              "heart", "The muscular pump that drives blood around the body."),
    Structure("aorta", "UBERON:0000947", "aorta", "cardiovascular", (0.80, 0.08, 0.08),
              "main artery",
              "The body's largest artery, carrying blood from the heart down through the chest "
              "and belly."),
    Structure("inferior_vena_cava", "UBERON:0001072", "inferior vena cava", "cardiovascular",
              (0.15, 0.20, 0.65), "main vein of the lower body",
              "The large vein that returns blood from the legs and belly to the heart."),
)  # fmt: skip

# Respiratory: the five lung lobes and the trachea.

RESPIRATORY: tuple[Structure, ...] = ()

# Digestive: the gut in the order food takes (oesophagus, stomach, duodenum,
# small intestine, colon), then the liver, gallbladder and pancreas. The
# "small_bowel" label is the small intestine after the duodenum, which has a
# label of its own. The spleen is a lymphoid organ; it goes with the digestive
# organs it lies among, as the systems have no lymphoid list.

DIGESTIVE: tuple[Structure, ...] = (
    Structure("esophagus", "UBERON:0001043", "esophagus", "digestive", (0.66, 0.34, 0.38),
              "food pipe",
              "The muscular tube that carries swallowed food from the throat down to the stomach."),
    Structure("stomach", "UBERON:0000945", "stomach", "digestive", (0.80, 0.45, 0.40),
              "stomach",
              "A muscular bag that churns food with acid and passes it on to the gut."),
    Structure("duodenum", "UBERON:0002114", "duodenum", "digestive", (0.90, 0.48, 0.22),
              "first part of the small bowel",
              "The short first bend of the gut after the stomach, where bile and pancreatic juice "
              "meet the food."),
    Structure("small_bowel", "UBERON:0002108", "small intestine", "digestive", (0.90, 0.62, 0.58),
              "small bowel",
              "The long coiled tube after the duodenum, where most of the food's nutrients pass "
              "into the blood."),
    Structure("colon", "UBERON:0001155", "colon", "digestive", (0.60, 0.40, 0.26),
              "large bowel",
              "The wide last stretch of the gut; it takes back water and salts and forms the "
              "stool."),
    Structure("liver", "UBERON:0002107", "liver", "digestive", (0.45, 0.12, 0.08),
              "liver",
              "Processes what the gut absorbs, stores energy and clears the blood of toxins."),
    Structure("gallbladder", "UBERON:0002110", "gallbladder", "digestive", (0.18, 0.40, 0.12),
              "gallbladder",
              "A small bag under the liver that stores bile and releases it to digest fat."),
    Structure("pancreas", "UBERON:0001264", "pancreas", "digestive", (0.85, 0.65, 0.35),
              "pancreas",
              "Makes digestive juices for the gut and hormones such as insulin for the blood."),
    Structure("spleen", "UBERON:0002106", "spleen", "digestive", (0.35, 0.08, 0.20),
              "spleen",
              "Filters the blood, recycles worn-out red cells and holds immune cells; it lies "
              "behind the stomach."),
)  # fmt: skip

# Urinary: the kidneys, kidney cysts if present, the bladder and the prostate.

URINARY: tuple[Structure, ...] = (
    Structure("kidney_left", "UBERON:0004538", "left kidney", "urinary", (0.55, 0.20, 0.15),
              "left kidney",
              "Filters waste and surplus water out of the blood to make urine."),
    Structure("kidney_right", "UBERON:0004539", "right kidney", "urinary", (0.55, 0.20, 0.15),
              "right kidney",
              "Filters waste and surplus water out of the blood to make urine."),
)  # fmt: skip

# Nervous: the spinal cord, and the brain only when the scan holds it whole.

NERVOUS: tuple[Structure, ...] = ()

# Endocrine: the thyroid and the adrenal glands.

ENDOCRINE: tuple[Structure, ...] = ()

# Muscular: the deep back muscles (autochthon), the iliopsoas and the gluteal
# muscles.

MUSCULAR: tuple[Structure, ...] = ()

SYSTEMS: dict[str, tuple[Structure, ...]] = {
    "skeleton": SKELETON,
    "cardiovascular": CARDIOVASCULAR,
    "respiratory": RESPIRATORY,
    "digestive": DIGESTIVE,
    "urinary": URINARY,
    "nervous": NERVOUS,
    "endocrine": ENDOCRINE,
    "muscular": MUSCULAR,
}
BODY: tuple[Structure, ...] = tuple(s for system in SYSTEMS.values() for s in system)
BY_LABEL = {s.label: s for s in BODY}

# The first render's ten organs, in its order; the organ demo and mesh_check.py build them.
ORGANS: tuple[Structure, ...] = tuple(
    BY_LABEL[label]
    for label in ("liver", "gallbladder", "stomach", "pancreas", "spleen", "kidney_left",
                  "kidney_right", "heart", "aorta", "inferior_vena_cava")
)  # fmt: skip
STRUCTURES = ORGANS
REFERENCE = ORGANS + SKELETON
SETS: dict[str, tuple[Structure, ...]] = {
    "organs": ORGANS,
    "reference": REFERENCE,
    **SYSTEMS,
    "body": BODY,
}
RULED_SETS = ("reference", *SYSTEMS, "body")  # the sets built under the cut and symmetry rules

# TotalSegmentator tasks whose weights are Apache-2.0. Licensed tasks are never run.
# The README lists them under "Openly available for any usage (Apache-2.0 license)":
# https://github.com/wasserth/TotalSegmentator#subtasks
ALLOWED_TASKS = frozenset({"total", "body", "vertebrae_body", "head_glands_cavities"})


def node_name(term: str) -> str:
    """glTF node name for a term: UBERON:0002107 -> UBERON_0002107, FMA:7987 -> FMA_7987."""
    return term.replace(":", "_")
