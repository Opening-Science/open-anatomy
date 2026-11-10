# Personalisation rule map

**Status: design only.** Nothing here is built yet. This document fixes what
the rule map is, what it may do and what it must never do, so that the viewer
work in TwinLLM and OpenTwin can start from it.

## Goal

The reference bodies are generic: an adult male and an adult female with every
organ in place. A person's own body can differ. Someone who has had a
nephrectomy has one kidney, and someone with a hip replacement has an implant.
Once a clinician has verified such a fact, the person's view should show it.

The rule map is a small, public JSON file. It says which coded facts change
which nodes, and how. The viewer applies it when it loads a body, on the
person's own device. The released models never change.

## Principles

1. **Verified facts only.** A fact edits the body only once a clinician has
   verified it (see [What counts as verified](#what-counts-as-verified)).
   Self-reported or inferred facts never do.
2. **Local-first.** The person's facts never leave their device. The rule map
   and the models are public and identical for everyone, and the edits are
   computed where the facts already are.
3. **Generic base, data-only edits.** The rule map holds no geometry and no
   code: only codes, node names and edits from a fixed vocabulary.
4. **No change when unsure.** If a rule needs a side and the record gives
   none, or two facts disagree, nothing changes and the viewer says why.
5. **Traceable and reversible.** Every edit shown names the fact and the
   sign-off behind it, and can be switched off to show the reference body.
6. **Illustrative, not clinical.** The view is not clinically validated and
   must not guide care. An edit only hides, marks, mirrors or adds a stock
   part. It never infers a finding the record does not state.

## What counts as verified

The inputs are FHIR R4 resources in the person's own record:

- a `Condition` with `verificationStatus` = `confirmed`, or
- a `Procedure` with `status` = `completed`,

and, for either, a `Provenance` that targets it with an `agent` whose `type` is
`verifier` (system
`http://terminology.hl7.org/CodeSystem/provenance-participant-type`) and whose
`who` is a `Practitioner`. That `Provenance` is the clinician's sign-off. The
viewer shows who verified each fact and when (`Provenance.recorded`).

A resolved `Condition` still counts for absences: a kidney removed years ago
is still absent.

## The rule map

```json
{
  "version": 1,
  "rules": [
    {
      "id": "kidney-absent",
      "facts": [
        { "system": "http://snomed.info/sct", "code": "175905003", "side": "from-body-site" },
        { "system": "http://snomed.info/sct", "code": "765472003", "side": "left" },
        { "system": "http://snomed.info/sct", "code": "765473008", "side": "right" },
        { "system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "Z90.5", "side": "from-body-site" }
      ],
      "edits": {
        "left": [{ "op": "hide", "node": "UBERON_0004538" }],
        "right": [{ "op": "hide", "node": "UBERON_0004539" }]
      }
    }
  ]
}
```

- **facts:** codes that trigger the rule. A fact matches on `system` and `code`
  only (no subsumption in v1, so every code is listed on purpose).
- **side:**
  - `left` or `right` when the code itself carries the side;
  - `from-body-site` when it must come from the resource's `bodySite`
    (SNOMED CT 18639004 "Left kidney structure", 9846003 "Right kidney
    structure");
  - `none` for unpaired structures.

  With `from-body-site` and no usable `bodySite`, the rule does not apply and
  the viewer asks for the side.
- **edits:** keyed by side, or by `any` for unpaired structures. Nodes are the
  manifest's node names, which are the UBERON or FMA IDs.

### Edit vocabulary (v1)

| Op | Effect | Example |
| --- | --- | --- |
| `hide` | the node is not drawn | a removed kidney |
| `mark` | the node stays, with a label and a hatched surface | partial pancreatectomy |
| `mirror` | a layer is mirrored left to right | situs inversus (organs layer only) |
| `add` | a stock part from the asset library is attached to an ANNY bone | hip implant |

`mark` exists so that a partial removal is shown as a fact, without inventing
geometry the record does not describe. `add` uses only library parts under
CC0 or CC BY. The library does not exist yet; until it does, `add` rules fall
back to `mark`.

### Order and conflicts

The viewer applies edits in a fixed order, `mirror`, then `hide`, then `add`,
then `mark`, so the result does not depend on the order of the facts. Applying
the same edit twice changes nothing.

If verified facts contradict each other, every edit on the affected node is
dropped and the viewer flags it for the clinician. An example is a verified
left nephrectomy alongside a verified left kidney finding dated after it.

## First rules

The codes below were looked up on 9 Nov 2026: SNOMED CT on tx.fhir.org and
ICD-10-CM on the NLM Clinical Tables service.

| Fact | Codes | Edit |
| --- | --- | --- |
| Kidney removed | SNOMED 175905003 Total nephrectomy (side from body site); 765472003 Excision of left kidney; 765473008 Excision of right kidney; ICD-10-CM Z90.5 Acquired absence of kidney (side from body site) | `hide` UBERON_0004538 (left) or UBERON_0004539 (right) |
| Kidney never formed | SNOMED 762907005 Left renal agenesis; 762908000 Right renal agenesis; ICD-10-CM Q60.0 Renal agenesis, unilateral (side from body site) | `hide` that kidney |
| One working kidney | SNOMED 880055007 Single functional kidney | none: the kidney is still there; shown as a note only |
| Spleen removed or absent | SNOMED 234319005 Splenectomy; ICD-10-CM Z90.81 Acquired absence of spleen, Q89.01 Asplenia (congenital) | `hide` UBERON_0002106 |
| Gallbladder removed | SNOMED 38102005 Cholecystectomy | `hide` UBERON_0002110 |
| Pancreas removed | ICD-10-CM Z90.410 Acquired total absence of pancreas | `hide` UBERON_0001264 |
| Part of pancreas removed | ICD-10-CM Z90.411 Acquired partial absence of pancreas | `mark` UBERON_0001264 |
| Situs inversus | SNOMED 43876007 Situs inversus viscerum; ICD-10-CM Q89.3 Situs inversus | `mirror` the organs layer |
| Hip replaced | SNOMED 52734007 Total replacement of hip (side from body site); ICD-10-CM Z96.641 right, Z96.642 left, Z96.643 bilateral | `add` a hip implant at upperleg01 (falls back to `mark` FMA_24474 or FMA_24475) |
| Knee replaced | SNOMED 609588000 Total knee replacement; ICD-10-CM Z96.651 right, Z96.652 left | none yet: the reference bodies have no knee |

ICD-10-CM Z90.49, "acquired absence of other specified parts of digestive
tract", is deliberately left out: it names no organ, so it can't select a
node. SNOMED 880055007, "single functional kidney", never hides a kidney,
because the second one may still be there.

## Privacy

- The rule map, the models and the asset library are public and the same for
  everyone. Fetching them reveals nothing about a person.
- The person's `Condition`, `Procedure` and `Provenance` resources are read on
  their device and are never sent anywhere to compute an edit.
- The viewer keeps no log of which rules fired, beyond what the person chooses
  to save locally.

## Licences

- **ICD-10-CM** is published by the US CDC's National Center for Health
  Statistics for free use.
- **SNOMED CT** needs a licence to use in a system. It is free in SNOMED
  International member countries, Switzerland among them, and an implementer
  must hold one before shipping SNOMED-coded rules. The rule map stores the
  codes as identifiers only, with no descriptions copied from SNOMED CT beyond
  the short names in this document.

## Open questions

1. Should a resolved `Condition` with no matching `Procedure` (for example
   Z90.5 recorded once, years ago) count, or does an absence need its
   procedure too?
2. Which codes for the side are accepted in `bodySite`: only the SNOMED CT
   "structure" concepts above, or also laterality qualifiers?
3. Where do stock implant parts come from (CC0 or CC BY), and who reviews
   them?
4. Should the clinician sign-off be per fact, or can one `Provenance` cover a
   reviewed list?
