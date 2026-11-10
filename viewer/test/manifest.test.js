import { test } from "node:test";
import assert from "node:assert/strict";

import {
  aboutLine,
  attachGeometry,
  nodeId,
  nodeLabel,
  normaliseManifest,
  plainName,
  searchWords,
  sourceText,
  validateManifest,
} from "../manifest.js";

/** A manifest shaped like the current skeleton build's. */
function currentBuild() {
  return {
    case: "s0086",
    set: "skeleton",
    file: "s0086.skeleton.glb",
    sha256: "0".repeat(64),
    units: "metres",
    nodes: [
      { name: "FMA_7485", term: "FMA:7485", label: "sternum", system: "skeletal", licence: "CC-BY-4.0", volume_ml: 90.2, modelled: false, pieces: "one", triangles: 8880 },
      { name: "FMA_85213", term: "FMA:85213", label: "set of costal cartilages", system: "skeletal", licence: "CC-BY-4.0", volume_ml: 118.4, modelled: true, pieces: "several" },
      { name: "FMA_7987", term: "FMA:7987", label: "left first rib", system: "skeletal", licence: "CC-BY-4.0", volume_ml: 8.8, modelled: false, pieces: "one" },
    ],
  };
}

/** A manifest with the contract's fields. */
function contractBuild() {
  return {
    contract: 1,
    file: "body.glb",
    nodes: [
      {
        name: "liver", node: "UBERON_0002107", term: "UBERON:0002107", plain: "Liver", about: "Cleans the blood.",
        system: "digestive", layer: "organs", licence: "CC-BY-4.0", centreM: [-0.06, 1.18, 0.03],
        bbox: [[-0.1, 1.1, 0], [0, 1.26, 0.06]], volumeMl: 1497.2, pieces: "one", modelled: false,
        explode: { direction: [-0.82, -0.31, 0.48], distance: 0.13 },
      },
    ],
  };
}

test("validateManifest accepts the current build and the contract shape", () => {
  assert.deepEqual(validateManifest(currentBuild()), { ok: true, errors: [], warnings: [] });
  assert.deepEqual(validateManifest(contractBuild()), { ok: true, errors: [], warnings: [] });
  const skewed = contractBuild();
  skewed.nodes[0].explode.direction = [0, 2, 0];
  const check = validateManifest(skewed);
  assert.equal(check.ok, true);
  assert.ok(check.warnings.some((w) => w.includes("unit length")), "a direction far from unit length is only a warning");
});

test("validateManifest reports what stops the viewer", () => {
  assert.equal(validateManifest(null).ok, false);
  assert.equal(validateManifest([]).ok, false);
  assert.match(validateManifest({}).errors[0], /nodes/);
  assert.match(validateManifest({ nodes: [] }).errors[0], /no structures/);

  const dup = currentBuild();
  dup.nodes.push({ ...dup.nodes[0] });
  assert.match(validateManifest(dup).errors[0], /duplicate id FMA_7485/);

  const mismatch = currentBuild();
  mismatch.nodes[0].term = "FMA:1";
  assert.match(validateManifest(mismatch).errors[0], /does not match/);

  const noSystem = currentBuild();
  delete noSystem.nodes[1].system;
  assert.match(validateManifest(noSystem).errors[0], /no system/);

  const badExplode = currentBuild();
  badExplode.nodes[2].explode = { direction: [1, 0], distance: -1 };
  assert.match(validateManifest(badExplode).errors[0], /explode/);

  const badBounds = currentBuild();
  badBounds.nodes[2].bounds = [[0, 0, 0], [-1, 0, 0]];
  assert.match(validateManifest(badBounds).errors[0], /bounds/);

  const odd = currentBuild();
  odd.nodes[0].system = "plumbing";
  delete odd.file;
  const check = validateManifest(odd);
  assert.equal(check.ok, true);
  assert.ok(check.warnings.some((w) => w.includes("unknown system")));
  assert.ok(check.warnings.some((w) => w.includes("file")));
});

test("ids come from the node name, an explicit node field or the term", () => {
  assert.equal(nodeId({ name: "FMA_7485" }), "FMA_7485");
  assert.equal(nodeId({ name: "sternum", term: "FMA:7485" }), "FMA_7485");
  assert.equal(nodeId({ node: "UBERON_0002107", name: "liver" }), "UBERON_0002107");
  assert.equal(nodeId({ name: "liver" }), null);
  assert.equal(nodeLabel({ name: "FMA_7485", label: "sternum" }), "sternum");
  assert.equal(nodeLabel({ name: "sternum" }), "sternum");
  assert.equal(nodeLabel({ name: "FMA_7485" }), null);
});

test("plain names read as everyday English", () => {
  const cases = [
    ["left first rib", "Left rib 1"],
    ["right twelfth rib", "Right rib 12"],
    ["sixth cervical vertebra", "Neck bone 6"],
    ["first thoracic vertebra", "Upper back bone 1"],
    ["fifth lumbar vertebra", "Lower back bone 5"],
    ["atlas", "Top neck bone"],
    ["sacrum", "Base of the spine"],
    ["sternum", "Breastbone"],
    ["set of costal cartilages", "Rib cartilage"],
    ["left clavicle", "Left collarbone"],
    ["right scapula", "Right shoulder blade"],
    ["left humerus", "Left upper arm bone"],
    ["right hip bone", "Right hip bone"],
    ["left femur", "Left thigh bone"],
    ["left kidney", "Left kidney"],
    ["heart", "Heart"],
    ["aorta", "Main artery (aorta)"],
    ["inferior vena cava", "Main vein from the lower body"],
    ["upper lobe of left lung", "Left lung, upper part"],
    ["urinary bladder", "Bladder"],
    ["esophagus", "Food pipe"],
    ["liver", "Liver"],
    ["some new structure", "Some new structure"],
    ["", ""],
  ];
  for (const [label, plain] of cases) assert.equal(plainName(label), plain, label);
});

test("search words add the spine, vertebra codes, the pelvis and the chest", () => {
  assert.equal(searchWords("first thoracic vertebra"), "spine backbone T1");
  assert.equal(searchWords("fifth lumbar vertebra"), "spine backbone L5");
  assert.equal(searchWords("atlas"), "spine backbone C1");
  assert.match(searchWords("sacrum"), /spine.*pelvis/);
  assert.equal(searchWords("left hip bone"), "pelvis");
  assert.match(searchWords("right ninth rib"), /chest/);
  assert.match(searchWords("sternum"), /chest/);
  assert.equal(searchWords("heart"), "");
  assert.equal(normaliseManifest(currentBuild()).nodes[0].keywords, "chest");
});

test("about lines come from the label's kind, then the system", () => {
  assert.match(aboutLine("sixth cervical vertebra", "skeleton"), /spine/);
  assert.match(aboutLine("left first rib", "skeleton"), /chest wall/);
  assert.match(aboutLine("set of costal cartilages", "skeleton"), /cartilage/);
  assert.equal(aboutLine("odd bone", "skeleton"), "Part of the skeleton.");
  assert.equal(aboutLine("heart", "cardiovascular"), "The muscular pump that drives blood around the body.");
  assert.equal(aboutLine("gizmo", "plumbing"), "A structure of the body.");
});

test("normaliseManifest maps systems, fills plain names and about lines and marks them derived", () => {
  const { meta, nodes, warnings } = normaliseManifest(currentBuild());
  assert.equal(meta.case, "s0086");
  assert.equal(meta.file, "s0086.skeleton.glb");
  assert.deepEqual(warnings, []);
  assert.equal(nodes.length, 3);
  const sternum = nodes[0];
  assert.equal(sternum.id, "FMA_7485");
  assert.equal(sternum.term, "FMA:7485");
  assert.equal(sternum.label, "sternum");
  assert.equal(sternum.plain, "Breastbone");
  assert.equal(sternum.system, "skeleton");
  assert.equal(sternum.volume_ml, 90.2);
  assert.equal(sternum.triangles, 8880);
  assert.equal(sternum.modelled, false);
  assert.deepEqual(sternum.derived, ["plain", "about"]);
  assert.equal(sternum.centre, null);
  assert.equal(sternum.bounds, null);
  assert.equal(sternum.explode, null);
  assert.equal(sourceText(sternum), "segmented from CT");
  const cartilage = nodes[1];
  assert.equal(cartilage.modelled, true);
  assert.equal(cartilage.pieces, "several");
  assert.equal(sourceText(cartilage), "modelled");
});

test("normaliseManifest keeps every contract field and derives nothing", () => {
  const { meta, nodes } = normaliseManifest(contractBuild());
  assert.equal(meta.contract, 1);
  assert.equal(meta.bodyCentre, null);
  assert.deepEqual(normaliseManifest({ ...contractBuild(), body_centre_m: [0, 1.2, 0.01] }).meta.bodyCentre, [0, 1.2, 0.01]);
  assert.deepEqual(normaliseManifest({ ...contractBuild(), bodyCentreM: [0, 1.21, 0.01] }).meta.bodyCentre, [0, 1.21, 0.01]);
  assert.equal(normaliseManifest({ ...contractBuild(), body_centre_m: [0, "x", 0] }).meta.bodyCentre, null);
  const liver = nodes[0];
  assert.equal(liver.id, "UBERON_0002107");
  assert.equal(liver.label, "liver");
  assert.equal(liver.plain, "Liver");
  assert.equal(liver.about, "Cleans the blood.");
  assert.equal(liver.layer, "organs");
  assert.deepEqual(liver.centre, [-0.06, 1.18, 0.03]);
  assert.deepEqual(liver.bounds, [[-0.1, 1.1, 0], [0, 1.26, 0.06]]);
  assert.equal(liver.volume_ml, 1497.2);
  assert.deepEqual(liver.explode, { direction: [-0.82, -0.31, 0.48], distance: 0.13 });
  assert.deepEqual(liver.derived, []);
});

test("normaliseManifest throws on an invalid manifest", () => {
  assert.throws(() => normaliseManifest({ nodes: [{ name: "x" }] }), /no usable id/);
});

test("attachGeometry fills centre and bounds from the mesh and flags nodes without any", () => {
  const { nodes } = normaliseManifest(currentBuild());
  const boxes = new Map([
    ["FMA_7485", { min: [-0.03, 0.08, 0.04], max: [0.03, 0.26, 0.11] }],
    ["FMA_85213", { min: [-0.12, 0.05, 0.03], max: [0.12, 0.3, 0.12] }],
  ]);
  const out = attachGeometry(nodes, boxes);
  const sternum = out[0];
  assert.deepEqual(sternum.bounds, [[-0.03, 0.08, 0.04], [0.03, 0.26, 0.11]]);
  assert.deepEqual(sternum.centre.map((v) => Math.round(v * 1000) / 1000), [0, 0.17, 0.075]);
  assert.deepEqual(sternum.derived, ["plain", "about", "bounds", "centre"]);
  assert.equal(sternum.geometry, true);
  assert.equal(out[2].geometry, false, "no mesh for the rib");
  assert.equal(out[2].bounds, null);

  // The manifest's centre wins over the measured box centre; the box still fills the bounds.
  const withCentre = attachGeometry([{ ...nodes[0], centre: [0.01, 0.2, 0.08] }], { FMA_7485: boxes.get("FMA_7485") });
  assert.deepEqual(withCentre[0].centre, [0.01, 0.2, 0.08]);
  assert.deepEqual(withCentre[0].derived, ["plain", "about", "bounds"]);
});
