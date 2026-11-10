/**
 * Reads a body's manifest into the records the page uses, and fills in what
 * the manifest does not carry yet. Pure: no fetch, no three.js.
 *
 * The contract (docs/explode-view.md) gives every node a plain-language name,
 * a one-line "about", a centre, bounds and an explode direction and distance.
 * A manifest from the current build has only the term, label, system, licence,
 * volume and the modelled flag, so normaliseManifest() derives a plain name and
 * an about line from the label, attachGeometry() takes the centre and bounds
 * from the loaded mesh, and explode.js derives the explode fields. Every
 * derived value is listed in the node's `derived` array, so the page can say
 * which words came from the manifest and which from the viewer.
 */

import { isBounds, isVec3, boxCentre } from "./explode.js";

/** Manifest system names mapped onto the contract's closed list. */
export const SYSTEM_ALIASES = {
  skeleton: "skeleton",
  skeletal: "skeleton",
  cardiovascular: "cardiovascular",
  circulatory: "cardiovascular",
  respiratory: "respiratory",
  digestive: "digestive",
  urinary: "urinary",
  nervous: "nervous",
  endocrine: "endocrine",
  muscular: "muscular",
  lymphoid: "lymphoid",
  lymphatic: "lymphoid",
};

export const ID_PATTERN = /^[A-Za-z]+_\d+$/;

/** Node id for a manifest entry: "FMA_7485", from the id, the node name or the term. */
export function nodeId(entry) {
  for (const key of ["id", "node"]) {
    if (typeof entry[key] === "string" && ID_PATTERN.test(entry[key])) return entry[key];
  }
  if (typeof entry.name === "string" && ID_PATTERN.test(entry.name)) return entry.name;
  if (typeof entry.term === "string" && /^[A-Za-z]+:\d+$/.test(entry.term)) return entry.term.replace(":", "_");
  return null;
}

/** Anatomical name for an entry: the label, or a name that is not an id. */
export function nodeLabel(entry) {
  if (typeof entry.label === "string" && entry.label.trim()) return entry.label.trim();
  if (typeof entry.name === "string" && !ID_PATTERN.test(entry.name) && entry.name.trim()) return entry.name.trim();
  return null;
}

const ORDINALS = {
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8,
  ninth: 9, tenth: 10, eleventh: 11, twelfth: 12,
};

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

const PLAIN_WORDS = [
  [/^sternum$/, "Breastbone"],
  [/^(set of )?costal cartilages?$/, "Rib cartilage"],
  [/^sacrum$/, "Base of the spine"],
  [/^coccyx$/, "Tailbone"],
  [/^atlas$/, "Top neck bone"],
  [/^axis$/, "Second neck bone"],
  [/^clavicle$/, "Collarbone"],
  [/^scapula$/, "Shoulder blade"],
  [/^humerus$/, "Upper arm bone"],
  [/^hip bone$/, "Hip bone"],
  [/^femur$/, "Thigh bone"],
  [/^kidney$/, "Kidney"],
  [/^adrenal gland$/, "Adrenal gland"],
  [/^lung$/, "Lung"],
  [/^heart$/, "Heart"],
  [/^aorta$/, "Main artery (aorta)"],
  [/^inferior vena cava$/, "Main vein from the lower body"],
  [/^superior vena cava$/, "Main vein from the upper body"],
  [/^portal vein$/, "Vein from the gut to the liver"],
  [/^trachea$/, "Windpipe"],
  [/^o?esophagus$/, "Food pipe"],
  [/^urinary bladder$/, "Bladder"],
  [/^small intestine$/, "Small bowel"],
  [/^colon$/, "Large bowel"],
  [/^duodenum$/, "First part of the small bowel"],
  [/^thyroid( gland)?$/, "Thyroid"],
  [/^spinal cord$/, "Spinal cord"],
  [/^brain$/, "Brain"],
  [/^(upper|middle|lower) lobe of (left|right) lung$/, (m) => `${cap(m[2])} lung, ${m[1]} part`],
];

/**
 * A plain-language name for an anatomical label: "left first rib" -> "Left rib 1",
 * "first thoracic vertebra" -> "Upper back bone 1". No Latin and no letter codes;
 * the anatomical name stays in `label` for the clinician view and the search.
 */
export function plainName(label) {
  const text = String(label ?? "").trim().toLowerCase();
  if (!text) return "";
  let m = /^(left|right) (\w+) rib$/.exec(text);
  if (m && ORDINALS[m[2]]) return `${cap(m[1])} rib ${ORDINALS[m[2]]}`;
  m = /^(\w+) (cervical|thoracic|lumbar) vertebra$/.exec(text);
  if (m && ORDINALS[m[1]]) {
    const region = { cervical: "Neck bone", thoracic: "Upper back bone", lumbar: "Lower back bone" };
    return `${region[m[2]]} ${ORDINALS[m[1]]}`;
  }
  m = /^(left|right) (.+)$/.exec(text);
  const side = m ? cap(m[1]) + " " : "";
  const base = m ? m[2] : text;
  for (const [pattern, replacement] of PLAIN_WORDS) {
    const hit = pattern.exec(base);
    if (hit) {
      const word = typeof replacement === "function" ? replacement(hit) : replacement;
      return side ? side + word[0].toLowerCase() + word.slice(1) : word;
    }
  }
  return cap(side.toLowerCase() + base);
}

/**
 * Extra words the search should find a structure by, though the main view does
 * not show them: "spine" and the short code for a vertebra ("T1"), "pelvis" for
 * the hip bones and the sacrum, "chest" for the ribs and the breastbone.
 */
export function searchWords(label) {
  const text = String(label ?? "").trim().toLowerCase();
  const m = /^(\w+) (cervical|thoracic|lumbar) vertebra$/.exec(text);
  if (m && ORDINALS[m[1]]) return `spine backbone ${m[2][0].toUpperCase()}${ORDINALS[m[1]]}`;
  if (text === "atlas") return "spine backbone C1";
  if (text === "axis") return "spine backbone C2";
  if (text === "sacrum") return "spine backbone pelvis";
  if (text === "coccyx") return "spine backbone";
  if (/hip bone$/.test(text)) return "pelvis";
  if (/(rib|sternum|costal cartilages?)$/.test(text)) return "chest";
  return "";
}

const ABOUT = [
  [/vertebra$/, "One of the bones of the spine. It carries the body's weight and protects the spinal cord."],
  [/^(atlas|axis)$/, "One of the two topmost neck bones, which let the head nod and turn."],
  [/rib$/, "A curved bone of the chest wall. The ribs shield the heart and lungs and swing with each breath."],
  [/^sternum$/, "The flat bone at the front of the chest where the rib cartilages meet."],
  [/costal cartilage/, "Flexible cartilage that joins the ribs to the breastbone and lets the chest expand."],
  [/^sacrum$/, "The wedge of fused vertebrae that joins the spine to the pelvis."],
  [/clavicle$/, "The strut between the breastbone and the shoulder that holds the arm away from the chest."],
  [/scapula$/, "The flat triangular bone on the back of the shoulder; the arm's socket sits on it."],
  [/humerus$/, "The long bone of the upper arm, from the shoulder to the elbow."],
  [/hip bone$/, "One half of the pelvis. It links the spine to the legs and carries the hip socket."],
  [/femur$/, "The thigh bone, the longest bone in the body."],
  [/^heart$/, "The muscular pump that drives blood around the body."],
  [/^aorta$/, "The main artery, carrying blood from the heart to the rest of the body."],
  [/vena cava$/, "A great vein that returns blood to the heart."],
  [/^liver$/, "The large organ under the right ribs that processes nutrients and clears the blood."],
  [/kidney$/, "Filters the blood and makes urine."],
  [/lung/, "Takes oxygen from the air into the blood and lets carbon dioxide out."],
  [/^trachea$/, "The tube that carries air from the throat to the lungs."],
  [/^spleen$/, "Filters the blood and stores immune cells."],
];

const SYSTEM_ABOUT = {
  skeleton: "Part of the skeleton.",
  cardiovascular: "Part of the heart and vessels.",
  respiratory: "Part of the breathing system.",
  digestive: "Part of the digestive system.",
  urinary: "Part of the urinary system.",
  nervous: "Part of the nervous system.",
  endocrine: "A hormone gland.",
  muscular: "A muscle.",
  lymphoid: "Part of the immune and lymphatic system.",
};

/** One plain line saying what a structure is, from its label and system. */
export function aboutLine(label, system) {
  const text = String(label ?? "").trim().toLowerCase();
  for (const [pattern, line] of ABOUT) {
    if (pattern.test(text)) return line;
  }
  return SYSTEM_ABOUT[system] ?? "A structure of the body.";
}

/** "segmented from CT" or "modelled", for the panel. */
export function sourceText(node) {
  return node.modelled ? "modelled" : "segmented from CT";
}

/** Problems with a manifest: errors stop the viewer, warnings are shown in the console. */
export function validateManifest(manifest) {
  const errors = [];
  const warnings = [];
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return { ok: false, errors: ["the manifest is not a JSON object"], warnings };
  }
  if (!Array.isArray(manifest.nodes)) {
    errors.push("the manifest has no `nodes` array");
    return { ok: false, errors, warnings };
  }
  if (manifest.nodes.length === 0) errors.push("the manifest lists no structures");
  if (typeof manifest.file !== "string") warnings.push("the manifest names no GLB `file`");
  const seen = new Set();
  manifest.nodes.forEach((entry, i) => {
    const where = `node ${i}`;
    if (!entry || typeof entry !== "object") {
      errors.push(`${where} is not an object`);
      return;
    }
    const id = nodeId(entry);
    if (!id) errors.push(`${where} has no usable id (name, node or term)`);
    else if (seen.has(id)) errors.push(`${where}: duplicate id ${id}`);
    else seen.add(id);
    if (typeof entry.term === "string" && id && entry.term.replace(":", "_") !== id) {
      errors.push(`${where}: term ${entry.term} does not match id ${id}`);
    }
    if (!nodeLabel(entry)) warnings.push(`${where} (${id ?? "?"}) has no label`);
    if (typeof entry.system !== "string") errors.push(`${where} (${id ?? "?"}) has no system`);
    else if (!(entry.system in SYSTEM_ALIASES)) warnings.push(`${where} (${id}): unknown system "${entry.system}"`);
    const centre = entry.centre ?? entry.centreM ?? entry.centre_m;
    if (centre !== undefined && !isVec3(centre)) errors.push(`${where} (${id}): centre is not three numbers`);
    const bounds = entry.bounds ?? entry.bbox;
    if (bounds !== undefined && !isBounds(bounds)) errors.push(`${where} (${id}): bounds are not [[min],[max]]`);
    if (entry.explode !== undefined) {
      const e = entry.explode;
      if (!e || !isVec3(e.direction) || !Number.isFinite(e.distance) || e.distance < 0) {
        errors.push(`${where} (${id}): explode needs a direction of three numbers and a distance >= 0`);
      } else if (Math.abs(Math.hypot(...e.direction) - 1) > 1e-3) {
        warnings.push(`${where} (${id}): explode direction is not unit length`);
      }
    }
  });
  return { ok: errors.length === 0, errors, warnings };
}

/**
 * The page's records from a manifest: id, term, label, plain, about, keywords, system,
 * layer, licence, modelled, pieces, volume_ml, triangles, centre, bounds and
 * explode (the last three null until the manifest or the mesh supplies them),
 * with `derived` naming every field the viewer filled in. Throws on errors.
 */
export function normaliseManifest(manifest) {
  const check = validateManifest(manifest);
  if (!check.ok) throw new Error(`manifest: ${check.errors.join("; ")}`);
  const nodes = manifest.nodes.map((entry) => {
    const id = nodeId(entry);
    const term = typeof entry.term === "string" ? entry.term : id.replace("_", ":");
    const label = nodeLabel(entry) ?? term;
    const derived = [];
    const system = SYSTEM_ALIASES[entry.system] ?? entry.system;
    let plain = typeof entry.plain === "string" && entry.plain.trim() ? entry.plain.trim() : null;
    if (!plain) {
      plain = plainName(label);
      derived.push("plain");
    }
    let about = typeof entry.about === "string" && entry.about.trim() ? entry.about.trim() : null;
    if (!about) {
      about = aboutLine(label, system);
      derived.push("about");
    }
    const centre = entry.centre ?? entry.centreM ?? entry.centre_m ?? null;
    const bounds = entry.bounds ?? entry.bbox ?? null;
    const volume = entry.volume_ml ?? entry.volumeMl ?? null;
    return {
      id,
      term,
      label,
      plain,
      about,
      keywords: searchWords(label),
      system,
      layer: typeof entry.layer === "string" ? entry.layer : null,
      licence: typeof entry.licence === "string" ? entry.licence : null,
      modelled: entry.modelled === true,
      pieces: typeof entry.pieces === "string" ? entry.pieces : "one",
      volume_ml: Number.isFinite(volume) ? volume : null,
      triangles: Number.isFinite(entry.triangles) ? entry.triangles : null,
      centre,
      bounds,
      explode: entry.explode ?? null,
      derived,
    };
  });
  const meta = {
    contract: manifest.contract ?? null,
    case: manifest.case ?? null,
    set: manifest.set ?? null,
    file: manifest.file ?? null,
    sha256: manifest.sha256 ?? null,
    attribution: manifest.attribution ?? null,
    units: manifest.units ?? "metres",
    axes: manifest.axes ?? null,
    bodyCentre: [manifest.body_centre_m, manifest.bodyCentreM, manifest.bodyCentre].find(isVec3) ?? null,
  };
  return { meta, nodes, warnings: check.warnings };
}

/**
 * Give every node a centre and bounds. `boxes` maps an id to {min, max} in
 * the body frame, measured from the loaded mesh. The manifest's own values win;
 * the mesh fills the gaps (marked derived); a node with neither is flagged
 * `geometry: false` and left out of the scene.
 */
export function attachGeometry(nodes, boxes) {
  return nodes.map((n) => {
    const box = boxes.get ? boxes.get(n.id) : boxes[n.id];
    const measured = box && isVec3(box.min) && isVec3(box.max) ? [box.min, box.max] : null;
    const derived = [...n.derived];
    let bounds = isBounds(n.bounds) ? n.bounds : null;
    if (!bounds && measured) {
      bounds = measured;
      derived.push("bounds");
    }
    let centre = isVec3(n.centre) ? n.centre : null;
    if (!centre && bounds) {
      centre = boxCentre(bounds);
      derived.push("centre");
    }
    return { ...n, centre, bounds, derived, geometry: Boolean(bounds) };
  });
}
