/**
 * Writes the viewer's synthetic test body: tests/fixtures/viewer/body.glb and
 * body.manifest.json. Ten simple shapes stand in for structures (a breastbone
 * plate, three vertebrae, four ribs, the rib cartilage and a heart), named and
 * described as mesh.py names them, in the body frame (metres, +Y up, +Z
 * forward, +X the body's left). Nothing here comes from a scan. The output is
 * byte for byte the same on every run, and fixture.test.js checks that the
 * committed files are what this script writes.
 *
 *   node viewer/test/make-fixture.mjs
 */

import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const FIXTURE_DIR = join(HERE, "..", "..", "tests", "fixtures", "viewer");

const BONE = [0.86, 0.82, 0.7];
const CARTILAGE = [0.78, 0.84, 0.86];
const HEART = [0.42, 0.14, 0.13];
const LICENCE = "CC-BY-4.0";

const norm = (v) => {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
};

/** A closed ellipsoid: {positions, normals, indices}. */
function ellipsoid(centre, radii, rings = 10, segments = 16) {
  const positions = [];
  const normals = [];
  const indices = [];
  for (let i = 0; i <= rings; i++) {
    const phi = (Math.PI * i) / rings;
    for (let j = 0; j <= segments; j++) {
      const theta = (2 * Math.PI * j) / segments;
      const unit = [Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)];
      positions.push(centre[0] + radii[0] * unit[0], centre[1] + radii[1] * unit[1], centre[2] + radii[2] * unit[2]);
      normals.push(...norm([unit[0] / radii[0], unit[1] / radii[1], unit[2] / radii[2]]));
    }
  }
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j;
      const b = a + segments + 1;
      // Counter-clockwise seen from outside, so the faces look outward.
      indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  return { positions, normals, indices };
}

/** A closed tube of the given radius along a polyline given by a function of s in [0, 1]. */
function tube(path, radius, steps = 24, ring = 8) {
  const positions = [];
  const normals = [];
  const indices = [];
  for (let i = 0; i <= steps; i++) {
    const s = i / steps;
    const p = path(s);
    const ahead = path(Math.min(1, s + 1e-3));
    const behind = path(Math.max(0, s - 1e-3));
    const t = norm([ahead[0] - behind[0], ahead[1] - behind[1], ahead[2] - behind[2]]);
    // A side vector that is never parallel to the tangent for these paths.
    const side = norm([t[1] * 0 - t[2] * 1, t[2] * 0 - t[0] * 0, t[0] * 1 - t[1] * 0]); // t x (0,1,0)
    const up = norm([t[1] * side[2] - t[2] * side[1], t[2] * side[0] - t[0] * side[2], t[0] * side[1] - t[1] * side[0]]);
    // The tube tapers to a point at both ends so the surface is closed.
    const r = radius * Math.sin(Math.PI * Math.min(1, Math.max(0, s)) * 0.9999 + 1e-4) ** 0.35;
    for (let j = 0; j <= ring; j++) {
      const a = (2 * Math.PI * j) / ring;
      const n = norm([
        side[0] * Math.cos(a) + up[0] * Math.sin(a),
        side[1] * Math.cos(a) + up[1] * Math.sin(a),
        side[2] * Math.cos(a) + up[2] * Math.sin(a),
      ]);
      positions.push(p[0] + r * n[0], p[1] + r * n[1], p[2] + r * n[2]);
      normals.push(...n);
    }
  }
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < ring; j++) {
      const a = i * (ring + 1) + j;
      const b = a + ring + 1;
      // Counter-clockwise seen from outside, so the faces look outward.
      indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  return { positions, normals, indices };
}

function merge(parts) {
  const out = { positions: [], normals: [], indices: [] };
  for (const part of parts) {
    const base = out.positions.length / 3;
    out.positions.push(...part.positions);
    out.normals.push(...part.normals);
    out.indices.push(...part.indices.map((i) => i + base));
  }
  return out;
}

/** A rib: a half ring from the spine (back, -Z) round to the front, sloping down. */
function rib(side, y0, r) {
  const sign = side === "left" ? 1 : -1;
  return tube((s) => {
    const phi = Math.PI * s;
    return [sign * r * Math.sin(phi), y0 - 0.025 * s, -0.06 - r * Math.cos(phi) + r * 0.55];
  }, 0.006);
}

/** The structures: [{id, term, label, system, colour, mesh, modelled, pieces}]. */
export function structures() {
  const list = [];
  list.push({
    id: "FMA_7485", term: "FMA:7485", label: "sternum", colour: BONE,
    mesh: ellipsoid([0, 0.1, 0.065], [0.022, 0.085, 0.007], 12, 16),
  });
  const vertebrae = [
    ["FMA_9165", "first thoracic vertebra", 0.18],
    ["FMA_9187", "second thoracic vertebra", 0.15],
    ["FMA_9209", "third thoracic vertebra", 0.12],
  ];
  for (const [id, label, y] of vertebrae) {
    list.push({
      id, term: id.replace("_", ":"), label, colour: BONE,
      mesh: merge([ellipsoid([0, y, -0.06], [0.02, 0.016, 0.018]), ellipsoid([0, y - 0.004, -0.085], [0.006, 0.01, 0.012], 6, 8)]),
    });
  }
  const ribs = [
    ["FMA_7987", "left first rib", "left", 0.175, 0.07],
    ["FMA_8012", "left second rib", "left", 0.145, 0.085],
    ["FMA_7857", "right first rib", "right", 0.175, 0.07],
    ["FMA_7882", "right second rib", "right", 0.145, 0.085],
  ];
  for (const [id, label, side, y, r] of ribs) {
    list.push({ id, term: id.replace("_", ":"), label, colour: BONE, mesh: rib(side, y, r) });
  }
  // Cartilage: short bars from each rib's front end to the breastbone's edge.
  const bars = [];
  for (const [, , side, y, r] of ribs) {
    const sign = side === "left" ? 1 : -1;
    const end = [sign * r * Math.sin(Math.PI), y - 0.025, -0.06 + r + r * 0.55];
    bars.push(tube((s) => [end[0] * (1 - s) + sign * 0.02 * s, end[1] - 0.01 * s, end[2] * (1 - s) + 0.065 * s], 0.005, 8, 6));
  }
  list.push({
    id: "FMA_85213", term: "FMA:85213", label: "set of costal cartilages", colour: CARTILAGE,
    mesh: merge(bars), modelled: true, pieces: "several",
  });
  list.push({
    id: "UBERON_0000948", term: "UBERON:0000948", label: "heart", system: "cardiovascular", colour: HEART,
    mesh: ellipsoid([0.015, 0.1, 0.01], [0.04, 0.045, 0.035], 12, 18),
  });
  return list.map((s) => ({ system: "skeletal", modelled: false, pieces: "one", ...s }));
}

const pad4 = (n) => (4 - (n % 4)) % 4;

/** The GLB bytes for the structures: one node, one mesh and one material each. */
export function buildGlb(list) {
  const chunks = [];
  const bufferViews = [];
  const accessors = [];
  const meshes = [];
  const materials = [];
  const nodes = [];
  let offset = 0;
  const addView = (bytes, target) => {
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length, target });
    chunks.push(bytes);
    const padding = pad4(bytes.length);
    if (padding) chunks.push(Buffer.alloc(padding));
    offset += bytes.length + padding;
    return bufferViews.length - 1;
  };
  list.forEach((s, i) => {
    const pos = Float32Array.from(s.mesh.positions);
    const nor = Float32Array.from(s.mesh.normals);
    const count = pos.length / 3;
    const idx = count < 65535 ? Uint16Array.from(s.mesh.indices) : Uint32Array.from(s.mesh.indices);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < count; v++) {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], pos[v * 3 + k]);
        max[k] = Math.max(max[k], pos[v * 3 + k]);
      }
    }
    const posView = addView(Buffer.from(pos.buffer), 34962);
    const norView = addView(Buffer.from(nor.buffer), 34962);
    const idxView = addView(Buffer.from(idx.buffer), 34963);
    accessors.push({ bufferView: posView, componentType: 5126, count, type: "VEC3", min, max });
    accessors.push({ bufferView: norView, componentType: 5126, count, type: "VEC3" });
    accessors.push({ bufferView: idxView, componentType: count < 65535 ? 5123 : 5125, count: idx.length, type: "SCALAR" });
    materials.push({
      name: s.label,
      pbrMetallicRoughness: { baseColorFactor: [...s.colour, 1], metallicFactor: 0, roughnessFactor: 0.6 },
    });
    meshes.push({
      name: s.id,
      primitives: [{ attributes: { POSITION: i * 3, NORMAL: i * 3 + 1 }, indices: i * 3 + 2, material: i }],
    });
    nodes.push({ name: s.id, mesh: i, extras: { term: s.term, name: s.label, system: s.system, licence: LICENCE } });
  });
  const bin = Buffer.concat(chunks);
  const json = {
    asset: { version: "2.0", generator: "open-anatomy viewer fixture", copyright: "Synthetic test shapes, not from any scan" },
    scene: 0,
    scenes: [{ name: "Scene", nodes: nodes.map((_, i) => i) }],
    nodes,
    materials,
    meshes,
    accessors,
    bufferViews,
    buffers: [{ byteLength: bin.length }],
  };
  let jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  jsonBytes = Buffer.concat([jsonBytes, Buffer.alloc(pad4(jsonBytes.length), 0x20)]);
  const header = Buffer.alloc(12);
  header.write("glTF", 0, "ascii");
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + jsonBytes.length + 8 + bin.length, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonBytes.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(bin.length, 0);
  binHeader.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, jsonHeader, jsonBytes, binHeader, bin]);
}

/** The manifest in the shape the current skeleton build writes (no contract fields). */
export function buildManifest(list, glb) {
  return {
    case: "synthetic",
    set: "viewer-fixture",
    file: "body.glb",
    sha256: createHash("sha256").update(glb).digest("hex"),
    attribution: "Synthetic shapes written by viewer/test/make-fixture.mjs. No scan data.",
    units: "metres",
    axes: "+Y superior, +Z anterior, +X the body's left",
    nodes: list.map((s) => ({
      name: s.id,
      term: s.term,
      label: s.label,
      system: s.system,
      licence: LICENCE,
      triangles: s.mesh.indices.length / 3,
      volume_ml: null,
      modelled: s.modelled,
      pieces: s.pieces,
    })),
    missing: [],
    fragments: [],
    dropped: [],
  };
}

export function buildFixture() {
  const list = structures();
  const glb = buildGlb(list);
  const manifest = buildManifest(list, glb);
  return { glb, manifest: `${JSON.stringify(manifest, null, 2)}\n` };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const { glb, manifest } = buildFixture();
  mkdirSync(FIXTURE_DIR, { recursive: true });
  writeFileSync(join(FIXTURE_DIR, "body.glb"), glb);
  writeFileSync(join(FIXTURE_DIR, "body.manifest.json"), manifest);
  console.log(`wrote ${join(FIXTURE_DIR, "body.glb")} (${glb.length} bytes) and body.manifest.json`);
}
