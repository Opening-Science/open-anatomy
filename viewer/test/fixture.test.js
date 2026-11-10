import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { FIXTURE_DIR, buildFixture, structures } from "./make-fixture.mjs";
import { boundsAt, layout, overlapping } from "../explode.js";
import { attachGeometry, normaliseManifest } from "../manifest.js";

/** The JSON chunk of a GLB. */
function glbJson(buf) {
  assert.equal(buf.toString("ascii", 0, 4), "glTF");
  assert.equal(buf.readUInt32LE(4), 2);
  assert.equal(buf.readUInt32LE(8), buf.length, "the header's length is the file's");
  const jsonLength = buf.readUInt32LE(12);
  assert.equal(buf.readUInt32LE(16), 0x4e4f534a, "first chunk is JSON");
  const json = JSON.parse(buf.toString("utf8", 20, 20 + jsonLength));
  const binLength = buf.readUInt32LE(20 + jsonLength);
  assert.equal(buf.readUInt32LE(24 + jsonLength), 0x004e4942, "second chunk is BIN");
  assert.equal(28 + jsonLength + binLength, buf.length);
  assert.equal(json.buffers[0].byteLength, binLength);
  return json;
}

const committedGlb = () => readFileSync(join(FIXTURE_DIR, "body.glb"));
const committedManifest = () => JSON.parse(readFileSync(join(FIXTURE_DIR, "body.manifest.json"), "utf8"));

test("the committed fixture is what make-fixture.mjs writes, and small", () => {
  const { glb, manifest } = buildFixture();
  assert.ok(committedGlb().equals(glb), "tests/fixtures/viewer/body.glb is stale: run node viewer/test/make-fixture.mjs");
  assert.equal(readFileSync(join(FIXTURE_DIR, "body.manifest.json"), "utf8"), manifest);
  assert.ok(glb.length < 200 * 1024, `the GLB is ${glb.length} bytes`);
});

test("the fixture GLB is well formed and its nodes match the manifest", () => {
  const json = glbJson(committedGlb());
  const manifest = committedManifest();
  assert.deepEqual(
    json.nodes.map((n) => n.name),
    manifest.nodes.map((n) => n.name),
  );
  assert.equal(json.scenes[0].nodes.length, json.nodes.length);
  for (const view of json.bufferViews) assert.equal(view.byteOffset % 4, 0, "buffer views are 4-byte aligned");
  for (const node of json.nodes) {
    assert.equal(node.extras.term.replace(":", "_"), node.name);
    assert.ok(typeof node.extras.system === "string");
    const position = json.accessors[json.meshes[node.mesh].primitives[0].attributes.POSITION];
    assert.ok(position.min.every(Number.isFinite) && position.max.every(Number.isFinite));
    assert.ok(position.min.every((lo, k) => lo < position.max[k]));
  }
  assert.ok(json.nodes.some((n) => n.extras.system === "cardiovascular"), "two systems, so the groups and the block offsets are exercised");
});

test("the fixture lays out with no overlapping bounds at t = 1", () => {
  const json = glbJson(committedGlb());
  const boxes = new Map(
    json.nodes.map((n) => {
      const position = json.accessors[json.meshes[n.mesh].primitives[0].attributes.POSITION];
      return [n.name, { min: position.min, max: position.max }];
    }),
  );
  const nodes = attachGeometry(normaliseManifest(committedManifest()).nodes, boxes);
  assert.ok(nodes.every((n) => n.geometry));
  const laidOut = layout(nodes);
  assert.ok(overlapping(boundsAt(laidOut, 0)).length > 0, "the assembled fixture overlaps, so the test means something");
  assert.deepEqual(overlapping(boundsAt(laidOut, 1)), []);
  assert.deepEqual(laidOut.unresolved, []);
  assert.equal(laidOut.derivedCount, nodes.length);
  assert.ok(laidOut.systemOffsets.get("cardiovascular").some((v) => v !== 0), "the heart's system moves as a block");
});

test("every fixture shape is closed side out: positive volume, normals along the faces", () => {
  for (const s of structures()) {
    const { positions: p, normals: n, indices } = s.mesh;
    const at = (list, i) => [list[i * 3], list[i * 3 + 1], list[i * 3 + 2]];
    let volume = 0;
    let agree = 0;
    for (let f = 0; f < indices.length; f += 3) {
      const [a, b, c] = [at(p, indices[f]), at(p, indices[f + 1]), at(p, indices[f + 2])];
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const face = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      volume += (a[0] * face[0] + a[1] * face[1] + a[2] * face[2]) / 6;
      const m = at(n, indices[f]);
      if (face[0] * m[0] + face[1] * m[1] + face[2] * m[2] >= 0) agree++;
    }
    assert.ok(volume > 0, `${s.label}: signed volume ${volume}`);
    assert.ok(agree / (indices.length / 3) > 0.95, `${s.label}: ${agree} of ${indices.length / 3} faces agree with their normals`);
  }
});
