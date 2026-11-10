import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ANTERIOR,
  MARGIN_M,
  bodyCentre,
  boundsAt,
  boxRadius,
  deriveExplode,
  firstClear,
  layout,
  normalise,
  offsetAt,
  overlapInterval,
  overlapping,
  resolveOverlaps,
  systemOffsets,
} from "../explode.js";

/** A box of the given half sizes around a centre. */
const box = (c, h) => [
  [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
  [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
];

const node = (id, centre, half = [0.02, 0.02, 0.02], extra = {}) => ({
  id,
  system: "skeleton",
  label: id,
  centre,
  bounds: box(centre, half),
  ...extra,
});

const close = (a, b, tol = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < tol, `${a} vs ${b} at ${i}`));
};

/** A small skeleton-like body whose boxes overlap heavily when assembled. */
function stackedBody() {
  const nodes = [];
  for (let i = 0; i < 6; i++) {
    // Vertebrae: boxes 4 cm tall every 2.5 cm, so neighbours overlap by 1.5 cm.
    nodes.push(node(`V${i}`, [0, i * 0.025, -0.05], [0.02, 0.02, 0.02], { volume_ml: 30 }));
  }
  // A front plate whose box covers the whole front.
  nodes.push(node("STERNUM", [0, 0.07, 0.05], [0.02, 0.07, 0.01], { volume_ml: 90, label: "sternum" }));
  for (let i = 0; i < 3; i++) {
    // Ribs: boxes 4 cm tall every 3 cm, so same-side neighbours overlap too.
    const y = 0.04 + i * 0.03;
    nodes.push(node(`RL${i}`, [0.08, y, 0], [0.08, 0.02, 0.06], { label: `left rib ${i}`, volume_ml: 20 }));
    nodes.push(node(`RR${i}`, [-0.08, y, 0], [0.08, 0.02, 0.06], { label: `right rib ${i}`, volume_ml: 20 }));
  }
  return nodes;
}

test("bodyCentre weights centres by volume and falls back to equal weights", () => {
  close(bodyCentre([node("a", [0, 0, 0], undefined, { volume_ml: 1 }), node("b", [4, 0, 0], undefined, { volume_ml: 3 })]), [3, 0, 0]);
  close(bodyCentre([node("a", [0, 0, 0]), node("b", [4, 2, 0])]), [2, 1, 0]);
  close(bodyCentre([]), [0, 0, 0]);
});

test("a structure off to one side moves straight away from the body centre", () => {
  const nodes = [node("mid", [0, 0, 0], undefined, { volume_ml: 1000 }), node("side", [0.2, 0, 0], [0.03, 0.04, 0.05])];
  const out = deriveExplode(nodes, { bodyCentre: [0, 0, 0] });
  const side = out.find((n) => n.id === "side");
  close(side.explode.direction, [1, 0, 0]);
  assert.ok(Math.abs(side.explode.distance - (boxRadius(side.bounds) + MARGIN_M)) < 1e-12);
  assert.equal(side.explode.derived, true);
});

test("a structure at the centre takes its system's mean direction, or +Z when the mean cancels", () => {
  const centred = node("c", [0.004, 0, 0]);
  const withMean = deriveExplode([centred, node("b", [0.1, 0, 0]), node("d", [0.1, 0.1, 0])], { bodyCentre: [0, 0, 0] });
  const expected = normalise([1 + Math.SQRT1_2, Math.SQRT1_2, 0]);
  close(withMean[0].explode.direction, expected, 1e-9);

  const cancelled = deriveExplode([centred, node("l", [0.1, 0, 0]), node("r", [-0.1, 0, 0])], { bodyCentre: [0, 0, 0] });
  close(cancelled[0].explode.direction, [...ANTERIOR]);

  const alone = deriveExplode([centred], { bodyCentre: [0, 0, 0] });
  close(alone[0].explode.direction, [...ANTERIOR]);
});

test("explode fields from the manifest are kept and only normalised", () => {
  const given = node("g", [0.3, 0, 0], undefined, { explode: { direction: [0, 2, 0], distance: 0.5 } });
  const out = deriveExplode([given, node("other", [0, 0.3, 0])]);
  close(out[0].explode.direction, [0, 1, 0]);
  assert.equal(out[0].explode.distance, 0.5);
  assert.equal(out[0].explode.derived, false);
  assert.equal(out[1].explode.derived, true);
});

test("left and right partners mirror each other and move the same distance", () => {
  const nodes = [
    node("L", [0.1, 0.05, 0.01], [0.02, 0.02, 0.02], { label: "left clavicle" }),
    node("R", [-0.1, 0.02, 0.0], [0.04, 0.03, 0.02], { label: "right clavicle" }),
    node("S", [0, 0, 0], undefined, { label: "sternum", volume_ml: 500 }),
  ];
  const out = deriveExplode(nodes, { bodyCentre: [0, 0, 0] });
  const left = out.find((n) => n.id === "L").explode;
  const right = out.find((n) => n.id === "R").explode;
  close([-left.direction[0], left.direction[1], left.direction[2]], right.direction);
  assert.equal(left.distance, right.distance);
  assert.ok(left.distance >= boxRadius(nodes[1].bounds) + MARGIN_M - 1e-12);
});

test("firstClear jumps over chained intervals in one pass", () => {
  const intervals = [
    [0.7, 0.8],
    [0.1, 0.3],
    [0.25, 0.5],
  ];
  assert.equal(firstClear(0.2, intervals), 0.5);
  assert.equal(firstClear(0.6, intervals), 0.6);
  assert.equal(firstClear(0.75, intervals), 0.8);
  assert.equal(firstClear(0.1, intervals), 0.1);
});

test("overlapInterval gives the travel range where two boxes meet", () => {
  // Box of half 1 at x = -5 moving along +X towards a placed box of half 1 at x = 0.
  const placed = { id: "p", centre: [0, 0, 0], half: [1, 1, 1] };
  const iv = overlapInterval([-5, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], placed, 0);
  close(iv, [3, 7]);
  // Separated along an axis the motion does not change: never overlaps.
  assert.equal(overlapInterval([-5, 3, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], placed, 0), null);
  // Already past it: the interval lies at negative travel.
  const behind = overlapInterval([5, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], placed, 0);
  close(behind, [-7, -3]);
});

test("overlapping() lists pairs that share volume and ignores touching faces", () => {
  const items = [
    { id: "a", bounds: box([0, 0, 0], [1, 1, 1]) },
    { id: "b", bounds: box([2, 0, 0], [1, 1, 1]) },
    { id: "c", bounds: box([1.5, 0, 0], [1, 1, 1]) },
  ];
  assert.deepEqual(overlapping(items), [
    ["a", "c"],
    ["b", "c"],
  ]);
});

test("at t = 1 no two structures' bounds overlap, though many do at t = 0", () => {
  const laidOut = layout(stackedBody());
  assert.ok(overlapping(boundsAt(laidOut, 0)).length > 5, "the assembled body overlaps");
  assert.deepEqual(overlapping(boundsAt(laidOut, 1)), []);
  assert.deepEqual(laidOut.unresolved, []);
  assert.ok(laidOut.adjusted.length > 0, "some distances had to grow");
  for (const a of laidOut.adjusted) assert.ok(a.to > a.from);
  assert.equal(laidOut.derivedCount, laidOut.nodes.length);
});

test("the slider moves every structure linearly along its final offset", () => {
  const laidOut = layout(stackedBody());
  const entry = laidOut.entries.get("STERNUM");
  close(offsetAt(entry, 0), [0, 0, 0]);
  close(offsetAt(entry, 0.5), entry.offset.map((v) => v / 2));
  close(offsetAt(entry, 1), entry.offset);
  close(offsetAt(entry, 7), entry.offset);
  assert.ok(Math.abs(Math.hypot(...entry.offset) - entry.distance) < 1e-9, "single system: no block offset");
});

test("layout is deterministic and does not depend on the input order", () => {
  const body = stackedBody();
  const dump = (l) => JSON.stringify([...l.entries.entries()].sort(([a], [b]) => a.localeCompare(b)));
  assert.equal(dump(layout(body)), dump(layout(body)));
  const shuffled = [...body].reverse();
  const a = layout(body);
  const b = layout(shuffled);
  for (const [id, e] of a.entries) close(e.offset, b.entries.get(id).offset, 1e-9);
});

test("manifest distances that overlap are increased and reported", () => {
  const nodes = [
    node("a", [0, 0, 0], [0.05, 0.05, 0.05], { explode: { direction: [0, 0, 1], distance: 0.01 }, volume_ml: 100 }),
    node("b", [0, 0, 0.02], [0.05, 0.05, 0.05], { explode: { direction: [0, 0, 1], distance: 0.01 } }),
  ];
  const laidOut = layout(nodes);
  assert.deepEqual(overlapping(boundsAt(laidOut, 1)), []);
  assert.equal(laidOut.adjusted.length, 1);
  assert.equal(laidOut.adjusted[0].id, "b");
  assert.ok(laidOut.adjusted[0].to > 0.08);
  assert.equal(laidOut.entries.get("a").distance, 0.01);
  assert.equal(laidOut.derivedCount, 0);
});

test("a cap on the distance leaves the overlap reported instead of resolved", () => {
  const entries = [
    { id: "a", bounds: box([0, 0, 0], [1, 1, 1]), start: [0, 0, 0], direction: [1, 0, 0], distance: 0, rank: 0 },
    { id: "b", bounds: box([0.5, 0, 0], [1, 1, 1]), start: [0, 0, 0], direction: [1, 0, 0], distance: 0, rank: 1 },
  ];
  const solved = resolveOverlaps(entries, { maxDistance: 0.1, clearance: 0 });
  assert.equal(solved.distances.get("b"), 0.1);
  assert.deepEqual(solved.unresolved, [["a", "b"]]);
  const free = resolveOverlaps(entries, { clearance: 0 });
  assert.ok(Math.abs(free.distances.get("b") - 1.5) < 1e-9);
  assert.deepEqual(free.unresolved, []);
});

test("systems move as blocks only when there are several", () => {
  const single = systemOffsets(stackedBody());
  for (const v of single.values()) close(v, [0, 0, 0]);
  const two = [
    node("bone", [0, 0, -0.05], undefined, { volume_ml: 100 }),
    node("heart", [0, 0, 0.05], undefined, { system: "cardiovascular", volume_ml: 100 }),
  ];
  const offsets = systemOffsets(two, 0.1);
  close(offsets.get("skeleton"), [0, 0, -0.1]);
  close(offsets.get("cardiovascular"), [0, 0, 0.1]);
  const laidOut = layout(two, { systemGap: 0.1 });
  assert.deepEqual(overlapping(boundsAt(laidOut, 1)), []);
});

test("layout takes the manifest's body centre when it is given", () => {
  const nodes = [node("a", [0.1, 0, 0]), node("b", [0.3, 0, 0])];
  const own = layout(nodes);
  close(own.bodyCentre, [0.2, 0, 0]);
  close(own.entries.get("a").direction, [-1, 0, 0]);
  const given = layout(nodes, { bodyCentre: [0, 0, 0] });
  close(given.bodyCentre, [0, 0, 0]);
  close(given.entries.get("a").direction, [1, 0, 0]);
  close(given.entries.get("b").direction, [1, 0, 0]);
  assert.deepEqual(overlapping(boundsAt(given, 1)), []);
});

test("layout skips nodes without a centre or bounds", () => {
  const laidOut = layout([node("ok", [0.1, 0, 0]), { id: "no-geometry", system: "skeleton", centre: null, bounds: null }]);
  assert.deepEqual([...laidOut.entries.keys()], ["ok"]);
});
