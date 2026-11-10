import { test } from "node:test";
import assert from "node:assert/strict";

import {
  displayName,
  filterNodes,
  groupBySystem,
  initialState,
  matches,
  paramsFromState,
  reduce,
  stateFromParams,
  systemTitle,
  visibility,
} from "../state.js";

const nodes = [
  { id: "FMA_7485", term: "FMA:7485", label: "sternum", plain: "Breastbone", system: "skeleton" },
  { id: "FMA_7987", term: "FMA:7987", label: "left first rib", plain: "Left rib 1", system: "skeleton" },
  { id: "UBERON_0000948", term: "UBERON:0000948", label: "heart", plain: "Heart", system: "cardiovascular" },
  { id: "UBERON_0002107", term: "UBERON:0002107", label: "liver", plain: "Liver", system: "digestive" },
  { id: "X_1", term: "X:1", label: "thing", plain: "Thing", system: "mystery" },
];

test("the initial state is assembled, nothing isolated, everything shown", () => {
  assert.deepEqual(initialState(), { t: 0, isolated: null, hidden: [], clinician: false, query: "" });
});

test("setT clamps to [0, 1] and ignores values that are not numbers", () => {
  const s = initialState();
  assert.equal(reduce(s, { type: "setT", t: 0.5 }).t, 0.5);
  assert.equal(reduce(s, { type: "setT", t: "0.25" }).t, 0.25);
  assert.equal(reduce(s, { type: "setT", t: 3 }).t, 1);
  assert.equal(reduce(s, { type: "setT", t: -1 }).t, 0);
  assert.equal(reduce(s, { type: "setT", t: "abc" }), s);
  assert.equal(s.t, 0, "the old state is untouched");
});

test("isolate, toggle and show all move between the two views", () => {
  let s = initialState();
  s = reduce(s, { type: "isolate", id: "FMA_7485" });
  assert.equal(s.isolated, "FMA_7485");
  s = reduce(s, { type: "isolate", id: "UBERON_0000948" });
  assert.equal(s.isolated, "UBERON_0000948");
  s = reduce(s, { type: "toggleIsolate", id: "UBERON_0000948" });
  assert.equal(s.isolated, null);
  s = reduce(s, { type: "toggleIsolate", id: "FMA_7987" });
  assert.equal(s.isolated, "FMA_7987");
  s = reduce(s, { type: "showAll" });
  assert.equal(s.isolated, null);
  assert.equal(reduce(s, { type: "showAll" }), s, "show all on an assembled view is a no-op");
  assert.equal(reduce(s, { type: "isolate", id: "" }), s);
  assert.equal(reduce(s, { type: "isolate" }), s);
  assert.equal(reduce(s, { type: "nonsense" }), s);
});

test("hidden systems are a sorted set", () => {
  let s = initialState();
  s = reduce(s, { type: "setSystemVisible", system: "urinary", visible: false });
  s = reduce(s, { type: "setSystemVisible", system: "digestive", visible: false });
  s = reduce(s, { type: "setSystemVisible", system: "digestive", visible: false });
  assert.deepEqual(s.hidden, ["digestive", "urinary"]);
  s = reduce(s, { type: "toggleSystem", system: "urinary" });
  assert.deepEqual(s.hidden, ["digestive"]);
  s = reduce(s, { type: "toggleSystem", system: "skeleton" });
  assert.deepEqual(s.hidden, ["digestive", "skeleton"]);
  s = reduce(s, { type: "setSystemVisible", system: "skeleton", visible: true });
  assert.deepEqual(s.hidden, ["digestive"]);
});

test("visibility: isolation ghosts the rest, hidden systems stay hidden, the isolated one always shows", () => {
  const all = initialState();
  assert.equal(visibility(all, nodes[0]), "shown");
  const hidden = reduce(all, { type: "setSystemVisible", system: "skeleton", visible: false });
  assert.equal(visibility(hidden, nodes[0]), "hidden");
  assert.equal(visibility(hidden, nodes[2]), "shown");
  const isolated = reduce(hidden, { type: "isolate", id: "FMA_7485" });
  assert.equal(visibility(isolated, nodes[0]), "shown");
  assert.equal(visibility(isolated, nodes[1]), "hidden");
  assert.equal(visibility(isolated, nodes[2]), "ghost");
});

test("the main view uses plain names; clinicians see anatomical names", () => {
  assert.equal(displayName(nodes[0], false), "Breastbone");
  assert.equal(displayName(nodes[0], true), "sternum");
  assert.equal(displayName({ id: "FMA_1", label: "x" }, false), "x");
  assert.equal(displayName({ id: "FMA_1" }, true), "FMA_1");
});

test("search matches every word against plain name, label, term, id and system title", () => {
  assert.ok(matches(nodes[1], "rib"));
  assert.ok(matches(nodes[1], "LEFT 1"));
  assert.ok(matches(nodes[1], "first rib"));
  assert.ok(matches(nodes[1], "FMA:7987"));
  assert.ok(matches(nodes[1], "fma_7987"));
  assert.ok(matches(nodes[2], "heart vessels"));
  assert.ok(!matches(nodes[2], "rib"));
  assert.ok(matches(nodes[2], ""));
  const t1 = { id: "FMA_9165", label: "first thoracic vertebra", plain: "Upper back bone 1", system: "skeleton", keywords: "spine backbone T1" };
  assert.ok(matches(t1, "spine"));
  assert.ok(matches(t1, "t1"));
  assert.ok(matches(t1, "vertebra"));
  assert.ok(!matches(t1, "t2"));
  assert.deepEqual(filterNodes(nodes, "rib").map((n) => n.id), ["FMA_7987"]);
  assert.deepEqual(filterNodes(nodes, "heart").map((n) => n.id), ["UBERON_0000948"]);
  assert.equal(filterNodes(nodes, "").length, nodes.length);
});

test("groups follow the contract's system order, unknown systems last, nodes in manifest order", () => {
  const groups = groupBySystem([nodes[4], nodes[3], nodes[2], nodes[1], nodes[0]]);
  assert.deepEqual(
    groups.map((g) => g.system),
    ["skeleton", "cardiovascular", "digestive", "mystery"],
  );
  assert.deepEqual(groups[0].nodes.map((n) => n.id), ["FMA_7987", "FMA_7485"]);
  assert.equal(groups[0].title, "Skeleton");
  assert.equal(groups[1].title, "Heart and vessels");
  assert.equal(groups[3].title, "Mystery");
  assert.equal(systemTitle("respiratory"), "Lungs");
  assert.equal(systemTitle(""), "Other");
});

test("URL parameters set the state and the state writes back only what differs from the defaults", () => {
  const params = new URLSearchParams("t=0.5&isolate=FMA_7485&hide=digestive,urinary&clinician=1&q=rib");
  const ids = new Set(nodes.map((n) => n.id));
  const s = stateFromParams((k) => params.get(k), ids);
  assert.deepEqual(s, { t: 0.5, isolated: "FMA_7485", hidden: ["digestive", "urinary"], clinician: true, query: "rib" });
  assert.deepEqual(paramsFromState(s), { t: "0.5", isolate: "FMA_7485", hide: "digestive,urinary", clinician: "1", q: "rib" });
  assert.deepEqual(paramsFromState(initialState()), {});

  const unknown = new URLSearchParams("isolate=NOPE_1&clinician=false&t=2");
  const u = stateFromParams((k) => unknown.get(k), ids);
  assert.deepEqual(u, { ...initialState(), t: 1 });
  const unchecked = stateFromParams((k) => unknown.get(k));
  assert.equal(unchecked.isolated, "NOPE_1", "without an id list the isolate is kept for the page to check");
});
