/**
 * Geometry for the "Take apart" slider. Pure functions: no three.js, no DOM.
 *
 * Frame: metres, +Y up, +Z forward (anterior), +X the body's left. A node is
 *   {id, system, centre: [x, y, z], bounds: [[minx, miny, minz], [maxx, maxy, maxz]],
 *    volume_ml?, label?, explode?: {direction: [x, y, z], distance}}
 * and the manifest's explode fields are kept when present. When they are
 * missing, deriveExplode() applies the contract's rules (docs/explode-view.md):
 * the direction runs from the body centre to the structure's centre, a
 * structure within 1 cm of the body centre takes its system's mean direction,
 * a mean shorter than 0.1 falls back to +Z, and the distance is the bounding
 * radius plus 2 cm. Left and right partners move symmetrically.
 *
 * layout() then makes sure no two structures' bounds overlap at t = 1: each
 * structure, nearest to the body centre first, keeps its direction and moves
 * only as far along it as it needs to clear the structures already placed.
 * Distances that had to grow are reported so the page can say so.
 */

export const SYSTEM_ORDER = [
  "skeleton",
  "cardiovascular",
  "respiratory",
  "digestive",
  "urinary",
  "nervous",
  "endocrine",
  "muscular",
  "lymphoid",
];
export const NEAR_CENTRE_M = 0.01;
export const MIN_MEAN = 0.1;
export const MARGIN_M = 0.02;
export const CLEARANCE_M = 0.001;
export const SYSTEM_GAP_M = 0.08;
export const MAX_DISTANCE_M = 2.0;
export const ANTERIOR = Object.freeze([0, 0, 1]);

const EPS = 1e-9;

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const length = (a) => Math.hypot(a[0], a[1], a[2]);
export const mirrorX = (a) => [-a[0], a[1], a[2]];

/** Unit vector, or null when the vector is too short to have a direction. */
export function normalise(a) {
  const n = length(a);
  return n < EPS ? null : [a[0] / n, a[1] / n, a[2] / n];
}

export const isVec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);

export const isBounds = (b) =>
  Array.isArray(b) && b.length === 2 && isVec3(b[0]) && isVec3(b[1]) && b[0].every((lo, k) => lo <= b[1][k]);

export const boxCentre = (b) => [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, (b[0][2] + b[1][2]) / 2];
export const boxHalf = (b) => [(b[1][0] - b[0][0]) / 2, (b[1][1] - b[0][1]) / 2, (b[1][2] - b[0][2]) / 2];
/** Radius of the sphere around the box: half its diagonal. */
export const boxRadius = (b) => length(boxHalf(b));

export function translateBounds(b, offset) {
  return [add(b[0], offset), add(b[1], offset)];
}

export function unionBounds(list) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const b of list) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], b[0][k]);
      hi[k] = Math.max(hi[k], b[1][k]);
    }
  }
  return list.length ? [lo, hi] : null;
}

/** True when two boxes share volume (touching faces do not count). */
export function boundsOverlap(a, b, clearance = 0) {
  for (let k = 0; k < 3; k++) {
    if (a[1][k] + clearance <= b[0][k] || b[1][k] + clearance <= a[0][k]) return false;
  }
  return true;
}

/** Every overlapping pair among [{id, bounds}], as [idA, idB] with idA < idB. */
export function overlapping(items, clearance = 0) {
  const found = [];
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      if (boundsOverlap(items[i].bounds, items[j].bounds, clearance)) {
        found.push([items[i].id, items[j].id].sort());
      }
    }
  }
  return found.sort((a, b) => (a[0] + a[1]).localeCompare(b[0] + b[1]));
}

/** Volume-weighted centroid of the structures' centres (weight 1 without a volume). */
export function bodyCentre(nodes) {
  let weight = 0;
  const sum = [0, 0, 0];
  for (const n of nodes) {
    if (!isVec3(n.centre)) continue;
    const w = n.volume_ml > 0 ? n.volume_ml : 1;
    weight += w;
    for (let k = 0; k < 3; k++) sum[k] += n.centre[k] * w;
  }
  return weight > 0 ? scale(sum, 1 / weight) : [0, 0, 0];
}

/** "left first rib" -> {side: "left", base: "first rib"}; null when unsided. */
export function sideOf(label) {
  const m = /^(left|right)\s+(.+)$/i.exec(String(label ?? "").trim());
  return m ? { side: m[1].toLowerCase(), base: m[2].toLowerCase() } : null;
}

/** Map from a node's id to its left/right partner's id, by label. */
export function pairs(nodes) {
  const byKey = new Map();
  for (const n of nodes) {
    const s = sideOf(n.label);
    if (!s) continue;
    const entry = byKey.get(s.base) ?? {};
    if (!(s.side in entry)) entry[s.side] = n.id;
    byKey.set(s.base, entry);
  }
  const mates = new Map();
  for (const { left, right } of byKey.values()) {
    if (left && right) {
      mates.set(left, right);
      mates.set(right, left);
    }
  }
  return mates;
}

/** A manifest explode field the viewer can use as it is. */
export function validExplode(e) {
  return (
    !!e &&
    isVec3(e.direction) &&
    length(e.direction) > EPS &&
    Number.isFinite(e.distance) &&
    e.distance >= 0
  );
}

/**
 * Fill in explode {direction, distance} where the manifest has none, following
 * the contract's rules, and mark each as derived or not. Returns new nodes.
 */
export function deriveExplode(nodes, options = {}) {
  const centre = options.bodyCentre ?? bodyCentre(nodes);
  const near = options.nearCentre ?? NEAR_CENTRE_M;
  const minMean = options.minMean ?? MIN_MEAN;
  const margin = options.margin ?? MARGIN_M;

  const raw = new Map();
  const sums = new Map();
  for (const n of nodes) {
    const v = sub(n.centre, centre);
    const d = length(v) >= near ? normalise(v) : null;
    raw.set(n.id, d);
    if (d) {
      const s = sums.get(n.system) ?? { sum: [0, 0, 0], count: 0 };
      s.sum = add(s.sum, d);
      s.count += 1;
      sums.set(n.system, s);
    }
  }
  const systemDirection = (system) => {
    const s = sums.get(system);
    if (!s) return [...ANTERIOR];
    const mean = scale(s.sum, 1 / s.count);
    return length(mean) >= minMean ? normalise(mean) : [...ANTERIOR];
  };

  const out = nodes.map((n) => {
    if (validExplode(n.explode)) {
      return {
        ...n,
        explode: { direction: normalise(n.explode.direction), distance: n.explode.distance, derived: false },
      };
    }
    const direction = raw.get(n.id) ?? systemDirection(n.system);
    const distance = boxRadius(n.bounds) + margin;
    return { ...n, explode: { direction, distance, derived: true } };
  });

  // Left and right partners mirror each other across the midline and move as far.
  const byId = new Map(out.map((n) => [n.id, n]));
  const mates = pairs(out);
  for (const n of out) {
    const m = byId.get(mates.get(n.id));
    if (!m || n.id > m.id || !n.explode.derived || !m.explode.derived) continue;
    const left = sideOf(n.label).side === "left" ? n : m;
    const right = left === n ? m : n;
    const dl = normalise(add(left.explode.direction, mirrorX(right.explode.direction)));
    if (dl) {
      left.explode.direction = dl;
      right.explode.direction = mirrorX(dl);
    }
    const distance = Math.max(left.explode.distance, right.explode.distance);
    left.explode.distance = distance;
    right.explode.distance = distance;
  }
  return out;
}

/** Systems present, in the contract's order, then any others alphabetically. */
export function systemsOf(nodes) {
  const seen = new Set(nodes.map((n) => n.system));
  const rank = (s) => (SYSTEM_ORDER.includes(s) ? SYSTEM_ORDER.indexOf(s) : SYSTEM_ORDER.length);
  return [...seen].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/**
 * How far each system moves as a block at t = 1, so systems come apart before
 * their structures do. Zero when the body has a single system.
 */
export function systemOffsets(nodes, gap = SYSTEM_GAP_M, centre = bodyCentre(nodes)) {
  const systems = systemsOf(nodes);
  const offsets = new Map();
  for (const system of systems) {
    if (systems.length < 2) {
      offsets.set(system, [0, 0, 0]);
      continue;
    }
    const own = bodyCentre(nodes.filter((n) => n.system === system));
    const v = sub(own, centre);
    const dir = length(v) >= NEAR_CENTRE_M ? normalise(v) : [...ANTERIOR];
    offsets.set(system, scale(dir, gap));
  }
  return offsets;
}

/**
 * The range of travel d along `direction` where a box (centre c, half size h,
 * already shifted by `start`) overlaps a placed box; null when it never does.
 */
export function overlapInterval(c, h, start, direction, placed, clearance = CLEARANCE_M) {
  let lo = -Infinity;
  let hi = Infinity;
  for (let k = 0; k < 3; k++) {
    const reach = h[k] + placed.half[k] + clearance;
    const delta0 = c[k] + start[k] - placed.centre[k];
    if (Math.abs(direction[k]) < EPS) {
      if (Math.abs(delta0) >= reach) return null;
      continue;
    }
    const a = (-reach - delta0) / direction[k];
    const b = (reach - delta0) / direction[k];
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
    if (lo >= hi) return null;
  }
  return [lo, hi];
}

/**
 * Smallest d >= d0 that lies in none of the open intervals. One pass over the
 * intervals sorted by their start is enough: a jump past an interval's end can
 * never land inside an interval that started earlier.
 */
export function firstClear(d0, intervals) {
  const sorted = [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let d = d0;
  for (const [lo, hi] of sorted) {
    if (lo < d && d < hi) d = hi;
  }
  return d;
}

/**
 * Give every entry {id, bounds, start, direction, distance, rank} the smallest
 * distance >= its own at which its bounds at t = 1 clear every entry placed
 * before it (lower rank first, then id). Deterministic for the same input.
 */
export function resolveOverlaps(entries, options = {}) {
  const clearance = options.clearance ?? CLEARANCE_M;
  const maxDistance = options.maxDistance ?? MAX_DISTANCE_M;
  const order = [...entries].sort((a, b) => a.rank - b.rank || a.id.localeCompare(b.id));
  const placed = [];
  const distances = new Map();
  const adjusted = [];
  const unresolved = [];
  for (const e of order) {
    const c = boxCentre(e.bounds);
    const h = boxHalf(e.bounds);
    const direction = normalise(e.direction) ?? [...ANTERIOR];
    const intervals = [];
    for (const p of placed) {
      const iv = overlapInterval(c, h, e.start, direction, p, clearance);
      if (iv) intervals.push(iv);
    }
    let d = firstClear(e.distance, intervals);
    if (d > maxDistance) {
      d = maxDistance;
      for (const p of placed) {
        const iv = overlapInterval(c, h, e.start, direction, p, clearance);
        if (iv && iv[0] < d && d < iv[1]) unresolved.push([e.id, p.id].sort());
      }
    }
    if (d > e.distance + 1e-12) adjusted.push({ id: e.id, from: e.distance, to: d });
    distances.set(e.id, d);
    placed.push({ id: e.id, centre: add(c, add(e.start, scale(direction, d))), half: h });
  }
  return { distances, adjusted, unresolved };
}

/**
 * Everything the page needs to move structures: for each id the start (its
 * system's block offset), direction, final distance and offset at t = 1.
 * `options.bodyCentre` is the manifest's body centre, when it gives one.
 */
export function layout(nodes, options = {}) {
  const usable = nodes.filter((n) => isVec3(n.centre) && isBounds(n.bounds));
  const centre = isVec3(options.bodyCentre) ? options.bodyCentre : bodyCentre(usable);
  const withExplode = deriveExplode(usable, { ...options, bodyCentre: centre });
  const offsets = systemOffsets(withExplode, options.systemGap ?? SYSTEM_GAP_M, centre);
  const entries = withExplode.map((n) => ({
    id: n.id,
    bounds: n.bounds,
    start: offsets.get(n.system) ?? [0, 0, 0],
    direction: n.explode.direction,
    distance: n.explode.distance,
    rank: length(sub(n.centre, centre)),
  }));
  const solved = resolveOverlaps(entries, options);
  const result = new Map();
  for (const e of entries) {
    const distance = solved.distances.get(e.id);
    const node = withExplode.find((n) => n.id === e.id);
    result.set(e.id, {
      start: e.start,
      direction: e.direction,
      distance,
      requested: e.distance,
      derived: node.explode.derived,
      offset: add(e.start, scale(e.direction, distance)),
    });
  }
  return {
    bodyCentre: centre,
    systemOffsets: offsets,
    nodes: withExplode,
    entries: result,
    adjusted: solved.adjusted,
    unresolved: solved.unresolved,
    derivedCount: withExplode.filter((n) => n.explode.derived).length,
  };
}

/** Where a structure sits at slider value t, relative to its home. */
export function offsetAt(entry, t) {
  return scale(entry.offset, Math.min(1, Math.max(0, t)));
}

/** Bounds of every laid-out structure at slider value t, as [{id, bounds}]. */
export function boundsAt(laidOut, t) {
  return laidOut.nodes.map((n) => ({
    id: n.id,
    bounds: translateBounds(n.bounds, offsetAt(laidOut.entries.get(n.id), t)),
  }));
}
