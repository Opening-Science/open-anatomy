/**
 * The page's state and its transitions, as pure functions: the slider value,
 * which structure is isolated, which systems are hidden, the clinician toggle
 * and the search text. main.js owns the one state object and calls reduce();
 * everything here can be tested without a browser.
 */

import { SYSTEM_ORDER } from "./explode.js";

/** Group titles for the main view, by system key. */
export const SYSTEM_TITLES = {
  skeleton: "Skeleton",
  cardiovascular: "Heart and vessels",
  respiratory: "Lungs",
  digestive: "Digestive",
  urinary: "Urinary",
  nervous: "Nervous",
  endocrine: "Endocrine",
  muscular: "Muscles",
  lymphoid: "Lymphatic",
};

export function systemTitle(system) {
  return SYSTEM_TITLES[system] ?? (system ? system[0].toUpperCase() + system.slice(1) : "Other");
}

export function initialState() {
  return { t: 0, isolated: null, hidden: [], clinician: false, query: "" };
}

const clamp01 = (t) => Math.min(1, Math.max(0, t));

/** A new state from an action; the state passed in is never changed. */
export function reduce(state, action) {
  switch (action?.type) {
    case "setT": {
      const t = Number(action.t);
      return Number.isFinite(t) ? { ...state, t: clamp01(t) } : state;
    }
    case "isolate":
      return typeof action.id === "string" && action.id ? { ...state, isolated: action.id } : state;
    case "toggleIsolate":
      if (typeof action.id !== "string" || !action.id) return state;
      return { ...state, isolated: state.isolated === action.id ? null : action.id };
    case "showAll":
      return state.isolated === null ? state : { ...state, isolated: null };
    case "setSystemVisible": {
      if (typeof action.system !== "string") return state;
      const hidden = new Set(state.hidden);
      if (action.visible) hidden.delete(action.system);
      else hidden.add(action.system);
      return { ...state, hidden: [...hidden].sort() };
    }
    case "toggleSystem":
      return reduce(state, {
        type: "setSystemVisible",
        system: action.system,
        visible: state.hidden.includes(action.system),
      });
    case "setClinician":
      return { ...state, clinician: Boolean(action.on) };
    case "setQuery":
      return { ...state, query: String(action.query ?? "") };
    case "reset":
      return initialState();
    default:
      return state;
  }
}

/** How a structure is drawn: "shown", "ghost" (faint, while another is isolated) or "hidden". */
export function visibility(state, node) {
  if (state.isolated !== null) {
    if (node.id === state.isolated) return "shown";
    return state.hidden.includes(node.system) ? "hidden" : "ghost";
  }
  return state.hidden.includes(node.system) ? "hidden" : "shown";
}

/** The name the main view shows: plain language, or the anatomical name for clinicians. */
export function displayName(node, clinician) {
  return clinician ? node.label || node.plain || node.id : node.plain || node.label || node.id;
}

/** True when every word of the query appears in the node's names, IDs, system title or search words. */
export function matches(node, query) {
  const words = String(query ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = [node.plain, node.label, node.term, node.id, systemTitle(node.system), node.keywords]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return words.every((w) => hay.includes(w));
}

export function filterNodes(nodes, query) {
  return nodes.filter((n) => matches(n, query));
}

/** [{system, title, nodes}] in the contract's system order; nodes keep the manifest's order. */
export function groupBySystem(nodes) {
  const groups = new Map();
  for (const n of nodes) {
    const key = n.system || "other";
    if (!groups.has(key)) groups.set(key, { system: key, title: systemTitle(key), nodes: [] });
    groups.get(key).nodes.push(n);
  }
  const rank = (s) => (SYSTEM_ORDER.includes(s) ? SYSTEM_ORDER.indexOf(s) : SYSTEM_ORDER.length);
  return [...groups.values()].sort((a, b) => rank(a.system) - rank(b.system) || a.system.localeCompare(b.system));
}

/**
 * State from URL parameters: ?t=0.5&isolate=FMA_7485&hide=digestive,urinary
 * &clinician=1&q=rib. `get(name)` is URLSearchParams.get or alike; `ids`,
 * when given, drops an isolate that names no structure.
 */
export function stateFromParams(get, ids = null) {
  let state = initialState();
  const t = get("t");
  if (t !== null && t !== "") state = reduce(state, { type: "setT", t });
  const isolate = get("isolate");
  if (isolate && (!ids || ids.has(isolate))) state = reduce(state, { type: "isolate", id: isolate });
  const hide = get("hide");
  for (const system of (hide ?? "").split(",").map((s) => s.trim()).filter(Boolean)) {
    state = reduce(state, { type: "setSystemVisible", system, visible: false });
  }
  const clinician = get("clinician");
  if (clinician !== null) state = reduce(state, { type: "setClinician", on: ["1", "true", "yes", "on"].includes(clinician) });
  const q = get("q");
  if (q) state = reduce(state, { type: "setQuery", query: q });
  return state;
}

/** The URL parameters for a state, leaving out defaults. */
export function paramsFromState(state) {
  const out = {};
  if (state.t !== 0) out.t = String(Math.round(state.t * 1000) / 1000);
  if (state.isolated) out.isolate = state.isolated;
  if (state.hidden.length) out.hide = state.hidden.join(",");
  if (state.clinician) out.clinician = "1";
  if (state.query) out.q = state.query;
  return out;
}
