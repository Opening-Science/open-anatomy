/**
 * The page. Loads a body's GLB and manifest, lays the structures out with
 * explode.js, draws them with three.js and wires the controls to state.js.
 *
 * URL parameters: glb and manifest (paths or URLs; default ./body.glb and
 * ./body.manifest.json), t (0..1), isolate (a node id), hide (systems, comma
 * separated), clinician (1) and q (search text). The page keeps them up to
 * date, so the address bar is always a link to the current view.
 *
 * For tests: document.documentElement.dataset.ready is "1" once the first
 * frame with the model has been drawn, dataset.error holds a load error, and
 * window.viewer exposes the state, the layout, the camera, dispatch(),
 * screenPoint(id) and structureAt(x, y).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { boundsAt, boxCentre, boxRadius, layout, offsetAt, translateBounds, unionBounds } from "./explode.js";
import { attachGeometry, normaliseManifest, sourceText } from "./manifest.js";
import {
  displayName,
  groupBySystem,
  initialState,
  matches,
  paramsFromState,
  reduce,
  stateFromParams,
  visibility,
} from "./state.js";

const DEFAULT_GLB = "./body.glb";
const DEFAULT_MANIFEST = "./body.manifest.json";
const GHOST_OPACITY = 0.07;
const FOLLOW_MS = 110; // time constant of the camera's glide to a new framing
const FADE_MS = 70; // time constant of the fade to and from a ghost
const FIT_MARGIN = 1.12;
const CLICK_PX = 6;
const CLICK_MS = 600;
const KEY_TURN = THREE.MathUtils.degToRad(12);
const KEY_ZOOM = 1.2;

const $ = (selector) => document.querySelector(selector);
const ui = {
  canvas: $("#view"),
  stage: $("#stage"),
  slider: $("#take-apart"),
  sliderValue: $("#take-apart-value"),
  clinician: $("#clinician"),
  search: $("#search"),
  list: $("#structures"),
  info: $("#info"),
  status: $("#status"),
  picker: $("#picker"),
  files: $("#files"),
  panel: $("#panel"),
  panelToggle: $("#panel-toggle"),
  count: $("#count"),
  hint: $("#hint"),
  hover: $("#hover"),
};
const root = document.documentElement;
const params = new URLSearchParams(location.search);
const errors = [];

function setStatus(text, isError = false) {
  ui.status.textContent = text;
  ui.status.hidden = !text;
  ui.status.classList.toggle("error", isError);
}

function fail(message) {
  errors.push(message);
  root.dataset.error = message;
  console.error(message);
  setStatus(message, true);
}

window.addEventListener("error", (e) => {
  errors.push(String(e.message));
  root.dataset.error = String(e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  errors.push(String(e.reason));
  root.dataset.error = String(e.reason);
});

// Scene -----------------------------------------------------------------------

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
} catch (err) {
  fail("This browser cannot draw 3D here (WebGL is off or missing).");
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color("#f3f1ec");
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
camera.position.set(0, 0, 2);
const controls = new OrbitControls(camera, ui.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.2;
controls.screenSpacePanning = true;

// Soft studio light: a warm key from the upper front left, a cool fill from the
// right and a little from behind, over a dim hemisphere so shading stays visible.
// (three.js divides light by pi on a matt surface, so the intensities add up to
// about three for a well-lit face.)
scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9087, 0.7));
const key = new THREE.DirectionalLight(0xfff3e2, 2.2);
key.position.set(-1.5, 2, 2.5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xf4f6ff, 0.6);
fill.position.set(2, 0.5, 1);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.4);
rim.position.set(0, 1, -2);
scene.add(rim);

// State -----------------------------------------------------------------------

let state = initialState();
let model = null; // {nodes, byId: Map id -> {node, object, materials, home, entry}, laidOut, root, meta}
let cameraGoal = null; // {position, target} the camera is gliding to, or null
let dirty = true; // the scene changed and needs drawing
let settled = true; // false while the camera or a fade is still moving

const invalidate = () => {
  dirty = true;
};
controls.addEventListener("change", invalidate);
// The camera is the user's as soon as they touch it.
controls.addEventListener("start", () => {
  cameraGoal = null;
});

// Loading ---------------------------------------------------------------------

async function readManifest(source) {
  if (typeof source !== "string") return source;
  const response = await fetch(source);
  if (!response.ok) throw new Error(`no manifest at ${source} (${response.status})`);
  return response.json();
}

/** Find the object for each manifest node by name; a node may be a mesh or a group of meshes. */
function indexObjects(sceneRoot, ids) {
  const objects = new Map();
  sceneRoot.traverse((o) => {
    if (ids.has(o.name) && !objects.has(o.name)) objects.set(o.name, o);
  });
  return objects;
}

/** Body-frame bounds of every object, from its geometry at the assembled pose. */
function measure(objects) {
  const boxes = new Map();
  for (const [id, object] of objects) {
    const box = new THREE.Box3().setFromObject(object, true);
    if (!box.isEmpty()) boxes.set(id, { min: box.min.toArray(), max: box.max.toArray() });
  }
  return boxes;
}

function ownMaterials(object) {
  const materials = [];
  object.traverse((o) => {
    if (!o.isMesh) return;
    const list = Array.isArray(o.material) ? o.material : [o.material];
    const clones = list.map((m) => {
      const c = m.clone();
      c.transparent = false;
      c.opacity = 1;
      c.userData.opacityTarget = 1;
      return c;
    });
    o.material = Array.isArray(o.material) ? clones : clones[0];
    materials.push(...clones);
  });
  return materials;
}

async function load(glbSource, manifestSource) {
  root.dataset.ready = "0";
  delete root.dataset.error;
  errors.length = 0;
  setStatus("Loading the body…");
  ui.picker.hidden = true;
  if (model) {
    scene.remove(model.root);
    model.root.traverse((o) => {
      if (!o.isMesh) return;
      o.geometry.dispose();
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
    });
    model = null;
    ui.list.replaceChildren();
    ui.info.replaceChildren();
    ui.count.textContent = "";
    document.body.classList.remove("isolated");
    invalidate();
  }
  const manifest = normaliseManifest(await readManifest(manifestSource));
  for (const w of manifest.warnings) console.warn(`manifest: ${w}`);
  const gltf = await new GLTFLoader().loadAsync(glbSource);
  const sceneRoot = gltf.scene;
  sceneRoot.updateMatrixWorld(true);

  const ids = new Set(manifest.nodes.map((n) => n.id));
  const objects = indexObjects(sceneRoot, ids);
  const nodes = attachGeometry(manifest.nodes, measure(objects));
  const missing = nodes.filter((n) => !n.geometry).map((n) => n.id);
  if (missing.length) console.warn(`${missing.length} manifest nodes have no mesh in the GLB: ${missing.join(", ")}`);
  for (const child of sceneRoot.children) {
    if (!ids.has(child.name)) {
      console.warn(`GLB node ${child.name || "(unnamed)"} is not in the manifest; hidden`);
      child.visible = false;
    }
  }

  const laidOut = layout(
    nodes.filter((n) => n.geometry),
    { bodyCentre: manifest.meta.bodyCentre },
  );
  const total = laidOut.nodes.length;
  if (!total) throw new Error("the GLB has none of the manifest's structures");
  if (laidOut.derivedCount) {
    console.info(`Explode direction and distance derived in the viewer for ${laidOut.derivedCount} of ${total} structures (the manifest has none).`);
  }
  if (laidOut.adjusted.length) {
    const factor = Math.max(...laidOut.adjusted.map((a) => a.to / Math.max(a.from, 1e-9)));
    console.info(
      `Increased the explode distance of ${laidOut.adjusted.length} of ${total} structures (largest ×${factor.toFixed(2)}) so no two bounds overlap at 1.`,
    );
  }
  if (laidOut.unresolved.length) {
    console.warn(`${laidOut.unresolved.length} pairs of bounds still overlap at 1: ${laidOut.unresolved.map((p) => p.join(" / ")).join(", ")}`);
  }

  const byId = new Map();
  for (const node of laidOut.nodes) {
    const object = objects.get(node.id);
    byId.set(node.id, { node, object, materials: ownMaterials(object), home: object.position.clone(), entry: laidOut.entries.get(node.id) });
  }
  scene.add(sceneRoot);
  model = { nodes: laidOut.nodes, byId, laidOut, root: sceneRoot, meta: manifest.meta };

  ui.count.textContent = `${total} part${total === 1 ? "" : "s"}`;
  state = stateFromParams((k) => params.get(k), new Set(byId.keys()));
  if (params.get("isolate") && !byId.has(params.get("isolate"))) console.warn(`isolate=${params.get("isolate")} names no structure`);
  applyAll();
  // The view asked for in the URL appears at once, without the fades.
  for (const { materials } of byId.values()) for (const m of materials) applyOpacity(m, m.userData.opacityTarget);
  // A new body is first seen from the front.
  camera.position.set(0, 0, 2);
  controls.target.set(0, 0, 0);
  frame(focusBounds(), false);
  controls.update();
  setStatus("");
  ui.hint.hidden = false;
  render();
  requestAnimationFrame(() => {
    render();
    root.dataset.ready = "1";
  });
}

// Applying state ----------------------------------------------------------------

function applyPositions() {
  for (const { object, home, entry } of model.byId.values()) {
    const [x, y, z] = offsetAt(entry, state.t);
    object.position.set(home.x + x, home.y + y, home.z + z);
  }
  invalidate();
}

function applyVisibility() {
  for (const { node, object, materials } of model.byId.values()) {
    const v = visibility(state, node);
    object.visible = v !== "hidden";
    for (const m of materials) m.userData.opacityTarget = v === "ghost" ? GHOST_OPACITY : 1;
  }
  invalidate();
}

/** Fade materials towards their target opacity; true while any is still moving. */
function settleMaterials(dt) {
  let moving = false;
  const k = 1 - Math.exp(-dt / FADE_MS);
  for (const { materials } of model.byId.values()) {
    for (const m of materials) {
      const target = m.userData.opacityTarget;
      if (m.opacity === target) continue;
      if (Math.abs(m.opacity - target) < 0.005) applyOpacity(m, target);
      else applyOpacity(m, m.opacity + (target - m.opacity) * k);
      moving = true;
    }
  }
  return moving;
}

function applyOpacity(m, value) {
  const transparent = value < 0.999;
  // three.js compiles an opaque material to ignore its opacity, so the switch needs a new program.
  if (m.transparent !== transparent) m.needsUpdate = true;
  m.opacity = value;
  m.transparent = transparent;
  m.depthWrite = value > 0.5;
}

function applyIsolatedClass() {
  document.body.classList.toggle("isolated", state.isolated !== null);
  // The card is laid out now, so the 3D view has its new size before the camera frames.
  resize(false);
}

function applyAll() {
  applyPositions();
  applyVisibility();
  ui.slider.value = String(state.t);
  updateSliderText();
  ui.clinician.checked = state.clinician;
  ui.search.value = state.query;
  document.body.classList.toggle("clinician", state.clinician);
  renderInfo();
  renderList();
  applyIsolatedClass();
  updateUrl();
}

function dispatch(action) {
  const next = reduce(state, action);
  if (next === state) return;
  const prev = state;
  state = next;
  if (!model) return;
  if (prev.t !== state.t) {
    applyPositions();
    ui.slider.value = String(state.t);
    updateSliderText();
  }
  const isolatedChanged = prev.isolated !== state.isolated;
  const hiddenChanged = prev.hidden.join() !== state.hidden.join();
  if (isolatedChanged || hiddenChanged) {
    applyVisibility();
    renderInfo();
    renderList();
    applyIsolatedClass();
  }
  if (prev.clinician !== state.clinician) {
    document.body.classList.toggle("clinician", state.clinician);
    renderInfo();
    renderList();
  }
  if (prev.query !== state.query) renderList();
  if (prev.t !== state.t || isolatedChanged || (hiddenChanged && state.isolated === null)) frame(focusBounds(), true);
  if (isolatedChanged) showSelectedInList();
  updateUrl();
}

function updateUrl() {
  const next = new URLSearchParams();
  for (const key of ["glb", "manifest"]) if (params.get(key)) next.set(key, params.get(key));
  for (const [k, v] of Object.entries(paramsFromState(state))) next.set(k, v);
  const query = next.toString();
  history.replaceState(null, "", `${location.pathname}${query ? `?${query}` : ""}${location.hash}`);
}

function updateSliderText() {
  const pct = Math.round(state.t * 100);
  const text = pct === 0 ? "together" : pct === 100 ? "fully apart" : `${pct}% apart`;
  ui.sliderValue.textContent = text;
  ui.slider.setAttribute("aria-valuetext", text);
}

// Camera ------------------------------------------------------------------------

/** What the camera should show: the isolated structure, or every visible one, at the current t. */
function focusBounds() {
  if (!model) return null;
  if (state.isolated !== null && model.byId.has(state.isolated)) {
    const { node, entry } = model.byId.get(state.isolated);
    return translateBounds(node.bounds, offsetAt(entry, state.t));
  }
  const shown = boundsAt(model.laidOut, state.t).filter(({ id }) => visibility(state, model.byId.get(id).node) !== "hidden");
  return unionBounds(shown.map((b) => b.bounds)) ?? unionBounds(boundsAt(model.laidOut, state.t).map((b) => b.bounds));
}

/** Unit vector from the point looked at to the camera, as it is or as it will be. */
function viewDirection() {
  const from = cameraGoal ?? { position: camera.position, target: controls.target };
  const direction = from.position.clone().sub(from.target);
  if (direction.lengthSq() < 1e-12) direction.set(0, 0, 1);
  return direction.normalize();
}

/** How far back the camera must stand, looking along -direction at the box's centre, to see all of it. */
function fitDistance(bounds, centre, direction) {
  const right = new THREE.Vector3().crossVectors(camera.up, direction);
  if (right.lengthSq() < 1e-9) right.set(1, 0, 0);
  right.normalize();
  const up = new THREE.Vector3().crossVectors(direction, right);
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const tanH = tanV * camera.aspect;
  const d = new THREE.Vector3();
  let distance = 0;
  for (let i = 0; i < 8; i++) {
    d.set(bounds[i & 1][0], bounds[(i >> 1) & 1][1], bounds[(i >> 2) & 1][2]).sub(centre);
    const depth = d.dot(direction);
    distance = Math.max(distance, Math.abs(d.dot(right)) / tanH + depth, Math.abs(d.dot(up)) / tanV + depth);
  }
  return distance;
}

/** Point the camera at the bounds from where it looks now, gliding there or at once. */
function frame(bounds, animate) {
  if (!bounds) return;
  const centre = new THREE.Vector3(...boxCentre(bounds));
  const radius = Math.max(boxRadius(bounds), 0.01);
  const direction = viewDirection();
  const distance = Math.max(fitDistance(bounds, centre, direction) * FIT_MARGIN, radius * 0.5);
  const position = centre.clone().addScaledVector(direction, distance);
  camera.near = Math.max(0.001, distance / 200);
  camera.far = distance * 50;
  camera.updateProjectionMatrix();
  controls.minDistance = radius * 0.2;
  controls.maxDistance = distance * 6;
  glideTo(position, centre, animate);
}

/** Send the camera to a new place: gliding, or at once (also for people who ask for less motion). */
function glideTo(position, target, animate = true) {
  // Finish what is left of a drag's glide first, so it does not pull against the new goal.
  controls.enableDamping = false;
  controls.update();
  controls.enableDamping = true;
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    cameraGoal = { position, target };
  } else {
    camera.position.copy(position);
    controls.target.copy(target);
    cameraGoal = null;
  }
  invalidate();
}

/** Move the camera a step towards its goal; true while it is on its way. */
function stepCamera(dt) {
  if (!cameraGoal) return false;
  const k = 1 - Math.exp(-dt / FOLLOW_MS);
  camera.position.lerp(cameraGoal.position, k);
  controls.target.lerp(cameraGoal.target, k);
  const left = camera.position.distanceTo(cameraGoal.position) + controls.target.distanceTo(cameraGoal.target);
  if (left < 1e-4 * cameraGoal.position.distanceTo(cameraGoal.target)) {
    camera.position.copy(cameraGoal.position);
    controls.target.copy(cameraGoal.target);
    cameraGoal = null;
  }
  return true;
}

/** Turn the camera about what it looks at, or move it closer (zoom < 1) or away: the keyboard's orbit. */
function orbitBy(dTheta, dPhi, zoom) {
  const from = cameraGoal ?? { position: camera.position, target: controls.target };
  const target = from.target.clone();
  const s = new THREE.Spherical().setFromVector3(from.position.clone().sub(target));
  s.theta += dTheta;
  s.phi = THREE.MathUtils.clamp(s.phi + dPhi, 0.05, Math.PI - 0.05);
  s.radius = THREE.MathUtils.clamp(s.radius * zoom, controls.minDistance, controls.maxDistance);
  glideTo(target.clone().add(new THREE.Vector3().setFromSpherical(s)), target);
}

// Picking -------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();
let pointerDown = null;
let hoverAt = null; // the mouse position to look under at the next frame

ui.canvas.addEventListener("pointerdown", (e) => {
  hideHover();
  if (e.button !== 0 && e.pointerType === "mouse") return;
  pointerDown = { x: e.clientX, y: e.clientY, time: performance.now() };
});
ui.canvas.addEventListener("pointerup", (e) => {
  if (!pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  const held = performance.now() - pointerDown.time;
  pointerDown = null;
  if (moved > CLICK_PX || held > CLICK_MS) return;
  pick(e.clientX, e.clientY);
});
ui.canvas.addEventListener("pointercancel", () => {
  pointerDown = null;
});
ui.canvas.addEventListener("pointermove", (e) => {
  if (e.pointerType === "mouse" && e.buttons === 0) hoverAt = { x: e.clientX, y: e.clientY };
});
ui.canvas.addEventListener("pointerleave", hideHover);

/** The id of the structure drawn at a point of the window, or null. The isolated one wins over ghosts in front of it. */
function structureAt(clientX, clientY) {
  if (!model) return null;
  const rect = ui.canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const candidates = [...model.byId.values()].filter((s) => s.object.visible).map((s) => s.object);
  let first = null;
  for (const hit of raycaster.intersectObjects(candidates, true)) {
    let o = hit.object;
    while (o && !model.byId.has(o.name)) o = o.parent;
    if (!o) continue;
    if (state.isolated === null || o.name === state.isolated) return o.name;
    first ??= o.name;
  }
  return first;
}

function pick(clientX, clientY) {
  const id = structureAt(clientX, clientY);
  if (id) dispatch({ type: "toggleIsolate", id });
  else dispatch({ type: "showAll" });
}

function hideHover() {
  hoverAt = null;
  ui.hover.hidden = true;
  ui.canvas.classList.remove("over-part");
}

/** Name the part under the mouse, once per frame at most. */
function updateHover() {
  if (!hoverAt || !model) return;
  const { x, y } = hoverAt;
  hoverAt = null;
  const id = structureAt(x, y);
  ui.canvas.classList.toggle("over-part", id !== null);
  ui.hover.hidden = id === null;
  if (id === null) return;
  ui.hover.textContent = displayName(model.byId.get(id).node, state.clinician);
  const stage = ui.stage.getBoundingClientRect();
  const left = Math.min(x - stage.left + 14, stage.width - ui.hover.offsetWidth - 4);
  const top = Math.max(y - stage.top - ui.hover.offsetHeight - 10, 4);
  ui.hover.style.transform = `translate(${Math.max(4, left)}px, ${top}px)`;
}

/** Where a structure's centre is drawn, in window coordinates (for tests). */
function screenPoint(id) {
  const entry = model?.byId.get(id);
  if (!entry) return null;
  camera.updateMatrixWorld();
  const centre = boxCentre(translateBounds(entry.node.bounds, offsetAt(entry.entry, state.t)));
  const p = new THREE.Vector3(...centre).project(camera);
  const rect = ui.canvas.getBoundingClientRect();
  return { x: rect.left + ((p.x + 1) / 2) * rect.width, y: rect.top + ((1 - p.y) / 2) * rect.height };
}

// Panel -------------------------------------------------------------------------

function text(tag, content, className) {
  const el = document.createElement(tag);
  el.textContent = content;
  if (className) el.className = className;
  return el;
}

function renderInfo() {
  ui.info.replaceChildren();
  if (!model) return;
  if (state.isolated === null || !model.byId.has(state.isolated)) {
    ui.info.append(text("h2", "The whole body"));
    ui.info.append(text("p", "Tap or click a part to see it on its own. Drag the slider to take the body apart.", "muted"));
    if (model.meta.attribution && state.clinician) ui.info.append(text("p", model.meta.attribution, "small muted"));
    return;
  }
  const { node } = model.byId.get(state.isolated);
  ui.info.append(text("h2", displayName(node, state.clinician)));
  ui.info.append(text("p", node.about));
  if (state.clinician) {
    const dl = document.createElement("dl");
    const rows = [
      ["Anatomical name", node.label],
      ["ID", node.term],
      ["System", node.system],
      ["Licence", node.licence],
      ["Volume", node.volume_ml === null ? null : `${node.volume_ml} ml`],
      ["Pieces", node.pieces],
      ["Triangles", node.triangles === null ? null : String(node.triangles)],
    ];
    for (const [k, v] of rows) {
      if (v === null || v === undefined) continue;
      dl.append(text("dt", k), text("dd", v));
    }
    ui.info.append(dl);
    if (node.derived.length) {
      const what = node.derived.includes("plain") || node.derived.includes("about") ? "The plain name and description" : "Centre and bounds";
      ui.info.append(text("p", `${what} come from the viewer, not the manifest (derived: ${node.derived.join(", ")}).`, "small muted"));
    }
  }
  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(text("span", sourceText(node), `source ${node.modelled ? "modelled" : "segmented"}`));
  const back = text("button", "Show all", "button primary");
  back.type = "button";
  back.id = "show-all";
  back.addEventListener("click", () => dispatch({ type: "showAll" }));
  actions.append(back);
  ui.info.append(actions);
}

function renderList() {
  if (!model) return;
  const focused = document.activeElement?.dataset?.id;
  ui.list.replaceChildren();
  const groups = groupBySystem(model.nodes);
  let shown = 0;
  for (const group of groups) {
    const nodes = group.nodes.filter((n) => matches(n, state.query));
    if (!nodes.length) continue;
    shown += nodes.length;
    const visible = !state.hidden.includes(group.system);
    const section = document.createElement("section");
    section.className = visible ? "group" : "group off";
    const header = document.createElement("label");
    header.className = "group-header";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = visible;
    box.dataset.system = group.system;
    box.setAttribute("aria-label", `Show ${group.title}`);
    box.addEventListener("change", () => dispatch({ type: "setSystemVisible", system: group.system, visible: box.checked }));
    header.append(box, text("span", group.title, "group-title"), text("span", String(group.nodes.length), "muted small"));
    section.append(header);
    const ul = document.createElement("ul");
    for (const node of nodes) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "item";
      button.dataset.id = node.id;
      button.setAttribute("aria-pressed", String(state.isolated === node.id));
      button.append(text("span", displayName(node, state.clinician), "item-name"));
      if (state.clinician) button.append(text("span", node.term, "term"));
      if (node.modelled) button.append(text("span", "modelled", "tag"));
      button.addEventListener("click", () => {
        dispatch({ type: "toggleIsolate", id: node.id });
        // On a phone the list is a sheet over the 3D view: put it away to show the part.
        if (state.isolated === node.id) setPanelOpen(false);
      });
      li.append(button);
      ul.append(li);
    }
    section.append(ul);
    ui.list.append(section);
  }
  if (!shown) ui.list.append(text("p", "No part matches.", "muted"));
  if (focused) ui.list.querySelector(`[data-id="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
}

/** Scroll the list to the part that was just picked in the 3D view. */
function showSelectedInList() {
  if (state.isolated === null) return;
  ui.list.querySelector(`[data-id="${CSS.escape(state.isolated)}"]`)?.scrollIntoView({ block: "nearest" });
}

function setPanelOpen(open) {
  if (document.body.classList.contains("panel-open") === open) return;
  document.body.classList.toggle("panel-open", open);
  ui.panelToggle.setAttribute("aria-expanded", String(open));
  resize(true);
}

// Controls ----------------------------------------------------------------------

ui.slider.addEventListener("input", () => dispatch({ type: "setT", t: ui.slider.value }));
ui.clinician.addEventListener("change", () => dispatch({ type: "setClinician", on: ui.clinician.checked }));
ui.search.addEventListener("input", () => dispatch({ type: "setQuery", query: ui.search.value }));
ui.panelToggle.addEventListener("click", () => setPanelOpen(!document.body.classList.contains("panel-open")));
document.addEventListener("keydown", (e) => {
  const typing = e.target instanceof HTMLInputElement && e.target.type !== "range" && e.target.type !== "checkbox";
  if (e.key === "Escape") {
    if (typing && ui.search.value) {
      dispatch({ type: "setQuery", query: "" });
      ui.search.value = "";
    } else if (document.body.classList.contains("panel-open") && state.isolated === null) {
      setPanelOpen(false);
    } else {
      dispatch({ type: "showAll" });
    }
  } else if (e.key === "/" && !typing) {
    e.preventDefault();
    setPanelOpen(true);
    ui.search.focus();
  }
});
// With the 3D view focused: arrows turn the body, plus and minus zoom.
ui.canvas.addEventListener("keydown", (e) => {
  if (!model || e.ctrlKey || e.metaKey || e.altKey) return;
  const moves = {
    ArrowLeft: [-KEY_TURN, 0, 1],
    ArrowRight: [KEY_TURN, 0, 1],
    ArrowUp: [0, -KEY_TURN, 1],
    ArrowDown: [0, KEY_TURN, 1],
    "+": [0, 0, 1 / KEY_ZOOM],
    "=": [0, 0, 1 / KEY_ZOOM],
    "-": [0, 0, KEY_ZOOM],
  };
  const move = moves[e.key];
  if (!move) return;
  e.preventDefault();
  orbitBy(...move);
});
ui.files.addEventListener("change", async () => {
  const files = [...ui.files.files];
  const glb = files.find((f) => /\.glb$/i.test(f.name));
  const manifestFile = files.find((f) => /\.json$/i.test(f.name));
  if (!glb || !manifestFile) {
    setStatus("Choose both the .glb file and its .manifest.json.", true);
    return;
  }
  delete root.dataset.error;
  setStatus("Loading the body…");
  const url = URL.createObjectURL(glb);
  try {
    await load(url, JSON.parse(await manifestFile.text()));
  } catch (err) {
    loadFailed(err);
  } finally {
    URL.revokeObjectURL(url);
  }
});

function loadFailed(err) {
  fail(`Could not show the body: ${err.message || err}`);
  ui.hint.hidden = true;
  ui.picker.hidden = false;
}

// Rendering loop ------------------------------------------------------------------

/** Match the drawing size to the 3D view's box; re-frame when the box changed shape. */
function resize(reframe = true) {
  const { clientWidth: w, clientHeight: h } = ui.stage;
  if (!w || !h) return;
  const size = renderer.getSize(new THREE.Vector2());
  if (size.x === w && size.y === h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  invalidate();
  if (reframe && model) frame(focusBounds(), false);
}
new ResizeObserver(() => resize(true)).observe(ui.stage);
resize(false);

function render() {
  renderer.render(scene, camera);
  dirty = false;
}

let last = performance.now();

function tick(now) {
  const dt = Math.min(250, now - last);
  last = now;
  const gliding = stepCamera(dt);
  // True, and a "change" event that sets dirty, while a drag or what is left of it moves the camera.
  const turning = controls.update();
  const fading = model ? settleMaterials(dt) : false;
  settled = !gliding && !turning && !fading;
  if (dirty || gliding || fading) render();
  updateHover();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// Start -------------------------------------------------------------------------

window.viewer = {
  get state() {
    return state;
  },
  get layout() {
    return model?.laidOut ?? null;
  },
  get nodes() {
    return model?.nodes ?? null;
  },
  get meta() {
    return model?.meta ?? null;
  },
  get settled() {
    return settled && !dirty;
  },
  get camera() {
    return { position: camera.position.toArray(), target: controls.target.toArray(), aspect: camera.aspect };
  },
  errors,
  dispatch,
  screenPoint,
  structureAt,
};

load(params.get("glb") || DEFAULT_GLB, params.get("manifest") || DEFAULT_MANIFEST).catch((err) => {
  loadFailed(err);
  // Nothing was asked for and nothing sits next to the page: not an error worth a red box.
  if (!params.get("glb") && !params.get("manifest")) setStatus("No body to show yet.");
});
