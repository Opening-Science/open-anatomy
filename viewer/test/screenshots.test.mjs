/**
 * The page in headless Chromium. The test serves the repository over a local
 * static server and drives the browser through its DevTools protocol, so it
 * can wait until the model is drawn (the page sets data-ready="1") before each
 * capture, and can click, tap, drag and type like a person.
 *
 * - Screenshots, driven by URL parameters only: the body at t = 0, 0.5 and 1
 *   and one isolated structure. The PNGs must exist, be drawn and shaded, and
 *   differ from one another in the 3D view; the page must report no errors and
 *   no two bounds may overlap at t = 1. One more with the clinician names.
 * - Mouse and keyboard at 1280 px: hover, click to isolate, Escape, "Show
 *   all", drag to turn, wheel to zoom, arrow keys on the 3D view and on the
 *   slider, search, show and hide a system, the clinician toggle.
 * - Touch at 390 px: tap to isolate, the part's card, the parts sheet, pinch
 *   to zoom, no sideways scrolling.
 *
 * Environment:
 *   CHROMIUM          the browser binary; else the first of chromium,
 *                     chromium-browser, google-chrome, google-chrome-stable on PATH
 *   VIEWER_GLB        a GLB to show instead of tests/fixtures/viewer/body.glb
 *   VIEWER_MANIFEST   its manifest (both or neither)
 *   VIEWER_ISOLATE    the node to isolate (default FMA_7485, else the first node)
 *   VIEWER_SHOTS      output folder (default out/viewer-screenshots)
 *
 *   node --test --test-timeout=300000 viewer/test/screenshots.test.mjs
 */

import { after, before, describe, test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const GLB = resolve(process.env.VIEWER_GLB || join(ROOT, "tests", "fixtures", "viewer", "body.glb"));
const MANIFEST = resolve(process.env.VIEWER_MANIFEST || join(ROOT, "tests", "fixtures", "viewer", "body.manifest.json"));
const OUT = resolve(process.env.VIEWER_SHOTS || join(ROOT, "out", "viewer-screenshots"));
const DESKTOP = { width: 1280, height: 800, mobile: false };
const PHONE = { width: 390, height: 844, mobile: true };
const READY_MS = 150_000;
// Software rendering draws a full-size body at a few frames a second, and the
// camera's glide after a drag is counted in frames.
const SETTLE_MS = 90_000;

// Static server ------------------------------------------------------------------

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".glb": "model/gltf-binary",
  ".png": "image/png",
  ".css": "text/css; charset=utf-8",
};

function fileFor(pathname) {
  if (pathname === "/fixture/body.glb") return GLB;
  if (pathname === "/fixture/body.manifest.json") return MANIFEST;
  const abs = normalize(join(ROOT, decodeURIComponent(pathname)));
  return abs.startsWith(ROOT) ? abs : null;
}

function startServer() {
  const server = createServer((req, res) => {
    const file = fileFor(new URL(req.url, "http://localhost").pathname);
    if (!file || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", "cache-control": "no-store" });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

// Chromium and its DevTools protocol ---------------------------------------------------

function findChromium() {
  const candidates = [process.env.CHROMIUM, "chromium", "chromium-browser", "google-chrome", "google-chrome-stable"].filter(Boolean);
  for (const c of candidates) {
    if (c.includes("/")) {
      if (existsSync(c)) return c;
      continue;
    }
    for (const dir of (process.env.PATH || "").split(":")) {
      if (dir && existsSync(join(dir, c))) return join(dir, c);
    }
  }
  return null;
}

function launchChromium(bin, profile) {
  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--hide-scrollbars",
    "--no-first-run",
    "--disable-extensions",
    "--disable-background-networking",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "--window-size=1280,800",
    "about:blank",
  ];
  const child = spawn(bin, args, { stdio: ["ignore", "ignore", "pipe"] });
  return new Promise((ok, fail) => {
    let text = "";
    const timer = setTimeout(() => fail(new Error(`Chromium did not announce DevTools:\n${text}`)), 30_000);
    child.stderr.on("data", (chunk) => {
      text += chunk;
      const m = /DevTools listening on (ws:\/\/\S+)/.exec(text);
      if (m) {
        clearTimeout(timer);
        ok({ child, wsUrl: m[1] });
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      fail(new Error(`Chromium exited with ${code}:\n${text}`));
    });
  });
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.addEventListener("message", (e) => {
      const msg = JSON.parse(e.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { ok, fail } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) fail(new Error(`${msg.error.message}${msg.error.data ? `: ${msg.error.data}` : ""}`));
        else ok(msg.result);
      } else if (msg.method) {
        for (const l of this.listeners) l(msg);
      }
    });
  }

  static connect(url) {
    return new Promise((ok, fail) => {
      const ws = new WebSocket(url);
      ws.addEventListener("open", () => ok(new Cdp(ws)));
      ws.addEventListener("error", () => fail(new Error(`could not connect to ${url}`)));
    });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.id;
    const message = { id, method, params };
    if (sessionId) message.sessionId = sessionId;
    this.ws.send(JSON.stringify(message));
    return new Promise((ok, fail) => this.pending.set(id, { ok, fail }));
  }

  on(handler) {
    this.listeners.push(handler);
  }

  close() {
    this.ws.close();
  }
}

class Page {
  constructor(cdp, sessionId) {
    this.cdp = cdp;
    this.sessionId = sessionId;
    this.problems = [];
    cdp.on((msg) => {
      if (msg.sessionId !== sessionId) return;
      if (msg.method === "Runtime.exceptionThrown") {
        const d = msg.params.exceptionDetails;
        this.problems.push(`exception: ${d.exception?.description || d.text}`);
      } else if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
        this.problems.push(`console.error: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
      } else if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
        this.problems.push(`${msg.params.entry.source}: ${msg.params.entry.text} ${msg.params.entry.url || ""}`);
      }
    });
  }

  static async open(cdp) {
    const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
    const page = new Page(cdp, sessionId);
    await page.send("Page.enable");
    await page.send("Runtime.enable");
    await page.send("Log.enable");
    return page;
  }

  send(method, params) {
    return this.cdp.send(method, params, this.sessionId);
  }

  async evaluate(expression, awaitPromise = false) {
    const { result, exceptionDetails } = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
    return result.value;
  }

  /** "ready" once the model is drawn, or "error: ..." when the page gave up. */
  async loaded(what) {
    const started = Date.now();
    for (;;) {
      const status = await this.evaluate(
        "(() => { const d = document.documentElement?.dataset; return !d ? 'wait' : d.error ? 'error: ' + d.error : d.ready === '1' && window.viewer ? 'ready' : 'wait'; })()",
      );
      if (status !== "wait") return status;
      if (Date.now() - started > READY_MS) throw new Error(`${what} was not ready after ${READY_MS / 1000} s`);
      await new Promise((r) => setTimeout(r, 250));
    }
  }

  async open(url, viewport) {
    await this.send("Emulation.setDeviceMetricsOverride", { ...viewport, deviceScaleFactor: 1 });
    await this.send("Emulation.setTouchEmulationEnabled", { enabled: viewport.mobile, maxTouchPoints: 5 });
    await this.send("Page.navigate", { url });
    return this.loaded(url);
  }

  async show(url, viewport) {
    const status = await this.open(url, viewport);
    if (status !== "ready") throw new Error(`${url}\n${status}`);
    await this.settle();
  }

  /** Choose files in a file input, as a person would in the file dialog. */
  async chooseFiles(selector, files) {
    const { result } = await this.send("Runtime.evaluate", { expression: `document.querySelector(${JSON.stringify(selector)})` });
    await this.send("DOM.setFileInputFiles", { files, objectId: result.objectId });
  }

  /** Wait until the camera and the fades have come to rest and the frame is drawn. */
  async settle() {
    const started = Date.now();
    await this.frames();
    while (!(await this.evaluate("window.viewer.settled"))) {
      if (Date.now() - started > SETTLE_MS) throw new Error(`the view did not settle after ${SETTLE_MS / 1000} s`);
      await new Promise((r) => setTimeout(r, 50));
    }
    await this.frames();
  }

  frames() {
    return this.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))", true);
  }

  /** The value of a JavaScript expression in the page, through JSON. */
  async json(expression) {
    const value = await this.evaluate(`JSON.stringify(${expression} ?? null)`);
    return JSON.parse(value);
  }

  /** The middle of an element, in window coordinates. */
  async middle(selector) {
    const r = await this.json(`(() => { const r = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height}; })()`);
    assert.ok(r.w > 0 && r.h > 0, `${selector} is laid out`);
    return r;
  }

  mouse(type, x, y, extra = {}) {
    return this.send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons: 1, clickCount: 1, ...extra });
  }

  async moveTo(x, y) {
    await this.mouse("mouseMoved", x, y, { button: "none", buttons: 0, clickCount: 0 });
  }

  async click(x, y) {
    await this.moveTo(x, y);
    await this.mouse("mousePressed", x, y);
    await this.mouse("mouseReleased", x, y, { buttons: 0 });
  }

  async drag(x0, y0, x1, y1, steps = 8) {
    await this.moveTo(x0, y0);
    await this.mouse("mousePressed", x0, y0);
    for (let i = 1; i <= steps; i++) {
      await this.mouse("mouseMoved", x0 + ((x1 - x0) * i) / steps, y0 + ((y1 - y0) * i) / steps, { clickCount: 0 });
    }
    await this.mouse("mouseReleased", x1, y1, { buttons: 0 });
  }

  wheel(x, y, deltaY) {
    return this.mouse("mouseWheel", x, y, { button: "none", buttons: 0, clickCount: 0, deltaX: 0, deltaY });
  }

  async tap(x, y) {
    await this.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y, id: 1 }] });
    await this.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  }

  /** Two fingers on a horizontal line through (x, y), moving from `from` to `to` pixels apart. */
  async pinch(x, y, from, to, steps = 6) {
    const points = (gap) => [
      { x: x - gap / 2, y, id: 1 },
      { x: x + gap / 2, y, id: 2 },
    ];
    await this.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: points(from) });
    for (let i = 1; i <= steps; i++) {
      await this.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: points(from + ((to - from) * i) / steps) });
    }
    await this.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  }

  async key(key, windowsVirtualKeyCode, text) {
    const base = { key, code: key.length === 1 ? undefined : key, windowsVirtualKeyCode, nativeVirtualKeyCode: windowsVirtualKeyCode };
    await this.send("Input.dispatchKeyEvent", { type: text ? "keyDown" : "rawKeyDown", text, ...base });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
  }

  async screenshot(file) {
    const { data } = await this.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(file, Buffer.from(data, "base64"));
    return decodePng(readFileSync(file));
  }
}

// PNG reading ---------------------------------------------------------------------

/** {width, height, rgba} from an 8-bit, non-interlaced RGB or RGBA PNG. */
export function decodePng(buf) {
  assert.equal(buf.subarray(0, 8).toString("hex"), "89504e470d0a1a0a", "PNG signature");
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idat = [];
  while (pos < buf.length) {
    const length = buf.readUInt32BE(pos);
    const type = buf.subarray(pos + 4, pos + 8).toString("ascii");
    const data = buf.subarray(pos + 8, pos + 8 + length);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      assert.equal(data[8], 8, "8 bits per channel");
      colorType = data[9];
      assert.ok(colorType === 2 || colorType === 6, `RGB or RGBA, got colour type ${colorType}`);
      assert.equal(data[12], 0, "not interlaced");
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
      break;
    }
    pos += 12 + length;
  }
  const bpp = colorType === 6 ? 4 : 3;
  const stride = width * bpp;
  const raw = inflateSync(Buffer.concat(idat));
  const rgba = Buffer.alloc(width * height * 4);
  let prev = Buffer.alloc(stride);
  let at = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[at++];
    const line = Buffer.from(raw.subarray(at, at + stride));
    at += stride;
    for (let i = 0; i < stride; i++) {
      const left = i >= bpp ? line[i - bpp] : 0;
      const up = prev[i];
      const upLeft = i >= bpp ? prev[i - bpp] : 0;
      let add = 0;
      if (filter === 1) add = left;
      else if (filter === 2) add = up;
      else if (filter === 3) add = (left + up) >> 1;
      else if (filter === 4) {
        const p = left + up - upLeft;
        const pa = Math.abs(p - left);
        const pb = Math.abs(p - up);
        const pc = Math.abs(p - upLeft);
        add = pa <= pb && pa <= pc ? left : pb <= pc ? up : upLeft;
      }
      line[i] = (line[i] + add) & 0xff;
    }
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      rgba[o] = line[x * bpp];
      rgba[o + 1] = line[x * bpp + 1];
      rgba[o + 2] = line[x * bpp + 2];
      rgba[o + 3] = bpp === 4 ? line[x * bpp + 3] : 255;
    }
    prev = line;
  }
  return { width, height, rgba };
}

/** Number of distinct colours (5 bits per channel) inside a region; flat pages have few. */
export function distinctColours(img, region) {
  const seen = new Set();
  for (let y = region.y0; y < region.y1; y++) {
    for (let x = region.x0; x < region.x1; x++) {
      const o = (y * img.width + x) * 4;
      seen.add(((img.rgba[o] >> 3) << 10) | ((img.rgba[o + 1] >> 3) << 5) | (img.rgba[o + 2] >> 3));
    }
  }
  return seen.size;
}

/** Share of pixels in a region that are not the region's background (its top-left pixel). */
export function drawnShare(img, region) {
  const o0 = (region.y0 * img.width + region.x0) * 4;
  const bg = [img.rgba[o0], img.rgba[o0 + 1], img.rgba[o0 + 2]];
  let drawn = 0;
  let total = 0;
  for (let y = region.y0; y < region.y1; y++) {
    for (let x = region.x0; x < region.x1; x++) {
      const o = (y * img.width + x) * 4;
      const d = Math.abs(img.rgba[o] - bg[0]) + Math.abs(img.rgba[o + 1] - bg[1]) + Math.abs(img.rgba[o + 2] - bg[2]);
      if (d > 12) drawn++;
      total++;
    }
  }
  return drawn / total;
}

/** Share of pixels in a region that differ clearly between two same-size images. */
export function differingShare(a, b, region) {
  assert.equal(a.width, b.width);
  assert.equal(a.height, b.height);
  let differ = 0;
  let total = 0;
  for (let y = region.y0; y < region.y1; y++) {
    for (let x = region.x0; x < region.x1; x++) {
      const o = (y * a.width + x) * 4;
      const d = Math.abs(a.rgba[o] - b.rgba[o]) + Math.abs(a.rgba[o + 1] - b.rgba[o + 1]) + Math.abs(a.rgba[o + 2] - b.rgba[o + 2]);
      if (d > 30) differ++;
      total++;
    }
  }
  return differ / total;
}

// The tests ----------------------------------------------------------------------

const manifestNodes = () => JSON.parse(readFileSync(MANIFEST, "utf8")).nodes;

/** The structure the tests isolate: VIEWER_ISOLATE, else the sternum, else the first node. */
function isolateId() {
  if (process.env.VIEWER_ISOLATE) return process.env.VIEWER_ISOLATE;
  const ids = manifestNodes().map((n) => n.name);
  return ids.includes("FMA_7485") ? "FMA_7485" : ids[0];
}

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("the viewer in headless Chromium", () => {
  let server;
  let browser;
  let cdp;
  let page;
  let profile;
  let base;
  const report = { glb: GLB, manifest: MANIFEST, shots: {} };
  let isolatedByUrl = null; // the screenshot of the structure isolated through the URL

  before(async () => {
    assert.ok(existsSync(GLB), `no GLB at ${GLB}`);
    assert.ok(existsSync(MANIFEST), `no manifest at ${MANIFEST}`);
    const bin = findChromium();
    assert.ok(bin, "no Chromium found; set CHROMIUM=/path/to/chromium");
    mkdirSync(OUT, { recursive: true });
    server = await startServer();
    base = `http://127.0.0.1:${server.address().port}/viewer/index.html?glb=/fixture/body.glb&manifest=/fixture/body.manifest.json`;
    profile = mkdtempSync(join(tmpdir(), "viewer-chromium-"));
    browser = await launchChromium(bin, profile);
    cdp = await Cdp.connect(browser.wsUrl);
    page = await Page.open(cdp);
  });

  after(async () => {
    writeFileSync(join(OUT, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
    cdp?.close();
    if (browser) {
      browser.child.kill("SIGTERM");
      await new Promise((r) => browser.child.once("exit", r));
    }
    server?.close();
    // Chromium's helpers may still be writing as it exits; a leftover profile must not fail the run.
    try {
      if (profile) rmSync(profile, { recursive: true, force: true, maxRetries: 20, retryDelay: 250 });
    } catch (err) {
      console.warn(`could not remove ${profile}: ${err.message}`);
    }
  });

  /** The 3D view's pixels, as the page lays them out. */
  async function stageRegion() {
    const r = await page.json("(() => { const r = document.querySelector('#stage').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; })()");
    return { x0: Math.ceil(r[0]), y0: Math.ceil(r[1]), x1: Math.floor(r[2]), y1: Math.floor(r[3]) };
  }

  const viewerState = () => page.json("window.viewer.state");
  const nodeOf = (id) => page.json(`window.viewer.nodes.find((n) => n.id === ${JSON.stringify(id)})`);
  const pointOf = (id) => page.json(`window.viewer.screenPoint(${JSON.stringify(id)})`);
  const textOf = (selector) => page.evaluate(`document.querySelector(${JSON.stringify(selector)})?.textContent ?? ''`);
  const listed = () => page.json("[...document.querySelectorAll('#structures .item .item-name')].map((e) => e.textContent)");

  /** Capture the page as it is now and check the picture is a drawn, shaded body. */
  async function capture(name, viewport, extra = {}) {
    const file = join(OUT, `${name}.png`);
    const img = await page.screenshot(file);
    const state = await viewerState();
    const layoutInfo = await page.json(
      "({unresolved: window.viewer.layout.unresolved, adjusted: window.viewer.layout.adjusted.length, derived: window.viewer.layout.derivedCount, structures: window.viewer.nodes.length, errors: window.viewer.errors})",
    );
    const region = await stageRegion();
    const drawn = drawnShare(img, region);
    const colours = distinctColours(img, region);
    report.shots[name] = { file, ...extra, viewport, region, state, layout: layoutInfo, drawn, colours, problems: [...page.problems] };
    assert.ok(existsSync(file) && statSync(file).size > 0, `${file} exists`);
    assert.equal(img.width, viewport.width);
    assert.equal(img.height, viewport.height);
    assert.deepEqual(page.problems, [], `${name}: the page reported problems`);
    assert.deepEqual(layoutInfo.errors, [], `${name}: window.viewer.errors`);
    assert.ok(drawn > 0.02, `${name} is not blank (${(drawn * 100).toFixed(1)} % of the 3D view is drawn, ${colours} colours)`);
    assert.ok(colours > 40, `${name} is shaded, not flat (${colours} colours)`);
    return { img, state, layout: layoutInfo, region };
  }

  /** Open the page with the given parameters and capture it: nothing is clicked. */
  async function shoot(name, query, viewport) {
    page.problems.length = 0;
    await page.show(`${base}&${query}`, viewport);
    return capture(name, viewport, { query });
  }

  test("screenshots from URL parameters: t = 0, 0.5 and 1, and one isolated structure", async () => {
    const isolate = isolateId();
    const shots = {
      t0: await shoot("t0", "t=0", DESKTOP),
      t05: await shoot("t05", "t=0.5", DESKTOP),
      t1: await shoot("t1", "t=1", DESKTOP),
      isolate: await shoot("isolate", `t=0&isolate=${encodeURIComponent(isolate)}`, DESKTOP),
    };
    assert.equal(shots.t0.state.t, 0);
    assert.equal(shots.t05.state.t, 0.5);
    assert.equal(shots.t1.state.t, 1);
    assert.equal(shots.isolate.state.isolated, isolate);
    isolatedByUrl = shots.isolate;
    assert.deepEqual(shots.t1.layout.unresolved, [], "no two bounds overlap at t = 1");
    const names = Object.keys(shots);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const share = differingShare(shots[names[i]].img, shots[names[j]].img, shots.t0.region);
        report.shots[names[j]][`differs_from_${names[i]}`] = share;
        assert.ok(share > 0.01, `${names[i]} and ${names[j]} differ in the 3D view (${(share * 100).toFixed(2)} % of pixels)`);
      }
    }
    // The isolated part's card: plain name, one line, where the shape comes from.
    const node = await nodeOf(isolate);
    assert.equal(await textOf("#info h2"), node.plain);
    assert.equal(await textOf("#info h2 + p"), node.about);
    assert.equal(await textOf("#info .source"), node.modelled ? "modelled" : "segmented from CT");
    assert.equal(await textOf("#show-all"), "Show all");
    // Everything fetched from outside the page is in the import map's integrity list.
    const outside = await page.json(`performance.getEntriesByType("resource").map((e) => e.name).filter((n) => !n.startsWith(location.origin) && !n.startsWith("blob:") && !n.startsWith("data:"))`);
    const pinned = await page.json(`Object.keys(JSON.parse(document.querySelector('script[type="importmap"]').textContent).integrity)`);
    assert.ok(outside.length >= 3, `three.js came from the CDN (${outside.length} files)`);
    assert.deepEqual(outside.filter((url) => !pinned.includes(url)), [], "every outside file has a pinned hash");
  });

  test("the clinician view names parts anatomically and shows their IDs", async () => {
    const isolate = isolateId();
    await shoot("clinician", `t=0.4&clinician=1&isolate=${encodeURIComponent(isolate)}`, DESKTOP);
    const node = await nodeOf(isolate);
    assert.equal(await textOf("#info h2"), node.label);
    const card = await textOf("#info");
    assert.ok(card.includes(node.term), `the card shows ${node.term}`);
    assert.ok((await listed()).includes(node.label), "the list uses anatomical names");
    assert.ok((await textOf("#structures")).includes(node.term), "the list shows IDs");
  });

  test("mouse and keyboard: hover, click, Escape, Show all, drag, wheel, arrows, search, hide, clinician", async () => {
    const id = isolateId();
    page.problems.length = 0;
    await page.show(`${base}&t=0`, DESKTOP);
    const node = await nodeOf(id);
    const plainNames = (await page.json("window.viewer.nodes.map((n) => n.plain)")).sort();
    assert.deepEqual((await listed()).sort(), plainNames, "the main view lists every part by its plain name");
    const mainView = `${await textOf("header")} ${await textOf("#info")} ${await textOf("#structures")} ${await textOf("footer")}`;
    assert.ok(!/(FMA|UBERON)[:_]\d+/.test(mainView), "no IDs in the main view");
    assert.ok(!(await listed()).includes(node.label) || node.label.toLowerCase() === node.plain.toLowerCase(), "no anatomical names in the main view");

    // Hover names the part under the mouse.
    let p = await pointOf(id);
    assert.equal(await page.json(`window.viewer.structureAt(${p.x}, ${p.y})`), id, "the structure is drawn where screenPoint says");
    await page.moveTo(p.x, p.y);
    await page.frames();
    assert.equal(await page.evaluate("document.querySelector('#hover').hidden"), false);
    assert.equal(await textOf("#hover"), node.plain);
    await capture("hover", DESKTOP, { query: "t=0, the mouse over the part" });

    // A click isolates it; the rest fade to ghosts and the camera frames it.
    const whole = await page.json("window.viewer.camera");
    await page.click(p.x, p.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, id);
    assert.equal(await textOf("#info h2"), node.plain);
    assert.equal(await page.evaluate(`document.querySelector('.item[data-id="${id}"]').getAttribute('aria-pressed')`), "true");
    const close = await page.json("window.viewer.camera");
    assert.ok(distance(close.position, close.target) < distance(whole.position, whole.target), "the camera moved in to frame the part");
    assert.ok(new URL(await page.evaluate("location.href")).searchParams.get("isolate") === id, "the address carries the view");
    // The click ends in the same picture as the link: the rest are ghosts, the part fills the view.
    await page.moveTo(5, 5);
    const clicked = await capture("isolate-click", DESKTOP, { query: "t=0, clicked" });
    if (isolatedByUrl) {
      const share = differingShare(clicked.img, isolatedByUrl.img, clicked.region);
      report.shots["isolate-click"].differs_from_isolate = share;
      assert.ok(share < 0.01, `a click isolates as the link does (${(share * 100).toFixed(2)} % of pixels differ)`);
    }

    // Escape brings the body back.
    await page.key("Escape", 27);
    await page.settle();
    assert.equal((await viewerState()).isolated, null);
    assert.equal(await textOf("#info h2"), "The whole body");

    // Click again, then the "Show all" button.
    p = await pointOf(id);
    await page.click(p.x, p.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, id);
    const showAll = await page.middle("#show-all");
    await page.click(showAll.x, showAll.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, null);

    // A drag turns the body and is not a click; the wheel zooms.
    const stage = await page.middle("#stage");
    const before = await page.json("window.viewer.camera");
    await page.drag(stage.x, stage.y, stage.x + 180, stage.y + 40);
    await page.settle();
    const turned = await page.json("window.viewer.camera");
    assert.equal((await viewerState()).isolated, null, "a drag does not isolate");
    assert.ok(distance(before.position, turned.position) > 0.05 * distance(before.position, before.target), "the drag turned the camera");
    await page.wheel(stage.x, stage.y, -400);
    await page.settle();
    const zoomed = await page.json("window.viewer.camera");
    assert.ok(distance(zoomed.position, zoomed.target) < distance(turned.position, turned.target), "the wheel zoomed in");

    // Keyboard on the 3D view: arrows turn, plus and minus zoom.
    await page.evaluate("document.querySelector('#view').focus()");
    await page.key("ArrowLeft", 37);
    await page.settle();
    const keyed = await page.json("window.viewer.camera");
    assert.ok(distance(keyed.position, zoomed.position) > 0.05 * distance(zoomed.position, zoomed.target), "an arrow key turned the camera");
    await page.key("-", 189, "-");
    await page.settle();
    const out = await page.json("window.viewer.camera");
    assert.ok(distance(out.position, out.target) > distance(keyed.position, keyed.target), "minus zoomed out");

    // Keyboard on the slider.
    await page.evaluate("document.querySelector('#take-apart').focus()");
    for (let i = 0; i < 5; i++) await page.key("ArrowRight", 39);
    await page.settle();
    assert.equal((await viewerState()).t, 0.05);
    assert.equal(await textOf("#take-apart-value"), "5% apart");
    await page.key("End", 35);
    await page.settle();
    assert.equal((await viewerState()).t, 1);
    assert.equal(await textOf("#take-apart-value"), "fully apart");
    await page.key("Home", 36);
    await page.settle();
    assert.equal((await viewerState()).t, 0);

    // Search by a plain word, then by the anatomical name.
    await page.key("/", 191, "/");
    assert.equal(await page.evaluate("document.activeElement.id"), "search", "/ focuses the search box");
    await page.send("Input.insertText", { text: node.plain.toLowerCase() });
    await page.frames();
    assert.deepEqual(await listed(), [node.plain]);
    await page.key("Escape", 27);
    assert.equal(await page.evaluate("document.querySelector('#search').value"), "", "Escape clears the search");
    await page.send("Input.insertText", { text: node.label });
    await page.frames();
    assert.ok((await listed()).includes(node.plain), "the anatomical name finds the part too");
    await page.key("Escape", 27);
    assert.equal((await listed()).length, plainNames.length);

    // Hide the part's system, then show it again.
    const box = await page.middle(`#structures input[data-system="${node.system}"]`);
    await page.click(box.x, box.y);
    await page.settle();
    assert.deepEqual((await viewerState()).hidden, [node.system]);
    p = await pointOf(id);
    assert.notEqual(await page.json(`window.viewer.structureAt(${p.x}, ${p.y})`), id, "a hidden system cannot be picked");
    const boxAgain = await page.middle(`#structures input[data-system="${node.system}"]`);
    await page.click(boxAgain.x, boxAgain.y);
    await page.settle();
    assert.deepEqual((await viewerState()).hidden, []);

    // The clinician toggle swaps the names and adds the IDs; a second click swaps back.
    const toggle = await page.middle("#clinician");
    await page.click(toggle.x, toggle.y);
    await page.frames();
    assert.equal((await viewerState()).clinician, true);
    assert.ok((await listed()).includes(node.label));
    assert.ok((await textOf("#structures")).includes(node.term));
    await page.click(toggle.x, toggle.y);
    await page.frames();
    assert.deepEqual((await listed()).sort(), plainNames);

    // A part picked from the list; a modelled one says so.
    const modelled = manifestNodes().find((n) => n.modelled)?.name;
    if (modelled) {
      await page.evaluate(`document.querySelector('.item[data-id="${modelled}"]').scrollIntoView({block: 'center'})`);
      const item = await page.middle(`.item[data-id="${modelled}"]`);
      await page.click(item.x, item.y);
      await page.settle();
      assert.equal((await viewerState()).isolated, modelled);
      assert.equal(await textOf("#info .source"), "modelled");
    }
    assert.deepEqual(page.problems, []);
  });

  test("with no body to load the page offers a file picker, and shows the files chosen", async () => {
    page.problems.length = 0;
    const origin = new URL(base).origin;
    const status = await page.open(`${origin}/viewer/index.html`, DESKTOP);
    assert.match(status, /^error: /, "there is no body.glb next to the page");
    assert.equal(await page.evaluate("document.querySelector('#picker').hidden"), false, "the picker is offered");
    assert.equal(await textOf("#status"), "No body to show yet.");
    await page.screenshot(join(OUT, "picker.png"));
    // Only the GLB: the page asks for both.
    await page.chooseFiles("#files", [GLB]);
    assert.match(await textOf("#status"), /both/);
    page.problems.length = 0;
    await page.chooseFiles("#files", [GLB, MANIFEST]);
    assert.equal(await page.loaded("the chosen files"), "ready");
    await page.settle();
    assert.equal(await page.evaluate("document.querySelector('#picker').hidden"), true);
    assert.equal(await page.json("window.viewer.nodes.length"), manifestNodes().length);
    await capture("picked", DESKTOP, { query: "files chosen in the picker" });

    // A manifest that is not one stops with a message and the picker again.
    const bad = `${origin}/viewer/index.html?glb=/fixture/body.glb&manifest=/viewer/package.json`;
    assert.match(await page.open(bad, DESKTOP), /^error: Could not show the body: manifest: /);
    assert.equal(await page.evaluate("document.querySelector('#picker').hidden"), false);
  });

  test("touch at phone width (390 px): tap, the part's card, the parts sheet, pinch", async () => {
    const id = isolateId();
    const whole = await shoot("phone-whole", "t=0", PHONE);
    assert.equal(whole.state.isolated, null);
    const node = await nodeOf(id);
    const noSideways = async (when) =>
      assert.equal(await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), false, `no sideways scrolling ${when}`);
    await noSideways("with the whole body");
    assert.equal(await page.evaluate("getComputedStyle(document.querySelector('#panel')).visibility"), "hidden", "the list is put away on a phone");
    for (const selector of ["#take-apart", "#panel-toggle", "#clinician"]) {
      const r = await page.middle(selector === "#clinician" ? ".switch" : selector);
      assert.ok(r.h >= 44, `${selector} is at least 44 px tall for a finger (${r.h})`);
    }
    const slider = await page.middle("#take-apart");
    assert.ok(slider.w >= 300, `the slider spans the phone's width (${slider.w} px)`);

    // Tap a part: its card appears under the 3D view, which still shows all of it.
    let p = await pointOf(id);
    await page.tap(p.x, p.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, id);
    assert.equal(await textOf("#info h2"), node.plain);
    assert.equal(await textOf("#info .source"), node.modelled ? "modelled" : "segmented from CT");
    const stage = await stageRegion();
    const card = await page.middle("#info");
    assert.ok(card.y - card.h / 2 >= stage.y1 - 1, "the card does not cover the 3D view");
    await capture("phone", PHONE, { query: "t=0, tapped" });
    await noSideways("with one part");

    // Pinch out zooms in.
    const before = await page.json("window.viewer.camera");
    const centre = { x: (stage.x0 + stage.x1) / 2, y: (stage.y0 + stage.y1) / 2 };
    await page.pinch(centre.x, centre.y, 80, 240);
    await page.settle();
    const after = await page.json("window.viewer.camera");
    assert.ok(distance(after.position, after.target) < distance(before.position, before.target), "pinching out zoomed in");
    assert.equal((await viewerState()).isolated, id, "a pinch is not a tap");

    // "Show all" on the card.
    const showAll = await page.middle("#show-all");
    await page.tap(showAll.x, showAll.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, null);

    // The parts sheet: open it, pick a part, and the sheet makes way for it.
    const parts = await page.middle("#panel-toggle");
    await page.tap(parts.x, parts.y);
    await new Promise((r) => setTimeout(r, 400));
    await page.settle();
    assert.equal(await page.evaluate("document.body.classList.contains('panel-open')"), true);
    await capture("phone-parts", PHONE, { query: "t=0, parts sheet open" });
    await noSideways("with the parts sheet");
    const first = await page.json("document.querySelector('#structures .item').dataset.id");
    const item = await page.middle("#structures .item");
    await page.tap(item.x, item.y);
    await page.settle();
    assert.equal((await viewerState()).isolated, first);
    assert.equal(await page.evaluate("document.body.classList.contains('panel-open')"), false, "the sheet closes to show the part");
    assert.deepEqual(page.problems, []);
  });
});
