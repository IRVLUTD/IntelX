/* IntelX project page, live demo, part 3: the worker.

   The model and the explorer (static/js/live_core.js) run here, off the page's
   thread, so the page stays responsive and the run is not paced by the screen.
   The page sends the build of ONNX Runtime Web it chose (WebGPU or CPU), the
   model and then the map; the worker loads OpenCV.js (findContours) from the
   CDN, checks a WebGPU session against a reference result before trusting it,
   and sends a snapshot of the run about fifteen times a second: the pose, the
   new trail and coverage points, the cells explored since the last snapshot
   and, when there is a new one, the predicted field. */
'use strict';
importScripts('live_core.js');
var C = self.IntelXCore;
var ort = null, sess = null, cvP = Promise.resolve(false);
var ex = null, runId = 0, running = false, looping = false, sent = null, lastSnap = 0;
var SNAP_MS = 66, SLICE_MS = 25;

function fail(e) { running = false; self.postMessage({ type: 'error', runId: runId, message: String(e && e.message || e) }); }

self.onmessage = function (e) {
  var m = e.data;
  try {
    if (m.type === 'boot') boot(m);
    else if (m.type === 'init') init(m).catch(fail);
    else if (m.type === 'start') start(m);
    else if (m.type === 'resume') { running = true; if (!looping) run(); }
    else if (m.type === 'pause') running = false;
    else if (m.type === 'pace') { if (ex) ex.idleEvery = m.idleEvery; }
    else if (m.type === 'drop') { running = false; ex = null; }
  } catch (err) { fail(err); }
};

// ONNX Runtime and OpenCV.js; OpenCV keeps loading while the page downloads
// the model
function boot(m) {
  importScripts(m.ort);
  ort = self.ort;
  cvP = new Promise(function (res) {
    try { importScripts(m.cv); } catch (e) { res(false); return; }
    var cv = self.cv;
    // never `await` the module object itself: its then() resolves to itself
    if (cv instanceof Promise) cv.then(function (mod) { self.cv = mod; res(true); }, function () { res(false); });
    else if (cv && cv.Mat) res(true);
    else if (cv) { cv.onRuntimeInitialized = function () { res(true); }; setTimeout(function () { res(!!(self.cv && self.cv.Mat)); }, 30000); }
    else res(false);
  });
}

async function init(m) {
  ort.env.wasm.wasmPaths = m.wasmPaths;      // the .mjs loader; the binary comes with the message
  ort.env.wasm.wasmBinary = m.wasm;
  ort.env.wasm.numThreads = 1;
  var weights = C.rebuildWeights(m.info, new Uint8Array(m.blob));
  sess = await ort.InferenceSession.create(new Uint8Array(m.graph), { executionProviders: [m.ep], externalData: [{ path: 'weights.bin', data: weights }] });
  weights = null;
  var test = null;
  if (m.test) {                              // make_selftest.py: the field, mean-pooled
    self.postMessage({ type: 'status', text: 'Checking WebGPU against a reference result…' });
    var t0 = performance.now(), y = (await predict([new Uint8Array(m.test.input)]))[0];
    var k = m.test.pool, n = 400 / k, d = 0;
    for (var by = 0; by < n; by++) for (var bx = 0; bx < n; bx++) {
      var s = 0;
      for (var yy = 0; yy < k; yy++) for (var xx = 0; xx < k; xx++) s += y[(by * k + yy) * 400 + bx * k + xx];
      d = Math.max(d, Math.abs(s / (k * k) - m.test.ref[by * n + bx]));
    }
    test = { diff: d, ms: performance.now() - t0 };
    if (!(d <= m.test.tol)) throw new Error('its result differs from the reference by ' + d.toExponential(2));
  }
  self.postMessage({ type: 'ready', ep: m.ep, cv: await cvP, test: test });
}

async function predict(imgs) {
  var out = [];
  for (var k = 0; k < imgs.length; k++) {      // one at a time: modest memory on any GPU
    var x = new Float32Array(160000), im = imgs[k];
    for (var j = 0; j < 160000; j++) x[j] = (im[j] / 255 - 0.5) / 0.5;
    var r = await sess.run({ input: new ort.Tensor('float32', x, [1, 1, 400, 400]) });
    out.push(r.heatmap.data);
    // a junction's eleven virtual poses take a while on the CPU: keep the
    // page's counter moving meanwhile
    if (imgs.length > 1 && ex && sent && performance.now() - lastSnap >= SNAP_MS) snapshot(ex, k + 1);
  }
  return out;
}

function start(m) {
  runId = m.runId;
  ex = new C.Explorer({ W: m.W, H: m.H, occ: m.occ }, m.spawn, predict, { idleEvery: m.idleEvery });
  sent = { trail: 0, curve: 0, path: undefined, last: undefined };
  running = true;
  if (!looping) run();
}

// a task boundary without setTimeout's clamping, so messages (pause, pace)
// get in between slices of work
var chan = new MessageChannel(), wakers = [];
chan.port1.onmessage = function () { var w = wakers.shift(); if (w) w(); };
function yieldTask() { return new Promise(function (r) { wakers.push(r); chan.port2.postMessage(0); }); }

async function run() {
  looping = true;
  try {
    while (running && ex && !ex.done) {
      var e = ex, t0 = performance.now();
      while (running && e === ex && !e.done && performance.now() - t0 < SLICE_MS) await e.iter();
      if (e === ex && (e.done || !running || performance.now() - lastSnap >= SNAP_MS)) snapshot(e);
      await yieldTask();
    }
  } catch (err) { fail(err); }
  looping = false;
}

function snapshot(e, pending) {
  lastSnap = performance.now();
  var msg = { type: 'tick', runId: runId, pose: e.pose, pathLen: e.pathLen, preds: e.preds + (pending || 0), inferMs: e.inferMs,
    cov: e.coverage(), done: e.done, state: e.state, goal: e.goal || null, pathIndex: e.pathIndex,
    nav: e.lastNav ? e.lastNav.src : null, nodes: e.nodes,
    trail: e.trail.slice(sent.trail), curve: e.curve.slice(sent.curve) }, tr = [];
  sent.trail = e.trail.length; sent.curve = e.curve.length;
  if (e.path !== sent.path) { msg.path = e.path || null; sent.path = e.path; }
  if (e.last && e.last !== sent.last) {
    var L = e.last, heat = new Float32Array(L.heat);         // its own buffer, handed over
    msg.last = { pose: L.pose, heat: heat, lc: { keep: L.lc.keep, cent: L.lc.cent }, fviz: L.fviz || null };
    tr.push(heat.buffer); sent.last = L;
  }
  var b = e.revealed;
  if (b) {
    var W = e.map.W, w = b[2] - b[0], buf = new Uint8Array(w * (b[3] - b[1]));
    for (var y = b[1]; y < b[3]; y++) buf.set(e.explored.subarray(y * W + b[0], y * W + b[2]), (y - b[1]) * w);
    msg.box = b; msg.exp = buf; tr.push(buf.buffer);
    e.revealed = null;
  }
  self.postMessage(msg, tr);
}
