/* IntelX project page, live demo, part 2: the page.

   The visitor picks an example, draws a floor plan or uploads one; on Start the
   page downloads the U-Net (ONNX; 26 MB, its convolution weights stored as int8
   with a scale per 64 weights) and ONNX Runtime Web's WebGPU or CPU build, all
   from its own static/live/, and hands them to static/js/live_worker.js, which
   runs the model and the explorer (static/js/live_core.js, the simulator and
   the planner) off the page's thread and sends back snapshots to draw.
   OpenCV.js (findContours for the frontier search) comes from the CDN.

   Opened from localhost with ?dev=<port>, predictions go to the development
   server in live_src/dev_infer_server.py instead (same checkpoint, local GPU). */
(function () {
  'use strict';
  var el = document.getElementById('live');
  if (!el || !window.IntelXCore) return;
  var C = window.IntelXCore, RES = C.RES, PI = Math.PI;
  var $ = function (s) { return el.querySelector(s); }, $$ = function (s) { return Array.prototype.slice.call(el.querySelectorAll(s)); };
  var ORT_LOCAL = 'static/live/ort/', JS_DIR = 'static/js/';
  var CV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';
  var MODEL = 'static/live/model/', MAPS = 'static/live/maps/';
  var DRAW_W = 600, DRAW_H = 400;            // a new drawing: 30 x 20 m
  var MAX_SIDE = 1600;                        // 80 m at 0.05 m per cell
  var devPort = (function () {
    var p = new URLSearchParams(location.search).get('dev');
    return p && /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? p : null;
  })();

  var canvas = $('.live__canvas'), ctx = canvas.getContext('2d'), stage = $('.live__stage');
  var statusEl = $('.live__status'), runBtn = $('.live__run'), resetBtn = $('.live__reset');
  var hudCov = $('.live__cov'), hudPath = $('.live__path'), hudMap = $('.live__mapname'), hudMeta = $('.live__mapmeta');
  var hudDec = $('.live__decision'), chartHost = $('.live__chart');
  var S = {
    map: null, title: '', spawn: null, mode: 'edit', tool: 'wall', undo: [],
    ex: null, running: false, engine: null, loading: null, idleEvery: 6,
    showField: true, showMarks: true, dirty: true, layer: document.createElement('canvas'),
    field: document.createElement('canvas'), fieldFor: null, drag: null, fan: new Float64Array(C.NB)
  };
  S.lctx = S.layer.getContext('2d'); S.field.width = S.field.height = 400; S.fctx = S.field.getContext('2d');

  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function setStatus(s) { statusEl.textContent = s; }
  function fmt(v, d) { return v.toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }

  /* --------------------------------------------------------------- maps */
  function grayOf(img) {
    var c = document.createElement('canvas'); c.width = img.naturalWidth || img.width; c.height = img.naturalHeight || img.height;
    var g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0);
    var d = g.getImageData(0, 0, c.width, c.height).data, v = new Uint8Array(c.width * c.height);
    for (var i = 0; i < v.length; i++) v[i] = Math.round(0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]);
    return { w: c.width, h: c.height, v: v };
  }
  // ROS map_server values (254 free, 0 wall, 205 unknown); otherwise dark is
  // wall, light is floor and mid grey (a SLAM map's unknown) is unknown
  function occFromGray(g) {
    var n = g.v.length, ros = 0, floor = 0, occ = new Int8Array(n), i;
    for (i = 0; i < n; i += 7) { var x = g.v[i]; if (x === 254 || x === 0 || x === 205) ros++; if (x === 254) floor++; }
    ros = floor > 0 && ros > 0.9 * (n / 7);
    for (i = 0; i < n; i++) {
      var v = g.v[i];
      occ[i] = ros ? (v === 254 ? 0 : v === 0 ? 1 : -1) : (v < 100 ? 1 : v > 200 ? 0 : -1);
    }
    return C.sealByAdjacent(occ, g.w, g.h);
  }
  function setMap(map, title, spawn) {
    S.map = map; S.title = title; S.spawn = spawn || autoSpawn(map); S.undo = [];
    dropRun(); S.mode = 'edit';
    S.layer.width = map.W; S.layer.height = map.H; paintAll();
    fit(); updateHud(); drawChart(); S.dirty = true;
  }
  function freeArea(map) { var n = 0; for (var i = 0; i < map.occ.length; i++) if (map.occ[i] === 0) n++; return n * RES * RES; }
  function autoSpawn(map) {
    var W = map.W, H = map.H, sx = 0, sy = 0, n = 0, x, y;
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) if (map.occ[y * W + x] === 0) { sx += x; sy += y; n++; }
    if (!n) return null;
    sx /= n; sy /= n;
    var best = null, bd = Infinity;
    for (y = 0; y < H; y += 3) for (x = 0; x < W; x += 3) {
      var d = (x - sx) * (x - sx) + (y - sy) * (y - sy);
      if (d < bd && C.spawnOk(map, x, y)) { bd = d; best = [x, y]; }
    }
    if (!best) for (y = 0; y < H && !best; y++) for (x = 0; x < W; x++) if (map.occ[y * W + x] === 0) { best = [x, y]; break; }
    return best ? [best[0], best[1], openHeading(map, best[0], best[1])] : null;
  }
  function openHeading(map, x, y) {        // face the longest free ray
    var best = 0, bl = -1;
    for (var k = 0; k < 16; k++) {
      var a = k * PI / 8, l = 0;
      while (l < 200) { var px = Math.trunc(x + l * Math.cos(a)), py = Math.trunc(y + l * Math.sin(a)); if (px < 0 || py < 0 || px >= map.W || py >= map.H || map.occ[py * map.W + px] === 1) break; l++; }
      if (l > bl) { bl = l; best = a; }
    }
    return best;
  }
  function loadExample(ex) {
    var im = new Image();
    im.onload = function () {
      var g = grayOf(im);
      setMap({ W: g.w, H: g.h, occ: occFromGray(g) }, ex.title, ex.spawn.slice());
      hudMeta.textContent = fmt(ex.area_m2) + ' m² · ' + ex.group + ' · Seed ' + ex.seed + ' Spawn';
      setStatus(ex.title + ': the spawn is the one the paper\'s seed ' + ex.seed + ' trial used. Press Start, or switch to Draw to edit it.');
    };
    im.onerror = function () { setStatus('Could not load ' + ex.file + '.'); };
    im.src = MAPS + ex.file;
  }
  function newDrawing() {
    var occ = new Int8Array(DRAW_W * DRAW_H), x, y;
    for (x = 0; x < DRAW_W; x++) for (y = 0; y < 3; y++) { occ[y * DRAW_W + x] = 1; occ[(DRAW_H - 1 - y) * DRAW_W + x] = 1; }
    for (y = 0; y < DRAW_H; y++) for (x = 0; x < 3; x++) { occ[y * DRAW_W + x] = 1; occ[y * DRAW_W + DRAW_W - 1 - x] = 1; }
    setMap({ W: DRAW_W, H: DRAW_H, occ: occ }, 'Your Floor Plan', null);
    hudMeta.textContent = '30 × 20 m';
    setStatus('Draw walls and rooms, erase to open doorways, place the robot with Spawn, then press Start.');
  }
  function parsePGM(buf) {
    var b = new Uint8Array(buf), p = 0, tok = [];
    function word() {
      for (;;) {
        while (p < b.length && /\s/.test(String.fromCharCode(b[p]))) p++;
        if (b[p] === 35) { while (p < b.length && b[p] !== 10) p++; continue; }
        break;
      }
      var s = ''; while (p < b.length && !/\s/.test(String.fromCharCode(b[p]))) s += String.fromCharCode(b[p++]);
      return s;
    }
    var magic = word(), w = +word(), h = +word(), mx = +word();
    if (magic !== 'P5' && magic !== 'P2') throw new Error('not a PGM file');
    var v = new Uint8Array(w * h), sc = 255 / mx;
    if (magic === 'P5') { p++; for (var i = 0; i < w * h; i++) v[i] = mx > 255 ? Math.round(((b[p + 2 * i] << 8) | b[p + 2 * i + 1]) * sc) : Math.round(b[p + i] * sc); }
    else for (var j = 0; j < w * h; j++) v[j] = Math.round(+word() * sc);
    return { w: w, h: h, v: v };
  }
  // resample to 0.05 m per cell: a cell is wall if any source pixel in it is,
  // else free if any is free, else unknown; then crop to the building
  function rescale(map, mpp) {
    var f = mpp / RES;
    var W = Math.max(1, Math.round(map.W * f)), H = Math.max(1, Math.round(map.H * f)), occ = new Int8Array(W * H).fill(-1);
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var sx0 = Math.floor(x / f), sx1 = Math.max(sx0 + 1, Math.floor((x + 1) / f)), sy0 = Math.floor(y / f), sy1 = Math.max(sy0 + 1, Math.floor((y + 1) / f));
      var wall = false, free = false;
      for (var yy = sy0; yy < Math.min(sy1, map.H) && !wall; yy++) for (var xx = sx0; xx < Math.min(sx1, map.W); xx++) {
        var v = map.occ[yy * map.W + xx]; if (v === 1) { wall = true; break; } if (v === 0) free = true;
      }
      occ[y * W + x] = wall ? 1 : free ? 0 : -1;
    }
    return crop({ W: W, H: H, occ: occ });
  }
  function crop(map) {
    var W = map.W, H = map.H, x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) if (map.occ[y * W + x] === 0) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (x1 < 0) return null;
    x0 = Math.max(0, x0 - 10); y0 = Math.max(0, y0 - 10); x1 = Math.min(W - 1, x1 + 10); y1 = Math.min(H - 1, y1 + 10);
    var w = x1 - x0 + 1, h = y1 - y0 + 1, occ = new Int8Array(w * h);
    for (var yy = 0; yy < h; yy++) occ.set(map.occ.subarray((yy + y0) * W + x0, (yy + y0) * W + x0 + w), yy * w);
    return { W: w, H: h, occ: occ };
  }
  function loadUpload(file) {
    var mpp = parseFloat($('.live__mpp').value) || RES;
    var done = function (g) {
      if (!g) return;
      var m = rescale({ W: g.w, H: g.h, occ: occFromGray(g) }, mpp);
      if (!m) { setStatus('No free space found in that image. Walls should be dark and floor light.'); return; }
      if (m.W > MAX_SIDE || m.H > MAX_SIDE) { setStatus('That map is ' + fmt(m.W * RES) + ' × ' + fmt(m.H * RES) + ' m; the demo takes up to 80 m a side. Check the metres per pixel.'); return; }
      setMap(m, file.name.replace(/\.[^.]+$/, ''), null);
      hudMeta.textContent = fmt(freeArea(m)) + ' m² · ' + fmt(m.W * RES) + ' × ' + fmt(m.H * RES) + ' m';
      setStatus('Loaded at ' + mpp + ' m per pixel. Check the spawn (the Spawn tool moves it), then press Start.');
    };
    var r = new FileReader();
    if (/\.pgm$/i.test(file.name)) {
      r.onload = function () { try { done(parsePGM(r.result)); } catch (e) { setStatus('Could not read that PGM: ' + e.message); } };
      r.readAsArrayBuffer(file);
    } else {
      r.onload = function () { var im = new Image(); im.onload = function () { done(grayOf(im)); }; im.onerror = function () { setStatus('Could not read that image.'); }; im.src = r.result; };
      r.readAsDataURL(file);
    }
  }

  /* ------------------------------------------------------------- drawing */
  function paintAll() { paintBox(0, 0, S.map.W, S.map.H); }
  function paintBox(x0, y0, x1, y1) {
    var ex = S.mode === 'run' ? S.ex : null, m = ex ? ex.map : S.map;
    if (!m) return;
    var W = m.W;
    x0 = Math.max(0, x0 | 0); y0 = Math.max(0, y0 | 0); x1 = Math.min(W, Math.ceil(x1)); y1 = Math.min(m.H, Math.ceil(y1));
    if (x1 <= x0 || y1 <= y0) return;
    var img = S.lctx.createImageData(x1 - x0, y1 - y0), d = img.data;
    var U = S.cU, G = S.cG, F = S.cF, Wl = S.cW, i = 0, occ = m.occ, H = m.H;
    // only wall cells that face free space are drawn as walls, as in the
    // race replays; the solid mass inside and around the building stays quiet
    var surf = function (k, x, y) {
      for (var dy = -2; dy <= 2; dy++) for (var dx = -2; dx <= 2; dx++) {
        var nx = x + dx, ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < W && ny < H && occ[ny * W + nx] !== 1) return true;
      }
      return false;
    };
    for (var y = y0; y < y1; y++) for (var x = x0; x < x1; x++, i += 4) {
      var k = y * W + x, o = occ[k], c;
      if (ex) c = ex.explored[k] ? (o === 0 ? F : Wl) : (o === 1 ? (surf(k, x, y) ? G : U) : U);
      else c = o === 1 ? (surf(k, x, y) ? Wl : U) : o === 0 ? F : U;
      d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
    }
    S.lctx.putImageData(img, x0, y0);
  }
  function rgb(s) { var h = s.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function palette() { S.cU = rgb(css('--map-unknown') || '#E9ECF1'); S.cG = rgb(css('--map-ghost') || '#D3D8E0'); S.cF = [255, 255, 255]; S.cW = rgb(css('--map-wall') || '#1E232C'); }
  function fit() {
    if (!S.map) return;
    var cw = stage.clientWidth || 800, a = S.map.W / S.map.H;
    var maxH = Math.min(window.innerHeight * 0.68, 720), h = Math.min(maxH, cw / a), dpr = Math.min(window.devicePixelRatio || 1, 2);
    stage.style.height = Math.max(240, h).toFixed(0) + 'px';
    var ch = stage.clientHeight;
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
    S.dpr = dpr; S.s = Math.min(cw / S.map.W, ch / S.map.H);
    S.ox = (cw - S.map.W * S.s) / 2; S.oy = (ch - S.map.H * S.s) / 2;
    S.dirty = true;
  }
  var VIR = [[68, 1, 84], [71, 45, 123], [59, 82, 139], [44, 114, 142], [33, 145, 140], [40, 174, 128], [94, 201, 98], [173, 220, 48], [253, 231, 37]];
  function viridis(t) { var f = Math.min(0.9999, Math.max(0, t)) * 8, i = f | 0, u = f - i, a = VIR[i], b = VIR[i + 1]; return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]; }
  // the field as the method video drew it: viridis, opacity rising with
  // intensity (gamma 0.8, at most 0.78), nothing below 4% of its maximum;
  // colours from a 256-step table, since on a GPU new fields come quickly
  var LUT = null;
  function paintField(last) {
    if (S.fieldFor === last) return;
    S.fieldFor = last;
    if (!LUT) {
      LUT = new Uint8ClampedArray(1024);
      for (var v = 0; v < 256; v++) { var c = viridis(v / 255); LUT[v * 4] = c[0]; LUT[v * 4 + 1] = c[1]; LUT[v * 4 + 2] = c[2]; LUT[v * 4 + 3] = 255 * Math.pow(v / 255, 0.8) * 0.78; }
    }
    var h = last.heat, mx = 0, i;
    for (i = 0; i < h.length; i++) if (h[i] > mx) mx = h[i];
    var img = S.fctx.createImageData(400, 400), d = img.data;
    if (mx > 0) for (i = 0; i < h.length; i++) {
      var n = h[i] / mx; if (n < 0.04) continue;
      var q = ((n * 255 + 0.5) | 0) * 4, o = i * 4;
      d[o] = LUT[q]; d[o + 1] = LUT[q + 1]; d[o + 2] = LUT[q + 2]; d[o + 3] = LUT[q + 3];
    }
    S.fctx.putImageData(img, 0, 0);
  }
  function cross(x, y, r, w, col) { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.lineWidth = w; ctx.strokeStyle = col; ctx.stroke(); }
  function draw() {
    if (!S.map) return;
    var dpr = S.dpr, s = S.s, px = 1 / s, m = S.map, ex = S.mode === 'run' ? S.ex : null;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = css('--map-unknown') || '#E9ECF1'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(s * dpr, 0, 0, s * dpr, S.ox * dpr, S.oy * dpr);
    ctx.imageSmoothingEnabled = s < 1.5; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(S.layer, 0, 0);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (!ex) {
      if (s * 20 > 6) {                                        // 1 m grid while editing
        ctx.strokeStyle = 'rgba(26, 29, 35, .06)'; ctx.lineWidth = px; ctx.beginPath();
        for (var gx = 0; gx <= m.W; gx += 20) { ctx.moveTo(gx, 0); ctx.lineTo(gx, m.H); }
        for (var gy = 0; gy <= m.H; gy += 20) { ctx.moveTo(0, gy); ctx.lineTo(m.W, gy); }
        ctx.stroke();
      }
      if (S.drag && S.drag.preview) {
        var p = S.drag.preview; ctx.strokeStyle = 'rgba(213, 94, 0, .8)'; ctx.lineWidth = 3;
        if (S.tool === 'room') ctx.strokeRect(Math.min(p[0], p[2]), Math.min(p[1], p[3]), Math.abs(p[2] - p[0]), Math.abs(p[3] - p[1]));
        else { ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[2], p[3]); ctx.stroke(); }
      }
      if (S.spawn) robot(S.spawn[0], S.spawn[1], S.spawn[2], px, true);
      return;
    }
    var last = ex.last, i;
    if (S.showField && last) {
      paintField(last);
      ctx.save(); ctx.translate(last.pose[0], last.pose[1]); ctx.rotate(PI / 2 - last.pose[2]);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(S.field, -200, -200); ctx.restore();
    }
    if (ex.path && ex.pathIndex < ex.path.length) {
      ctx.setLineDash([4 * px, 4 * px]); ctx.strokeStyle = 'rgba(26, 29, 35, .45)'; ctx.lineWidth = 1.2 * px;
      ctx.beginPath(); ctx.moveTo(ex.pose[0], ex.pose[1]);
      for (i = ex.pathIndex; i < ex.path.length; i++) ctx.lineTo(ex.path[i][0], ex.path[i][1]);
      ctx.stroke(); ctx.setLineDash([]);
    }
    // the laser, as the race replays draw it
    if (!ex.done) {
      C.scanRanges(m, ex.pose[0], ex.pose[1], ex.pose[2], S.fan);
      ctx.beginPath(); ctx.moveTo(ex.pose[0], ex.pose[1]);
      for (i = 0; i < C.NB; i += 6) { var a = C.ANGLES0[i] + ex.pose[2]; ctx.lineTo(ex.pose[0] + S.fan[i] * Math.cos(a), ex.pose[1] + S.fan[i] * Math.sin(a)); }
      ctx.closePath(); ctx.fillStyle = css('--map-scan') || 'rgba(213, 94, 0, .16)'; ctx.fill();
    }
    ctx.strokeStyle = css('--m-ours') || '#D55E00'; ctx.globalAlpha = .85; ctx.lineWidth = 2 * px;
    ctx.beginPath(); ex.trail.forEach(function (q, j) { if (j) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
    ctx.lineTo(ex.pose[0], ex.pose[1]); ctx.stroke(); ctx.globalAlpha = 1;
    if (S.showMarks) marks(ex, px);
    star(m, px);
    robot(ex.pose[0], ex.pose[1], ex.pose[2], px, false);
  }
  function star(m, px) {
    var sp = S.spawn; if (!sp) return;
    var r = 6 * px; ctx.beginPath();
    for (var k = 0; k < 10; k++) { var a = -PI / 2 + k * PI / 5, rr = k % 2 ? r * .45 : r; ctx.lineTo(sp[0] + Math.cos(a) * rr, sp[1] + Math.sin(a) * rr); }
    ctx.closePath(); ctx.fillStyle = '#2E9E4F'; ctx.fill();
  }
  function robot(x, y, a, px, ghost) {
    var r = 7 * px; ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    ctx.beginPath(); ctx.moveTo(r * 1.35, 0); ctx.lineTo(-r * .9, r * .95); ctx.lineTo(-r * .35, 0); ctx.lineTo(-r * .9, -r * .95); ctx.closePath();
    ctx.fillStyle = css('--m-ours') || '#D55E00'; ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 1.6 * px; ctx.globalAlpha = ghost ? .9 : 1;
    ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function marks(ex, px) {
    var rank = [css('--ov-rank-1'), css('--ov-rank-2'), css('--ov-rank-3'), css('--ov-rank-4')], last = ex.last;
    ex.nodes.forEach(function (nd) {
      nd.components.forEach(function (c) {
        var col = rank[(c.rank - 1) % 4]; ctx.globalAlpha = c.explored ? .4 : 1;
        ctx.beginPath(); ctx.moveTo(nd.x, nd.y); ctx.lineTo(c.x, c.y); ctx.lineWidth = 1.1 * px; ctx.strokeStyle = col; ctx.stroke();
        ctx.beginPath(); ctx.arc(c.x, c.y, 3 * px, 0, 2 * PI); ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = px; ctx.strokeStyle = '#FFFFFF'; ctx.stroke();
      });
      ctx.globalAlpha = 1;
      var q = 5 * px; ctx.beginPath(); ctx.moveTo(nd.x, nd.y - q); ctx.lineTo(nd.x + q, nd.y); ctx.lineTo(nd.x, nd.y + q); ctx.lineTo(nd.x - q, nd.y); ctx.closePath();
      ctx.fillStyle = css('--ov-node-fill'); ctx.fill(); ctx.lineWidth = 1.6 * px; ctx.strokeStyle = css('--ov-node'); ctx.stroke();
    });
    if (last) {
      last.lc.keep.forEach(function (cc, r) {
        var c = last.lc.cent[cc], w = C.img2world(Math.trunc(c[0]), Math.trunc(c[1]), last.pose);
        ctx.beginPath(); ctx.arc(w[0], w[1], 3.6 * px, 0, 2 * PI); ctx.fillStyle = '#FFFFFF'; ctx.fill();
        ctx.lineWidth = 1.8 * px; ctx.strokeStyle = rank[r % 4]; ctx.stroke();
      });
      (last.fviz || []).forEach(function (f) { cross(f[0], f[1], 3 * px, 3.4 * px, '#FFFFFF'); cross(f[0], f[1], 3 * px, 1.7 * px, css('--ov-frontier')); });
    }
    if (ex.goal && ex.state === 'MOVE') { cross(ex.goal[0], ex.goal[1], 4 * px, 4 * px, '#FFFFFF'); cross(ex.goal[0], ex.goal[1], 4 * px, 2.2 * px, css('--ov-goal')); }
  }
  function updateHud() {
    var ex = S.mode === 'run' ? S.ex : null;
    hudMap.textContent = S.title || 'Map';
    hudCov.innerHTML = (ex ? ex.coverage().toFixed(1) : '0.0') + '<small>%</small>';
    hudPath.innerHTML = (ex ? fmt(ex.pathLen) : '0') + '<small> m</small>';
    if (!ex) { hudDec.hidden = true; return; }
    var nav = ex.lastNav;
    hudDec.hidden = false;
    hudDec.textContent = (nav ? nav.src : 'Starting') + ' · ' + ex.nodes.length + (ex.nodes.length === 1 ? ' junction' : ' junctions') +
      ' · ' + fmt(ex.preds) + ' predictions' + (ex.inferMs ? ' · ' + ex.inferMs.toFixed(0) + ' ms' : '');
  }

  /* --------------------------------------------------------------- chart */
  var NS = 'http://www.w3.org/2000/svg';
  function mk(tag, at, parent) { var e = document.createElementNS(NS, tag); for (var k in at) e.setAttribute(k, at[k]); parent.appendChild(e); return e; }
  function drawChart() {
    var ex = S.mode === 'run' ? S.ex : null, Wv = 1000, Hv = 220, L = 46, R = 18, T = 12, B = 36;
    var dmax = Math.max(20, ex ? ex.pathLen * 1.15 : 20);
    var steps = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000], st = steps.find(function (q) { return dmax / q <= 7; }) || 2000;
    dmax = Math.ceil(dmax / st) * st;
    var X = function (v) { return L + v / dmax * (Wv - L - R); }, Y = function (v) { return T + (1 - v / 100) * (Hv - T - B); };
    var svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', '0 0 ' + Wv + ' ' + Hv);
    var g = mk('g', { 'class': 'grid' }, svg);
    [0, 25, 50, 75, 100].forEach(function (v) { mk('line', { x1: L, x2: Wv - R, y1: Y(v), y2: Y(v) }, g); mk('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v + '%'; });
    for (var v = 0; v <= dmax + 1e-6; v += st) { mk('line', { x1: X(v), x2: X(v), y1: T, y2: Hv - B }, g); mk('text', { x: X(v), y: Hv - B + 18, 'text-anchor': 'middle' }, svg).textContent = fmt(v); }
    mk('text', { x: (L + Wv - R) / 2, y: Hv - 2, 'text-anchor': 'middle' }, svg).textContent = 'Distance Travelled [m]';
    mk('line', { 'class': 'target', x1: L, x2: Wv - R, y1: Y(99), y2: Y(99) }, svg);
    mk('text', { x: L + 6, y: Y(99) - 6 }, svg).textContent = '99% Target';
    if (ex && ex.curve.length > 1) {
      var step = Math.max(1, Math.ceil(ex.curve.length / 700)), pts = [];
      for (var i = 0; i < ex.curve.length; i += step) pts.push(X(ex.curve[i][0]).toFixed(1) + ',' + Y(ex.curve[i][1]).toFixed(1));
      pts.push(X(ex.pathLen).toFixed(1) + ',' + Y(ex.coverage()).toFixed(1));
      mk('polyline', { 'class': 'curve is-ours', points: pts.join(' '), stroke: 'var(--m-ours)' }, svg);
      mk('circle', { 'class': 'head', cx: X(ex.pathLen), cy: Y(ex.coverage()), r: 4.5, fill: 'var(--m-ours)' }, svg);
    }
    chartHost.innerHTML = ''; chartHost.appendChild(svg);
  }

  /* --------------------------------------------------------------- model */
  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement('script'); s.src = src; s.async = true; s.crossOrigin = 'anonymous';
      s.onload = res; s.onerror = function () { rej(new Error('Could not load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function loadCV() {
    if (window.cv && window.cv.Mat) return Promise.resolve(true);
    return loadScript(CV_URL).then(function () {
      // never `await` a thenable Emscripten module: its then() resolves to itself
      if (window.cv instanceof Promise) return window.cv.then(function (m) { window.cv = m; return true; });
      if (window.cv && window.cv.Mat) return true;
      return new Promise(function (res) {
        window.cv.onRuntimeInitialized = function () { res(true); };
        setTimeout(function () { res(!!(window.cv && window.cv.Mat)); }, 30000);
      });
    }).catch(function () { return false; });
  }
  async function fetchProgress(url, onBytes) {
    var r = await fetch(url);
    if (!r.ok) throw new Error('Could not load ' + url);
    if (!r.body || !r.body.getReader) { var b = new Uint8Array(await r.arrayBuffer()); onBytes(b.length); return b; }
    var rd = r.body.getReader(), parts = [], n = 0;
    for (;;) { var x = await rd.read(); if (x.done) break; parts.push(x.value); n += x.value.length; onBytes(x.value.length); }
    var out = new Uint8Array(n), o = 0; parts.forEach(function (p) { out.set(p, o); o += p.length; });
    return out;
  }
  // pack_png.py stores the model's bytes in the RGB channels of opaque PNGs,
  // since the page's host serves only web file types; read them back exactly
  async function pngBytes(url, nbytes, onBytes) {
    var raw = await fetchProgress(url, onBytes);
    var bmp = await createImageBitmap(new Blob([raw], { type: 'image/png' }), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    var c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
    var g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(bmp, 0, 0);
    var d = g.getImageData(0, 0, c.width, c.height).data, out = new Uint8Array(nbytes);
    for (var i = 0, j = 0; j < nbytes; i += 4) {
      out[j++] = d[i];
      if (j < nbytes) out[j++] = d[i + 1];
      if (j < nbytes) out[j++] = d[i + 2];
    }
    return out;
  }
  function getJSON(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error('Could not load ' + url); return r.json(); }); }
  function abs(u) { return new URL(u, location.href).href; }
  function joined(parts) {
    var n = 0, o = 0; parts.forEach(function (p) { n += p.length; });
    var all = new Uint8Array(n); parts.forEach(function (p) { all.set(p, o); o += p.length; });
    return all;
  }
  function sizeOf(parts) { return parts.reduce(function (a, p) { return a + p.size; }, 0); }
  async function predictWith(ort, sess, imgs) {
    var out = [];
    for (var k = 0; k < imgs.length; k++) {        // one at a time: modest memory on any GPU
      var x = new Float32Array(160000), im = imgs[k];
      for (var j = 0; j < 160000; j++) x[j] = (im[j] / 255 - 0.5) / 0.5;
      var r = await sess.run({ input: new ort.Tensor('float32', x, [1, 1, 400, 400]) });
      out.push(r.heatmap.data);
    }
    return out;
  }

  // Everything comes from the page's own files: the model, and ONNX Runtime
  // Web's WebGPU and CPU builds packed by pack_runtime.py, so nothing is
  // fetched from a CDN. WebGPU when the browser offers an
  // adapter and the session passes make_selftest.py's check, the CPU
  // otherwise; either runs in live_worker.js together with the explorer. Only
  // a host that refuses workers gets the CPU build on the page's own thread.
  // ?ort=cpu skips WebGPU and ?worker=0 the worker.
  async function loadEngine() {
    if (devPort) return devEngine();
    var q = location.search, noGpu = /[?&]ort=(cpu|local)\b/.test(q), useWorker = !/[?&]worker=0\b/.test(q) && typeof Worker === 'function';
    setStatus('Loading the model…');
    var gpu = false;
    if (!noGpu && navigator.gpu) { try { gpu = !!(await navigator.gpu.requestAdapter()); } catch (e) { /* no adapter */ } }
    var info = await getJSON(MODEL + 'export.json'), builds = await getJSON(ORT_LOCAL + 'builds.json');
    var test = gpu ? await getJSON(MODEL + 'selftest.json') : null;
    var keys = gpu ? ['gpu', 'cpu'] : ['cpu'], bins = {}, got = 0, total = sizeOf(info.png_parts) + (test ? test.input.size : 0);
    var tick = function (n) { got += n; setStatus('Downloading the model and its runtime: ' + (Math.min(got, total) / 1e6).toFixed(1) + ' of ' + (total / 1e6).toFixed(1) + ' MB'); };
    var build = function (key) {
      if (!bins[key]) {
        total += sizeOf(builds[key].parts);
        bins[key] = Promise.all(builds[key].parts.map(function (p) { return pngBytes(ORT_LOCAL + p.file, p.bytes, tick); })).then(joined);
      }
      return bins[key];
    };
    var h = useWorker ? bootWorker(keys[0], builds[keys[0]]) : null;     // it loads OpenCV.js meanwhile
    build(keys[0]);
    var model = joined(await Promise.all(info.png_parts.map(function (p) { return pngBytes(MODEL + p.file, p.bytes, tick); })));
    var input = test ? await pngBytes(MODEL + test.input.file, test.input.bytes, tick) : null;
    var why = null;
    if (useWorker) for (var i = 0; i < keys.length; i++) {
      var key = keys[i];
      try {
        if (!h) h = bootWorker(key, builds[key]);
        if (h.failed) throw h.failed;
        var bin = await build(key);
        setStatus('Starting the model on ' + (key === 'gpu' ? 'WebGPU' : 'the CPU (WebAssembly)') + '…');
        var d = await initWorker(h, bin, model, info, key === 'gpu' ? { input: input, pool: test.pool, ref: test.ref, tol: test.tol } : null);
        return { kind: key === 'gpu' ? 'webgpu' : 'wasm', worker: h.w, cv: d.cv, test: d.test, why: why };
      } catch (e) {
        if (window.console) console.warn('IntelX demo: ' + key + ' worker:', e);
        if (key === 'gpu') why = /reference/.test(e.message) ? 'gave a wrong result in its self-test here' : 'did not start here';
        h = null;
      }
    }
    var eng = await mainEngine(builds.cpu, await build('cpu'), model, info);
    eng.why = why; eng.refused = useWorker;
    return eng;
  }
  function bootWorker(key, build) {
    var w;
    try { w = new Worker(JS_DIR + 'live_worker.js'); } catch (e) { return { failed: e }; }
    var h = { w: w, key: key, settled: false };
    h.ready = new Promise(function (res, rej) {
      var end = function (e) { if (h.settled) return; h.settled = true; w.terminate(); rej(e); };
      w.onerror = function (ev) { if (ev && ev.preventDefault) ev.preventDefault(); end(new Error((ev && ev.message) || 'the worker did not start')); };
      w.onmessage = function (ev) {
        var d = ev.data;
        if (d.type === 'status') setStatus(d.text);
        else if (d.type === 'error') end(new Error(d.message));
        else if (d.type === 'ready' && !h.settled) { h.settled = true; w.onmessage = onWorker; w.onerror = onWorkerError; res(d); }
      };
    });
    h.ready.catch(function () { /* awaited in loadEngine */ });
    w.postMessage({ type: 'boot', ort: abs(ORT_LOCAL + build.script), cv: CV_URL });
    return h;
  }
  function initWorker(h, bin, model, info, test) {
    // copies: the worker takes over these buffers, and a fallback needs the originals
    var g = model.slice(0, info.graph_bytes), b = model.slice(info.graph_bytes), wb = bin.slice();
    h.w.postMessage({ type: 'init', ep: h.key === 'gpu' ? 'webgpu' : 'wasm', wasmPaths: abs(ORT_LOCAL), wasm: wb.buffer, graph: g.buffer, blob: b.buffer,
      info: { weights_fp32_file_bytes: info.weights_fp32_file_bytes, q8_manifest: info.q8_manifest }, test: test }, [wb.buffer, g.buffer, b.buffer]);
    return h.ready;
  }
  async function mainEngine(build, bin, model, info) {
    setStatus('Starting the model on the CPU (WebAssembly)…');
    var cvP = loadCV();
    window.ort = undefined;
    await loadScript(ORT_LOCAL + build.script);
    var ort = window.ort;
    ort.env.wasm.wasmPaths = abs(ORT_LOCAL); ort.env.wasm.wasmBinary = bin;
    if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;
    var sess = await ort.InferenceSession.create(model.slice(0, info.graph_bytes), { executionProviders: ['wasm'],
      externalData: [{ path: 'weights.bin', data: C.rebuildWeights(info, model.slice(info.graph_bytes)) }] });
    return { kind: 'wasm', worker: null, cv: await cvP, predict: function (imgs) { return predictWith(ort, sess, imgs); } };
  }
  async function devEngine() {
    var cvP = loadCV(), base = 'http://localhost:' + devPort;
    await fetch(base + '/');
    return { kind: 'dev', worker: null, cv: await cvP, predict: async function (imgs) {
      var body = new Uint8Array(imgs.length * 160000); imgs.forEach(function (im, k) { body.set(im, k * 160000); });
      var r = await fetch(base + '/infer', { method: 'POST', headers: { 'X-Batch': String(imgs.length) }, body: body });
      var y = new Float32Array(await r.arrayBuffer()), out = [];
      for (var k = 0; k < imgs.length; k++) out.push(y.subarray(k * 160000, (k + 1) * 160000));
      return out;
    } };
  }

  /* ----------------------------------------------------------------- run */
  function reachable(map, sp) {               // drop free space the robot can never reach
    var W = map.W, H = map.H, seen = new Uint8Array(W * H), st = [Math.trunc(sp[1]) * W + Math.trunc(sp[0])], dropped = 0;
    seen[st[0]] = 1;
    while (st.length) {
      var k = st.pop(), x = k % W, y = (k - x) / W;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        var nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        var nk = ny * W + nx; if (seen[nk] || map.occ[nk] === 1) continue; seen[nk] = 1; st.push(nk);
      }
    }
    var occ = Int8Array.from(map.occ);
    for (var i = 0; i < occ.length; i++) if (occ[i] === 0 && !seen[i]) { occ[i] = 1; dropped++; }
    return { map: { W: W, H: H, occ: occ }, dropped: dropped * RES * RES };
  }
  async function startRun() {
    if (!S.map || !S.spawn) { setStatus('Place the robot with the Spawn tool first.'); return; }
    if (!C.spawnOk(S.map, S.spawn[0], S.spawn[1])) setStatus('The spawn is close to a wall; IntelX will start there anyway.');
    if (!S.engine) {
      if (!S.loading) S.loading = loadEngine();
      runBtn.disabled = true;
      try { S.engine = await S.loading; }
      catch (e) { S.loading = null; runBtn.disabled = false; setStatus('The model could not start: ' + e.message); return; }
      runBtn.disabled = false;
      if (S.engine.kind === 'wasm' && !S.paceChosen) {     // a second or more per prediction on the CPU
        S.idleEvery = 12;
        $$('.live__pace button').forEach(function (x) { x.setAttribute('aria-pressed', x.getAttribute('data-px') === '12' ? 'true' : 'false'); });
      }
    }
    var fresh = !S.ex || S.ex.done, w = S.engine.worker;
    if (fresh) {
      var r = reachable(S.map, S.spawn);
      S.mode = 'run'; S.fieldFor = null;
      S.note = r.dropped > 1 ? ' ' + fmt(r.dropped) + ' m² the robot cannot reach is left out.' : '';
      newRun(r.map); paintAll();
    }
    S.running = true; setRunLabel();
    var m = S.engine, where = m.kind === 'webgpu' ? 'WebGPU, on this device\'s graphics card' : m.kind === 'dev' ? 'the local development GPU'
      : 'the CPU (WebAssembly), ' + (m.why ? 'since WebGPU ' + m.why : 'since this browser has no WebGPU');
    setStatus('IntelX is exploring on ' + where + '.' + (m.kind === 'wasm' && m.refused ? ' This host does not allow a background worker, so the page pauses during each prediction.' : '') +
      (m.cv ? '' : ' OpenCV.js did not load, so frontiers are found approximately.') + S.note);
    if (w) { if (!fresh) { w.postMessage({ type: 'pace', idleEvery: S.idleEvery }); w.postMessage({ type: 'resume' }); } }
    else { S.ex.idleEvery = S.idleEvery; loop(); }
  }
  // the worker's run is mirrored here by an object with the Explorer fields
  // the drawing reads, updated from its snapshots
  function newRun(map) {
    S.runId = (S.runId || 0) + 1;
    var w = S.engine.worker;
    if (!w) { S.ex = new C.Explorer(map, S.spawn.slice(), S.engine.predict, { idleEvery: S.idleEvery }); return; }
    S.ex = { map: map, explored: new Uint8Array(map.W * map.H), pose: S.spawn.slice(), pathLen: 0, preds: 0, inferMs: 0, cov: 0,
      done: null, state: 'NEED', goal: null, path: null, pathIndex: 0, trail: [[S.spawn[0], S.spawn[1]]], curve: [[0, 0]],
      nodes: [], last: null, lastNav: null, coverage: function () { return this.cov; } };
    var occ = Int8Array.from(map.occ);
    w.postMessage({ type: 'start', runId: S.runId, W: map.W, H: map.H, occ: occ, spawn: S.spawn.slice(), idleEvery: S.idleEvery }, [occ.buffer]);
  }
  function onWorker(ev) {
    var d = ev.data, ex = S.ex, i;
    if (d.type === 'error') { workerFailed(d.message); return; }
    if (d.type !== 'tick' || !ex || d.runId !== S.runId) return;
    ex.pose = d.pose; ex.pathLen = d.pathLen; ex.preds = d.preds; ex.inferMs = d.inferMs; ex.cov = d.cov; ex.done = d.done;
    ex.state = d.state; ex.goal = d.goal; ex.pathIndex = d.pathIndex; ex.nodes = d.nodes; ex.lastNav = d.nav ? { src: d.nav } : null;
    if ('path' in d) ex.path = d.path;
    if (d.last) ex.last = d.last;
    for (i = 0; i < d.trail.length; i++) ex.trail.push(d.trail[i]);
    for (i = 0; i < d.curve.length; i++) ex.curve.push(d.curve[i]);
    if (d.box) {
      var b = d.box, bw = b[2] - b[0], W = ex.map.W;
      for (var y = b[1]; y < b[3]; y++) ex.explored.set(d.exp.subarray((y - b[1]) * bw, (y - b[1] + 1) * bw), y * W + b[0]);
      paintBox(b[0], b[1], b[2], b[3]);
    }
    S.dirty = true;
    var now = performance.now();
    if (d.done || now - (S.hudAt || 0) > 250) { updateHud(); drawChart(); S.hudAt = now; }
    if (d.done) finishRun(ex);
  }
  function onWorkerError(ev) { if (ev && ev.preventDefault) ev.preventDefault(); workerFailed((ev && ev.message) || 'the worker stopped'); }
  function workerFailed(msg) {
    if (S.engine && S.engine.worker) S.engine.worker.terminate();
    S.engine = null; S.loading = null;
    if (S.ex && !S.ex.done) S.ex.done = 'error';
    S.running = false; setRunLabel();
    setStatus('The demo stopped: ' + msg + '. Press Run Again to start over.');
  }
  function stopRun() {
    if (S.running && S.engine && S.engine.worker) S.engine.worker.postMessage({ type: 'pause' });
    S.running = false; setRunLabel();
  }
  function dropRun() {                         // the run is over for good: a new map, an edit, Reset
    stopRun();
    if (S.ex && S.engine && S.engine.worker) S.engine.worker.postMessage({ type: 'drop' });
    S.ex = null; setRunLabel();
  }
  function setRunLabel() {
    runBtn.querySelector('span').textContent = S.running ? 'Pause' : (S.ex && !S.ex.done && S.mode === 'run' ? 'Resume' : (S.ex && S.ex.done ? 'Run Again' : 'Start'));
    runBtn.setAttribute('aria-pressed', S.running ? 'true' : 'false');
  }
  async function loop() {                      // the explorer on the page's own thread
    var ex = S.ex, lastHud = 0;
    while (S.running && ex === S.ex && !ex.done) {
      var t0 = performance.now();
      while (performance.now() - t0 < 24 && S.running && !ex.done) await ex.iter();
      var b = ex.revealed; if (b) { ex.revealed = null; paintBox(b[0], b[1], b[2], b[3]); }
      S.dirty = true;
      if (performance.now() - lastHud > 250) { updateHud(); drawChart(); lastHud = performance.now(); }
      // next frame, or a short timer when the tab is hidden and frames stop
      await new Promise(function (r) { var done = false, go = function () { if (!done) { done = true; r(); } }; requestAnimationFrame(go); setTimeout(go, 60); });
    }
    if (ex === S.ex && ex.done) finishRun(ex);
  }
  function finishRun(ex) {
    paintAll(); updateHud(); drawChart(); S.running = false; setRunLabel(); S.dirty = true;
    var cov = ex.coverage();
    setStatus(ex.done === 'complete'
      ? 'IntelX reached ' + cov.toFixed(1) + '% coverage after ' + fmt(ex.pathLen, 1) + ' m, with ' + fmt(ex.preds) + ' predictions and ' + ex.nodes.length + ' junctions.'
      : 'IntelX stopped at ' + cov.toFixed(1) + '% after ' + fmt(ex.pathLen, 1) + ' m: ' + ({ exhausted: 'no reachable frontier was left', stalled: 'it stopped making progress', 'boxed in': 'it was boxed in' }[ex.done] || ex.done) + '.');
  }

  /* -------------------------------------------------------------- editing */
  function cell(e) {
    var r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left - S.ox) / S.s, (e.clientY - r.top - S.oy) / S.s];
  }
  function snapPt(p) { return [Math.round(p[0] / 5) * 5, Math.round(p[1] / 5) * 5]; }
  function axisSnap(a, b) {
    var dx = b[0] - a[0], dy = b[1] - a[1], ang = Math.abs(Math.atan2(dy, dx)) % (PI / 2);
    if (ang < 0.17 || ang > PI / 2 - 0.17) return Math.abs(dx) > Math.abs(dy) ? [b[0], a[1]] : [a[0], b[1]];
    return b;
  }
  function stamp(x, y, r, v) {
    var m = S.map, x0 = Math.max(0, Math.round(x - r)), x1 = Math.min(m.W - 1, Math.round(x + r)), y0 = Math.max(0, Math.round(y - r)), y1 = Math.min(m.H - 1, Math.round(y + r));
    for (var yy = y0; yy <= y1; yy++) for (var xx = x0; xx <= x1; xx++) m.occ[yy * m.W + xx] = v;
  }
  function line(a, b, r, v) {
    var n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2));
    for (var i = 0; i <= n; i++) stamp(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, r, v);
  }
  function pushUndo() { S.undo.push(Int8Array.from(S.map.occ)); if (S.undo.length > 30) S.undo.shift(); }
  function editable() {
    if (S.mode === 'run') {
      if (S.running) return false;
      dropRun(); S.mode = 'edit'; paintAll(); updateHud(); drawChart(); setRunLabel();
    }
    return true;
  }
  canvas.addEventListener('pointerdown', function (e) {
    if (!S.map || S.panel !== 'draw' || !editable()) return;
    var p = cell(e);
    if (S.tool === 'spawn') {
      var x = Math.trunc(p[0]), y = Math.trunc(p[1]);
      if (x >= 0 && y >= 0 && x < S.map.W && y < S.map.H && S.map.occ[y * S.map.W + x] === 0) {
        S.spawn = [x, y, openHeading(S.map, x, y)];
        setStatus(C.spawnOk(S.map, x, y) ? 'Spawn placed. Press Start.' : 'Spawn placed, but within 0.55 m of a wall; the paper\'s spawns keep more clearance.');
      } else setStatus('The robot has to start on the floor.');
      S.dirty = true; return;
    }
    canvas.setPointerCapture(e.pointerId);
    pushUndo();
    S.drag = { start: S.tool === 'erase' ? p : snapPt(p), last: p };
    if (S.tool === 'erase') { stamp(p[0], p[1], 9, 0); paintBox(p[0] - 12, p[1] - 12, p[0] + 13, p[1] + 13); }
    S.dirty = true;
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!S.drag) return;
    var p = cell(e);
    if (S.tool === 'erase') { line(S.drag.last, p, 9, 0); paintBox(Math.min(p[0], S.drag.last[0]) - 12, Math.min(p[1], S.drag.last[1]) - 12, Math.max(p[0], S.drag.last[0]) + 13, Math.max(p[1], S.drag.last[1]) + 13); S.drag.last = p; }
    else { var b = snapPt(p); if (S.tool === 'wall') b = axisSnap(S.drag.start, b); S.drag.preview = [S.drag.start[0], S.drag.start[1], b[0], b[1]]; }
    S.dirty = true;
  });
  function endDrag(e) {
    if (!S.drag) return;
    var d = S.drag; S.drag = null;
    if (S.tool !== 'erase' && d.preview) {
      var p = d.preview, a = [p[0], p[1]], b = [p[2], p[3]];
      if (S.tool === 'wall') line(a, b, 1, 1);
      else { line(a, [b[0], a[1]], 1, 1); line([b[0], a[1]], b, 1, 1); line(b, [a[0], b[1]], 1, 1); line([a[0], b[1]], a, 1, 1); }
      paintAll();
    }
    if (S.spawn && S.map.occ[Math.trunc(S.spawn[1]) * S.map.W + Math.trunc(S.spawn[0])] !== 0) { S.spawn = autoSpawn(S.map); setStatus('A wall covered the robot, so it moved.'); }
    S.dirty = true;
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  /* ------------------------------------------------------------ controls */
  function showPanel(name) {
    S.panel = name;
    $$('.live__src button').forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute('data-src') === name ? 'true' : 'false'); });
    $$('.live__panel').forEach(function (p) { p.hidden = p.getAttribute('data-panel') !== name; });
    canvas.classList.toggle('is-drawing', name === 'draw');
  }
  $$('.live__src button').forEach(function (b) {
    b.addEventListener('click', function () {
      if (S.running) stopRun();
      var name = b.getAttribute('data-src');
      showPanel(name);
      if (name === 'draw' && S.mode === 'run') editable();
      if (name === 'draw') setStatus('Wall draws straight walls, Room a rectangle, Erase opens doorways, Spawn places the robot.');
    });
  });
  $$('.live__tools button[data-tool]').forEach(function (b) {
    b.addEventListener('click', function () {
      S.tool = b.getAttribute('data-tool');
      $$('.live__tools button[data-tool]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
    });
  });
  $('.live__new').addEventListener('click', function () { if (!S.running) newDrawing(); });
  $('.live__undo').addEventListener('click', function () { if (!S.running && S.undo.length && editable()) { S.map.occ = S.undo.pop(); paintAll(); S.dirty = true; } });
  $('.live__file').addEventListener('change', function (e) { var f = e.target.files && e.target.files[0]; if (f) loadUpload(f); e.target.value = ''; });
  $('.live__example').addEventListener('change', function (e) { if (S.running) stopRun(); var ex = S.examples[+e.target.value]; if (ex) loadExample(ex); });
  runBtn.addEventListener('click', function () { if (S.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', function () {
    dropRun(); S.mode = 'edit'; paintAll(); updateHud(); drawChart(); setRunLabel(); S.dirty = true;
    setStatus('Reset. Press Start to explore again.');
  });
  $$('.live__pace button').forEach(function (b) {
    b.addEventListener('click', function () {
      S.idleEvery = +b.getAttribute('data-px'); S.paceChosen = true;
      if (S.engine && S.engine.worker) S.engine.worker.postMessage({ type: 'pace', idleEvery: S.idleEvery }); else if (S.ex) S.ex.idleEvery = S.idleEvery;
      $$('.live__pace button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
    });
  });
  $('.live__showfield').addEventListener('change', function (e) { S.showField = e.target.checked; S.dirty = true; });
  $('.live__showmarks').addEventListener('change', function (e) { S.showMarks = e.target.checked; S.dirty = true; });
  window.addEventListener('resize', function () { fit(); });
  if ('ResizeObserver' in window) new ResizeObserver(function () { fit(); }).observe(el);

  (function frame() { if (S.dirty) { S.dirty = false; draw(); } requestAnimationFrame(frame); })();

  palette(); showPanel('examples');
  fetch(MAPS + 'index.json').then(function (r) { return r.json(); }).then(function (list) {
    S.examples = list;
    var sel = $('.live__example');
    sel.innerHTML = list.map(function (m, i) { return '<option value="' + i + '">' + m.title + ' (' + m.group + ')</option>'; }).join('');
    loadExample(list[0]);
  }).catch(function () { newDrawing(); });
  drawChart(); setRunLabel();
  S.loadFile = loadUpload;                    // for scripted tests of the upload path
  window.IntelXLive = S;
})();
