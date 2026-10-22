/* IntelX project page, live demo, part 1: the simulator and the planner.

   A port of what the paper's runs executed for IntelX, so the page can run it on
   any floor plan inside the visitor's browser:
     intelx.sim.simulator  the laser (Bresenham ray tracing), the exploration
                           update, the network's north-oriented input image and
                           its virtual-pose twin
     intelx.policy         plan_local; plan_global with the frontier prepass and
                           junction cases A and B; the prediction worker's goal
                           arbitration, frontier fallback and guards
     intelx.simulation     the fixed-step state machine that drives the robot
     intelx.common         A* with wall clearance, and the plain A* used for costs
   with the benchmark's settings (intelx.config plus run_sweep_all.PIN_CONFIG).
   Python's int() truncation, numpy's linspace/percentile/histogram, scipy's
   gaussian_filter/find_peaks and OpenCV's warpAffine rounding are reproduced
   where results depend on them; OpenCV.js, when loaded, supplies findContours.

   One deliberate difference: the paper's predictor ran an idle pass after every
   pixel of motion, with the robot allowed up to 12 px ahead of it
   (PRED_MAX_LAG_PX). Each idle pass is a forward pass of the network, so here
   idle passes run every `idleEvery` px (6 by default) instead. */
(function (root) {
  'use strict';
  var PI = Math.PI, TAU = 2 * PI;

  /* ------------------------------------------------------------ settings */
  var RES = 0.05, WIN = 400, C0 = 200, RAD = 6, FOV = 220 * PI / 180, NB = 662, MAX_R = 200;
  var BUF = Math.trunc(400 * Math.SQRT2 * 1.1), BC = BUF >> 1;      // 622, 311
  var STEP_PX = 0.75;          // STEP_SIZE_M / DT * FIXED_DT * 15, in pixels per step
  var MAP_EVERY = 5;           // MAP_UPDATE_INTERVAL
  var GOAL_TOL = 3.0;          // Explorer.goal_tolerance, px
  var TARGET_PCT = 99;
  var K = {
    TOP_PCT: 96, SIGMA: 2, NBINS: 72, ANG_K: 3, PEAK_H: 0.1, MIN_AREA: 10, REL_PEAK: 0.25,
    COV_THR: 0.0005, COV_MAJ: 0.1, MAX_COMP: 4, MIN_ARC_GAP: 5,
    LOCK_DIST_M: 3.0, LOCK_TOL_DEG: 45, PREPASS_EVERY_PX: 20, SNAP_SEARCH_PX: 100,
    STICK_TOL_M: 2.0, STICK_MARGIN: 0.25, UTURN_DEG: 110, PREEMPT_MIN_GOAL_PX: 50,
    PREEMPT_MIN_SCORE: 0.005, SWITCH_MARGIN: 0.35, FALLBACK_CANDS: 10, NO_FRONTIER_EXHAUST: 5,
    MIN_GOAL_ADVANCE_PX: 15, NO_GOAL_NUDGE_PX: 60, ASTAR_END_TOL: 20, ASTAR_MARGIN: 150,
    DIRECT_MARGIN_PX: 0, START_SNAP_PX: 3, CLEARANCE_PX: 1.0 / RES, WALL_PENALTY: 3.0,
    SHORTCUT_LOOKAHEAD: 60, STALL_GOALS: 50
  };

  function linspace(a, b, n) {
    var o = new Float64Array(n), d = (b - a) / (n - 1);
    for (var i = 0; i < n; i++) o[i] = i * d + a;
    o[n - 1] = b;
    return o;
  }
  var ANGLES0 = linspace(-FOV / 2, FOV / 2, NB);

  // The robot as drawn into every network input: cv2.circle(radius 6, 200)
  // then cv2.arrowedLine(value 1) toward the top. Extracted from OpenCV.
  var MARKER = (function () {
    var s = '0,-6,200;-1,-10,1;0,-10,1;1,-10,1;-2,-9,1;-1,-9,1;0,-9,1;1,-9,1;2,-9,1;-3,-8,1;-2,-8,1;-1,-8,1;0,-8,1;1,-8,1;2,-8,1;3,-8,1;-3,-7,1;-2,-7,1;-1,-7,1;0,-7,1;1,-7,1;2,-7,1;3,-7,1;-2,-6,1;-1,-6,1;0,-6,1;1,-6,1;2,-6,1;-3,-5,200;-2,-5,200;-1,-5,1;0,-5,1;1,-5,1;2,-5,200;3,-5,200;-4,-4,200;-3,-4,200;-2,-4,200;-1,-4,1;0,-4,1;1,-4,1;2,-4,200;3,-4,200;4,-4,200;-5,-3,200;-4,-3,200;-3,-3,200;-2,-3,200;-1,-3,1;0,-3,1;1,-3,1;2,-3,200;3,-3,200;4,-3,200;5,-3,200;-5,-2,200;-4,-2,200;-3,-2,200;-2,-2,200;-1,-2,1;0,-2,1;1,-2,1;2,-2,200;3,-2,200;4,-2,200;5,-2,200;-5,-1,200;-4,-1,200;-3,-1,200;-2,-1,200;-1,-1,1;0,-1,1;1,-1,1;2,-1,200;3,-1,200;4,-1,200;5,-1,200;-6,0,200;-5,0,200;-4,0,200;-3,0,200;-2,0,200;-1,0,1;0,0,1;1,0,1;2,0,200;3,0,200;4,0,200;5,0,200;6,0,200;-5,1,200;-4,1,200;-3,1,200;-2,1,200;-1,1,200;0,1,1;1,1,200;2,1,200;3,1,200;4,1,200;5,1,200;-5,2,200;-4,2,200;-3,2,200;-2,2,200;-1,2,200;0,2,200;1,2,200;2,2,200;3,2,200;4,2,200;5,2,200;-5,3,200;-4,3,200;-3,3,200;-2,3,200;-1,3,200;0,3,200;1,3,200;2,3,200;3,3,200;4,3,200;5,3,200;-4,4,200;-3,4,200;-2,4,200;-1,4,200;0,4,200;1,4,200;2,4,200;3,4,200;4,4,200;-3,5,200;-2,5,200;-1,5,200;0,5,200;1,5,200;2,5,200;3,5,200;0,6,200';
    // the disc pixel at (0,-6) is drawn first and then covered by the arrow's
    // tip row below it, exactly as OpenCV leaves it
    return s.split(';').map(function (t) { return t.split(',').map(Number); });
  })();

  /* ----------------------------------------- the network's input geometry */
  // Every input image has the robot at (200, 200), so the per-pixel angle and
  // its histogram bin are constants. Case B uses np.histogram(bins=72), whose
  // edges span the data range rather than [-pi, pi], hence the second table.
  var ANG = new Float32Array(WIN * WIN), BIN_A = new Uint8Array(WIN * WIN), BIN_B = new Uint8Array(WIN * WIN);
  var CENTERS_A;
  (function () {
    var amin = Infinity, amax = -Infinity, i;
    for (var y = 0; y < WIN; y++) for (var x = 0; x < WIN; x++) {
      var a = Math.fround(Math.atan2(y - C0, x - C0));
      ANG[y * WIN + x] = a;
      if (a < amin) amin = a; if (a > amax) amax = a;
    }
    var eA = linspace(-PI, PI, K.NBINS + 1), eB = linspace(amin, amax, K.NBINS + 1);
    for (i = 0; i < WIN * WIN; i++) { BIN_A[i] = histBin(ANG[i], eA); BIN_B[i] = histBin(ANG[i], eB); }
    CENTERS_A = new Float64Array(K.NBINS);
    for (i = 0; i < K.NBINS; i++) CENTERS_A[i] = 0.5 * (eA[i] + eA[i + 1]);
  })();
  function histBin(v, e) {                  // numpy.histogram: [e_i, e_i+1), the last bin closed
    var n = e.length - 1;
    if (v < e[0] || v > e[n]) return 255;
    var i = Math.trunc(((v - e[0]) / (e[n] - e[0])) * n);
    if (i === n) i = n - 1;
    if (v < e[i]) i--;
    else if (i !== n - 1 && v >= e[i + 1]) i++;
    return i;
  }

  /* -------------------------------------------------------- the simulator */
  // gpu_simulate_scan / raytrace_kernel: one Bresenham ray per beam
  function scanRanges(map, x, y, th, out) {
    var W = map.W, H = map.H, occ = map.occ, x0 = Math.trunc(x), y0 = Math.trunc(y);
    for (var i = 0; i < NB; i++) {
      var ang = ANGLES0[i] + th;
      var x1 = Math.trunc(x + MAX_R * Math.cos(ang)), y1 = Math.trunc(y + MAX_R * Math.sin(ang));
      var dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
      var err = dx - dy, px = x0, py = y0, o = MAX_R, ex, ey;
      for (;;) {
        if (px < 0 || px >= W || py < 0 || py >= H) {
          ex = px - x; ey = py - y;
          var d = Math.sqrt(ex * ex + ey * ey);
          if (d < o) o = d;
          break;
        }
        if (occ[py * W + px] === 1) { ex = px - x; ey = py - y; o = Math.sqrt(ex * ex + ey * ey); break; }
        if (px === x1 && py === y1) break;
        var e2 = 2 * err;
        if (e2 > -dy) { err -= dy; px += sx; }
        if (e2 < dx) { err += dx; py += sy; }
      }
      out[i] = o;
    }
    return out;
  }
  // update_exploration_numba: march each ray in unit steps, rounding to cells
  // also widens ex.revealed, the box of cells newly explored since the page
  // last drew them ([x0, y0, x1, y1], exclusive end; null when none)
  function updateExplored(ex, x, y, th, ranges) {
    var map = ex.map, W = map.W, H = map.H, occ = map.occ, explored = ex.explored;
    var x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (var i = 0; i < NB; i++) {
      var r = ranges[i], ang = ANGLES0[i] + th, c = Math.cos(ang), s = Math.sin(ang);
      var maxSteps = Math.trunc(r) + 1;
      for (var step = 0; step < maxSteps; step++) {
        if (step > r) break;
        var px = Math.trunc(x + step * c + 0.5), py = Math.trunc(y + step * s + 0.5);
        if (px >= 0 && px < W && py >= 0 && py < H) {
          var k = py * W + px;
          if (!explored[k]) {
            explored[k] = 1; if (occ[k] === 0) ex.freeSeen++;
            if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
          }
          if (occ[k] === 1) break;
        } else break;
      }
    }
    if (x1 >= 0) {
      var b = ex.revealed;
      ex.revealed = b ? [Math.min(b[0], x0), Math.min(b[1], y0), Math.max(b[2], x1 + 1), Math.max(b[3], y1 + 1)] : [x0, y0, x1 + 1, y1 + 1];
    }
  }

  /* ------------------------------------------------- the network's input */
  function cvRound(v) {                     // OpenCV saturate_cast<int>(double): half to even
    var f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }
  var BASE = new Uint8Array(BUF * BUF);
  // build_origin_image_north_oriented (ranges given) and
  // build_virtual_pose_image_north_oriented (ranges null): the explored map
  // around (cx, cy), the live scan on top, rotated so `theta` points up with
  // cv2.warpAffine(INTER_NEAREST, border 127), cropped to 400 x 400, robot drawn.
  function buildImage(map, explored, cx, cy, theta, ranges, out) {
    var W = map.W, H = map.H, occ = map.occ, base = BASE, dx, dy, k;
    base.fill(127);
    for (dy = -BC; dy < BC; dy++) {
      var gy = Math.trunc(cy + dy);
      if (gy < 0 || gy >= H) continue;
      var rb = (BC + dy) * BUF + BC, rg = gy * W;
      for (dx = -BC; dx < BC; dx++) {
        var gx = Math.trunc(cx + dx);
        if (gx < 0 || gx >= W) continue;
        k = rg + gx;
        if (explored[k]) base[rb + dx] = occ[k] === 0 ? 255 : 0;
      }
    }
    if (ranges) {
      var ix = Math.trunc(cx), iy = Math.trunc(cy);
      for (var i = 0; i < NB; i++) {
        var ang = ANGLES0[i] + theta, r = ranges[i];
        var x1 = Math.trunc(cx + r * Math.cos(ang)), y1 = Math.trunc(cy + r * Math.sin(ang));
        var ddx = Math.abs(x1 - ix), ddy = Math.abs(y1 - iy), sx = ix < x1 ? 1 : -1, sy = iy < y1 ? 1 : -1;
        var err = ddx - ddy, x = ix, y = iy;
        for (;;) {
          var ux = BC + (x - ix), uy = BC + (y - iy);
          if (ux < 0 || ux >= BUF || uy < 0 || uy >= BUF) break;
          if (y < 0 || y >= H || x < 0 || x >= W) break;
          if (occ[y * W + x] === 1) { base[uy * BUF + ux] = 0; break; }
          base[uy * BUF + ux] = 255;
          if (x === x1 && y === y1) break;
          var e2 = 2 * err;
          if (e2 > -ddy) { err -= ddy; x += sx; }
          if (e2 < ddx) { err += ddx; y += sy; }
        }
      }
    }
    // cv2.getRotationMatrix2D((311, 311), 90 - theta_deg, 1), inverted the way
    // warpAffine does, then its fixed-point nearest-neighbour sampling
    var a = (90.0 - (theta * 180.0 / PI)) * PI / 180, al = Math.cos(a), be = Math.sin(a);
    var m0 = al, m1 = be, m2 = (1 - al) * BC - be * BC, m3 = -be, m4 = al, m5 = be * BC + (1 - al) * BC;
    var D = m0 * m4 - m1 * m3; D = D !== 0 ? 1 / D : 0;
    var A11 = m4 * D, A22 = m0 * D, A12 = -m1 * D, A21 = -m3 * D;
    var b1 = -A11 * m2 - A12 * m5, b2 = -A21 * m2 - A22 * m5;
    var off = BC - C0, adelta = new Int32Array(WIN), bdelta = new Int32Array(WIN);
    for (var u = 0; u < WIN; u++) { adelta[u] = cvRound(A11 * (u + off) * 1024); bdelta[u] = cvRound(A21 * (u + off) * 1024); }
    for (var v = 0; v < WIN; v++) {
      var yy = v + off, X0 = cvRound((A12 * yy + b1) * 1024) + 512, Y0 = cvRound((A22 * yy + b2) * 1024) + 512;
      for (u = 0; u < WIN; u++) {
        var X = (X0 + adelta[u]) >> 10, Y = (Y0 + bdelta[u]) >> 10;
        out[v * WIN + u] = (X >= 0 && X < BUF && Y >= 0 && Y < BUF) ? base[Y * BUF + X] : 127;
      }
    }
    for (var m = 0; m < MARKER.length; m++) out[(C0 + MARKER[m][1]) * WIN + C0 + MARKER[m][0]] = MARKER[m][2];
    return out;
  }

  // _transform_image_goal_to_world and _transform_world_to_image_goal
  function img2world(px, py, pose) {
    var rot = (90.0 - (pose[2] * 180.0 / PI)) * PI / 180.0, c = Math.cos(rot), s = Math.sin(rot);
    var dx = px - C0, dy = py - C0;
    return [pose[0] + dx * c - dy * s, pose[1] + dx * s + dy * c];
  }
  function world2img(wx, wy, pose) {
    var rot = (90.0 - (pose[2] * 180.0 / PI)) * PI / 180.0, c = Math.cos(rot), s = Math.sin(rot);
    var dx = wx - pose[0], dy = wy - pose[1];
    return [Math.trunc(C0 + (dx * c + dy * s)), Math.trunc(C0 + (-dx * s + dy * c))];
  }
  function wrapPi(a) { var r = (a + PI) % TAU; if (r < 0) r += TAU; return r - PI; }

  /* --------------------------------------------- numpy / scipy, as needed */
  function gaussKernel(sigma) {
    var r = Math.trunc(4 * sigma + 0.5), w = new Float64Array(2 * r + 1), s = 0, i;
    for (i = -r; i <= r; i++) { w[i + r] = Math.exp(-0.5 / (sigma * sigma) * i * i); s += w[i + r]; }
    for (i = 0; i < w.length; i++) w[i] /= s;
    return w;
  }
  var GK2 = gaussKernel(K.SIGMA), GK3 = gaussKernel(K.ANG_K);
  function refl(i, n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; return i; }
  // scipy.ndimage.gaussian_filter(mode='reflect'): axis 0, then axis 1, float32 between
  function gauss2d(src, W, H, w) {
    var r = (w.length - 1) >> 1, tmp = new Float32Array(W * H), out = new Float32Array(W * H), x, y, k, s;
    for (y = 0; y < H; y++) {
      var inner = y >= r && y < H - r;
      for (x = 0; x < W; x++) {
        s = 0;
        if (inner) for (k = -r; k <= r; k++) s += w[k + r] * src[(y + k) * W + x];
        else for (k = -r; k <= r; k++) s += w[k + r] * src[refl(y + k, H) * W + x];
        tmp[y * W + x] = s;
      }
    }
    for (y = 0; y < H; y++) {
      var row = y * W;
      for (x = 0; x < W; x++) {
        s = 0;
        if (x >= r && x < W - r) for (k = -r; k <= r; k++) s += w[k + r] * tmp[row + x + k];
        else for (k = -r; k <= r; k++) s += w[k + r] * tmp[row + refl(x + k, W)];
        out[row + x] = s;
      }
    }
    return out;
  }
  function gauss1d(src, w) {
    var n = src.length, r = (w.length - 1) >> 1, out = new Float64Array(n);
    for (var i = 0; i < n; i++) { var s = 0; for (var k = -r; k <= r; k++) s += w[k + r] * src[refl(i + k, n)]; out[i] = s; }
    return out;
  }
  function percentile(a, q) {               // numpy 'linear'
    var s = Float32Array.from(a).sort(), n = s.length;
    var idx = n * (q / 100) + (1 - q / 100) - 1, lo = Math.floor(idx), t = idx - lo;
    if (lo + 1 >= n) return s[n - 1];
    return s[lo] + (s[lo + 1] - s[lo]) * t;
  }
  // scipy.signal.find_peaks(x, height=hmin, distance=dist)
  function findPeaks(x, hmin, dist) {
    var n = x.length, peaks = [], i = 1, imax = n - 1, k;
    while (i < imax) {
      if (x[i - 1] < x[i]) {
        var ia = i + 1;
        while (ia < imax && x[ia] === x[i]) ia++;
        if (x[ia] < x[i]) { peaks.push((i + ia - 1) >> 1); i = ia; }
      }
      i++;
    }
    peaks = peaks.filter(function (p) { return x[p] >= hmin; });
    if (dist > 1 && peaks.length > 1) {
      var keep = peaks.map(function () { return 1; }), order = peaks.map(function (p, j) { return j; });
      order.sort(function (a, b) { return x[peaks[a]] - x[peaks[b]]; });
      var d = Math.ceil(dist);
      for (var q = order.length - 1; q >= 0; q--) {
        var j = order[q];
        if (!keep[j]) continue;
        for (k = j - 1; k >= 0 && peaks[j] - peaks[k] < d; k--) keep[k] = 0;
        for (k = j + 1; k < peaks.length && peaks[k] - peaks[j] < d; k++) keep[k] = 0;
      }
      peaks = peaks.filter(function (p, j2) { return keep[j2]; });
    }
    return peaks;
  }

  /* ------------------------------------------------------ the local planner */
  // Angular components of one field: the mask of its top 4% (after smoothing),
  // a 72-bin angular histogram of its mass, peaks, nearest-peak assignment,
  // two gates, and a weighted centroid per surviving component. Shared by
  // plan_local (bins over [-pi, pi]) and Case B's per-pose check (data range).
  function components(heat, caseB) {
    var N = WIN * WIN, i, hmin = Infinity, hmax = -Infinity;
    for (i = 0; i < N; i++) { var v = heat[i]; if (v < hmin) hmin = v; if (v > hmax) hmax = v; }
    var hn = new Float32Array(N), den = hmax - hmin + 1e-6;
    for (i = 0; i < N; i++) hn[i] = (heat[i] - hmin) / den;
    var hs = gauss2d(hn, WIN, WIN, GK2), smin = Infinity, smax = -Infinity;
    for (i = 0; i < N; i++) { if (hs[i] < smin) smin = hs[i]; if (hs[i] > smax) smax = hs[i]; }
    var sden = smax - smin + 1e-6;
    for (i = 0; i < N; i++) hs[i] = (hs[i] - smin) / sden;
    var thr = percentile(hs, K.TOP_PCT), bins = caseB ? BIN_B : BIN_A;
    var hist = new Float64Array(K.NBINS), mask = new Uint8Array(N), nm = 0;
    for (i = 0; i < N; i++) if (hs[i] >= thr) { mask[i] = 1; nm++; var b = bins[i]; if (b !== 255) hist[b] += hn[i]; }
    var hm = 0;
    for (i = 0; i < K.NBINS; i++) if (hist[i] > hm) hm = hist[i];
    var tiled = new Float64Array(3 * K.NBINS);
    for (i = 0; i < 3 * K.NBINS; i++) tiled[i] = hist[i % K.NBINS] / (hm + 1e-9);
    var sm = gauss1d(tiled, GK3).slice(K.NBINS, 2 * K.NBINS);
    var peaks = findPeaks(sm, K.PEAK_H, K.MIN_ARC_GAP);
    if (!caseB) {
      var smx = 0, sarg = 0;
      for (i = 0; i < K.NBINS; i++) if (sm[i] > smx) { smx = sm[i]; sarg = i; }
      if (!peaks.length && smx > 1e-9) peaks = [sarg];
      if (peaks.length > K.MAX_COMP) {
        peaks = peaks.map(function (p, j) { return [sm[p], j, p]; })
          .sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; }).reverse()
          .slice(0, K.MAX_COMP).map(function (t) { return t[2]; });
      }
    }
    var res = { keep: [], cent: {}, cov: {}, label: null, peaks: peaks, hist: sm, thr: thr, max: hmax, n: peaks.length };
    if (!peaks.length) return res;
    var pa = peaks.map(function (p) { return CENTERS_A[p]; }), np = pa.length;
    var near = new Int8Array(N).fill(-1), cnt = new Int32Array(np);
    for (i = 0; i < N; i++) {
      if (!mask[i]) continue;
      var best = 0, bd = Infinity;
      for (var k = 0; k < np; k++) { var d = Math.abs(wrapPi(ANG[i] - pa[k])); if (d < bd) { bd = d; best = k; } }
      near[i] = best; cnt[best]++;
    }
    var label = new Uint8Array(N);
    for (var kk = 0; kk < np; kk++) {
      if (cnt[kk] < K.MIN_AREA) continue;
      var nmax = -Infinity, above = 0, sw = 0, sx = 0, sy = 0, sxm = 0, sym = 0, rawsum = 0;
      for (i = 0; i < N; i++) {
        if (near[i] !== kk) continue;
        var raw = heat[i], x = i % WIN, y = (i - x) / WIN;
        if (hn[i] > nmax) nmax = hn[i];
        if (raw > K.COV_THR) above++;
        var w = raw > 0 ? raw : 0;
        sw += w; sx += x * w; sy += y * w; sxm += x; sym += y; rawsum += raw;
      }
      if (nmax < K.REL_PEAK) continue;
      if (above / cnt[kk] < K.COV_MAJ) continue;
      var cc = kk + 1;
      res.cent[cc] = sw > 1e-9 ? [sx / sw, sy / sw] : [sxm / cnt[kk], sym / cnt[kk]];
      res.cov[cc] = rawsum;
      res.keep.push(cc);
      for (i = 0; i < N; i++) if (near[i] === kk) label[i] = cc;
    }
    res.keep.sort(function (a, b) { return res.cov[b] - res.cov[a]; });
    res.label = label;
    return res;
  }

  /* ------------------------------------------------------------------ A* */
  function Heap(W) { this.f = new Float64Array(1024); this.n = new Int32Array(1024); this.size = 0; this.W = W; }
  Heap.prototype.less = function (i, j) {       // Python tuples (priority, (x, y))
    var a = this.f[i], b = this.f[j];
    if (a !== b) return a < b;
    var W = this.W, ni = this.n[i], nj = this.n[j], xi = ni % W, xj = nj % W;
    if (xi !== xj) return xi < xj;
    return ni < nj;
  };
  Heap.prototype.swap = function (i, j) {
    var t = this.f[i]; this.f[i] = this.f[j]; this.f[j] = t;
    var u = this.n[i]; this.n[i] = this.n[j]; this.n[j] = u;
  };
  Heap.prototype.push = function (f, node) {
    if (this.size === this.f.length) {
      var nf = new Float64Array(this.size * 2), nn = new Int32Array(this.size * 2);
      nf.set(this.f); nn.set(this.n); this.f = nf; this.n = nn;
    }
    var i = this.size++;
    this.f[i] = f; this.n[i] = node;
    while (i > 0) { var p = (i - 1) >> 1; if (!this.less(i, p)) break; this.swap(i, p); i = p; }
  };
  Heap.prototype.pop = function () {          // returns the node; its priority is in this.top
    this.top = this.f[0];
    var node = this.n[0], last = --this.size;
    if (last > 0) {
      this.f[0] = this.f[last]; this.n[0] = this.n[last];
      var i = 0;
      for (;;) {
        var l = 2 * i + 1, r = l + 1, m = i;
        if (l < last && this.less(l, m)) m = l;
        if (r < last && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
    }
    return node;
  };
  var NX = [0, 0, 1, -1, 1, 1, -1, -1], NY = [1, -1, 0, 0, 1, -1, 1, -1], NC = [1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4];

  // intelx.common.nearest_free_cell: passable = not a wall
  function nearestFree(occ, W, H, x, y, maxR) {
    if (x >= 0 && x < W && y >= 0 && y < H && occ[y * W + x] !== 1) return [x, y];
    var best = null;
    for (var r = 1; r <= maxR; r++) {
      for (var dy = -r; dy <= r; dy++) for (var dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        var nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < W && ny >= 0 && ny < H && occ[ny * W + nx] !== 1) {
          var d = dx * dx + dy * dy;
          if (!best || d < best[0]) best = [d, nx, ny];
        }
      }
      if (best) return [best[1], best[2]];
    }
    return null;
  }
  // optimized_astar_path_planner(..., skip_postprocess=True): the raw grid path
  // through anything that is not a wall, or null. `pen` adds a per-cell cost
  // and `closedSet` the closed set of astar_with_clearance.
  function astarGrid(occ, W, H, sx, sy, gx, gy, pen) {
    sx = Math.trunc(sx); sy = Math.trunc(sy); gx = Math.trunc(gx); gy = Math.trunc(gy);
    if (!(sx >= 0 && sx < W && sy >= 0 && sy < H) || occ[sy * W + sx] === 1) {
      var sn = nearestFree(occ, W, H, sx, sy, K.START_SNAP_PX);
      if (!sn) return null;
      sx = sn[0]; sy = sn[1];
    }
    if (!(gx >= 0 && gx < W && gy >= 0 && gy < H) || occ[gy * W + gx] === 1) return null;
    var N = W * H, g = new Float64Array(N).fill(Infinity), from = new Int32Array(N).fill(-1);
    var closed = pen ? new Uint8Array(N) : null, heap = new Heap(W), s = sy * W + sx, goal = gy * W + gx;
    g[s] = 0; heap.push(0, s);
    while (heap.size) {
      var cur = heap.pop();
      if (cur === goal) {
        var path = [];
        while (cur !== s) { path.push([cur % W, (cur / W) | 0]); cur = from[cur]; }
        path.push([sx, sy]);
        return path.reverse();
      }
      var cx = cur % W, cy = (cur - cx) / W;
      if (closed) { if (closed[cur]) continue; closed[cur] = 1; }
      else if (heap.top > g[cur] + Math.hypot(cx - gx, cy - gy) + 1e-9) continue;   // stale entry
      var gc = g[cur];
      for (var k = 0; k < 8; k++) {
        var nx = cx + NX[k], ny = cy + NY[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        var nb = ny * W + nx;
        if (occ[nb] === 1) continue;
        if (closed && closed[nb]) continue;
        var nc = gc + NC[k] + (pen ? pen[nb] : 0);
        if (nc < g[nb]) { g[nb] = nc; from[nb] = cur; heap.push(nc + Math.hypot(nx - gx, ny - gy), nb); }
      }
    }
    return null;
  }
  // scipy.ndimage.distance_transform_edt(occ == 0), Felzenszwalb-Huttenlocher
  function edtFree(map) {
    var W = map.W, H = map.H, occ = map.occ, n = Math.max(W, H), INF = 1e20;
    var f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
    var g = new Float64Array(W * H), out = new Float32Array(W * H), x, y;
    function dt1(len) {
      var k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
      for (var q = 1; q < len; q++) {
        var s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
        k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
      }
      k = 0;
      for (q = 0; q < len; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
    }
    for (x = 0; x < W; x++) {
      for (y = 0; y < H; y++) f[y] = occ[y * W + x] === 0 ? INF : 0;
      dt1(H);
      for (y = 0; y < H; y++) g[y * W + x] = d[y];
    }
    for (y = 0; y < H; y++) {
      for (x = 0; x < W; x++) f[x] = g[y * W + x];
      dt1(W);
      for (x = 0; x < W; x++) out[y * W + x] = Math.sqrt(d[x]);
    }
    return out;
  }
  function clearanceMaps(map) {
    if (map.dist) return;
    map.dist = edtFree(map);
    var pen = new Float32Array(map.W * map.H), c = K.CLEARANCE_PX;
    for (var i = 0; i < pen.length; i++) {
      var dd = map.dist[i];
      pen[i] = dd < c ? K.WALL_PENALTY * Math.exp(-dd / (c * 0.3)) : 0;
    }
    map.pen = pen;
  }
  // intelx.common.astar_with_clearance: wall-distance penalty, then shortcuts
  // only along lines that keep 1 m from walls
  function astarClearance(map, sx, sy, gx, gy) {
    clearanceMaps(map);
    var W = map.W, H = map.H, occ = map.occ, dist = map.dist;
    var path = astarGrid(occ, W, H, sx, sy, gx, gy, map.pen);
    if (!path) return null;
    function free(p, q) {
      var x0 = p[0], y0 = p[1], x1 = q[0], y1 = q[1], dd = Math.hypot(x1 - x0, y1 - y0);
      if (dd < 1e-6) return true;
      var n = Math.max(Math.trunc(dd * 2), 2);
      for (var i = 0; i <= n; i++) {
        var t = i / n, cx = Math.trunc(x0 + t * (x1 - x0)), cy = Math.trunc(y0 + t * (y1 - y0));
        if (cx < 0 || cx >= W || cy < 0 || cy >= H) return false;
        var k = cy * W + cx;
        if (occ[k] === 1 || dist[k] < K.CLEARANCE_PX) return false;
      }
      return true;
    }
    if (path.length <= 2) return path;
    var out = [path[0]], i = 0, n = path.length;
    while (i < n - 1) {
      var j = Math.min(n - 1, i + K.SHORTCUT_LOOKAHEAD), jumped = false;
      while (j > i + 1) {
        if (free(path[i], path[j])) { out.push(path[j]); i = j; jumped = true; break; }
        j--;
      }
      if (!jumped) { i++; if (i < n) out.push(path[i]); }
    }
    return out;
  }
  // _astar_windowed: plain A* inside the box around both ends plus 150 px,
  // falling back to the whole map
  function astarWindowed(map, a, b) {
    var W = map.W, H = map.H, ax = Math.trunc(a[0]), ay = Math.trunc(a[1]), bx = Math.trunc(b[0]), by = Math.trunc(b[1]);
    var m = K.ASTAR_MARGIN;
    var x0 = Math.max(0, Math.min(ax, bx) - m), x1 = Math.min(W, Math.max(ax, bx) + m + 1);
    var y0 = Math.max(0, Math.min(ay, by) - m), y1 = Math.min(H, Math.max(ay, by) + m + 1);
    if ((x1 - x0) * (y1 - y0) < 0.6 * (H * W)) {
      var sw = x1 - x0, sh = y1 - y0, sub = map.subCache;
      if (!sub || sub.length < sw * sh) sub = map.subCache = new Int8Array(sw * sh);
      for (var y = 0; y < sh; y++) sub.set(map.occ.subarray((y + y0) * W + x0, (y + y0) * W + x1), y * sw);
      var p = astarGrid(sub, sw, sh, ax - x0, ay - y0, bx - x0, by - y0, null);
      if (p && p.length >= 2) return p.map(function (q) { return [q[0] + x0, q[1] + y0]; });
    }
    return astarGrid(map.occ, W, H, ax, ay, bx, by, null);
  }
  // _astar_len: true path length, or null when the planner did not get there
  function astarLen(map, a, b) {
    var A = [Math.trunc(a[0]), Math.trunc(a[1])], B = [Math.trunc(b[0]), Math.trunc(b[1])];
    var p = astarWindowed(map, A, B);
    if (!p || p.length < 2) return null;
    var e = p[p.length - 1];
    if (Math.hypot(e[0] - B[0], e[1] - B[1]) > K.ASTAR_END_TOL) return null;
    var seg = 0;
    for (var i = 0; i < p.length - 1; i++) seg += Math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]);
    if (seg < Math.hypot(B[0] - A[0], B[1] - A[1]) - 1.0) return null;
    return seg;
  }

  /* ------------------------------------------------------------ frontiers */
  // GlobalPriorityPlanner.find_global_frontiers: unknown cells next to known
  // free space, OpenCV's external contours with at least 30 points, each
  // contour's centroid nudged 30 px back into free space, nearest first.
  function findGlobalFrontiers(ex, rx, ry) {
    var map = ex.map, W = map.W, H = map.H, occ = map.occ, explored = ex.explored, N = W * H;
    var fr = new Uint8Array(N), x, y, k, i;
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) {
      k = y * W + x;
      if (explored[k]) continue;
      var hit = 0;
      for (var dy = -1; dy <= 1 && !hit; dy++) {
        var yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (var dx = -1; dx <= 1; dx++) {
          var xx = x + dx; if (xx < 0 || xx >= W) continue;
          var kk = yy * W + xx;
          if (explored[kk] && occ[kk] === 0) { hit = 1; break; }
        }
      }
      fr[k] = hit;
    }
    var blobs = [], cv = root.cv;
    if (cv && cv.Mat && cv.findContours) {
      var mat = new cv.Mat(H, W, cv.CV_8UC1), contours = new cv.MatVector(), hier = new cv.Mat();
      mat.data.set(fr);
      cv.findContours(mat, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      for (i = 0; i < contours.size(); i++) {
        var cnt = contours.get(i), npts = cnt.rows;
        if (npts >= 30) {
          var M = cv.moments(cnt, false);
          if (M.m00 !== 0) blobs.push([Math.trunc(M.m10 / M.m00), Math.trunc(M.m01 / M.m00)]);
        }
        cnt.delete();
      }
      mat.delete(); contours.delete(); hier.delete();
    } else {
      // without OpenCV.js: 8-connected blobs of 30+ cells, pixel centroid
      var lab = new Int32Array(N), stack = [];
      for (i = 0; i < N; i++) {
        if (!fr[i] || lab[i]) continue;
        var n = 0, sx = 0, sy = 0; lab[i] = 1; stack.push(i);
        while (stack.length) {
          var c = stack.pop(), cx = c % W, cy = (c - cx) / W; n++; sx += cx; sy += cy;
          for (var j = 0; j < 8; j++) {
            var nx = cx + NX[j], ny = cy + NY[j];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            var nk = ny * W + nx;
            if (fr[nk] && !lab[nk]) { lab[nk] = 1; stack.push(nk); }
          }
        }
        if (n >= 30) blobs.push([Math.trunc(sx / n), Math.trunc(sy / n)]);
      }
    }
    var out = [];
    blobs.forEach(function (b) {
      var cx = b[0], cy = b[1], win = 10;
      var y1 = Math.max(0, cy - win), y2 = Math.min(H, cy + win), x1 = Math.max(0, cx - win), x2 = Math.min(W, cx + win);
      var vx = 0, vy = 0;
      if (y2 > y1 && x2 > x1) {
        var m00 = 0, m10 = 0, m01 = 0;
        for (var yy = y1; yy < y2; yy++) for (var xx = x1; xx < x2; xx++) {
          var kk = yy * W + xx;
          if (explored[kk] && occ[kk] === 0) { m00++; m10 += xx - x1; m01 += yy - y1; }
        }
        if (m00 !== 0) { vx = cx - (m10 / m00 + x1); vy = cy - (m01 / m00 + y1); }
      }
      if (vx === 0 && vy === 0) { vx = cx - rx; vy = cy - ry; }
      var mag = Math.sqrt(vx * vx + vy * vy);
      if (mag === 0) return;
      vx /= mag; vy /= mag;
      out.push({ x: cx - vx * 30, y: cy - vy * 30, a: Math.atan2(-vy, vx), dist: Math.hypot(cx - rx, cy - ry), c: [cx, cy] });
    });
    out.sort(function (a, b) { return a.dist - b.dist; });
    return out;
  }
  // frontier cells inside a window: known free with an unexplored 8-neighbour;
  // cells outside the window count as explored, like BORDER_CONSTANT 0
  function frontierCells(ex, x0, y0, x1, y1) {
    var map = ex.map, W = map.W, occ = map.occ, explored = ex.explored, w = x1 - x0, h = y1 - y0;
    var fr = new Uint8Array(w * h), kf = new Uint8Array(w * h), any = 0, anyKf = 0;
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      var k = (y + y0) * W + x + x0;
      if (!(explored[k] && occ[k] === 0)) continue;
      kf[y * w + x] = 1; anyKf = 1;
      var hit = 0;
      for (var dy = -1; dy <= 1 && !hit; dy++) {
        var yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (var dx = -1; dx <= 1; dx++) {
          var xx = x + dx; if (xx < 0 || xx >= w) continue;
          if (!explored[(yy + y0) * W + xx + x0]) { hit = 1; break; }
        }
      }
      if (hit) { fr[y * w + x] = 1; any = 1; }
    }
    return { fr: fr, kf: kf, w: w, h: h, any: any, anyKf: anyKf };
  }
  // _frontier_boundary_point: snap onto the midpoint of the nearest frontier segment
  function frontierBoundaryPoint(ex, f) {
    var W = ex.map.W, H = ex.map.H, fx = Math.round(f[0]), fy = Math.round(f[1]), s = K.SNAP_SEARCH_PX;
    var x0 = Math.max(0, fx - s), x1 = Math.min(W, fx + s + 1), y0 = Math.max(0, fy - s), y1 = Math.min(H, fy + s + 1);
    if (x1 <= x0 || y1 <= y0) return [f[0], f[1]];
    var c = frontierCells(ex, x0, y0, x1, y1), w = c.w, h = c.h, cx = fx - x0, cy = fy - y0, i, x, y;
    if (!c.any) {
      if (!c.anyKf) return [f[0], f[1]];
      var bd = Infinity, bi = -1;
      for (i = 0; i < w * h; i++) if (c.kf[i]) { x = i % w; y = (i - x) / w; var dd = (x - cx) * (x - cx) + (y - cy) * (y - cy); if (dd < bd) { bd = dd; bi = i; } }
      return [bi % w + x0, ((bi - bi % w) / w) + y0];
    }
    var lab = new Int32Array(w * h), nl = 0, stack = [], best = -1, bestD = Infinity;
    for (i = 0; i < w * h; i++) {
      if (!c.fr[i] || lab[i]) continue;
      nl++; lab[i] = nl; stack.push(i);
      var dmin = Infinity;
      while (stack.length) {
        var q = stack.pop(), qx = q % w, qy = (q - qx) / w;
        var d2 = (qx - cx) * (qx - cx) + (qy - cy) * (qy - cy);
        if (d2 < dmin) dmin = d2;
        for (var j = 0; j < 8; j++) {
          var nx = qx + NX[j], ny = qy + NY[j];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          var nk = ny * w + nx;
          if (c.fr[nk] && !lab[nk]) { lab[nk] = nl; stack.push(nk); }
        }
      }
      if (dmin < bestD) { bestD = dmin; best = nl; }
    }
    var mx = 0, my = 0, n = 0;
    for (i = 0; i < w * h; i++) if (lab[i] === best) { x = i % w; mx += x; my += (i - x) / w; n++; }
    mx /= n; my /= n;
    var bj = -1, bdd = Infinity;
    for (i = 0; i < w * h; i++) if (lab[i] === best) {
      x = i % w; y = (i - x) / w;
      var e = (x - mx) * (x - mx) + (y - my) * (y - my);
      if (e < bdd) { bdd = e; bj = i; }
    }
    return [bj % w + x0, ((bj - bj % w) / w) + y0];
  }
  // _frontier_consumed: no frontier cell left within 1 m of the point
  function frontierConsumed(ex, f) {
    var W = ex.map.W, H = ex.map.H, r = Math.max(3, Math.round(1.0 / RES)), x = Math.trunc(f[0]), y = Math.trunc(f[1]);
    if (!(x >= 0 && x < W && y >= 0 && y < H)) return true;
    var c = frontierCells(ex, Math.max(0, x - r), Math.max(0, y - r), Math.min(W, x + r + 1), Math.min(H, y + r + 1));
    if (!c.anyKf) return true;
    return !c.any;
  }

  /* -------------------------------------------------------------- Explorer */
  // map: { W, H, occ (Int8Array: 0 free, 1 wall, -1 unknown) }
  // predict(images: Uint8Array[]) -> Promise<Float32Array[]>
  function Explorer(map, spawn, predict, opts) {
    opts = opts || {};
    this.map = map; this.predict = predict;
    this.idleEvery = opts.idleEvery || 6;
    var N = map.W * map.H;
    this.explored = new Uint8Array(N); this.freeSeen = 0; this.freeTotal = 0;
    for (var i = 0; i < N; i++) if (map.occ[i] === 0) this.freeTotal++;
    this.pose = [spawn[0], spawn[1], spawn[2] || 0];
    this.state = 'NEED'; this.path = null; this.pathIndex = 0;
    this.pathLen = 0; this.moves = 0; this.preds = 0; this.goals = 0;
    this.trail = [[this.pose[0], this.pose[1]]]; this.curve = [[0, 0]];
    this.nodes = []; this.junctionId = null; this.cache = null;
    this.enroute = null; this.preempt = false; this.enrouteDist = 0; this.committed = null;
    this.lastComp = null; this.noFrontier = 0; this.lastIdle = null;
    this.done = null; this.noGain = 0; this.lastGoalCov = 0; this.failedPlans = 0;
    this.scanBuf = new Float64Array(NB); this.img = new Uint8Array(WIN * WIN);
    this.last = null;           // what the last pass saw and decided, for drawing
    this.revealed = null;
    this.log = [];
  }
  Explorer.prototype.coverage = function () { return this.freeTotal ? 100 * this.freeSeen / this.freeTotal : 0; };
  Explorer.prototype.note = function (s) { this.log.push(s); if (this.log.length > 200) this.log.shift(); };

  // ---- the prediction worker's helpers
  Explorer.prototype.goalDegenerate = function (g, pose) {
    if (!g) return true;
    if (Math.hypot(g[0] - C0, g[1] - C0) < K.MIN_GOAL_ADVANCE_PX) return true;
    var w = img2world(g[0], g[1], pose), xi = Math.trunc(w[0]), yi = Math.trunc(w[1]), m = this.map;
    if (!(xi >= 0 && xi < m.W && yi >= 0 && yi < m.H)) return true;
    return m.occ[yi * m.W + xi] === 1;
  };
  Explorer.prototype.nudgeToFree = function (g, pose) {
    var dx = g ? g[0] - C0 : 0, dy = g ? g[1] - C0 : 0, d = Math.hypot(dx, dy), m = this.map;
    var ux = d > 1e-6 ? dx / d : 0, uy = d > 1e-6 ? dy / d : -1;
    for (var dist = K.MIN_GOAL_ADVANCE_PX; dist <= 120; dist += 4) {
      var gx = C0 + ux * dist, gy = C0 + uy * dist, w = img2world(Math.trunc(gx), Math.trunc(gy), pose);
      var xi = Math.trunc(w[0]), yi = Math.trunc(w[1]);
      if (xi >= 0 && xi < m.W && yi >= 0 && yi < m.H && m.occ[yi * m.W + xi] !== 1) return [Math.trunc(gx), Math.trunc(gy)];
    }
    return null;
  };
  // intelx.common.nudge_goal_along_heading / snap_goal_to_free
  Explorer.prototype.nudgeAlongHeading = function (pose) {
    var m = this.map, rx = pose[0], ry = pose[1], step = TAU / 8, offs = [0];
    for (var k = 1; k <= 4; k++) offs.push(k * step, -k * step);
    function isFree(x, y) { var xi = Math.trunc(x), yi = Math.trunc(y); return xi >= 0 && xi < m.W && yi >= 0 && yi < m.H && m.occ[yi * m.W + xi] === 0; }
    for (var o = 0; o < offs.length; o++) {
      var th = pose[2] + offs[o], gx = rx + K.NO_GOAL_NUDGE_PX * Math.cos(th), gy = ry + K.NO_GOAL_NUDGE_PX * Math.sin(th);
      var dist = Math.hypot(gx - rx, gy - ry);
      if (dist < 1.0) continue;
      var n = Math.max(2, Math.trunc(dist)), far = 0, hit = false;
      for (var i = 1; i <= n; i++) {
        var t = i / n;
        if (isFree(rx + t * (gx - rx), ry + t * (gy - ry))) far = t * dist; else { hit = true; break; }
      }
      if (far <= 0) continue;
      var reach = hit ? far - 4.0 : far;
      if (reach < 3.0) continue;
      var tt = reach / dist, sx = rx + tt * (gx - rx), sy = ry + tt * (gy - ry);
      if (isFree(sx, sy)) return [sx, sy];
    }
    return null;
  };
  Explorer.prototype.isBackward = function (g, pose) {
    var w = img2world(Math.trunc(g[0]), Math.trunc(g[1]), pose);
    var d = Math.abs(wrapPi(Math.atan2(w[1] - pose[1], w[0] - pose[0]) - pose[2]));
    return d * 180 / PI > K.UTURN_DEG;
  };
  Explorer.prototype.strongestForward = function (lc, pose) {
    var ut = K.UTURN_DEG * PI / 180;
    for (var i = 0; i < lc.keep.length; i++) {
      var c = lc.cent[lc.keep[i]]; if (!c) continue;
      var w = img2world(Math.trunc(c[0]), Math.trunc(c[1]), pose);
      if (Math.abs(wrapPi(Math.atan2(w[1] - pose[1], w[0] - pose[0]) - pose[2])) <= ut) return [Math.trunc(c[0]), Math.trunc(c[1])];
    }
    return null;
  };
  Explorer.prototype.sticky = function (lc, pose) {
    var keep = lc.keep; if (!keep.length) return null;
    var best = keep[0], prev = this.lastComp;
    if (!prev || keep.length < 2) return best;
    var tol = K.STICK_TOL_M / RES, bestSum = lc.cov[best] || 0;
    for (var i = 0; i < keep.length; i++) {
      var cc = keep[i], c = lc.cent[cc]; if (!c) continue;
      var w = img2world(Math.trunc(c[0]), Math.trunc(c[1]), pose);
      if (Math.hypot(w[0] - prev[0], w[1] - prev[1]) > tol) continue;
      if ((lc.cov[cc] || 0) >= bestSum * (1 - K.STICK_MARGIN)) return cc;
      break;
    }
    return best;
  };

  // ---- plan_global: frontier prepass, junction case A (inside a known
  // junction: lock onto its weakest unexplored branch) and case B (a possible
  // new junction: verify it from 10 virtual poses on a 0.5 m circle)
  Explorer.prototype.planGlobal = async function (heat, pose, lc, allowRefresh) {
    var self = this, map = this.map, cache = this.cache;
    var abm = {}, fviz = [], ad = [];
    var need = !cache || K.PREPASS_EVERY_PX <= 0 || cache.nNodes !== this.nodes.length ||
      Math.hypot(pose[0] - cache.pose[0], pose[1] - cache.pose[1]) >= K.PREPASS_EVERY_PX;
    if (!allowRefresh && cache) need = false;
    if (need) {
      if (this.nodes.length) {
        var fr = findGlobalFrontiers(this, pose[0], pose[1]), pts = [];
        fr.forEach(function (f) {
          var s = frontierBoundaryPoint(self, [f.x, f.y]);
          pts.push([s[0], s[1], Math.hypot(s[0] - pose[0], s[1] - pose[1])]);
        });
        pts.sort(function (a, b) { return a[2] - b[2]; });
        pts = pts.slice(0, 4);
        fviz = pts.map(function (p) { return [p[0], p[1]]; });
        var branches = [];
        this.nodes.forEach(function (nd) { nd.components.forEach(function (c) { branches.push({ node: nd.id, rank: c.rank, x: c.x, y: c.y }); }); });
        pts.forEach(function (p) {
          var fpx = [Math.trunc(p[0]), Math.trunc(p[1])];
          var cands = branches.slice().sort(function (a, b) { return Math.hypot(a.x - p[0], a.y - p[1]) - Math.hypot(b.x - p[0], b.y - p[1]); }).slice(0, 3);
          var bestB = null, minCost = Infinity;
          cands.forEach(function (b) {
            var path = astarWindowed(map, [Math.trunc(b.x), Math.trunc(b.y)], fpx);
            if (path && path.length < minCost) { minCost = path.length; bestB = b; }
          });
          if (bestB) abm[bestB.node + ':' + bestB.rank] = fpx;
        });
        this.nodes.forEach(function (nd) {
          nd.components.forEach(function (c) {
            var key = nd.id + ':' + c.rank;
            if (abm[key]) ad.push({ node: nd.id, rank: c.rank, x: c.x, y: c.y, dist: Math.hypot(c.x - pose[0], c.y - pose[1]), frontier: abm[key] });
          });
        });
      }
      this.cache = { pose: [pose[0], pose[1]], nNodes: this.nodes.length, abm: abm, ad: ad, fviz: fviz };
    } else {
      abm = cache.abm; fviz = cache.fviz;
      ad = cache.ad.map(function (d) { var e = Object.assign({}, d); e.dist = Math.hypot(d.x - pose[0], d.y - pose[1]); return e; });
    }

    var keep = lc.keep, cent = lc.cent, chosen = null;
    var mergePx = K.LOCK_DIST_M / RES, cx = pose[0], cy = pose[1], active = null, i;
    for (i = 0; i < this.nodes.length; i++) if (Math.hypot(this.nodes[i].x - cx, this.nodes[i].y - cy) < mergePx) { active = this.nodes[i]; break; }
    this.lockLive = false;
    function leastTurning(nd) {
      var best = null, md = Infinity;
      nd.components.forEach(function (c) {
        var d = Math.abs(wrapPi(Math.atan2(c.y - nd.y, c.x - nd.x) - pose[2]));
        if (d < md) { md = d; best = c; }
      });
      if (best) nd.lockedAngle = Math.atan2(best.y - nd.y, best.x - nd.x);
    }
    if (active) {                                                    // Case A
      if (this.junctionId !== active.id) {
        this.junctionId = active.id;
        active.count++;
        var unexp = active.components.filter(function (c) { return !c.explored; });
        if (unexp.length) {
          unexp.sort(function (a, b) { return b.rank - a.rank; });
          var pick = null;
          for (i = 0; i < unexp.length; i++) if (abm[active.id + ':' + unexp[i].rank]) { pick = unexp[i]; break; }
          if (!pick) { leastTurning(active); active.lockedRank = null; }
          else {
            active.components.forEach(function (c) { if (c.rank === pick.rank) c.explored = true; });
            active.lockedAngle = Math.atan2(pick.y - active.y, pick.x - active.x);
            active.lockedRank = pick.rank;
            this.note('Junction ' + active.id + ': locked onto rank ' + pick.rank);
          }
        } else { leastTurning(active); active.lockedRank = null; }
      }
      var la = active.lockedAngle, lr = active.lockedRank;
      var live = lr !== null && lr !== undefined && !!abm[active.id + ':' + lr];
      this.lockLive = live;
      if (live && la !== undefined && la !== null) {
        var tol = K.LOCK_TOL_DEG * PI / 180;
        for (i = 0; i < keep.length; i++) {
          var c = cent[keep[i]], w = img2world(Math.trunc(c[0]), Math.trunc(c[1]), pose);
          if (Math.abs(wrapPi(Math.atan2(w[1] - cy, w[0] - cx) - la)) <= tol) { chosen = [Math.trunc(c[0]), Math.trunc(c[1])]; break; }
        }
      }
    } else if (keep.length > 1) {                                   // Case B
      this.junctionId = null;
      var tests = [[pose[0], pose[1], pose[2]]];
      for (i = 0; i < 10; i++) {
        var th = pose[2] + TAU * i / 10;
        tests.push([pose[0] + (0.5 / RES) * Math.cos(th), pose[1] + (0.5 / RES) * Math.sin(th), pose[2]]);
      }
      var imgs = [], valid = [];
      tests.forEach(function (tp) {
        var tx = Math.trunc(tp[0]), ty = Math.trunc(tp[1]);
        if (!(ty >= 0 && ty < map.H && tx >= 0 && tx < map.W)) return;
        if (map.occ[ty * map.W + tx] === 1) return;
        imgs.push(buildImage(map, self.explored, tp[0], tp[1], pose[2], null, new Uint8Array(WIN * WIN)));
        valid.push(tp);
      });
      this.lastCheck = { poses: valid, verified: 0, at: [pose[0], pose[1]] };
      if (imgs.length >= 5) {
        var outs = await this.predict(imgs);
        this.preds += imgs.length;
        var nodeXs = [], nodeYs = [], comps = [];
        outs.forEach(function (vh, idx) {
          var r = components(vh, true);
          if (r.n > 1) {
            nodeXs.push(valid[idx][0]); nodeYs.push(valid[idx][1]);
            r.keep.forEach(function (cc) {
              var wv = img2world(Math.trunc(r.cent[cc][0]), Math.trunc(r.cent[cc][1]), valid[idx]);
              comps.push({ x: wv[0], y: wv[1], cov: r.cov[cc] });
            });
          }
        });
        this.lastCheck.verified = nodeXs.length;
        if (nodeXs.length >= 5) {
          var ax = nodeXs.reduce(function (a, b) { return a + b; }, 0) / nodeXs.length;
          var ay = nodeYs.reduce(function (a, b) { return a + b; }, 0) / nodeYs.length;
          var clusters = [], atol = 30 * PI / 180;
          comps.forEach(function (c) {
            var ca = Math.atan2(c.y - ay, c.x - ax), placed = false;
            for (var q = 0; q < clusters.length; q++) {
              var ba = Math.atan2(clusters[q][0].y - ay, clusters[q][0].x - ax);
              if (Math.abs(wrapPi(ca - ba)) < atol) { clusters[q].push(c); placed = true; break; }
            }
            if (!placed) clusters.push([c]);
          });
          var avg = clusters.filter(function (cl) { return cl.length >= 2; }).map(function (cl) {
            var s = cl.reduce(function (a, c) { return [a[0] + c.x, a[1] + c.y, a[2] + c.cov]; }, [0, 0, 0]);
            return { x: s[0] / cl.length, y: s[1] / cl.length, cov: s[2] / cl.length };
          });
          avg.sort(function (a, b) { return b.cov - a.cov; });
          if (avg.length) {
            var fc = avg.map(function (c, r) {
              var ip = world2img(c.x, c.y, pose);
              return { rank: r + 1, x: c.x, y: c.y, cxImg: ip[0], cyImg: ip[1], explored: false, cov: c.cov };
            });
            var pickB = fc[fc.length - 1];                         // weakest first
            pickB.explored = true;
            var node = { id: this.nodes.length, x: ax, y: ay, count: 1, components: fc,
              lockedAngle: Math.atan2(pickB.y - ay, pickB.x - ax), lockedRank: pickB.rank };
            this.nodes.push(node);
            this.junctionId = node.id;
            chosen = [Math.trunc(pickB.cxImg), Math.trunc(pickB.cyImg)];
            this.note('Junction ' + node.id + ' registered, ' + fc.length + ' branches; weakest first (rank ' + pickB.rank + ')');
          }
        }
      }
    } else this.junctionId = null;
    return { chosen: chosen, ad: ad, fviz: fviz };
  };

  // ---- one pass of the prediction worker; returns the image-frame goal on a
  // nav pass (null when there is none)
  Explorer.prototype.policyPass = async function (nav) {
    var pose = this.pose.slice(), map = this.map, self = this;
    var ranges = scanRanges(map, pose[0], pose[1], pose[2], this.scanBuf);
    buildImage(map, this.explored, pose[0], pose[1], pose[2], ranges, this.img);
    var t0 = performance.now();
    var heat = (await this.predict([this.img]))[0];
    this.inferMs = performance.now() - t0;
    this.preds++;
    var lc = components(heat, false), keep = lc.keep, maxScore = lc.max;
    var skip = false;
    if (!nav && keep.length <= 1) {
      var mp = K.LOCK_DIST_M / RES;
      skip = !this.nodes.some(function (nd) { return Math.hypot(nd.x - pose[0], nd.y - pose[1]) < mp; });
    }
    var g;
    if (skip) {
      this.junctionId = null;
      g = { chosen: null, ad: this.cache ? this.cache.ad : [], fviz: this.cache ? this.cache.fviz : [] };
    } else g = await this.planGlobal(heat, pose, lc, nav);
    var goal = null, src;
    if (g.chosen) { goal = g.chosen; src = 'Junction Lock'; }
    else if (keep.length >= 1) {
      this.noFrontier = 0; this.committed = null;
      var cc = this.sticky(lc, pose), c = lc.cent[cc];
      goal = [Math.trunc(c[0]), Math.trunc(c[1])]; src = 'Local Planner';
      this.lastComp = img2world(goal[0], goal[1], pose);
    } else if (maxScore > K.COV_THR) {
      var am = 0; for (var i = 1; i < heat.length; i++) if (heat[i] > heat[am]) am = i;
      goal = [am % WIN, (am - am % WIN) / WIN]; src = 'Field Maximum';
    } else src = 'None';
    var churn = this.goalDegenerate(goal, pose);
    if (nav) { this.enroute = null; this.preempt = false; }
    else if (this.enrouteDist >= K.PREEMPT_MIN_GOAL_PX && keep.length >= 1 && maxScore >= K.PREEMPT_MIN_SCORE && !churn) {
      this.preempt = true; this.enroute = null; this.enrouteDist = 0;
      this.note('Coverage on the way: switching back to the local planner');
    } else if (this.enroute && maxScore < K.PREEMPT_MIN_SCORE) {
      if (frontierConsumed(this, this.enroute)) {
        this.preempt = true; this.enroute = null; this.committed = null; this.enrouteDist = 0;
        this.note('Frontier explored on the way: replanning');
      }
    }
    var force = false;
    if (nav && goal && keep.length >= 1 && !churn && this.isBackward(goal, pose)) {
      var fwd = this.strongestForward(lc, pose);
      if (fwd) goal = fwd; else force = true;
    }
    var target = null;
    if (nav && (keep.length === 0 || churn || force)) {
      this.lastComp = null;
      var ad = g.ad;
      if (ad.length) {
        this.noFrontier = 0;
        var rpx = [Math.trunc(pose[0]), Math.trunc(pose[1])], scored = [];
        ad.forEach(function (d) { var cd = astarLen(map, rpx, d.frontier); if (cd !== null) scored.push([cd, d]); });
        if (scored.length) {
          scored.sort(function (a, b) { return a[0] - b[0]; });
          var costD = scored[0][0], best = scored[0][1], com = this.committed;
          if (com && !frontierConsumed(this, com)) {
            var kp = null;
            for (i = 0; i < scored.length; i++) if (Math.hypot(scored[i][1].frontier[0] - com[0], scored[i][1].frontier[1] - com[1]) <= 30) { kp = scored[i]; break; }
            if (kp && kp[0] <= costD * (1 + K.SWITCH_MARGIN)) { costD = kp[0]; best = kp[1]; }
          }
          this.committed = [Math.trunc(best.frontier[0]), Math.trunc(best.frontier[1])];
          var fpx = best.frontier, bpx = [Math.trunc(best.x), Math.trunc(best.y)];
          var crb = astarLen(map, rpx, bpx), cbf = astarLen(map, bpx, fpx);
          var via = (crb !== null && cbf !== null) ? crb + cbf : null;
          target = (via === null || costD <= via + K.DIRECT_MARGIN_PX) ? fpx : bpx;
          this.enroute = fpx;
        } else {
          ad.sort(function (a, b) { return a.dist - b.dist; });
          var b0 = ad[0];
          target = b0.dist < 8 ? b0.frontier : [Math.trunc(b0.x), Math.trunc(b0.y)];
          this.enroute = b0.frontier;
        }
        goal = world2img(target[0], target[1], pose); src = 'Branch Frontier';
      } else {
        var ff = null, rp = [Math.trunc(pose[0]), Math.trunc(pose[1])];
        var fc = findGlobalFrontiers(this, rp[0], rp[1]).map(function (f) {
          var s = frontierBoundaryPoint(self, [f.x, f.y]);
          return [Math.hypot(s[0] - rp[0], s[1] - rp[1]), [Math.trunc(s[0]), Math.trunc(s[1])]];
        });
        fc.sort(function (a, b) { return a[0] - b[0]; });
        for (i = 0; i < Math.min(K.FALLBACK_CANDS, fc.length); i++) if (astarLen(map, rp, fc[i][1]) !== null) { ff = fc[i][1]; break; }
        if (ff) {
          this.noFrontier = 0; target = ff;
          goal = world2img(ff[0], ff[1], pose); src = 'Frontier Fallback';
          this.enroute = ff; this.committed = [ff[0], ff[1]];
        } else {
          this.noFrontier++;
          if (this.noFrontier >= K.NO_FRONTIER_EXHAUST) { this.done = 'exhausted'; this.note('No reachable frontier left'); }
        }
      }
    }
    if (nav && this.goalDegenerate(goal, pose)) {
      var nud = this.nudgeToFree(goal, pose);
      if (!nud) { var w = this.nudgeAlongHeading(pose); if (w) nud = world2img(w[0], w[1], pose); }
      if (nud) { goal = nud; src += ' (nudged)'; }
    }
    if (nav && !goal && !this.done) { this.done = 'boxed in'; }
    var wg = goal ? img2world(goal[0], goal[1], pose) : null;
    if (nav) this.enrouteDist = wg ? Math.hypot(wg[0] - pose[0], wg[1] - pose[1]) : 0;
    this.last = { pose: pose, heat: heat, lc: lc, ad: g.ad, fviz: g.fviz, goal: wg, src: src, nav: nav,
      maxScore: maxScore, target: target, junction: this.junctionId };
    if (nav) this.lastNav = this.last;
    return goal;
  };

  // ---- one step of the simulation worker's state machine
  Explorer.prototype.iter = async function () {
    if (this.done) return;
    var map = this.map, W = map.W, H = map.H;
    if (this.state === 'NEED') {
      var reqPose = this.pose.slice();
      var g = await this.policyPass(true);
      if (this.done) return;
      if (!g) return;
      var wgl = img2world(g[0], g[1], reqPose), gx = Math.trunc(wgl[0]), gy = Math.trunc(wgl[1]);
      if (!(gx >= 0 && gx < W && gy >= 0 && gy < H) || map.occ[gy * W + gx] === 1) { this.failedPlans++; return this.stallCheck(); }
      var path = astarClearance(map, Math.trunc(this.pose[0]), Math.trunc(this.pose[1]), gx, gy);
      if (!path || !path.length) { this.failedPlans++; return this.stallCheck(); }
      this.failedPlans = 0;
      this.path = path; this.pathIndex = 0; this.goal = wgl; this.goals++;
      var cov = this.coverage();
      if (cov <= this.lastGoalCov + 1e-9) this.noGain++; else this.noGain = 0;
      this.lastGoalCov = cov;
      this.state = 'MOVE';
      return this.stallCheck();
    }
    if (this.preempt) { this.preempt = false; this.path = null; this.state = 'NEED'; return; }
    if (!this.path || this.pathIndex >= this.path.length) { this.state = 'NEED'; return; }
    var t = this.path[this.pathIndex], px = this.pose[0], py = this.pose[1];
    var dx = t[0] - px, dy = t[1] - py, d = Math.hypot(dx, dy);
    if (d < GOAL_TOL) {
      this.pathIndex++;
      if (this.pathIndex >= this.path.length) { this.path = null; this.state = 'NEED'; }
      return;
    }
    var step = Math.min(STEP_PX, d), ux = dx / d, uy = dy / d;
    this.pose = [px + ux * step, py + uy * step, Math.atan2(uy, ux)];
    if (this.moves % MAP_EVERY === 0) {
      scanRanges(map, this.pose[0], this.pose[1], this.pose[2], this.scanBuf);
      updateExplored(this, this.pose[0], this.pose[1], this.pose[2], this.scanBuf);
      if (this.coverage() >= TARGET_PCT) this.done = 'complete';
    }
    this.pathLen += Math.hypot(this.pose[0] - px, this.pose[1] - py) * RES;
    this.moves++;
    var tr = this.trail[this.trail.length - 1];
    if (Math.hypot(this.pose[0] - tr[0], this.pose[1] - tr[1]) >= 2) this.trail.push([this.pose[0], this.pose[1]]);
    var cv = this.curve[this.curve.length - 1];
    if (this.pathLen - cv[0] >= 0.25 || this.done) this.curve.push([this.pathLen, this.coverage()]);
    if (!this.done && (!this.lastIdle || Math.hypot(this.pose[0] - this.lastIdle[0], this.pose[1] - this.lastIdle[1]) >= this.idleEvery)) {
      this.lastIdle = [this.pose[0], this.pose[1]];
      await this.policyPass(false);
    }
  };
  Explorer.prototype.stallCheck = function () {
    if (this.noGain >= K.STALL_GOALS) { this.done = 'stalled'; this.note('No coverage gain over ' + K.STALL_GOALS + ' goals'); }
    if (this.failedPlans >= K.STALL_GOALS) { this.done = 'stalled'; this.note('No plannable goal'); }
  };

  /* ------------------------------------------------------------ weights */
  var F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
  function half(x) {                         // one fp16 value -> fp32
    var s = (x & 0x8000) << 16, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * m * 5.960464477539063e-8;
    U32[0] = e === 31 ? (s | 0x7f800000 | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return F32[0];
  }
  // quantize_weights.py stores every convolution kernel as int8 with an fp32
  // scale per 64 weights, the small tensors as fp16. Rebuilds the fp32
  // weights.bin the ONNX graph references, byte for byte at the same offsets.
  function rebuildWeights(info, blob) {
    var out = new Uint8Array(info.weights_fp32_file_bytes), f = new Float32Array(out.buffer);
    info.q8_manifest.forEach(function (t) {
      var dst = t.off / 4, i;
      if (t.kind === 'g8') {                 // one scale per t.g consecutive weights
        var ng = Math.ceil(t.n / t.g), gs = new Float32Array(blob.buffer, blob.byteOffset + t.at, ng);
        var gq = new Int8Array(blob.buffer, blob.byteOffset + t.at + 4 * ng, t.n);
        for (i = 0; i < t.n; i++) f[dst + i] = gq[i] * gs[(i / t.g) | 0];
      } else if (t.kind === 'q8') {
        var sc = new Float32Array(blob.buffer, blob.byteOffset + t.at, t.ch);
        var q = new Int8Array(blob.buffer, blob.byteOffset + t.at + 4 * t.ch, t.n), per = t.n / t.ch;
        for (var c = 0; c < t.ch; c++) { var s = sc[c], b = c * per; for (i = 0; i < per; i++) f[dst + b + i] = q[b + i] * s; }
      } else {
        var h = new Uint16Array(blob.buffer, blob.byteOffset + t.at, t.n);
        for (i = 0; i < t.n; i++) f[dst + i] = half(h[i]);
      }
    });
    return out;
  }
  /* ---------------------------------------------------- map preparation */
  // seal_by_adjacent: unknown cells that do not touch free space become wall
  function sealByAdjacent(occ, W, H) {
    var out = Int8Array.from(occ);
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var k = y * W + x;
      if (occ[k] !== -1) continue;
      var near = (x > 0 && occ[k - 1] === 0) || (x < W - 1 && occ[k + 1] === 0) ||
        (y > 0 && occ[k - W] === 0) || (y < H - 1 && occ[k + W] === 0);
      if (!near) out[k] = 1;
    }
    return out;
  }
  // find_spawn's clearance test: free, and no wall within 0.45 m + 2 px
  function spawnOk(map, x, y) {
    var r = Math.trunc(0.45 / RES) + 2, W = map.W, H = map.H, xi = Math.trunc(x), yi = Math.trunc(y);
    if (!(xi >= 0 && xi < W && yi >= 0 && yi < H) || map.occ[yi * W + xi] !== 0) return false;
    for (var dy = -r; dy <= r; dy++) for (var dx = -r; dx <= r; dx++) {
      var nx = xi + dx, ny = yi + dy;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H && map.occ[ny * W + nx] === 1) return false;
    }
    return true;
  }

  root.IntelXCore = {
    RES: RES, WIN: WIN, NB: NB, FOV: FOV, MAX_R: MAX_R, K: K, ANGLES0: ANGLES0,
    scanRanges: scanRanges, buildImage: buildImage, components: components,
    img2world: img2world, world2img: world2img, astarClearance: astarClearance, astarLen: astarLen,
    findGlobalFrontiers: findGlobalFrontiers, frontierBoundaryPoint: frontierBoundaryPoint,
    sealByAdjacent: sealByAdjacent, spawnOk: spawnOk, edtFree: edtFree, rebuildWeights: rebuildWeights, Explorer: Explorer
  };
})(typeof window !== 'undefined' ? window : self);       // the page, or live_worker.js
