/* IntelX project page: exploration replays.

   Each map ships three files (built by build_race.py from the paper's runs):
     run.json    trajectories, coverage curves and summary stats per method
     reveal.png  per method, the path distance at which each cell was first
                 seen (R,G = decimetres, 65535 = never) and B = wall
     map.png     ground truth: 255 free, 0 wall surface, 64 other blocked
   Sorting each method's cells by that distance once means any moment of a run
   is a pointer into the sorted list, so every method stays synchronized by
   distance travelled, and scrubbing either way repaints only what changed. */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var root = document.documentElement;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var TAU = Math.PI * 2;

  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function rgba(str) {
    str = String(str || '').trim();
    if (str[0] === '#') {
      var h = str.slice(1); if (h.length === 3) h = h.replace(/./g, '$&$&');
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), 1];
    }
    var m = str.match(/[\d.]+/g) || [0, 0, 0, 1];
    return [+m[0], +m[1], +m[2], m[3] != null ? +m[3] : 1];
  }
  var fmtInt = function (v) { return Math.round(v).toLocaleString('en-US'); };

  /* ------------------------------------------------------------- loading */
  var cache = {};
  function loadImg(url) {
    return new Promise(function (res, rej) {
      var im = new Image(); im.decoding = 'async';
      im.onload = function () { res(im); };
      im.onerror = function () { rej(new Error('Could not load ' + url)); };
      im.src = url;
    });
  }
  function pixels(img) {
    var c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
    var g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    return g.getImageData(0, 0, c.width, c.height).data;
  }
  function b64(s, T) {
    var bin = atob(s), u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new T(u8.buffer);
  }
  function loadMap(base) {
    if (cache[base]) return cache[base];
    cache[base] = Promise.all([
      fetch(base + 'run.json').then(function (r) { if (!r.ok) throw new Error('Could not load ' + base + 'run.json'); return r.json(); }),
      loadImg(base + 'reveal.png'), loadImg(base + 'map.png')
    ]).then(function (got) {
      var meta = got[0], W = meta.width, H = meta.height, N = W * H;
      var rp = pixels(got[1]), gp = pixels(got[2]);
      var ground = new Uint8Array(N);
      for (var i = 0; i < N; i++) ground[i] = gp[i * 4];
      meta.ground = ground;
      var counts = new Uint32Array(65536);
      meta.runs.forEach(function (run, r) {
        var off = r * N * 4, dm = new Uint16Array(N), wall = new Uint8Array(N);
        counts.fill(0);
        for (var i = 0; i < N; i++) {
          var o = off + i * 4, v = (rp[o] << 8) | rp[o + 1];
          dm[i] = v; wall[i] = rp[o + 2] > 127 ? 1 : 0;
          if (v !== 65535) counts[v]++;
        }
        var total = 0;
        for (var v = 0; v < 65535; v++) { var c = counts[v]; counts[v] = total; total += c; }
        var order = new Uint32Array(total), vals = new Uint16Array(total);
        for (i = 0; i < N; i++) { v = dm[i]; if (v !== 65535) { var p = counts[v]++; order[p] = i; vals[p] = v; } }
        run.order = order; run.vals = vals; run.wall = wall;
        run.xy = b64(run.xy, Int16Array); run.th = b64(run.th, Uint8Array); run.cov = b64(run.cov, Uint16Array);
        run.n = run.xy.length / 2;
        if (run.passes) decodePasses(run.passes);
      });
      return meta;
    });
    return cache[base];
  }
  // An ablation arm's recorded planner decisions (build_ablation.py), one per
  // pass in path order; positions are grid cells x 10, like the trajectory.
  function decodePasses(P) {
    P.d = b64(P.d, Uint16Array); P.src = b64(P.src, Uint8Array); P.fmax = b64(P.fmax, Uint16Array);
    if (P.goal) P.goal = b64(P.goal, Int16Array);
    [['comp_n', 'comp', Uint8Array], ['fr_n', 'fr', Uint16Array]].forEach(function (k) {
      if (!P[k[0]]) return;
      var cnt = P[k[0]] = b64(P[k[0]], k[2]), at = P[k[1] + '_o'] = new Uint32Array(P.n + 1);
      P[k[1]] = b64(P[k[1]], Int16Array);
      for (var i = 0; i < P.n; i++) at[i + 1] = at[i] + cnt[i] * 2;
    });
    if (P.node_k) P.node_k = b64(P.node_k, Uint16Array);
  }
  function covAt(run, d) {
    var c = run.cov, f = Math.min(d, run.final_path_m) / run.cov_step_m, i = Math.floor(f);
    if (i >= c.length - 1) return d >= run.final_path_m ? run.final_cov : c[c.length - 1] / 100;
    return (c[i] + (c[i + 1] - c[i]) * (f - i)) / 100;
  }

  /* -------------------------------------------------------------- the view */
  function MapView(canvas, opts) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.opts = opts || {};
    this.layer = document.createElement('canvas');
    this.lctx = this.layer.getContext('2d');
    this.cur = -1;
  }
  MapView.prototype.set = function (meta, run) {
    var sameMap = this.meta === meta;
    this.meta = meta; this.run = run;
    if (!sameMap) {
      this.layer.width = meta.width; this.layer.height = meta.height;
      this.img = this.lctx.createImageData(meta.width, meta.height);
    }
    this.cur = -1; this.palette(); this.reset(); this.fit();
  };
  MapView.prototype.palette = function () {
    var n = this.opts.night, m = this.run ? this.run.mode : 'ours';
    var P = {
      unknown: css(n ? '--night-2' : '--map-unknown'), ghost: css(n ? '--night-ghost' : '--map-ghost'),
      free: css(n ? '--night-free' : '--map-free'), wall: css(n ? '--night-wall' : '--map-wall'),
      scan: css(n ? '--night-scan' : '--map-scan'), trail: css(n ? '--night-trail' : '--m-' + m),
      spawn: '#2E9E4F', edge: n ? css('--night') : '#FFFFFF'
    };
    ['unknown', 'ghost', 'free', 'wall'].forEach(function (k) { P[k + 'RGB'] = rgba(P[k]); });
    P.rank = [1, 2, 3, 4].map(function (i) { return css('--ov-rank-' + i); });
    P.node = css('--ov-node'); P.nodeFill = css('--ov-node-fill'); P.frontier = css('--ov-frontier');
    P.goal = css('--ov-goal'); P.window = css('--ov-window');
    this.P = P;
  };
  MapView.prototype.reset = function () {
    var d = this.img.data, g = this.meta.ground, U = this.P.unknownRGB, G = this.P.ghostRGB;
    for (var i = 0, o = 0, N = g.length; i < N; i++, o += 4) {
      var c = g[i] === 0 ? G : U; d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
    }
    var target = this.cur; this.k = 0; this.cur = -1;
    if (target >= 0) this.paint(target, true);
    this.lctx.putImageData(this.img, 0, 0);
  };
  MapView.prototype.paint = function (dm, quiet) {
    if (dm === this.cur) return;
    var run = this.run, order = run.order, vals = run.vals, wall = run.wall, g = this.meta.ground, d = this.img.data;
    var F = this.P.freeRGB, Wc = this.P.wallRGB, U = this.P.unknownRGB, G = this.P.ghostRGB;
    var k = this.k, n = order.length, lo = Infinity, hi = -1, i, o, c;
    while (k < n && vals[k] <= dm) {
      i = order[k++]; o = i * 4; c = wall[i] ? Wc : F;
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
      if (i < lo) lo = i; if (i > hi) hi = i;
    }
    while (k > 0 && vals[k - 1] > dm) {
      i = order[--k]; o = i * 4; c = g[i] === 0 ? G : U;
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
      if (i < lo) lo = i; if (i > hi) hi = i;
    }
    this.k = k; this.cur = dm;
    if (hi >= 0 && !quiet) {
      var W = this.meta.width, y0 = Math.floor(lo / W), y1 = Math.floor(hi / W);
      this.lctx.putImageData(this.img, 0, 0, 0, y0, W, y1 - y0 + 1);
    }
  };
  MapView.prototype.fit = function () {
    if (!this.meta) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var cw = this.cv.clientWidth || 300, chCss;
    if (this.opts.fixedBox) chCss = this.cv.clientHeight || cw * this.meta.height / this.meta.width;
    else { chCss = cw * this.meta.height / this.meta.width; this.cv.style.height = chCss.toFixed(1) + 'px'; }
    this.cv.width = Math.round(cw * dpr); this.cv.height = Math.round(chCss * dpr);
    this.dpr = dpr;
    var s = Math.min(this.cv.width / this.meta.width, this.cv.height / this.meta.height);
    this.s = s;
    this.ox = (this.cv.width - this.meta.width * s) / 2;
    this.oy = (this.cv.height - this.meta.height * s) / 2;
  };
  MapView.prototype.pose = function (d) {
    var run = this.run, xy = run.xy, th = run.th, n = run.n;
    var f = Math.max(0, Math.min(d, run.final_path_m)) / run.traj_step_m;
    var i0 = Math.min(n - 1, Math.floor(f)), i1 = Math.min(n - 1, i0 + 1), t = Math.min(1, f - i0);
    var a0 = th[i0] / 256 * TAU, a1 = th[i1] / 256 * TAU, da = ((a1 - a0 + Math.PI * 3) % TAU) - Math.PI;
    return {
      i: i0,
      x: (xy[2 * i0] + (xy[2 * i1] - xy[2 * i0]) * t) / 10,
      y: (xy[2 * i0 + 1] + (xy[2 * i1 + 1] - xy[2 * i0 + 1]) * t) / 10,
      a: a0 + da * t
    };
  };
  MapView.prototype.draw = function (dist) {
    if (!this.run) return;
    var run = this.run, d = Math.min(dist, run.final_path_m);
    this.paint(Math.round(d * 10));
    var ctx = this.ctx, s = this.s, px = 1 / s * this.dpr;      // one CSS pixel, in cells
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.P.unknown; ctx.fillRect(0, 0, this.cv.width, this.cv.height);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.layer, this.ox, this.oy, this.meta.width * s, this.meta.height * s);
    ctx.setTransform(s, 0, 0, s, this.ox, this.oy);
    var p = this.pose(d);
    if (this.opts.scan !== false && dist < run.final_path_m + 1) this.scan(p, px);
    var ov = run.passes ? this.meta.overlays || [] : [], k = ov.length ? this.passAt(d) : -1;
    if (k >= 0 && ov.indexOf('window') >= 0) this.window(p, px);
    this.trail(p, px);
    if (k >= 0) this.marks(k, px, ov);
    if (this.meta.spawn) this.star(this.meta.spawn[0], this.meta.spawn[1], 6.5 * px);
    this.robot(p, px);
  };
  // the last pass the planner made at or before this distance
  MapView.prototype.passAt = function (d) {
    var a = this.run.passes.d, t = Math.round(d * 10), lo = 0, hi = a.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (a[mid] <= t) lo = mid + 1; else hi = mid; }
    return lo - 1;
  };
  // the planner's 20 m window, square to the robot's heading
  MapView.prototype.window = function (p, px) {
    var h = this.meta.win_cells / 2, ctx = this.ctx;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a);
    ctx.strokeStyle = this.P.window; ctx.lineWidth = 1.2 * px; ctx.strokeRect(-h, -h, 2 * h, 2 * h);
    ctx.restore();
  };
  // What the planner held at that pass, as the ablation videos drew it: junction
  // nodes with their ranked branches (taken ones faded), ranked components,
  // frontier fallback targets and the goal it chose.
  MapView.prototype.marks = function (k, px, ov) {
    var P = this.run.passes, ctx = this.ctx, C = this.P, i, r, x, y;
    function cross(x, y, s, w, col) {
      ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s); ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
      ctx.lineWidth = w; ctx.strokeStyle = col; ctx.stroke();
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (P.node_k && ov.indexOf('junctions') >= 0) {
      P.nodesets[P.node_k[k]].forEach(function (nd) {
        var nx = nd[0] / 10, ny = nd[1] / 10, q = 4.6 * px;
        nd[2].forEach(function (b) {
          var col = C.rank[(b[0] - 1) % C.rank.length], bx = b[1] / 10, by = b[2] / 10;
          ctx.globalAlpha = b[3] ? .4 : 1;
          ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(bx, by); ctx.lineWidth = 1.1 * px; ctx.strokeStyle = col; ctx.stroke();
          ctx.beginPath(); ctx.arc(bx, by, 2.8 * px, 0, TAU); ctx.fillStyle = col; ctx.fill();
          ctx.lineWidth = px; ctx.strokeStyle = '#FFFFFF'; ctx.stroke();
        });
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.moveTo(nx, ny - q); ctx.lineTo(nx + q, ny); ctx.lineTo(nx, ny + q); ctx.lineTo(nx - q, ny); ctx.closePath();
        ctx.fillStyle = C.nodeFill; ctx.fill(); ctx.lineWidth = 1.6 * px; ctx.strokeStyle = C.node; ctx.stroke();
      });
    }
    if (P.comp && ov.indexOf('components') >= 0) {
      for (i = P.comp_o[k], r = 0; i < P.comp_o[k + 1]; i += 2, r++) {
        if (P.comp[i] === -32768) continue;
        x = P.comp[i] / 10; y = P.comp[i + 1] / 10;
        ctx.beginPath(); ctx.arc(x, y, 3.4 * px, 0, TAU); ctx.fillStyle = '#FFFFFF'; ctx.fill();
        ctx.lineWidth = 1.8 * px; ctx.strokeStyle = C.rank[r % C.rank.length]; ctx.stroke();
      }
    }
    if (P.fr && ov.indexOf('frontiers') >= 0) {
      for (i = P.fr_o[k]; i < P.fr_o[k + 1]; i += 2) {
        x = P.fr[i] / 10; y = P.fr[i + 1] / 10;
        cross(x, y, 3 * px, 3.4 * px, '#FFFFFF'); cross(x, y, 3 * px, 1.7 * px, C.frontier);
      }
    }
    if (P.goal && ov.indexOf('goal') >= 0 && P.goal[2 * k] !== -32768) {
      x = P.goal[2 * k] / 10; y = P.goal[2 * k + 1] / 10;
      cross(x, y, 3.6 * px, 3.8 * px, '#FFFFFF'); cross(x, y, 3.6 * px, 2 * px, C.goal);
    }
  };
  MapView.prototype.scan = function (p, px) {
    var run = this.run, g = this.meta.ground, W = this.meta.width, H = this.meta.height;
    var fov = run.fov, R = run.range_cells, full = fov > 6.2;
    var n = Math.max(24, Math.round(fov / TAU * 150)), step = 0.85;
    var ctx = this.ctx;
    ctx.beginPath();
    if (!full) ctx.moveTo(p.x, p.y);
    for (var k = 0; k <= n; k++) {
      var a = p.a - fov / 2 + fov * k / n, cx = Math.cos(a), sy = Math.sin(a), r = 0, x = p.x, y = p.y;
      while (r < R) {
        var nx = p.x + cx * (r + step), ny = p.y + sy * (r + step), ix = nx | 0, iy = ny | 0;
        if (ix < 0 || iy < 0 || ix >= W || iy >= H || g[iy * W + ix] < 128) break;
        r += step; x = nx; y = ny;
      }
      if (k === 0 && full) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = this.P.scan; ctx.fill();
  };
  MapView.prototype.trail = function (p, px) {
    var run = this.run, xy = run.xy, ctx = this.ctx, w = (this.opts.trail || 2) * px;
    var recent = Math.round(30 / run.traj_step_m), j0 = Math.max(0, p.i - recent);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = this.P.trail;
    ctx.beginPath(); ctx.moveTo(xy[0] / 10, xy[1] / 10);
    for (var j = 1; j <= p.i; j++) ctx.lineTo(xy[2 * j] / 10, xy[2 * j + 1] / 10);
    ctx.lineTo(p.x, p.y);
    ctx.globalAlpha = this.opts.night ? .62 : .58; ctx.lineWidth = w; ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.moveTo(xy[2 * j0] / 10, xy[2 * j0 + 1] / 10);
    for (j = j0 + 1; j <= p.i; j++) ctx.lineTo(xy[2 * j] / 10, xy[2 * j + 1] / 10);
    ctx.lineTo(p.x, p.y);
    if (this.opts.night) { ctx.shadowColor = this.P.trail; ctx.shadowBlur = 10 * this.dpr; }
    ctx.lineWidth = w * 1.25; ctx.stroke();
    ctx.shadowBlur = 0;
  };
  MapView.prototype.robot = function (p, px) {
    var ctx = this.ctx, r = (this.opts.robot || 6) * px;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a);
    ctx.beginPath(); ctx.moveTo(r * 1.35, 0); ctx.lineTo(-r * .9, r * .95); ctx.lineTo(-r * .35, 0); ctx.lineTo(-r * .9, -r * .95); ctx.closePath();
    ctx.fillStyle = this.opts.night ? '#FFFFFF' : this.P.trail;
    ctx.strokeStyle = this.opts.night ? this.P.trail : this.P.edge;
    ctx.lineWidth = 1.6 * px; ctx.lineJoin = 'round';
    if (this.opts.night) { ctx.shadowColor = this.P.trail; ctx.shadowBlur = 14 * this.dpr; }
    ctx.fill(); ctx.shadowBlur = 0; ctx.stroke();
    ctx.restore();
  };
  MapView.prototype.star = function (x, y, r) {
    var ctx = this.ctx; ctx.beginPath();
    for (var k = 0; k < 10; k++) { var a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * .45 : r; ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
    ctx.closePath(); ctx.fillStyle = this.P.spawn; ctx.fill();
  };

  var views = [];
  function onTheme() { views.forEach(function (v) { if (v.meta) { v.palette(); v.reset(); if (v.redraw) v.redraw(); } }); }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onTheme);
  new MutationObserver(onTheme).observe(root, { attributes: true, attributeFilter: ['data-theme'] });

  /* ---------------------------------------------------------------- hero */
  (function hero() {
    var fig = $('#hero'); if (!fig) return;
    var canvas = $('#hero-canvas'), stage = $('.stage', fig);
    var cov = $('#hero-cov'), path = $('#hero-path'), rivalCov = $('#hero-rival-cov'), bar = $('#hero-bar');
    var done = $('#hero-done'), toggle = $('#hero-toggle'), scale = $('#hero-scale i');
    var view = new MapView(canvas, { trail: 2.4, robot: 7, fixedBox: true });
    views.push(view);
    var run, rival, d = 0, playing = !reduce.matches, visible = true, last = 0, hold = 0, raf = 0;
    var DUR = 24, HOLD = 5;
    function hud() {
      var c = covAt(run, d);
      cov.innerHTML = c.toFixed(1) + '<small>%</small>';
      path.innerHTML = fmtInt(d) + '<small> m</small>';
      rivalCov.textContent = covAt(rival, d).toFixed(1) + '%';
      bar.style.width = Math.min(100, c / 99 * 100).toFixed(2) + '%';
      done.hidden = d < run.final_path_m - 0.01;
    }
    function render() { view.draw(d); hud(); }
    view.redraw = render;
    function size() {
      view.fit();
      var m = view.meta, cssPerCell = view.s / view.dpr;
      scale.style.width = (10 / m.res_m * cssPerCell).toFixed(1) + 'px';
      render();
    }
    function frame(now) {
      raf = 0;
      if (!playing || !visible) { last = 0; return; }
      var dt = last ? Math.min(.1, (now - last) / 1000) : 0; last = now;
      var end = run.final_path_m;
      if (d >= end) { hold += dt; if (hold > HOLD) { hold = 0; d = 0; } }
      else d = Math.min(end, d + end / DUR * dt);
      render();
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (!raf && playing && visible) raf = requestAnimationFrame(frame); }
    function setPlaying(p) {
      playing = p;
      toggle.textContent = p ? 'Pause' : (d >= run.final_path_m ? 'Replay' : 'Play');
      toggle.setAttribute('aria-pressed', p ? 'false' : 'true');
      if (p) { if (d >= run.final_path_m) { d = 0; hold = 0; } kick(); }
    }
    toggle.addEventListener('click', function () { setPlaying(!playing); });
    loadMap(fig.getAttribute('data-base')).then(function (meta) {
      run = meta.runs.filter(function (r) { return r.mode === fig.getAttribute('data-run'); })[0];
      rival = meta.runs.filter(function (r) { return r.mode === fig.getAttribute('data-rival'); })[0];
      view.set(meta, run);
      // open on a frame that already shows the building, then keep going
      d = reduce.matches ? run.final_path_m : run.final_path_m * .42;
      size();
      if ('ResizeObserver' in window) new ResizeObserver(function () { size(); }).observe(stage);
      else window.addEventListener('resize', size);
      if ('IntersectionObserver' in window) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; kick(); }).observe(stage);
      setPlaying(playing);
    }).catch(function (e) { var cap = document.querySelector('.teaser__note span'); if (cap) cap.textContent = e.message; });
  })();

  /* ---------------------------------------------------------------- race */
  // A replay block: panes synchronized by distance travelled, the coverage
  // chart that grows with them, a legend and the play controls. The baselines
  // race switches maps with its tabs; each ablation is one fixed set of runs.
  function stopWord(run) {
    return { stalled: 'Stalled', no_progress: 'No Progress', exhausted: 'Frontiers Exhausted', timeout: 'Timed Out' }[run.stop] || 'Stopped';
  }
  function glyph(body, wide) {
    return '<svg class="glyph' + (wide ? ' glyph--wide' : '') + '" viewBox="0 0 ' + (wide ? 34 : 14) + ' 14" aria-hidden="true">' + body + '</svg>';
  }
  var X_ = 'M3.5 3.5l7 7M10.5 3.5l-7 7';
  var OVERLAY_KEY = {
    goal: [['Goal Chosen', glyph('<path d="' + X_ + '" stroke="#FFFFFF" stroke-width="4.4" stroke-linecap="round"/><path d="' + X_ + '" stroke="var(--ov-goal)" stroke-width="2.2" stroke-linecap="round"/>')]],
    frontiers: [['Frontier (Fallback)', glyph('<path d="' + X_ + '" stroke="var(--ov-frontier)" stroke-width="2" stroke-linecap="round"/>')]],
    components: [['Component, Rank 1 to 3', glyph([6, 17, 28].map(function (x, i) { return '<circle cx="' + x + '" cy="7" r="4" fill="#FFFFFF" stroke="var(--ov-rank-' + (i + 1) + ')" stroke-width="2"/>'; }).join(''), true)]],
    window: [['Local Window', glyph('<rect x="2" y="2" width="10" height="10" fill="none" stroke="var(--ov-window)" stroke-width="1.6"/>')]],
    junctions: [['Junction Node', glyph('<path d="M7 1.6L12.4 7 7 12.4 1.6 7z" fill="var(--ov-node-fill)" stroke="var(--ov-node)" stroke-width="1.6"/>')],
                ['Branch, Rank 1 to 3 (Taken Ones Fade)', glyph([6, 17, 28].map(function (x, i) { return '<circle cx="' + x + '" cy="7" r="3.6" fill="var(--ov-rank-' + (i + 1) + ')"/>'; }).join(''), true)]]
  };

  function Race(rootEl, cfg) {
    var tabs = $$('.race__maps button', rootEl), grid = $('.race__grid', rootEl), chartHost = $('.race__chart', rootEl);
    var legend = $('.race__legend', rootEl), status = $('.race__status', rootEl);
    var playBtn = $('.race__ctl .iconbtn', rootEl), scrub = $('.race__scrub input', rootEl), out = $('.race__dist', rootEl);
    var speeds = $$('.race__ctl .seg button', rootEl);
    var meta = null, d = 0, Dmax = 1, playing = false, speed = 1, visible = false;
    var started = false, raf = 0, last = 0, chart = null, token = 0, panes = [], made = '';
    var DUR = cfg.dur || 34, label = cfg.label || function (run) { return run.label; };

    function makePanes(modes) {
      if (modes.join() === made) return;
      made = modes.join();
      panes.forEach(function (p) { var i = views.indexOf(p.view); if (i >= 0) views.splice(i, 1); });
      grid.innerHTML = '';
      panes = modes.map(function (mode) {
        var fig = document.createElement('figure');
        fig.className = 'pane' + (mode === 'ours' ? ' pane--ours' : '');
        fig.style.setProperty('--c', 'var(--m-' + mode + ')');
        fig.innerHTML = '<div class="pane__stage"><canvas></canvas></div><figcaption class="pane__bar"><span class="pane__name"></span><span class="pane__stat"></span><span class="pane__badge" hidden></span><span class="pane__sub" hidden></span></figcaption>';
        grid.appendChild(fig);
        var view = new MapView($('canvas', fig), { trail: 1.6, robot: 4.5 });
        view.redraw = render;
        views.push(view);
        fig.addEventListener('mouseenter', function () { hl(mode, true); });
        fig.addEventListener('mouseleave', function () { hl(mode, false); });
        return { mode: mode, fig: fig, view: view, name: $('.pane__name', fig), stat: $('.pane__stat', fig), badge: $('.pane__badge', fig), sub: $('.pane__sub', fig), run: null };
      });
    }
    function hl(mode, on) {
      if (!chart || !chart.curves[mode]) return;
      chart.curves[mode].classList.toggle('is-hl', on);
      if (on) chart.svg.insertBefore(chart.curves[mode], chart.cursor);
    }

    function buildChart() {
      var Wv = 1000, Hv = 260, L = 46, R = 18, T = 12, Bm = 36, svgNS = 'http://www.w3.org/2000/svg';
      var X = function (v) { return L + v / Dmax * (Wv - L - R); }, Y = function (v) { return T + (1 - v / 100) * (Hv - T - Bm); };
      var svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('viewBox', '0 0 ' + Wv + ' ' + Hv);
      var mk = function (tag, attrs, parent) { var e = document.createElementNS(svgNS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
      var g = mk('g', { 'class': 'grid' });
      [0, 25, 50, 75, 100].forEach(function (v) {
        mk('line', { x1: L, x2: Wv - R, y1: Y(v), y2: Y(v) }, g);
        mk('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }).textContent = v + '%';
      });
      var steps = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000], si = steps.findIndex(function (s) { return Dmax / s <= 7; });
      var stepM = si < 0 ? 2000 : steps[si];
      for (var v = 0; v <= Dmax + 1e-6; v += stepM) {
        mk('line', { x1: X(v), x2: X(v), y1: T, y2: Hv - Bm }, g);
        mk('text', { x: X(v), y: Hv - Bm + 18, 'text-anchor': 'middle' }).textContent = fmtInt(v);
      }
      mk('text', { x: (L + Wv - R) / 2, y: Hv - 2, 'text-anchor': 'middle' }).textContent = 'Distance Travelled [m]';
      mk('line', { 'class': 'target', x1: L, x2: Wv - R, y1: Y(99), y2: Y(99) });
      mk('text', { x: L + 6, y: Y(99) - 6 }).textContent = '99% Target';
      // Every curve grows with the race, as in the paper's videos: it is drawn
      // up to the distance travelled so far, with a dot at its head, and its
      // end marker appears once that method finishes or stops.
      var curves = {}, series = {};
      var runs = meta.runs.slice().sort(function (a, b) { return (a.mode === 'ours') - (b.mode === 'ours'); });
      runs.forEach(function (run) {
        var c = run.cov, n = c.length, step = Math.max(1, Math.ceil(n / 700)), xs = [], pts = [];
        for (var i = 0; i < n; i += step) { xs.push(i * run.cov_step_m); pts.push(X(i * run.cov_step_m).toFixed(1) + ',' + Y(c[i] / 100).toFixed(1)); }
        curves[run.mode] = mk('polyline', { 'class': 'curve' + (run.mode === 'ours' ? ' is-ours' : ''), points: '', stroke: 'var(--m-' + run.mode + ')' });
        series[run.mode] = { run: run, xs: xs, pts: pts };
      });
      runs.forEach(function (run) {
        var sr = series[run.mode], big = run.mode === 'ours';
        sr.end = mk('circle', { 'class': 'end', cx: X(run.final_path_m), cy: Y(run.final_cov), r: big ? 5 : 3.6, fill: 'var(--m-' + run.mode + ')' });
        sr.head = mk('circle', { 'class': 'head', r: big ? 4.5 : 3.2, fill: 'var(--m-' + run.mode + ')' });
      });
      var cursor = mk('line', { 'class': 'cursor', x1: X(0), x2: X(0), y1: T, y2: Hv - Bm });
      chartHost.innerHTML = ''; chartHost.appendChild(svg);
      chart = { svg: svg, curves: curves, series: series, cursor: cursor, X: X, Y: Y, at: -1 };
      var keys = [];
      (meta.overlays || []).forEach(function (o) { keys = keys.concat(OVERLAY_KEY[o] || []); });
      legend.innerHTML = panes.map(function (p) { return '<span style="--c:var(--m-' + p.mode + ')"><i></i>' + label(p.run) + '</span>'; }).join('') +
        keys.map(function (k) { return '<span>' + k[1] + k[0] + '</span>'; }).join('');
    }

    function layout() {
      if (!meta) return;
      grid.style.setProperty('--cols', cfg.cols(grid.clientWidth || 900, meta.width / meta.height, panes.length));
      panes.forEach(function (p) { p.view.fit(); });
    }
    function growChart() {
      if (chart.at === d) return;
      chart.at = d;
      Object.keys(chart.series).forEach(function (mode) {
        var sr = chart.series[mode], run = sr.run, dd = Math.min(d, run.final_path_m), xs = sr.xs;
        var lo = 0, hi = xs.length;                      // points at or before dd
        while (lo < hi) { var mid = (lo + hi) >> 1; if (xs[mid] <= dd) lo = mid + 1; else hi = mid; }
        var hx = chart.X(dd), hy = chart.Y(covAt(run, dd));
        var pts = sr.pts.slice(0, lo); pts.push(hx.toFixed(1) + ',' + hy.toFixed(1));
        chart.curves[mode].setAttribute('points', pts.join(' '));
        var fin = d >= run.final_path_m;
        sr.end.style.display = fin ? '' : 'none';
        sr.head.style.display = fin ? 'none' : '';
        if (!fin) { sr.head.setAttribute('cx', hx.toFixed(1)); sr.head.setAttribute('cy', hy.toFixed(1)); }
      });
      var x = chart.X(Math.min(d, Dmax));
      chart.cursor.setAttribute('x1', x); chart.cursor.setAttribute('x2', x);
    }
    function updatePane(p) {
      var run = p.run, c = covAt(run, d), fin = d >= run.final_path_m;
      p.stat.textContent = c.toFixed(1) + '% · ' + fmtInt(Math.min(d, run.final_path_m)) + ' m';
      p.badge.hidden = !fin;
      if (fin) {
        var ok = run.final_cov >= 99;
        p.badge.className = 'pane__badge ' + (ok ? 'is-done' : 'is-stop');
        p.badge.textContent = ok ? 'Reached 99%' : stopWord(run) + ' at ' + run.final_cov.toFixed(1) + '%';
      }
      if (run.passes) {         // what the planner was doing, as the videos showed it
        var P = run.passes, k = Math.max(0, p.view.passAt(Math.min(d, run.final_path_m)));
        p.sub.textContent = 'Decision: ' + meta.decisions[P.src[k]] + ' · Field Max ' + (P.fmax[k] / 1e4).toFixed(4);
      }
    }
    function render() {
      if (!meta) return;
      panes.forEach(function (p) { p.view.draw(d); updatePane(p); });
      if (chart) growChart();
      out.textContent = fmtInt(d) + ' m';
      if (document.activeElement !== scrub) scrub.value = d;
    }

    function frame(now) {
      raf = 0;
      if (!playing || !visible) { last = 0; return; }
      var dt = last ? Math.min(.1, (now - last) / 1000) : 0; last = now;
      d = Math.min(Dmax, d + Dmax / DUR * speed * dt);
      render();
      if (d >= Dmax) { setPlaying(false); return; }
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (!raf && playing && visible) raf = requestAnimationFrame(frame); }
    function setPlaying(p) {
      if (p && d >= Dmax) d = 0;
      playing = p; playBtn.setAttribute('aria-pressed', p ? 'true' : 'false');
      playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
      if (p) kick();
    }

    function select(base, tabId) {
      var my = ++token;
      tabs.forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute('data-map') === tabId ? 'true' : 'false'); });
      status.textContent = 'Loading the replay…';
      loadMap(base).then(function (m) {
        if (my !== token) return;
        meta = m;
        makePanes(cfg.order || m.order);
        Dmax = Math.max.apply(null, m.runs.map(function (r) { return r.final_path_m; }));
        scrub.max = (Math.ceil(Dmax * 2) / 2).toFixed(1); d = 0;    // a whole step, so the end is reachable
        panes.forEach(function (p) {
          p.run = m.runs.filter(function (r) { return r.mode === p.mode; })[0];
          p.name.textContent = label(p.run);
          p.sub.hidden = !p.run.passes;
          p.view.set(m, p.run);
        });
        buildChart(); layout();
        status.textContent = cfg.status(m, panes);
        render();
        if (!reduce.matches && (started || visible)) { started = true; setPlaying(true); }
      }).catch(function (e) { if (my === token) status.textContent = e.message + '. Reload the page to try again.'; });
    }

    tabs.forEach(function (b) { b.addEventListener('click', function () { select('static/race/' + b.getAttribute('data-map') + '/', b.getAttribute('data-map')); }); });
    playBtn.addEventListener('click', function () { started = true; setPlaying(!playing); });
    scrub.addEventListener('input', function () { started = true; setPlaying(false); d = Math.min(Dmax, parseFloat(scrub.value) || 0); render(); });
    speeds.forEach(function (b) {
      b.addEventListener('click', function () {
        speed = parseFloat(b.getAttribute('data-speed'));
        speeds.forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      });
    });
    var onResize = function () { if (meta) { layout(); render(); } };
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(grid);
    else window.addEventListener('resize', onResize);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        visible = es[0].isIntersecting;
        if (visible && !started && meta && !reduce.matches) { started = true; setPlaying(true); }
        kick();
      }, { threshold: .15 }).observe(rootEl);
    } else { visible = true; }
    if (tabs.length) {
      var first = tabs.filter(function (b) { return b.getAttribute('aria-selected') === 'true'; })[0] || tabs[0];
      select('static/race/' + first.getAttribute('data-map') + '/', first.getAttribute('data-map'));
    } else select(rootEl.getAttribute('data-base'));
  }

  // the baselines: IntelX first, then the baselines in the paper's order
  var main = $('#race');
  if (main) Race(main, {
    order: ['ours', 'random', 'nearest', 'cost_utility', 'geom_ig', 'frontier', 'hectoraug', 'upen', 'mapex'],
    label: function (run) { return run.mode === 'ours' ? 'IntelX (Ours)' : run.label; },
    cols: function (w, a) {
      if (w < 560) return a >= 1.2 ? 1 : 2;
      if (w < 860) return a >= 2.4 ? 2 : 3;
      var pw = (w - 24) / 3; return 3 * (pw / a + 34) > 1000 ? 5 : 3;
    },
    status: function (m, panes) {
      var ours = panes[0].run;
      var oursTxt = ours.final_cov >= 99 ? 'IntelX reaches 99% at ' + fmtInt(ours.p99) + ' m'
        : 'IntelX stops at ' + ours.final_cov.toFixed(1) + '% after ' + fmtInt(ours.final_path_m) + ' m' +
          (m.map === 'houseexpo_large_3' ? ', the stall the paper reports when the frontier it needed became unreachable' : '');
      var done = m.runs.filter(function (r) { return r.mode !== 'ours' && r.final_cov >= 99; }).length;
      return m.title + ', seed ' + m.seed + ', ' + fmtInt(m.area_m2) + ' m² of free space. ' + oursTxt + '; ' + done + ' of the 8 baselines also reach 99%.';
    }
  });
  // the ablations: each a fixed set of arms, IntelX first
  $$('.ablation .race[data-base]').forEach(function (el) {
    Race(el, {
      dur: 20,
      cols: function (w, a, n) {      // as many columns as keep every map 150 px tall
        for (var c = n; c > 1; c--) if ((w - (c - 1) * 12) / c / a >= 150) return c;
        return 1;
      },
      status: function (m) { return m.title + ', seed ' + m.seed + ', ' + m.note; }
    });
  });
})();
