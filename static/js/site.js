/* IntelX project page: navigation, the architecture tour, steppers, charts,
   the paper's tables, the lightbox and the citation. The exploration replays
   live in race.js. */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var el = function (tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  /* ------------------------------------------------------------ the paper */
  var METHODS = [
    { key: 'random', label: 'Random Frontier', group: 'Geometry Only' },
    { key: 'nearest', label: 'Nearest Frontier', group: 'Geometry Only' },
    { key: 'cost_utility', label: 'Cost-Utility', group: 'Geometry Only' },
    { key: 'geom_ig', label: 'Geometric IG', group: 'Geometry Only' },
    { key: 'frontier', label: 'Frontier (AutoX)', group: 'Our Previous Work' },
    { key: 'hectoraug', label: 'IG-Hector', group: 'Prediction Based' },
    { key: 'upen', label: 'UPEN', group: 'Prediction Based' },
    { key: 'mapex', label: 'MapEx', group: 'Prediction Based' },
    { key: 'ours', label: 'IntelX (Ours)', group: 'Ours' }
  ];
  // Table 3: P99 [m], rho [%], eta_d [m2/m], eta_t [m2/s]; value, std.
  var T3 = {
    'Small Office 3': { area: '667 m²', rows: [
      [[478.1, 234.3], [91.6, 4.8], [1.45, .91], [.67, .45]],
      [[253.7, 38.9], [98.0, 2.4], [3.11, 1.55], [1.05, .29]],
      [[208.8, 27.8], [95.6, 3.4], [3.29, 1.25], [1.31, .43]],
      [[199.0, 16.0], [95.1, 3.3], [2.97, .55], [1.30, .29]],
      [[234.2, 12.6], [99.0, 0.0], [2.83, .15], [1.37, .13]],
      [[250.0, 51.3], [99.0, 0.0], [2.75, .67], [1.38, .31]],
      [[251.2, 61.9], [99.0, 0.0], [2.72, 1.53], [1.29, .79]],
      [[152.4, 8.3], [95.1, 2.9], [4.28, .42], [1.38, 1.13]],
      [[155.7, 22.0], [99.0, 0.0], [4.10, .61], [1.89, .12]]] },
    'Medium Office 6': { area: '2,961 m²', rows: [
      [[1696.2, 336.8], [82.4, 8.9], [1.78, 1.07], [.46, .09]],
      [[1124.8, 269.3], [81.6, 6.6], [2.83, 1.21], [.64, .21]],
      [[1236.0, 225.0], [90.3, 5.6], [2.63, .60], [.55, .16]],
      [[992.9, 156.4], [87.7, 12.8], [2.96, .46], [.68, .21]],
      [[981.8, 102.2], [94.2, 4.3], [2.67, .56], [.89, .28]],
      [[1044.0, 221.6], [99.0, 0.0], [2.95, .83], [.77, .25]],
      [[863.5, 134.0], [99.0, 0.0], [3.47, .58], [1.30, .28]],
      [[687.5, 55.7], [99.0, 0.0], [4.29, .36], [1.20, .12]],
      [[546.3, 34.2], [99.0, 0.0], [5.19, .55], [2.18, .29]]] },
    'Large Office 7': { area: '7,669 m²', rows: [
      [[5547.7, 1653.6], [71.9, 13.9], [1.39, .69], [.70, .21]],
      [[3523.6, 756.1], [73.2, 12.6], [2.21, 1.79], [.83, .64]],
      [[3633.4, 433.1], [85.2, 10.9], [2.15, .39], [.62, .15]],
      [[3393.8, 261.2], [82.9, 12.8], [2.32, .63], [.96, .33]],
      [[3102.2, 253.2], [89.6, 8.5], [2.56, .55], [1.17, .27]],
      [[3004.1, 611.7], [99.0, 0.0], [2.61, .48], [.80, .16]],
      [[2812.4, 155.6], [99.0, 0.0], [2.71, .15], [1.05, .06]],
      [[3080.7, 161.8], [99.0, 0.0], [2.47, .13], [.73, .05]],
      [[1771.0, 115.0], [99.0, 0.0], [4.37, .28], [1.52, .24]]] }
  };
  // Table 4: P99', P95', P90', rho, AUC, eta_d, eta_t; value, std.
  var T4 = {
    'HouseExpo (9 Maps)': [
      [[5.96, 3.44], [5.39, 3.68], [5.54, 4.10], [85.2, 8.1], [.730, .253], [1.88, 1.20], [.70, .50]],
      [[5.67, 3.33], [5.87, 3.73], [6.20, 4.03], [83.7, 7.7], [.771, .255], [2.31, .69], [.88, .31]],
      [[6.62, 1.89], [7.13, 2.27], [6.03, 3.63], [93.6, 3.7], [.841, .082], [1.84, .34], [.54, .16]],
      [[5.53, 3.52], [4.34, 3.40], [4.67, 3.85], [87.9, 5.4], [.825, .187], [2.17, .34], [.77, .14]],
      [[4.01, 2.92], [3.99, 3.05], [4.07, 3.25], [93.4, 6.2], [.837, .188], [2.12, .31], [.86, .16]],
      [[2.16, .55], [2.23, .58], [2.07, .52], [99.0, 0.0], [.900, .007], [1.86, .41], [.70, .18]],
      [[2.66, 1.79], [2.29, 1.41], [2.37, 1.39], [99.0, 0.0], [.919, .113], [1.84, .88], [.74, .41]],
      [[1.14, .03], [1.08, .06], [1.04, .11], [99.0, 0.0], [.938, .010], [3.73, .48], [1.52, .22]],
      [[1.03, .05], [1.05, .06], [1.09, .11], [97.2, 1.2], [.946, .006], [3.80, .43], [1.69, .20]]],
    'MSD (9 Maps)': [
      [[6.08, 5.13], [5.57, 4.05], [4.61, 3.68], [81.6, 8.9], [.763, .140], [1.47, .21], [.53, .11]],
      [[2.27, 1.67], [2.30, 1.78], [2.64, 1.21], [84.8, 7.4], [.798, .116], [1.77, .38], [.61, .16]],
      [[4.15, 1.81], [3.52, 2.15], [3.65, 2.05], [82.7, 8.2], [.833, .207], [1.74, .60], [.55, .26]],
      [[5.27, 1.67], [4.64, 2.83], [4.16, 4.32], [86.1, 6.4], [.805, .111], [2.07, .55], [.67, .21]],
      [[1.57, .67], [1.71, .78], [1.84, 1.21], [87.8, 8.1], [.868, .110], [2.09, .73], [.74, .26]],
      [[1.27, .21], [1.41, .48], [1.67, .78], [99.0, 0.0], [.969, .008], [2.87, .32], [1.20, .13]],
      [[1.94, .41], [1.43, .38], [1.75, .88], [99.0, 0.0], [.955, .006], [1.94, .53], [.83, .23]],
      [[1.24, .18], [1.32, .33], [1.68, .69], [99.0, 0.0], [.967, .005], [2.92, .34], [1.23, .15]],
      [[1.07, .12], [1.17, .17], [1.45, .26], [99.0, 0.0], [.968, .004], [3.58, .33], [2.09, .38]]],
    'AWS RoboMaker (1 Map)': [
      [[4.79, 3.14], [6.18, 4.84], [7.52, 6.36], [80.8, 9.6], [.745, .236], [1.72, .41], [.64, .12]],
      [[1.65, .21], [1.98, .26], [2.07, .43], [99.0, .1], [.921, .015], [2.17, .14], [.86, .08]],
      [[3.11, 2.24], [4.17, 4.41], [4.75, 6.09], [95.9, 3.1], [.907, .061], [2.26, .76], [.83, .24]],
      [[5.69, 2.55], [3.84, 4.53], [4.49, 5.87], [95.1, 2.9], [.900, .055], [2.27, .56], [.84, .16]],
      [[1.59, .19], [1.62, .52], [1.81, .62], [99.0, 0.0], [.933, .013], [2.27, .32], [.97, .16]],
      [[1.70, .15], [1.70, .25], [1.56, .22], [99.0, 0.0], [.944, .005], [2.10, .09], [.90, .04]],
      [[2.91, 1.28], [1.65, .48], [1.77, .61], [96.8, 1.3], [.946, .011], [2.43, .26], [1.09, .12]],
      [[1.20, .10], [1.16, .11], [1.07, .08], [99.0, 0.0], [.958, .006], [2.98, .27], [1.25, .11]],
      [[1.00, 0.0], [1.05, .17], [1.15, .09], [99.0, 0.0], [.963, .002], [3.59, .29], [1.87, .23]]]
  };
  // Table 2.
  var T2 = [
    { name: 'All-Zero', note: '†', params: '–', v: [5.98, -0.22, null, null, null] },
    { name: 'DenseNet-121', params: '13.6M', v: [5.95, 0.14, 9.27, 20.1, 24.9] },
    { name: 'ResNet-18', params: '14.3M', v: [5.80, 0.22, 9.22, 25.5, 5.6] },
    { name: 'IntelX (ResNet-34)', params: '24.4M', v: [3.62, 0.44, 1.38, 66.3, 8.1], ours: true }
  ];
  // Table 1: name, rooms, area m2, kappa, shows.
  var T1 = [
    ['Small House AWS', 3, 153, 55, 133], ['Small Lounge 1', 8, 691, 346, 840], ['Small Lounge 2', 6, 940, 399, 969],
    ['Small Office 1', 13, 670, 421, 1025], ['Corridor 1', 19, 1128, 783, 1904], ['Small Office 2', 21, 1517, 1061, 2580],
    ['Medium Office 3', 28, 1339, 1063, 2585], ['Atrium 1', 19, 1971, 1296, 3152], ['Medium Office 5', 27, 1978, 1482, 3605],
    ['Medium Office 2', 33, 2132, 1706, 4148], ['Medium Office 1', 29, 3634, 2662, 6474], ['Large Office 6', 46, 3219, 2776, 6751],
    ['Large Office 3', 51, 3349, 2978, 7241], ['Large Office 5', 56, 3564, 3249, 7900], ['Large Office 2', 82, 4407, 4445, 10810],
    ['Medium Office 4', 47, 5482, 4568, 11109], ['Large Office 1', 70, 5540, 5206, 12659], ['Large Office 4', 99, 6441, 6627, 16115]
  ];
  var fmt = function (v, d) { return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); };
  var dec = function (v) { var s = String(v); return s.indexOf('.') < 0 ? 0 : Math.min(3, s.split('.')[1].length); };

  /* ---------------------------------------------------------- top nav */
  var nav = $('#topnav'), hero = $('#top');
  if (nav && hero && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (es) { nav.classList.toggle('is-on', !es[0].isIntersecting); },
      { rootMargin: '-80px 0px 0px 0px' }).observe(hero);
    var links = $$('.topnav__links a');
    var secs = links.map(function (a) { return document.querySelector(a.getAttribute('href')); }).filter(Boolean);
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle('is-here', a.getAttribute('href') === '#' + e.target.id); });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    secs.forEach(function (s) { io.observe(s); });
  }
  // a missing author photo falls back to the initials underneath it
  $$('.avatar img').forEach(function (im) {
    var drop = function () { im.remove(); };
    if (im.complete && !im.naturalWidth) drop(); else im.addEventListener('error', drop);
  });
  $$('[data-placeholder]').forEach(function (a) {
    if (a.tagName === 'A' && a.getAttribute('href') === '#') a.addEventListener('click', function (e) { e.preventDefault(); });
  });

  /* ---------------------------------------------------------- lightbox */
  var lb = $('#lightbox'), lbImg = lb && $('img', lb), lbFrom = null;
  function openLB(src, alt, from) { lbImg.src = src; lbImg.alt = alt || ''; lb.hidden = false; lbFrom = from; $('.lightbox__close', lb).focus(); document.documentElement.style.overflow = 'hidden'; }
  function closeLB() { lb.hidden = true; lbImg.removeAttribute('src'); document.documentElement.style.overflow = ''; if (lbFrom) lbFrom.focus(); }
  $$('[data-full]').forEach(function (b) {
    b.addEventListener('click', function () { var i = $('img', b); openLB(b.getAttribute('data-full'), i ? i.alt : '', b); });
  });
  if (lb) {
    lb.addEventListener('click', closeLB);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !lb.hidden) closeLB(); });
  }

  /* ------------------------------------------------------ architecture tour */
  var TOUR = [
    { t: 'The Whole Pipeline', r: [0, 0, 4030, 1680], c: 'End to end: (a) dataset collection, (b) the coverage predictor, and (c) the local and global planners built on it.' },
    { t: 'Simulate and Allocate', r: [30, 30, 830, 900], c: 'A 2D simulator renders a 220° laser into a 20 × 20 m robot-frame window at 0.05 m per pixel. Each of the 18 training maps gets samples in proportion to its complexity κ.' },
    { t: 'Shows, Seasons, Episodes', r: [840, 30, 740, 900], c: 'A show walks outward through 10 seasons. At each season, 157 goals on a 2.5 m ring are replayed as episodes from the frozen state.' },
    { t: 'Training Pairs', r: [1540, 30, 900, 900], c: 'Every season becomes one pair: the occupancy image I<sub>t</sub> and the field H<sub>t</sub> of coverage gain Δρ that its episodes actually measured.' },
    { t: 'The Predictor', r: [2470, 20, 1545, 920], c: 'A U-Net with a ResNet-34 encoder regresses Ĥ<sub>t</sub> ∈ [0, 1]<sup>W×W</sup> from I<sub>t</sub>, trained against H<sub>t</sub> with a weighted MSE.' },
    { t: 'Local Planner', r: [30, 960, 700, 705], c: 'When one angular component survives, its intensity-weighted centroid is the goal, g<sub>t</sub> = μ(Ĥ<sub>t</sub>), reached in a straight line.' },
    { t: 'Junction Verification', r: [690, 960, 2160, 705], c: 'More than one component proposes a junction. It registers only if at least 5 of 11 poses see it, and the robot then commits to the weakest branch.' },
    { t: 'Global Planner', r: [2850, 960, 1170, 705], c: 'Junction nodes keep their branches. New frontiers claim the branch that reaches them most cheaply, and when the window offers nothing the robot heads for the nearest active branch or frontier by A* distance.' }
  ];
  // Arrows of Fig. 2 traced in figure pixels, in the direction they flow. s: the tour stops
  // that light them up. k: the arrow's colour in the figure, red for data, gold for feedback.
  var FLOWS = [
    { k: 'red', s: [1], p: [[256,390],[256,424]] },
    { k: 'red', s: [1], p: [[256,530],[256,576],[444,576]] },
    { k: 'red', s: [1], p: [[696,524],[696,492]] },
    { k: 'red', s: [1], p: [[696,350],[696,318]] },
    { k: 'red', s: [1], p: [[290,744],[316,744]] },
    { k: 'red', s: [1], p: [[576,748],[610,748]] },
    { k: 'red', s: [2], p: [[1110,206],[1224,206]] },
    { k: 'red', s: [2], p: [[1352,224],[1352,390]] },
    { k: 'red', s: [2], p: [[1110,736],[1224,736]] },
    { k: 'red', s: [2], p: [[1352,720],[1352,590]] },
    { k: 'red', s: [2, 3], p: [[1436,480],[1800,480]] },
    { k: 'red', s: [3], p: [[2065,480],[2135,480]] },
    { k: 'red', s: [3, 4], p: [[2345,480],[2559,480]] },
    { k: 'red', s: [4], p: [[2575,480],[3495,480]] },
    { k: 'red', s: [4], p: [[3497,480],[3562,480],[3562,256],[3645,256]] },
    { k: 'red', s: [5, 6], p: [[680,1140],[724,1140]] },
    { k: 'red', s: [6], p: [[826,1236],[826,1266]] },
    { k: 'red', s: [6], p: [[936,1370],[960,1370],[960,1270],[992,1270]] },
    { k: 'red', s: [6], p: [[1995,1130],[2045,1130]] },
    { k: 'red', s: [6], p: [[1995,1436],[2040,1436]] },
    { k: 'red', s: [6], p: [[2215,1120],[2311,1120]] },
    { k: 'red', s: [6], p: [[2265,1436],[2311,1436]] },
    { k: 'red', s: [6], p: [[2509,1120],[2639,1120],[2639,1224]] },
    { k: 'red', s: [6], p: [[2509,1436],[2639,1436],[2639,1330]] },
    { k: 'red', s: [6, 7], p: [[2851,1284],[2889,1284],[2889,1150],[2921,1150]] },
    { k: 'red', s: [7], p: [[3125,1264],[3125,1296]] },
    { k: 'red', s: [7], p: [[3345,1424],[3401,1424],[3401,1150],[3459,1150]] },
    { k: 'red', s: [7], p: [[3671,1264],[3671,1300]] },
    { k: 'red', s: [5, 7], p: [[380,1030],[380,980],[3937,980],[3937,1427],[3887,1427]] },
    { k: 'gold', s: [6], p: [[2769,1260],[2769,996],[826,996],[826,1016]] },
    { k: 'gold', s: [5], p: [[2769,1320],[2769,1546],[655,1546],[655,1412]] },
    { k: 'gold', s: [5, 7], p: [[3670,1559],[3670,1580],[122,1580],[122,1412]] }
  ];
  (function tour() {
    var root = $('#tour'); if (!root) return;
    var stage = $('.tour__stage', root), cam = $('#tour-camera'), flowHost = $('#tour-flows');
    var steps = $('#tour-steps'), cap = $('#tour-cap'), play = $('#tour-play');
    var i = 0, timer = null, t0 = 0, DWELL = 6500, playing = !reduce.matches, visible = false, raf = 0;
    TOUR.forEach(function (s, k) {
      var b = el('button', null, s.t + '<i></i>');
      b.type = 'button'; b.setAttribute('role', 'tab');
      b.addEventListener('click', function () { go(k); setPlaying(false); });
      steps.appendChild(b);
    });
    var pills = $$('button', steps);
    // the arrow flows: a faint route under each arrow, and pulses travelling along it
    var flowEls = [], SVGNS = 'http://www.w3.org/2000/svg';
    FLOWS.forEach(function (f) {
      var d = 'M' + f.p.map(function (q) { return q[0] + ' ' + q[1]; }).join(' L');
      ['route', 'pulse'].forEach(function (kind) {
        var e = document.createElementNS(SVGNS, 'path');
        e.setAttribute('d', d); e.setAttribute('class', 'flow flow__' + kind + ' flow--' + f.k);
        flowHost.appendChild(e); flowEls.push({ el: e, s: f.s, kind: kind });
      });
    });
    function lightFlows(k) {
      flowEls.forEach(function (o) {
        var on = k === 0 || o.s.indexOf(k) >= 0;
        o.el.classList.toggle(o.kind === 'pulse' ? 'is-dim' : 'is-on', o.kind === 'pulse' ? !on : on);
      });
    }
    function frame() {
      var r = TOUR[i].r, W = stage.clientWidth, H = stage.clientHeight;
      var pad = i === 0 ? 1 : 0.94;
      var s = Math.min(W / r[2], H / r[3]) * pad;
      var tx = W / 2 - (r[0] + r[2] / 2) * s, ty = H / 2 - (r[1] + r[3] / 2) * s;
      if (i === 0) { tx = (W - 4030 * s) / 2; ty = (H - 1680 * s) / 2; }
      cam.style.transform = 'translate(' + tx.toFixed(1) + 'px,' + ty.toFixed(1) + 'px) scale(' + s.toFixed(5) + ')';
    }
    function go(k) {
      i = (k + TOUR.length) % TOUR.length; t0 = performance.now();
      pills.forEach(function (p, j) { p.setAttribute('aria-selected', j === i ? 'true' : 'false'); $('i', p).style.width = j < i ? '100%' : '0'; });
      cap.innerHTML = '<b>' + TOUR[i].t + '.</b>' + TOUR[i].c;
      lightFlows(i);
      frame();
    }
    function tick(now) {
      raf = 0;
      if (!playing || !visible) return;
      var u = Math.min(1, (now - t0) / DWELL);
      $('i', pills[i]).style.width = (u * 100).toFixed(1) + '%';
      if (u >= 1) go(i + 1);
      raf = requestAnimationFrame(tick);
    }
    function kick() { if (!raf && playing && visible) { t0 = performance.now() - (parseFloat($('i', pills[i]).style.width) || 0) / 100 * DWELL; raf = requestAnimationFrame(tick); } }
    function setPlaying(p) {
      playing = p; play.setAttribute('aria-pressed', p ? 'true' : 'false');
      play.setAttribute('aria-label', p ? 'Pause the tour' : 'Play the tour');
      if (p) kick();
    }
    play.addEventListener('click', function () { setPlaying(!playing); });
    window.addEventListener('resize', frame);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) { visible = es[0].isIntersecting; root.classList.toggle('is-offscreen', !visible); kick(); }, { threshold: .1 }).observe(stage);
    } else { visible = true; }
    go(0); setPlaying(playing);
  })();

  /* ---------------------------------------------------------- steppers */
  $$('[data-stepper]').forEach(function (root) {
    var btns = $$('.stepper__list button', root), img = $('.stepper__frame img', root), key = $('.stepper__key', root);
    var view = $('.stepper__view', root), cap = el('p', 'stepper__cap');
    view.appendChild(cap);
    function show(k) {
      btns.forEach(function (b, j) { if (j === k) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
      var b = btns[k];
      img.src = b.getAttribute('data-src');
      img.alt = $('b', b).textContent + '. ' + $('span', b).textContent;
      img.style.animation = 'none'; void img.offsetWidth; img.style.animation = '';
      if (key) { var kk = b.getAttribute('data-key'); key.hidden = !kk; if (kk) key.src = kk; }
      cap.innerHTML = '<b>' + $('b', b).textContent + '.</b> ' + $('span', b).innerHTML;
    }
    btns.forEach(function (b, k) {
      b.addEventListener('click', function () { show(k); });
      b.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); var n = Math.min(btns.length - 1, k + 1); btns[n].focus(); show(n); }
        if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); var p = Math.max(0, k - 1); btns[p].focus(); show(p); }
      });
    });
    btns.forEach(function (b) { var im = new Image(); im.src = b.getAttribute('data-src'); });   // warm the cache
    show(0);
  });

  /* ------------------------------------------------------ allocation chart */
  (function alloc() {
    var host = $('#alloc-chart'); if (!host) return;
    var max = T1[T1.length - 1][4], bars = [];
    T1.forEach(function (r) {
      var row = el('div', 'ab');
      row.innerHTML = '<span class="ab__name">' + r[0] + '</span><span class="ab__track"><span class="ab__bar"></span></span>' +
        '<span class="ab__val">' + fmt(r[4], 0) + '</span>' +
        '<span class="ab__tip">' + r[1] + ' rooms · ' + fmt(r[2], 0) + ' m² · κ ' + fmt(r[3], 0) + ' · ' + fmt(r[4] * 10, 0) + ' seasons</span>';
      host.appendChild(row); bars.push([$('.ab__bar', row), r[4] / max]);
    });
    var tot = el('div', 'ab ab--total');
    tot.innerHTML = '<span class="ab__name">Total</span><span></span><span class="ab__val">100,000</span>';
    host.appendChild(tot);
    function grow() { bars.forEach(function (b) { b[0].style.width = (b[1] * 100).toFixed(2) + '%'; }); }
    if (reduce.matches || !('IntersectionObserver' in window)) grow();
    else {
      var shown = false, o = new IntersectionObserver(function (es) { if (es[0].isIntersecting && !shown) { shown = true; grow(); o.disconnect(); } }, { threshold: .2 });
      o.observe(host);
      setTimeout(function () { if (!shown) grow(); }, 2500);   // never leave it empty
    }
  })();

  /* ------------------------------------------------------------- Table 2 */
  (function t2() {
    var t = $('#table2'); if (!t) return;
    var head = ['Model', 'Params', 'MAE ×10³ ↓', 'R² ↑', 'ε<sub>pk</sub> [m] ↓', 'α<sub>30</sub> [%] ↑', 'Time [ms] ↓'];
    var lower = [true, false, true, false, true];
    var best = [0, 1, 2, 3, 4].map(function (c) {
      var vs = T2.filter(function (r) { return r.v[c] != null && r.name !== 'All-Zero'; }).map(function (r) { return r.v[c]; });
      return lower[c] ? Math.min.apply(null, vs) : Math.max.apply(null, vs);
    });
    var h = '<thead><tr>' + head.map(function (x) { return '<th scope="col">' + x + '</th>'; }).join('') + '</tr></thead><tbody>';
    T2.forEach(function (r) {
      h += '<tr' + (r.ours ? ' class="is-ours"' : '') + '><td>' + r.name + (r.note ? '<sup>' + r.note + '</sup>' : '') + '</td><td>' + r.params + '</td>';
      r.v.forEach(function (v, c) {
        h += v == null ? '<td>–</td>' : '<td' + (v === best[c] ? ' class="best"' : '') + '>' + fmt(v, dec(v)) + '</td>';
      });
      h += '</tr>';
    });
    t.insertAdjacentHTML('beforeend', h + '</tbody>');
    var cap = $('caption', t); if (cap) cap.insertAdjacentHTML('beforeend', ' † Predicts zero everywhere.');
  })();

  /* --------------------------------------------------------- Tables 3, 4 */
  function cell(v, d, isBest) {
    return '<td' + (isBest ? ' class="best"' : '') + '>' + fmt(v[0], d) + ' <span class="pm">±' + fmt(v[1], d) + '</span></td>';
  }
  function groupRow(g, span) { return '<tr class="group"><td colspan="' + span + '">' + g + '</td></tr>'; }
  function tabs(host, names, onPick, initial) {
    names.forEach(function (n) {
      var b = el('button', null, n); b.type = 'button'; b.setAttribute('role', 'tab');
      b.addEventListener('click', function () { pick(n); });
      host.appendChild(b);
    });
    function pick(n) { $$('button', host).forEach(function (b) { b.setAttribute('aria-selected', b.textContent === n ? 'true' : 'false'); }); onPick(n); }
    pick(initial || names[0]);
  }
  (function t3() {
    var t = $('#table3'), host = $('#t3-tabs'); if (!t || !host) return;
    tabs(host, Object.keys(T3), function (name) {
      var rows = T3[name].rows;
      var complete = rows.map(function (r) { return r[1][0] >= 99.0 && r[1][1] === 0; });
      var p99 = Math.min.apply(null, rows.filter(function (r, i) { return complete[i]; }).map(function (r) { return r[0][0]; }));
      var rho = Math.max.apply(null, rows.map(function (r) { return r[1][0]; }));
      var ed = Math.max.apply(null, rows.map(function (r) { return r[2][0]; }));
      var et = Math.max.apply(null, rows.map(function (r) { return r[3][0]; }));
      var h = '<caption>' + name + ' (' + T3[name].area + '). Bold marks the best P99 among methods that complete every trial, the highest coverage and the best efficiencies.</caption>' +
        '<thead><tr><th scope="col">Method</th><th scope="col">P99 [m] ↓</th><th scope="col">ρ [%] ↑</th><th scope="col">η<sub>d</sub> [m²/m] ↑</th><th scope="col">η<sub>t</sub> [m²/s] ↑</th></tr></thead><tbody>';
      var g = '';
      METHODS.forEach(function (m, i) {
        if (m.group !== g && m.key !== 'ours') { g = m.group; h += groupRow(g, 5); }
        var r = rows[i];
        h += '<tr' + (m.key === 'ours' ? ' class="is-ours"' : '') + '><td><span class="dot" style="background:var(--m-' + m.key + ')"></span>' + m.label + '</td>' +
          cell(r[0], 1, complete[i] && r[0][0] === p99) + cell(r[1], 1, r[1][0] === rho) + cell(r[2], 2, r[2][0] === ed) + cell(r[3], 2, r[3][0] === et) + '</tr>';
      });
      t.innerHTML = h + '</tbody>';
    }, 'Large Office 7');
  })();
  (function t4() {
    var t = $('#table4'), host = $('#t4-tabs'); if (!t || !host) return;
    tabs(host, Object.keys(T4), function (name) {
      var rows = T4[name];
      var lower = [true, true, true, false, false, false, false], digits = [2, 2, 2, 1, 3, 2, 2];
      var best = lower.map(function (lo, c) {
        var vs = rows.map(function (r) { return r[c][0]; });
        return lo ? Math.min.apply(null, vs) : Math.max.apply(null, vs);
      });
      var h = '<caption>' + name + '. Bold marks the best value in each column.</caption><thead><tr><th scope="col">Method</th>' +
        ['P′<sub>99</sub> ↓', 'P′<sub>95</sub> ↓', 'P′<sub>90</sub> ↓', 'ρ [%] ↑', 'AUC ↑', 'η<sub>d</sub> ↑', 'η<sub>t</sub> ↑'].map(function (x) { return '<th scope="col">' + x + '</th>'; }).join('') +
        '</tr></thead><tbody>';
      var g = '';
      METHODS.forEach(function (m, i) {
        if (m.group !== g && m.key !== 'ours') { g = m.group; h += groupRow(g, 8); }
        h += '<tr' + (m.key === 'ours' ? ' class="is-ours"' : '') + '><td><span class="dot" style="background:var(--m-' + m.key + ')"></span>' + m.label + '</td>';
        rows[i].forEach(function (v, c) { h += cell(v, digits[c], best[c] != null && v[0] === best[c]); });
        h += '</tr>';
      });
      t.innerHTML = h + '</tbody>';
    });
  })();

  /* ------------------------------------------------- four candidate goals */
  (function goals() {
    var host = $('#goals'); if (!host) return;
    var G = [
      { tag: '(a)', name: 'Global Maxima', d: 2.5, g: .0168, note: 'Best gain, deep inside the room.' },
      { tag: '(b)', name: 'Weighted Centroid', d: .3, g: .0085, note: 'Opposing lobes cancel, so the goal lands almost on the robot.' },
      { tag: '(c)', name: 'Component Centroid', d: 1.5, g: .0168, note: 'IntelX: the best gain at the shortest distance.', ours: true },
      { tag: '(d)', name: 'Frontier Centroid', d: 6.6, g: .0168, note: 'Best gain, but more than four times as far.' }
    ];
    G.forEach(function (x) {
      var c = el('article', 'goal' + (x.ours ? ' goal--ours' : ''));
      c.innerHTML = '<span class="goal__tag">' + x.tag + (x.ours ? ' · IntelX' : '') + '</span><span class="goal__name">' + x.name + '</span>' +
        '<div class="goal__row"><span>d ' + x.d.toFixed(1) + ' m</span><span class="goal__track"><span class="goal__fill" style="width:' + (x.d / 6.6 * 100).toFixed(1) + '%"></span></span></div>' +
        '<div class="goal__row"><span>Δρ ' + x.g.toFixed(4) + '</span><span class="goal__track"><span class="goal__fill" style="width:' + (x.g / .0168 * 100).toFixed(1) + '%"></span></span></div>' +
        '<p class="goal__note">' + x.note + '</p>';
      host.appendChild(c);
    });
  })();

  /* --------------------------------------------------------- citation */
  var copy = $('#bib-copy');
  if (copy) copy.addEventListener('click', function () {
    var txt = $('#bibtex').textContent;
    var done = function () { copy.textContent = 'Copied'; copy.classList.add('is-done'); setTimeout(function () { copy.textContent = 'Copy'; copy.classList.remove('is-done'); }, 1800); };
    var fallback = function () { var r = document.createRange(); r.selectNodeContents($('#bibtex')); var s = getSelection(); s.removeAllRanges(); s.addRange(r); copy.textContent = 'Selected'; };
    try { navigator.clipboard.writeText(txt).then(done, fallback); } catch (e) { fallback(); }
  });

  /* ------------------------------------------- clips play while on screen */
  var clips = $$('video[data-autoplay]');
  if (clips.length && 'IntersectionObserver' in window && !reduce.matches) {
    var co = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) { if (v.preload === 'none') v.preload = 'auto'; var p = v.play(); if (p && p.catch) p.catch(function () {}); }
        else v.pause();
      });
    }, { threshold: .4 });
    clips.forEach(function (v) { co.observe(v); });
  }
  clips.forEach(function (v) { v.addEventListener('click', function () { if (v.paused) v.play(); else v.pause(); }); });
})();
