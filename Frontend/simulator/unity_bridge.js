/* unity_bridge.js — add a "3D engine: Three.js ⇄ Unity (C#)" toggle to the Simulation tab.
 *
 * This is the MAIN-repo (indoor-walk-test) port of the toggle. It is PURELY ADDITIVE: it inserts a
 * button above the 3D-sim viewport and, when switched to Unity, overlays an <iframe> running the Unity
 * WebGL C# RF engine. It does not touch the existing three.js sim, its solve worker, the faithful
 * preprocess, or the imported-model flow — those keep working exactly as before; Unity is just an
 * alternative renderer you can flip to and back.
 *
 * The Unity engine's built WebGL bundle is VENDORED into this repo at ./unity/ (the embed host page
 * unity/unity_embed.html + the build under unity/Build/WebGL). It is a copy of the output from the
 * separate Unity source repo (indoor-outdoor-walk-test-with-Unity-Engine); the C# source and Unity
 * project are NOT vendored here — only the ~32 MB runtime build — so a plain `git clone` of THIS repo
 * runs the Unity engine out of the box, with no second checkout. To refresh it, rebuild in the Unity
 * repo and re-copy unity/Build/WebGL (+ unity/unity_embed.html).
 *
 * Requirement: the dashboard must be served over http from the main-repo root (so unity/… is reachable);
 * on file:// the toggle shows a "build not found" status and the three.js engine is unaffected.
 *
 * PR #5 (Unity repo) fix carried over: Unity runs in its OWN document (unity_embed.html) inside an
 * <iframe> — its own WebGL context — so it never contends with this page's three.js-WebGPU + onnxruntime
 * for the GPU. The parent drives it with postMessage; Unity's .jslib status calls are forwarded back.
 *
 * Control surface (parity with Unity-repo PR #17 NoMa view + PR #18 instrument):
 *   Indoor  — band · cut height · volumetric · opacity · scheme (iBwave-style wall-masked heatmap).
 *   Outdoor — NoMa scene · 2D/3D view · donor base station · cascading sector picker (carrier → band →
 *             PCI, band drives the solve frequency) · antenna geometry (Panel/Omni) · heatmap on/off ·
 *             Move-Rx (click the street to reposition) · trajectory. The engine samples the baked 7/24
 *             walk points for the picked PCI and shows a live Spearman correlation vs measured RSRP.
 */
(function () {
  'use strict';

  // iframe host page, vendored in this repo (served from the main-repo root).
  var EMBED_URL = 'unity/unity_embed.html';
  // sector catalog, from the vendored Unity build's StreamingAssets.
  var CATALOG_URL = 'unity/Build/WebGL/StreamingAssets/base_stations.json';
  var BANDS_MHZ = [619, 1935, 2442, 3500, 5500, 6125];

  var state = { frame: null, active: false, ready: false, statusEl: null };

  function el(tag, attrs, css) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (css) e.style.cssText = css;
    return e;
  }

  // Unity (in the iframe) posts status/ready/progress back up here.
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.source !== 'unitySim') return;
    if (!state.statusEl) return;
    if (d.kind === 'status') state.statusEl.textContent = d.text;
    else if (d.kind === 'ready') { state.ready = true; state.statusEl.textContent = 'Unity engine ready.'; }
    else if (d.kind === 'progress') state.statusEl.textContent = 'loading Unity engine… ' + Math.round((d.value || 0) * 100) + '%';
    else if (d.kind === 'error') state.statusEl.textContent = 'Unity failed to load: ' + d.text;
  });

  function send(method, value) {
    if (!state.frame || !state.frame.contentWindow) return;
    try { state.frame.contentWindow.postMessage({ source: 'unitySim:cmd', method: method, value: String(value) }, '*'); }
    catch (e) { console.warn('[unitySim] postMessage failed', e); }
  }

  function init() {
    var wrap = document.querySelector('.sim3d-viewport-wrap');
    var three = document.getElementById('sim3dViewport');
    if (!wrap || !three || document.getElementById('unitySimHost')) return;

    if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';

    // Unity host (hidden until activated), overlaying the viewport — an <iframe>, not a shared canvas.
    var host = el('div', { id: 'unitySimHost' }, 'position:absolute; inset:0; display:none; background:#0f1728;');

    // Toolbar: band + cut-plane + status (wraps so the outdoor instrument controls fit).
    var bar = el('div', { id: 'unitySimBar' },
      'position:absolute; left:8px; top:8px; right:8px; z-index:2; display:flex; gap:10px; align-items:center;' +
      'flex-wrap:wrap; font:12px system-ui,sans-serif; color:#dfe6f2; background:rgba(10,18,36,.72); padding:6px 10px; border-radius:6px;');
    var title = el('strong'); title.textContent = 'Unity RF engine (C#)';
    var bandSel = el('select', { id: 'unitySimBand', title: 'Solve frequency (MHz)' });
    BANDS_MHZ.forEach(function (m) {
      var o = el('option'); o.value = m; o.textContent = m + ' MHz'; if (m === 3500) o.selected = true; bandSel.appendChild(o);
    });
    var sliceLabel = el('label', null, 'display:flex; gap:6px; align-items:center;'); sliceLabel.textContent = 'Cut height';
    var slice = el('input', { id: 'unitySimSlice', type: 'range', min: '0', max: '16', value: '6' });
    sliceLabel.appendChild(slice);

    // AARTOS-style volumetric heatmap controls.
    var volLabel = el('label', null, 'display:flex; gap:5px; align-items:center;');
    var vol = el('input', { id: 'unitySimVolumetric', type: 'checkbox' });
    volLabel.appendChild(vol); volLabel.appendChild(document.createTextNode('Volumetric'));

    var opLabel = el('label', null, 'display:flex; gap:5px; align-items:center;'); opLabel.textContent = 'Opacity';
    var opacity = el('input', { id: 'unitySimOpacity', type: 'range', min: '0', max: '100', value: '35' });
    opLabel.appendChild(opacity);

    var schemeLabel = el('label', null, 'display:flex; gap:5px; align-items:center;'); schemeLabel.textContent = 'Scheme';
    var scheme = el('select', { id: 'unitySimScheme' });
    ['rainbow', 'viridis'].forEach(function (s) {
      var o = el('option'); o.value = s; o.textContent = s.charAt(0).toUpperCase() + s.slice(1); scheme.appendChild(o);
    });
    schemeLabel.appendChild(scheme);

    // PR #6 — NoMa outdoor base-station view: scene + 2D/3D + donor + trajectory.
    var SITES = [
      ['BS7_forte_hall', 'Forte Hall (Verizon)'], ['BS6_1005_n_capitol', 'Sixth · 1005 N Capitol'],
      ['BS5_1140_n_capitol', 'Fifth · 1140 N Capitol'], ['BS4_55_m_st', 'Fourth · 55 M St'],
      ['BS2_900_nj_ave', 'Second · 900 NJ Ave']];
    var sceneLabel = el('label', null, 'display:flex; gap:5px; align-items:center;'); sceneLabel.textContent = 'Scene';
    var sceneSel = el('select', { id: 'unitySimScene', title: 'Indoor 7th-floor vs NoMa outdoor 3D' });
    [['indoor', 'Indoor floor'], ['outdoor', 'NoMa outdoor']].forEach(function (s) {
      var o = el('option'); o.value = s[0]; o.textContent = s[1]; sceneSel.appendChild(o);
    });
    sceneLabel.appendChild(sceneSel);
    var dimLabel = el('label', { id: 'unitySimDimWrap' }, 'display:none; gap:5px; align-items:center;'); dimLabel.textContent = 'View';
    var dimSel = el('select', { id: 'unitySimDim', title: '2D top-down vs 3D perspective' });
    [['3d', '3D perspective'], ['2d', '2D top-down']].forEach(function (s) {
      var o = el('option'); o.value = s[0]; o.textContent = s[1]; dimSel.appendChild(o);
    });
    dimLabel.appendChild(dimSel);
    var bsLabel = el('label', { id: 'unitySimBsWrap' }, 'display:none; gap:5px; align-items:center;'); bsLabel.textContent = 'Base stn';
    var bsSel = el('select', { id: 'unitySimBs', title: 'Donor base station' });
    SITES.forEach(function (s) { var o = el('option'); o.value = s[0]; o.textContent = s[1]; bsSel.appendChild(o); });
    bsLabel.appendChild(bsSel);
    var trajLabel = el('label', { id: 'unitySimTrajWrap' }, 'display:none; gap:5px; align-items:center;');
    var traj = el('input', { id: 'unitySimTraj', type: 'checkbox' }); traj.checked = true;
    trajLabel.appendChild(traj); trajLabel.appendChild(document.createTextNode('Trajectory'));

    // PR #6 v2 (instrument) — cascading sector picker (carrier → band → PCI) + antenna geometry + heatmap + move-Rx.
    function selWrap(id, text) {
      var l = el('label', { id: id + 'Wrap' }, 'display:none; gap:4px; align-items:center;');
      if (text) l.appendChild(document.createTextNode(text + ' '));
      var s = el('select', { id: id }); l.appendChild(s); return { label: l, sel: s };
    }
    var carrier = selWrap('unitySimCarrier', 'Carrier');
    var cellBand = selWrap('unitySimCellBand', 'Band');
    var pci = selWrap('unitySimPci', 'PCI');
    var geom = selWrap('unitySimGeom', 'Antenna');
    [['panel', 'Panel'], ['omni', 'Omni']].forEach(function (g) { var o = el('option'); o.value = g[0]; o.textContent = g[1]; geom.sel.appendChild(o); });
    var heatLabel = el('label', { id: 'unitySimHeatWrap' }, 'display:none; gap:5px; align-items:center;');
    var heat = el('input', { id: 'unitySimHeat', type: 'checkbox' }); heat.checked = true;
    heatLabel.appendChild(heat); heatLabel.appendChild(document.createTextNode('Heatmap'));
    var rxLabel = el('label', { id: 'unitySimRxWrap' }, 'display:none; gap:5px; align-items:center;');
    var rxChk = el('input', { id: 'unitySimRx', type: 'checkbox' });
    rxLabel.appendChild(rxChk); rxLabel.appendChild(document.createTextNode('Move Rx'));

    var status = el('span', { id: 'unitySimStatus' }, 'margin-left:auto; opacity:.9;'); status.textContent = 'idle';
    state.statusEl = status;
    [title, sceneLabel, dimLabel, bandSel, sliceLabel, bsLabel, carrier.label, cellBand.label, pci.label,
     geom.label, trajLabel, heatLabel, rxLabel, volLabel, opLabel, schemeLabel, status]
      .forEach(function (n) { bar.appendChild(n); });
    host.appendChild(bar);
    wrap.appendChild(host);

    // -- cascading sector catalog (fetched from the build's StreamingAssets) --------------------
    var CATALOG = null;
    function opts(sel, items) { sel.innerHTML = ''; items.forEach(function (it) { var o = el('option'); o.value = it[0]; o.textContent = it[1]; sel.appendChild(o); }); }
    function uniq(a) { return a.filter(function (v, i) { return a.indexOf(v) === i; }); }
    function siteSectors() { return (CATALOG && CATALOG[bsSel.value]) ? (CATALOG[bsSel.value].sectors || []).filter(function (s) { return s.freq_mhz > 1; }) : []; }
    function fillCarriers() { opts(carrier.sel, uniq(siteSectors().map(function (s) { return s.carrier || '?'; })).map(function (c) { return [c, c]; })); fillBands(); }
    function fillBands() {
      opts(cellBand.sel, uniq(siteSectors().filter(function (s) { return (s.carrier || '?') === carrier.sel.value; }).map(function (s) { return s.network + '|' + s.band; }))
        .map(function (b) { var p = b.split('|'); return [b, p[0].toUpperCase() + ' B' + p[1]]; }));
      fillPcis();
    }
    function fillPcis() {
      opts(pci.sel, siteSectors().filter(function (s) { return (s.carrier || '?') === carrier.sel.value && (s.network + '|' + s.band) === cellBand.sel.value; })
        .map(function (s) { return [s.pci + '|' + s.freq_mhz, 'PCI ' + s.pci + ' · ' + Math.round(s.freq_mhz) + ' MHz']; }));
      sendSector();
    }
    function sendSector() {
      if (!pci.sel.value) { send('SelectBaseStation', bsSel.value); return; }
      var pf = pci.sel.value.split('|');   // pci|freq
      send('SelectSector', bsSel.value + '|' + pf[0] + '|' + pf[1] + '|' + geom.sel.value);
    }
    fetch(CATALOG_URL)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d && d.sites) { CATALOG = {}; d.sites.forEach(function (s) { CATALOG[s.site_id] = s; }); if (sceneSel.value === 'outdoor') fillCarriers(); } })
      .catch(function () {});

    bandSel.addEventListener('change', function () { send('SetBand', bandSel.value); });
    slice.addEventListener('input', function () { send('SetSlice', slice.value); });
    vol.addEventListener('change', function () { send('SetVolumetric', vol.checked ? '1' : '0'); });
    opacity.addEventListener('input', function () { send('SetOpacity', opacity.value); });
    scheme.addEventListener('change', function () { send('SetColorScheme', scheme.value); });
    carrier.sel.addEventListener('change', fillBands);
    cellBand.sel.addEventListener('change', fillPcis);
    pci.sel.addEventListener('change', sendSector);
    geom.sel.addEventListener('change', sendSector);
    heat.addEventListener('change', function () { send('ShowHeatmap', heat.checked ? '1' : '0'); });
    rxChk.addEventListener('change', function () { send('SetRxMoveMode', rxChk.checked ? '1' : '0'); });

    function setOutdoorUI(outdoor) {
      dimLabel.style.display = outdoor ? 'flex' : 'none';
      bsLabel.style.display = outdoor ? 'flex' : 'none';
      [carrier.label, cellBand.label, pci.label, geom.label, heatLabel, rxLabel].forEach(function (n) { n.style.display = outdoor ? 'flex' : 'none'; });
      trajLabel.style.display = outdoor ? 'flex' : 'none';
      bandSel.style.display = outdoor ? 'none' : '';           // outdoor freq comes from the sector picker
      sliceLabel.style.display = 'flex';                        // elevation slice: indoor cut-height AND outdoor
    }
    sceneSel.addEventListener('change', function () {
      var outdoor = sceneSel.value === 'outdoor';
      send('SetScene', sceneSel.value);
      setOutdoorUI(outdoor);
      if (outdoor) { send('SetDimension', dimSel.value); if (CATALOG) fillCarriers(); else send('SelectBaseStation', bsSel.value); }
    });
    dimSel.addEventListener('change', function () { send('SetDimension', dimSel.value); });
    bsSel.addEventListener('change', function () { if (CATALOG) fillCarriers(); else send('SelectBaseStation', bsSel.value); });
    traj.addEventListener('change', function () { send('ShowTrajectory', traj.checked ? '1' : '0'); });

    // Engine toggle, above the viewport.
    var toggle = el('button', { type: 'button', id: 'unitySimToggle', 'class': 'sim-btn ghost' }, 'margin:0 0 8px 0;');
    toggle.textContent = '3D engine: Three.js  →  switch to Unity (C#)';
    // Insert the toggle ABOVE the .sim3d-layout grid, NOT into it. wrap.parentNode is the
    // `grid: 264px 1fr` layout; inserting the button there made it a grid cell that claimed the
    // 1fr column and pushed the viewport to the next row's 264px column — squeezing both the Unity
    // host and the three.js viewport to 264px. Placing it before the grid keeps the viewport full-width.
    var simLayout = wrap.parentNode;                       // .sim3d-layout
    (simLayout.parentNode || simLayout).insertBefore(toggle, simLayout);
    toggle.addEventListener('click', function () {
      state.active = !state.active;
      if (state.active) {
        three.style.visibility = 'hidden';
        host.style.display = 'block';
        toggle.textContent = '3D engine: Unity (C#)  →  switch to Three.js';
        load();
      } else {
        host.style.display = 'none';
        three.style.visibility = '';
        toggle.textContent = '3D engine: Three.js  →  switch to Unity (C#)';
      }
    });
  }

  function load() {
    if (state.frame) return;   // lazy-create the iframe once, on first activation
    if (state.statusEl) state.statusEl.textContent = 'loading Unity engine…';
    var frame = el('iframe', { id: 'unitySimFrame', src: EMBED_URL, title: 'Unity RF engine', allow: 'autoplay; fullscreen' },
      'position:absolute; inset:0; width:100%; height:100%; border:0; display:block; background:#0f1728;');
    frame.onerror = function () { if (state.statusEl) state.statusEl.textContent = 'Unity host page failed to load — is Unity_RF_Simulator/ present and served over http?'; };
    document.getElementById('unitySimHost').appendChild(frame);
    state.frame = frame;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
