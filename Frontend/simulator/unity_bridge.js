/* unity_bridge.js — add a "3D engine: Three.js ⇄ Unity (C#)" toggle to the Simulation tab.
 *
 * This is the MAIN-repo (indoor-walk-test) port of the toggle. It is PURELY ADDITIVE: it inserts a
 * button above the 3D-sim viewport and, when switched to Unity, overlays an <iframe> running the Unity
 * WebGL C# RF engine. It does not touch the existing three.js sim, its solve worker, the faithful
 * preprocess, or the imported-model flow — those keep working exactly as before; Unity is just an
 * alternative renderer you can flip to and back.
 *
 * The Unity engine itself lives in the SEPARATE Unity repo, checked out on disk at the (gitignored)
 * nested path Unity_RF_Simulator/  (repo: indoor-outdoor-walk-test-with-Unity-Engine). We reference its
 * already-built WebGL bundle at runtime via the embed host page below — we do NOT vendor the ~21 MB
 * build into this repo, and we do NOT pull in the fork's older copies of the dashboard/sim code.
 *
 * Requirements (both already true in this workspace):
 *   - the Unity repo is present at ./Unity_RF_Simulator/ with a built WebGL bundle, and
 *   - the dashboard is served over http from the main-repo root (so Unity_RF_Simulator/… is reachable).
 * A fresh clone of THIS repo alone won't have the Unity folder; the toggle then shows a "build not
 * found" status and the three.js engine is unaffected.
 *
 * PR #5 (Unity repo) fix carried over: Unity runs in its OWN document (unity_embed.html) inside an
 * <iframe> — its own WebGL context — so it never contends with this page's three.js-WebGPU + onnxruntime
 * for the GPU. The parent drives it with postMessage; Unity's .jslib status calls are forwarded back.
 */
(function () {
  'use strict';

  // iframe host page, inside the nested Unity repo (relative to this dashboard, served from repo root).
  var EMBED_URL = 'Unity_RF_Simulator/unity_embed.html';
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

    // Toolbar: band + cut-plane + status.
    var bar = el('div', { id: 'unitySimBar' },
      'position:absolute; left:8px; top:8px; right:8px; z-index:2; display:flex; gap:10px; align-items:center;' +
      'font:12px system-ui,sans-serif; color:#dfe6f2; background:rgba(10,18,36,.72); padding:6px 10px; border-radius:6px;');
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

    var status = el('span', { id: 'unitySimStatus' }, 'margin-left:auto; opacity:.9;'); status.textContent = 'idle';
    state.statusEl = status;
    [title, bandSel, sliceLabel, volLabel, opLabel, schemeLabel, status].forEach(function (n) { bar.appendChild(n); });
    host.appendChild(bar);
    wrap.appendChild(host);

    bandSel.addEventListener('change', function () { send('SetBand', bandSel.value); });
    slice.addEventListener('input', function () { send('SetSlice', slice.value); });
    vol.addEventListener('change', function () { send('SetVolumetric', vol.checked ? '1' : '0'); });
    opacity.addEventListener('input', function () { send('SetOpacity', opacity.value); });
    scheme.addEventListener('change', function () { send('SetColorScheme', scheme.value); });

    // Engine toggle, above the viewport.
    var toggle = el('button', { type: 'button', id: 'unitySimToggle', 'class': 'sim-btn ghost' }, 'margin:0 0 8px 0;');
    toggle.textContent = '3D engine: Three.js  →  switch to Unity (C#)';
    wrap.parentNode.insertBefore(toggle, wrap);
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
