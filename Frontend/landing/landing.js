/* landing.js — front-of-app screen router (classic script, shares window.*).
 *
 * Flow:  chooser (Indoor|Outdoor · 2D|3D)  →  [3D only] import  →  preprocess
 *        →  workspace (the existing sidebar + tabbed panel).
 *
 * Publishes the single source of truth other modules read:
 *   window.appMode   = { environment: 'indoor'|'outdoor', dim: '2d'|'3d' }
 *   window.appImport = { modelFiles: FileList|null, tabFile, csvFile }
 *
 * The workspace, Map-Coverage 2D/3D, and the Simulation tab all key off
 * window.appMode.dim; this file nudges the existing viewer (via the
 * window.__viewer3d hook exposed by viewer3d.js) to match the chosen mode. */
(function () {
  window.appMode = { environment: null, dim: null };
  window.appImport = { modelFiles: null, tabFile: null, csvFile: null };
  // Voxelization target the preprocess screen chooses (drives voxelize_city.py /
  // voxelize.py). Read by the 3D Simulation tab. 2.5d = extruded-footprint prisms
  // (robust for OSM/GIS); 3d = full volumetric (watertight/detailed meshes).
  window.appVoxel = { mode: '2.5d', cell_m: 1.0 };

  const SCREENS = ['screenChooser', 'screenImport', 'screenPreprocess', 'screenWorkspace'];
  function showScreen(id) {
    SCREENS.forEach((s) => { const el = document.getElementById(s); if (el) el.hidden = (s !== id); });
    window.scrollTo(0, 0);
  }
  window.showScreen = showScreen;

  /* ---- Screen 0: mode chooser ------------------------------------------ */
  const envButtons = document.querySelectorAll('#screenChooser [data-env]');
  const dimButtons = document.querySelectorAll('#screenChooser [data-dim]');
  const chooserGo  = document.getElementById('chooserGo');
  let pickEnv = 'indoor', pickDim = null;

  function refreshChooser() {
    envButtons.forEach((b) => b.classList.toggle('active', b.dataset.env === pickEnv));
    dimButtons.forEach((b) => b.classList.toggle('active', b.dataset.dim === pickDim));
    if (chooserGo) chooserGo.disabled = !(pickEnv && pickDim);
  }
  envButtons.forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return; pickEnv = b.dataset.env; refreshChooser();
  }));
  dimButtons.forEach((b) => b.addEventListener('click', () => { pickDim = b.dataset.dim; refreshChooser(); }));
  refreshChooser();

  if (chooserGo) chooserGo.addEventListener('click', () => {
    window.appMode.environment = pickEnv;
    window.appMode.dim = pickDim;
    if (pickDim === '3d') { showScreen('screenImport'); }
    else { enterWorkspace(); }
  });

  /* ---- Screen 1: import (3D) ------------------------------------------- */
  const modelInput = document.getElementById('importModel');
  const tabInput   = document.getElementById('importTab');
  const modelList  = document.getElementById('importModelList');
  const tabList    = document.getElementById('importTabList');

  function listFiles(el, files, fallback) {
    if (!el) return;
    el.innerHTML = '';
    if (!files || !files.length) { el.innerHTML = '<li class="dim">' + fallback + '</li>'; return; }
    Array.from(files).forEach((f) => {
      const li = document.createElement('li');
      li.textContent = f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
      el.appendChild(li);
    });
  }
  if (modelInput) modelInput.addEventListener('change', (e) => {
    window.appImport.modelFiles = e.target.files.length ? e.target.files : null;
    listFiles(modelList, e.target.files, 'Using built-in 7th-floor model');
  });
  if (tabInput) tabInput.addEventListener('change', (e) => {
    window.appImport.tabFile = e.target.files[0] || null;
    listFiles(tabList, e.target.files, 'No .TAB registration file');
  });
  listFiles(modelList, null, 'Using built-in 7th-floor model');
  listFiles(tabList, null, 'No .TAB registration file');

  bind('importBack', () => showScreen('screenChooser'));
  bind('importGo',   () => { prepPreprocess(); showScreen('screenPreprocess'); });

  /* ---- Screen 2: preprocess (3D) -------------------------------------- */
  const geoBox   = document.getElementById('geoSummary');
  const matBox   = document.getElementById('matSummary');
  const csvInput = document.getElementById('preCsv');
  const csvStat  = document.getElementById('preCsvStatus');
  const voxSeg   = document.getElementById('voxModeSeg');
  const voxCell  = document.getElementById('voxCell');
  const voxNote  = document.getElementById('voxNote');
  const voxSrc   = document.getElementById('voxSourceNote');
  const ENGINE_VOXEL_WARN = 40e6;   // matches voxelize_city.py ENGINE_VOXEL_WARN
  let voxExtentM = null;            // [X, Y↑, Z] metres, when a grid extent is known

  function estimateGrid(cell) {
    if (!voxExtentM || !(cell > 0)) return null;
    const g = voxExtentM.map((m) => Math.max(1, Math.ceil(m / cell) + 1));
    const n = g[0] * g[1] * g[2];
    return { g, n, memMB: Math.round(n / 1e6) };
  }
  function renderVoxNote() {
    if (!voxNote) return;
    const V = window.appVoxel;
    const outdoor = window.appMode.environment === 'outdoor';
    const modeTxt = V.mode === '3d'
      ? '3D · full volumetric'
      : '2.5D · extruded footprints';
    const est = estimateGrid(V.cell_m);
    let s = 'Target: <b>' + modeTxt + '</b> at <b>' + V.cell_m + ' m</b> cells.';
    let warn = false;
    if (est) {
      s += ' Grid ≈ <b>' + est.g.join(' × ') + '</b> = <b>' + est.n.toLocaleString() +
           '</b> voxels (' + est.memMB + ' MB).';
      if (est.n > ENGINE_VOXEL_WARN) {
        warn = true;
        s += ' Large for a dense per-Tx solve — coarsen to <b>2–3 m</b> for engine runs (the grid builds fine).';
      }
    } else {
      s += ' Grid is built at import by <code>' + (outdoor ? 'voxelize_city.py' : 'voxelize.py') + '</code>.';
    }
    if (outdoor && V.mode === '3d')
      s += ' Raw OSM extrusions are not watertight — 3D can leave hollow shells; 2.5D is recommended for GIS footprints.';
    voxNote.className = 'vox-note' + (warn ? ' warn' : '');
    voxNote.innerHTML = s;
  }
  if (voxSeg) voxSeg.querySelectorAll('.vox-opt').forEach((b) => b.addEventListener('click', () => {
    window.appVoxel.mode = b.dataset.vox;
    voxSeg.querySelectorAll('.vox-opt').forEach((x) => x.classList.toggle('active', x === b));
    renderVoxNote();
  }));
  if (voxCell) voxCell.addEventListener('input', () => {
    const v = parseFloat(voxCell.value);
    if (v > 0) { window.appVoxel.cell_m = v; renderVoxNote(); }
  });

  function prepPreprocess() {
    const outdoor = window.appMode.environment === 'outdoor';
    const voxer = outdoor ? 'voxelize_city.py' : 'voxelize.py';
    if (voxSrc) voxSrc.innerHTML = 'The model is voxelized into a material grid by <code>' +
      voxer + '</code>. Pick how buildings are turned into voxels:';
    // Imported model: reflect IT (name + estimated grid) here, not the built-in scene's grid.
    const imp = window.appImport && window.appImport.modelFiles;
    if (imp && imp.length) {
      const names = Array.from(imp).map((f) => f.name).join(', ');
      voxExtentM = [80, 30, 80];      // default sandbox bounds; the Simulation tab lets you resize
      if (voxSrc) voxSrc.innerHTML = 'Your imported model is voxelized into a material grid <b>in the browser</b> on the Simulation tab (analytic tier). Pick the voxelization:';
      if (geoBox) {
        const est = estimateGrid(window.appVoxel.cell_m);
        const rows = [
          ['Imported model', names],
          ['Processing', 'voxelized in-browser on import'],
          ['Sandbox bounds', voxExtentM.join(' × ') + ' m (default — adjust in the Simulation tab)'],
          ['Grid estimate', est ? (est.g.join(' × ') + ' = ' + est.n.toLocaleString() + ' voxels') : '—'],
        ];
        geoBox.innerHTML = rows.map((r) => '<div class="k">' + r[0] + '</div><div class="v">' + r[1] + '</div>').join('');
      }
      if (matBox) matBox.innerHTML = '<li>barrier (walls) · default multiwall loss (dB per crossing)</li>';
      renderVoxNote();
      return;
    }
    const A = window.SIM3D_ASSETS && window.SIM3D_ASSETS.manifest_3d;
    voxExtentM = (A && A.grid_shape && A.cell_size_m)
      ? A.grid_shape.map((n) => n * A.cell_size_m) : null;
    if (geoBox) {
      if (A && A.grid_shape) {
        const gs = A.grid_shape;
        const rows = [
          ['Loaded grid (X · Y↑ · Z)', gs.join(' × ')],
          ['Cell size', (A.cell_size_m != null ? A.cell_size_m : '?') + ' m'],
          [outdoor ? 'Domain height' : 'Ceiling height', (A.ceiling_height_m ? A.ceiling_height_m.toFixed(2) : '?') + ' m'],
          ['Voxels', (gs[0] * gs[1] * gs[2]).toLocaleString()],
          ['Source', voxer + (A.domain ? ' · ' + A.domain : ' (precomputed)')],
        ];
        geoBox.innerHTML = rows.map((r) => '<div class="k">' + r[0] + '</div><div class="v">' + r[1] + '</div>').join('');
      } else {
        geoBox.innerHTML = '<div class="k">Grid</div><div class="v">built at import · ' + voxer + '</div>';
      }
    }
    const mats = (A && A.materials) || [];
    if (matBox) matBox.innerHTML = mats.map((m) =>
      '<li>' + m.name + ' · ' + (m.loss_per_m_db ? (m.loss_per_m_db + ' dB/m') : ((m.loss_db || 0) + ' dB')) + '</li>').join('');
    renderVoxNote();
  }

  const csvSummary  = document.getElementById('csvSummary');
  const csvCoordSys = document.getElementById('csvCoordSys');
  const csvCols     = document.getElementById('csvCols');
  const csvDefaults = document.getElementById('csvDefaults');
  const csvValid    = document.getElementById('csvValidation');
  let csvText = null, csvCoordPref = 'auto', csvElevDefault = 1.5;

  function sysLabel(s) {
    return s === 'pixel' ? 'pixel (px/py)' : s === 'geographic' ? 'geographic (lon/lat)'
         : s === 'projected' ? 'projected x/y (m)' : 'unknown';
  }

  // Self-contained CSV inspection for the preprocess screen — detects the
  // coordinate system + column mapping, counts usable rows, and reports what is
  // missing. (Separate from viewer3d.parseRFCsv, which only returns points.)
  function inspectCsv(text, pref) {
    const lines = text.replace(/^﻿/, '').trim().split(/\r?\n/).filter((l) => l.length);
    if (lines.length < 2) return { ok: false, error: 'need a header row + at least one data row' };
    const H = lines[0].split(',').map((s) => s.trim().replace(/^["']|["']$/g, ''));
    const HL = H.map((s) => s.toLowerCase());
    const body = lines.slice(1).map((l) => l.split(','));
    const find  = (subs) => HL.findIndex((h) => subs.some((s) => h.includes(s)));
    const exact = (names) => HL.findIndex((h) => names.includes(h));
    const iPx = exact(['px']), iPy = exact(['py']);
    const iLon = find(['lon', 'lng', 'long', 'east']), iLat = find(['lat', 'north']);
    const iXp = exact(['x']), iYp = exact(['y']);

    let sys = pref;
    if (sys === 'auto')
      sys = (iPx >= 0 && iPy >= 0) ? 'pixel' : (iLon >= 0 && iLat >= 0) ? 'geographic'
          : (iXp >= 0 && iYp >= 0) ? 'projected' : null;
    let iX = -1, iY = -1;
    if (sys === 'pixel') { iX = iPx; iY = iPy; }
    else if (sys === 'geographic') { iX = iLon; iY = iLat; }
    else if (sys === 'projected') { iX = iXp; iY = iYp; }

    let iV = find(['rsrp', 'rsrq', 'cinr', 'rssi', 'dbm', 'signal', 'value']);
    if (iV < 0) for (let k = H.length - 1; k >= 0; k--)
      if (body.every((r) => r[k] !== undefined && r[k].trim() !== '' && isFinite(+r[k]))) { iV = k; break; }
    const iE = find(['elev', 'height', 'agl', 'floor_z', 'alt']);
    const iT = find(['t_sec', 'timestamp', 'time', 'sec', 'order', 'seq', 'index']);
    const unit = iV >= 0 && /rsrp|dbm|rssi/.test(HL[iV]) ? 'dBm'
               : iV >= 0 && /rsrq|cinr|s[in]nr/.test(HL[iV]) ? 'dB' : '';

    let nValid = 0;
    if (iX >= 0 && iY >= 0) for (const r of body)
      if (isFinite(+r[iX]) && isFinite(+r[iY])) nValid++;
    return {
      ok: iX >= 0 && iY >= 0, sys, nRows: body.length, nValid, nSkipped: body.length - nValid,
      valueMissing: iV < 0,
      cols: { x: iX >= 0 ? H[iX] : null, y: iY >= 0 ? H[iY] : null,
              value: iV >= 0 ? H[iV] : null, unit, elev: iE >= 0 ? H[iE] : null, time: iT >= 0 ? H[iT] : null },
    };
  }

  function updateElevText() {
    const el = document.getElementById('csvElevRowVal');
    if (el) el.textContent = 'default ' + csvElevDefault + ' m';
    if (window.appImport.csvMeta) window.appImport.csvMeta.elevDefault = csvElevDefault;
  }

  function renderCsv() {
    if (!csvText) return;
    const m = inspectCsv(csvText, csvCoordPref);
    window.appImport.csvMeta = m.ok
      ? { coordSys: m.sys, valueCol: m.cols.value, unit: m.cols.unit,
          elevDefault: m.cols.elev ? null : csvElevDefault, points: m.nValid }
      : null;
    if (csvSummary) csvSummary.hidden = false;
    if (csvCoordSys) csvCoordSys.value = csvCoordPref;

    if (csvStat) {
      if (!m.ok) {
        csvStat.className = 'step-status err';
        csvStat.textContent = m.error ? ('CSV error: ' + m.error)
          : 'No coordinate columns found — need px/py, lon/lat, or x/y.';
      } else {
        csvStat.className = 'step-status ok';
        csvStat.textContent = '✓ ' + m.nValid.toLocaleString() + ' points · ' + sysLabel(m.sys) +
          (m.cols.unit ? ' · ' + m.cols.unit : '');
      }
    }

    const axis = (m.sys === 'geographic') ? ['lon', 'lat'] : (m.sys === 'pixel') ? ['px', 'py'] : ['x', 'y'];
    const row = (k, v, cls) => '<div class="k">' + k + '</div><div class="v' + (cls ? ' ' + cls : '') + '">' + v + '</div>';
    const elevCell = m.cols.elev
      ? m.cols.elev
      : '<span id="csvElevRowVal">default ' + csvElevDefault + ' m</span>';
    if (csvCols) csvCols.innerHTML = [
      row('Data points', m.nValid.toLocaleString() + (m.nSkipped ? ' (' + m.nSkipped + ' skipped)' : ''), m.nSkipped ? 'miss' : ''),
      row('X · ' + axis[0], m.cols.x || '— missing —', m.cols.x ? '' : 'miss'),
      row('Y · ' + axis[1], m.cols.y || '— missing —', m.cols.y ? '' : 'miss'),
      row('Value' + (m.cols.unit ? ' (' + m.cols.unit + ')' : ''), m.cols.value || '— missing —', m.cols.value ? '' : 'miss'),
      row('Elevation', elevCell, m.cols.elev ? '' : 'def'),
      row('Time / order', m.cols.time || 'row order', m.cols.time ? '' : 'def'),
    ].join('');

    const defs = [];
    if (!m.cols.elev) defs.push('<div class="cd-row">No elevation column — placing points at Rx height ' +
      '<input type="number" id="csvElev" min="0" max="100" step="0.5" value="' + csvElevDefault + '"> m</div>');
    if (!m.cols.time) defs.push('<div class="cd-row">No time column — using <b>row order</b> for playback.</div>');
    if (m.valueMissing) defs.push('<div class="cd-row" style="color:#b0483f">No recognized value column — ' +
      'coverage needs a metric (RSRP / RSRQ / CINR / RSSI or a value column).</div>');
    if (csvDefaults) csvDefaults.innerHTML = defs.join('');
    const ei = document.getElementById('csvElev');
    if (ei) ei.addEventListener('input', () => { const v = parseFloat(ei.value); if (v >= 0) { csvElevDefault = v; updateElevText(); } });

    if (csvValid) {
      if (!m.ok) { csvValid.className = 'step-status err'; csvValid.textContent = '✗ Cannot register: no coordinate columns.'; }
      else if (m.valueMissing) { csvValid.className = 'step-status err'; csvValid.textContent = '✗ Missing a value column — pick a CSV with a metric.'; }
      else if (m.nSkipped) { csvValid.className = 'step-status'; csvValid.textContent = '⚠ ' + m.nValid + ' of ' + m.nRows + ' rows usable (' + m.nSkipped + ' had non-numeric coordinates).'; }
      else { csvValid.className = 'step-status ok'; csvValid.textContent = '✓ Ready to register ' + m.nValid + ' points to the grid.'; }
    }
  }

  if (csvInput) csvInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    window.appImport.csvFile = f || null;
    if (!f) {
      csvText = null; window.appImport.csvMeta = null;
      if (csvSummary) csvSummary.hidden = true;
      if (csvStat) { csvStat.textContent = ''; csvStat.className = 'step-status'; }
      return;
    }
    const reader = new FileReader();
    reader.onload = () => { csvText = String(reader.result); csvCoordPref = 'auto'; renderCsv(); };
    reader.readAsText(f);
  });
  if (csvCoordSys) csvCoordSys.addEventListener('change', () => { csvCoordPref = csvCoordSys.value; renderCsv(); });

  bind('preBack', () => showScreen('screenImport'));
  bind('preGo',   enterWorkspace);

  /* ---- Screen 3: workspace -------------------------------------------- */
  function enterWorkspace() { showScreen('screenWorkspace'); applyModeToWorkspace(); }

  function applyModeToWorkspace() {
    const dim = window.appMode.dim;
    const badge = document.getElementById('modeBadgeText');
    if (badge) badge.textContent =
      (window.appMode.environment === 'indoor' ? 'Indoor' : 'Outdoor') + ' · ' + (dim === '3d' ? '3D' : '2D');

    // Simulation tab: show the matching variant.
    const p2 = document.getElementById('sim2dPanel');
    const p3 = document.getElementById('sim3dPanel');
    if (p2) p2.hidden = (dim === '3d');
    if (p3) p3.hidden = (dim !== '3d');

    // Map Coverage: match the dimension (viewer3d wires these buttons).
    const btn = document.getElementById(dim === '3d' ? 'mapMode3dBtn' : 'mapMode2dBtn');
    if (btn) btn.click();

    // Outdoor mode: show the NoMa OSM outdoor view (2D map / 3D city with the whole 7/24 walk)
    // in the Map Coverage tab, replacing the indoor signal map. The iframe carries its own 2D/3D
    // toggle; we sync it to the chosen dimension on entry.
    const outdoor = window.appMode.environment === 'outdoor';
    const oHost = document.getElementById('outdoorMapHost');
    const mapTab = document.getElementById('mapTab');
    // CSS (#mapTab.outdoor-mode …) hides the indoor map + reveals the outdoor host with !important,
    // which beats the inline display that dashboard.js keeps re-applying to #plot on every refresh.
    if (mapTab) mapTab.classList.toggle('outdoor-mode', outdoor);
    if (oHost && outdoor) {
      let f = oHost.querySelector('iframe');
      if (!f) {
        f = document.createElement('iframe');
        f.src = 'Frontend/osm3d/outdoor_view.html';
        f.title = 'NoMa outdoor view';
        f.style.cssText = 'width:100%; height:100%; border:0; display:block;';
        oHost.appendChild(f);
      }
      const syncDim = () => { try { f.contentWindow.document.getElementById(dim === '3d' ? 'btn3d' : 'btn2d').click(); } catch (e) {} };
      if (f.contentWindow && f.contentWindow.document.readyState === 'complete') syncDim();
      else f.addEventListener('load', syncDim, { once: true });
    }

    // Feed an imported model + CSV into the 3D viewer, best-effort.
    if (dim === '3d' && window.__viewer3d) {
      if (window.appImport.modelFiles && window.__viewer3d.loadFiles) window.__viewer3d.loadFiles(window.appImport.modelFiles);
      if (window.appImport.csvFile && window.__viewer3d.importCsvText) {
        const r = new FileReader();
        r.onload = () => window.__viewer3d.importCsvText(String(r.result));
        r.readAsText(window.appImport.csvFile);
      }
    }

    // Let Plotly / three.js reclaim real width now the workspace is visible.
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  }

  bind('modeChange', () => showScreen('screenChooser'));

  function bind(id, fn) { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); }

  // Boot on the chooser.
  showScreen('screenChooser');
})();
