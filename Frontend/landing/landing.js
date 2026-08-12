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
    // Both 2D and 3D now go through Import (the project picker). 2D used to skip straight to
    // the workspace; it stops here first so a saved project can be chosen (or a new one made).
    prepImport();
    showScreen('screenImport');
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

  /* ---- Import project picker (2D + 3D) -------------------------------- */
  const NEW_PROJECT = '__new__';

  // Prepare the Import screen for the chosen mode: adapt the copy, populate the project
  // dropdown (backend + built-ins), preselect this mode's Default, and show/hide the
  // new-dataset cards. Called on entry from the chooser.
  function prepImport() {
    const env = window.appMode.environment, dim = window.appMode.dim;
    const title = document.getElementById('importTitle');
    const sub = document.getElementById('importSub');
    if (title) title.textContent = 'Import & dataset — ' +
      (env === 'indoor' ? 'Indoor' : 'Outdoor') + ' · ' + (dim === '3d' ? '3D' : '2D');
    if (sub) sub.textContent = dim === '3d'
      ? 'Load a saved project, or start a new one by importing a 3D model + its TAB registration.'
      : 'Load a saved project, or start a new one by importing measurement CSVs + a floor plan / area.';
    populateProjectPicker();
  }

  function populateProjectPicker() {
    const sel = document.getElementById('importProject');
    if (!sel) return;
    const env = window.appMode.environment, dim = window.appMode.dim;
    const fill = () => {
      const list = (window.getProjects ? window.getProjects(env, dim) : []);
      sel.innerHTML = list.map((p) => '<option value="' + p.id + '">' + p.label +
        (p.builtin ? '' : ' · saved') + '</option>').join('') +
        '<option value="' + NEW_PROJECT + '">➕ New project…</option>';
      const def = window.resolveDefaultProject && window.resolveDefaultProject(env, dim);
      if (def && list.some((p) => p.id === def.id)) sel.value = def.id;
      onProjectPick();
    };
    // Pull server/created projects first (best-effort; falls back to built-ins offline).
    (window.refreshProjects ? window.refreshProjects() : Promise.resolve()).then(fill, fill);
  }

  function onProjectPick() {
    const sel = document.getElementById('importProject');
    const legacyCards = document.getElementById('importNewCards');
    const builder = document.getElementById('importBuilder');
    const summary = document.getElementById('importProjectSummary');
    const goBtn = document.getElementById('importGo');
    const val = sel ? sel.value : null;
    const isNew = val === NEW_PROJECT;
    window.appImport.project = isNew ? null : (window.getProject ? window.getProject(val) : null);
    // Legacy 3D model/TAB cards belong to the old built-in 3D flow; keep them only for a saved
    // 3D project (optional model swap), hidden for 2D and for New (the builder replaces them).
    if (legacyCards) legacyCards.style.display = (!isNew && window.appMode.dim === '3d') ? '' : 'none';
    if (builder) builder.hidden = !isNew;
    if (summary) summary.style.display = isNew ? 'none' : '';
    if (goBtn) goBtn.style.display = isNew ? 'none' : '';   // New uses its own Create button
    if (isNew) renderBuilder(); else renderProjectSummary(window.appImport.project);
  }

  // Show the selected project's named data files + metadata — the "what will load" preview.
  function renderProjectSummary(m) {
    const box = document.getElementById('importProjectSummary');
    if (!box) return;
    if (!m) { box.innerHTML = ''; return; }
    const files = (m.data || []).map((d) => d.role + ' · ' + String(d.src).split('/').pop() +
      (d.optional ? ' (optional)' : '')).join('<br>');
    const rows = [
      ['Project', m.label + (m.builtin ? ' · built-in' : ' · saved')],
      ['Mode', (m.environment === 'indoor' ? 'Indoor' : 'Outdoor') + ' · ' + String(m.dim).toUpperCase()],
      ['Data files', files || '—'],
    ];
    if (m.model) rows.push(['3D model', m.model.label || m.model.src]);
    if (m.elevation) rows.push(['Elevation', m.elevation.mode === 'none' ? 'none (city heights)'
      : ((m.elevation.floor_z_m || 0) + ' m floor · ' + (m.elevation.ceiling_height_m || '?') + ' m ceiling')]);
    box.innerHTML = rows.map((r) => '<div class="k">' + r[0] + '</div><div class="v">' + r[1] + '</div>').join('');
  }

  const importProjectSel = document.getElementById('importProject');
  if (importProjectSel) importProjectSel.addEventListener('change', onProjectPick);

  /* ---- New-project builder (backend-powered) -------------------------- */
  // Renders this mode's required inputs, validates uploads live via POST /api/validate, and
  // POSTs /api/projects on Create. The backend shells the existing rasterize/bake scripts.
  let _bldTimer = null;

  function renderBuilder() {
    const box = document.getElementById('importBuilderInputs');
    if (!box) return;
    const env = window.appMode.environment, dim = window.appMode.dim;
    const fileRow = (id, label, attrs, hint) =>
      '<div class="field"><label for="' + id + '">' + label + '</label>' +
      '<input type="file" id="' + id + '" ' + attrs + '>' +
      (hint ? '<div class="step-note">' + hint + '</div>' : '') + '</div>';
    const numRow = (id, label, val, step, min) =>
      '<div class="field"><label for="' + id + '">' + label + '</label>' +
      '<input type="number" id="' + id + '" value="' + val + '" step="' + step + '"' +
      (min != null ? ' min="' + min + '"' : '') + '></div>';
    let html = '';
    if (dim === '2d') {
      html += fileRow('bldCsv', 'Measurement CSVs (folder)', 'multiple webkitdirectory',
        'Pick the folder of scanner CSVs — every .csv inside is read.');
      if (env === 'indoor') {
        html += fileRow('bldImage', 'Floor-plan image', 'accept=".png,.jpg,.jpeg"', 'PNG/JPG of the plan.');
        html += fileRow('bldTab', 'TAB georeference', 'accept=".tab,.TAB"',
          'MapInfo .TAB (≥3 GCPs) — sets scale + placement; crops the OSM basemap to it.');
      }
      html += numRow('bldCell', 'Grid cell (m)', env === 'indoor' ? '0.3' : '1', '0.1', '0.1');
    } else {
      html += fileRow('bldModel', '3D model',
        'accept=".dae,.obj,.mtl,.glb,.gltf,.fbx,.3ds,.stl,.kmz,.png,.jpg,.jpeg" multiple',
        'OBJ (+ .mtl & textures) / GLB / DAE / FBX / 3DS / STL / KMZ.');
      html += '<div class="field"><label for="bldElevMode">Elevation</label>' +
        '<select id="bldElevMode"><option value="none">None — flat / city building heights</option>' +
        '<option value="floor">Choose elevation…</option></select></div>' +
        '<div id="bldElevFields" style="display:none">' +
        numRow('bldFloorZ', 'Floor elevation (m)', '0', '0.5') +
        numRow('bldCeiling', 'Ceiling height (m)', '3', '0.1', '0.1') + '</div>';
      if (env === 'indoor')
        html += fileRow('bldTab', 'TAB georeference (optional)', 'accept=".tab,.TAB"', 'Optional for placement.');
      html += numRow('bldCell', 'Voxel cell (m)', '1', '0.5', '0.1');
    }
    box.innerHTML = html;
    box.querySelectorAll('input,select').forEach((el) => {
      el.addEventListener('change', scheduleValidate);
      el.addEventListener('input', scheduleValidate);
    });
    const em = document.getElementById('bldElevMode');
    if (em) em.addEventListener('change', () => {
      const f = document.getElementById('bldElevFields');
      if (f) f.style.display = em.value === 'none' ? 'none' : '';
    });
    const nm = document.getElementById('importNewName');
    if (nm && !nm._wired) { nm.addEventListener('input', scheduleValidate); nm._wired = true; }
    validateBuilder();
  }

  function scheduleValidate() { if (_bldTimer) clearTimeout(_bldTimer); _bldTimer = setTimeout(validateBuilder, 250); }

  function builderOpts() {
    const cell = parseFloat((document.getElementById('bldCell') || {}).value);
    const opts = {}; if (isFinite(cell)) opts.cell_m = cell;
    const em = document.getElementById('bldElevMode');
    if (em) opts.elevation = em.value === 'none' ? { mode: 'none' } : { mode: 'floor',
      floor_z_m: parseFloat((document.getElementById('bldFloorZ') || {}).value) || 0,
      ceiling_height_m: parseFloat((document.getElementById('bldCeiling') || {}).value) || 0 };
    return opts;
  }

  function builderFormData() {
    const fd = new FormData();
    fd.append('environment', window.appMode.environment);
    fd.append('dim', window.appMode.dim);
    fd.append('label', (document.getElementById('importNewName') || {}).value || '');
    fd.append('opts', JSON.stringify(builderOpts()));
    const add = (id, field) => {
      const el = document.getElementById(id);
      if (el && el.files) Array.from(el.files).forEach((f) => fd.append(field, f));
    };
    add('bldCsv', 'csv'); add('bldImage', 'image'); add('bldTab', 'tab'); add('bldModel', 'model');
    return fd;
  }

  function validateBuilder() {
    const report = document.getElementById('importBuilderReport');
    const create = document.getElementById('importCreate');
    const note = document.getElementById('importBuilderNote');
    if (!report) return;
    fetch(window.PROJECT_API_BASE + '/validate', { method: 'POST', body: builderFormData() })
      .then((r) => r.json())
      .then((rep) => {
        report.className = 'step-status';
        report.innerHTML = renderReport(rep);
        if (create) create.disabled = !rep.ok;
        if (note) note.textContent = rep.ok ? 'Ready to create.' : 'Provide the required inputs above.';
      })
      .catch(() => {
        report.className = 'step-status err';
        report.textContent = '✗ Backend not reachable — run  python Backend/server/app.py  then open http://127.0.0.1:8000';
        if (create) create.disabled = true;
        if (note) note.textContent = '';
      });
  }

  function renderReport(rep) {
    const row = (x, opt) => '<div style="color:' + (x.ok ? '#2a7d46' : '#b0483f') + '">' +
      (x.ok ? '✓' : '✗') + ' ' + x.role + (opt ? ' (optional)' : '') + ' — ' + (x.detail || '') + '</div>';
    let h = '';
    (rep.required || []).forEach((x) => { h += row(x, false); });
    (rep.optional || []).forEach((x) => { h += row(x, true); });
    (rep.warnings || []).forEach((w) => { h += '<div style="color:#b0771f">⚠ ' + w + '</div>'; });
    if (rep.grid) h += '<div style="color:#888">grid ≈ ' + rep.grid.cols + ' × ' + rep.grid.rows +
      ' @ ' + rep.grid.cell_m + ' m' + (rep.grid.extent_m ? ' (' + rep.grid.extent_m.join(' × ') + ' m)' : '') + '</div>';
    return h;
  }

  function createProject() {
    const create = document.getElementById('importCreate');
    const note = document.getElementById('importBuilderNote');
    if (create) create.disabled = true;
    if (note) note.textContent = 'Creating… rasterizing + baking on the backend (this can take a moment).';
    fetch(window.PROJECT_API_BASE + '/projects', { method: 'POST', body: builderFormData() })
      .then((r) => r.json().then((body) => ({ status: r.status, body })))
      .then(({ status, body }) => {
        if (status === 201 && body.id) {
          // A new dataset declares its own `const records`; reload into a fresh scope for it.
          location.search = '?project=' + encodeURIComponent(body.id);
        } else {
          if (note) note.textContent = '✗ ' + (body.error || ('failed (HTTP ' + status + ')'));
          if (create) create.disabled = false;
        }
      })
      .catch((e) => { if (note) note.textContent = '✗ ' + e.message; if (create) create.disabled = false; });
  }

  const importCreateBtn = document.getElementById('importCreate');
  if (importCreateBtn) importCreateBtn.addEventListener('click', createProject);

  bind('importBack', () => showScreen('screenChooser'));
  bind('importGo', () => {
    // The picker set window.appImport.project. 3D continues to Preprocess (voxel grid); 2D
    // enters the workspace directly (its lighter Preprocess/Exports view is added next phase).
    if (window.appMode.dim === '3d') { prepPreprocess(); showScreen('screenPreprocess'); }
    else { enterWorkspace(); }
  });

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
  // Boot the active project's data (via project_loader.js) — which also injects
  // dashboard.js/spectrum.js the first time — THEN apply the mode. window.appImport.project
  // is set by the Import project picker (added in a later phase); until then we resolve the
  // built-in Default for the chosen environment/dim. The loader dedupes, so re-entering the
  // workspace (change mode) reuses the already-loaded data and never re-wires the dashboard.
  function enterWorkspace() {
    showScreen('screenWorkspace');
    const manifest = (window.appImport && window.appImport.project) ||
      (window.resolveDefaultProject && window.resolveDefaultProject(window.appMode.environment, window.appMode.dim));
    const boot = (window.loadProject && manifest) ? window.loadProject(manifest) : Promise.resolve();
    const after = () => { applyModeToWorkspace(); wireSaveButton(); };
    boot.then(after).catch((e) => {
      console.error('[workspace] project load failed:', e);
      after();   // still show the shell so the failure is visible, not blank
    });
  }

  // SAVE button: visible only for a created project; commits a new version to the workspace-store
  // branch via the backend (never main). Publishing that branch to GitHub is a separate step.
  function wireSaveButton() {
    const btn = document.getElementById('wsSave');
    const note = document.getElementById('wsSaveNote');
    if (!btn) return;
    const proj = window.activeProject && window.activeProject();
    btn.hidden = !(proj && proj.id && !proj.builtin);
    if (note) note.textContent = '';
    if (btn._wired) return;
    btn._wired = true;
    btn.addEventListener('click', () => {
      const p = window.activeProject && window.activeProject();
      if (!p || !p.id) return;
      btn.disabled = true;
      if (note) note.textContent = 'saving…';
      fetch(window.PROJECT_API_BASE + '/projects/' + encodeURIComponent(p.id) + '/save', { method: 'POST' })
        .then((r) => r.json())
        .then((res) => {
          btn.disabled = false;
          if (res.error) { if (note) note.textContent = '✗ ' + res.error; return; }
          if (note) note.textContent = res.saved
            ? ('✓ saved v' + (res.version && res.version.version) + ' to workspace-store')
            : '✓ no changes since last save';
        })
        .catch((e) => { btn.disabled = false; if (note) note.textContent = '✗ ' + e.message; });
    });
  }

  // Outdoor mode: drive the outdoor iframe's dot filter from the Coverage sidebar. Repopulate the
  // sidebar selects from the outdoor walk's facets and intercept their changes in the CAPTURE phase
  // (stopImmediatePropagation) so dashboard.js's indoor network→band→pci cascade doesn't clobber
  // the outdoor options. Carrier ↔ operator (derived from PCI in the bake).
  let _outdoorSidebar = null;
  // Every outdoor iframe that reads the Coverage sidebar filter: the Map Coverage view AND the Time
  // Elapse replay. Both honour the same postMessage protocol, so the sidebar drives them together.
  function outdoorIframes() {
    return ['outdoorMapHost', 'outdoorTimeHost']
      .map((id) => { const h = document.getElementById(id); return h ? h.querySelector('iframe') : null; })
      .filter((f) => f && f.contentWindow);
  }
  function broadcastOutdoorFilter() {
    const v = (id) => { const el = document.getElementById(id); return el ? el.value : ''; };
    const msg = { source: 'outdoorFilter', operator: v('carrier'), network: v('network'), band: v('band'), pci: v('pci'), channel: v('channel') };
    outdoorIframes().forEach((f) => { try { f.contentWindow.postMessage(msg, '*'); } catch (e) {} });
  }
  function wireOutdoorSidebar(frame) {
    const SEL = { carrier: 'operators', network: 'networks', band: 'bands', pci: 'pcis', channel: 'channels' };
    const ids = Object.keys(SEL);
    const readAndPost = broadcastOutdoorFilter;   // broadcast to every outdoor iframe (map + time)
    const populate = (facets) => {
      ids.forEach((id) => {
        const el = document.getElementById(id); if (!el) return;
        const vals = facets[SEL[id]] || [];
        el.innerHTML = '<option value="">All</option>' + vals.map((x) => '<option value="' + x + '">' + x + '</option>').join('');
      });
      readAndPost();
    };
    unwireOutdoorSidebar();
    const onChange = (e) => { readAndPost(); e.stopImmediatePropagation(); };
    ids.forEach((id) => { const el = document.getElementById(id); if (el) el.addEventListener('change', onChange, true); });
    _outdoorSidebar = { onChange, ids, onMsg: null };
    const tryNow = () => { try { const ov = frame.contentWindow.__outdoorView; if (ov && ov.facets) { populate(ov.facets); return true; } } catch (e) {} return false; };
    if (!tryNow()) {
      const onMsg = (e) => { if (e.data && e.data.source === 'outdoorReady') populate(e.data.facets); };
      window.addEventListener('message', onMsg);
      _outdoorSidebar.onMsg = onMsg;
    }
  }
  function unwireOutdoorSidebar() {
    if (!_outdoorSidebar) return;
    _outdoorSidebar.ids.forEach((id) => { const el = document.getElementById(id); if (el) el.removeEventListener('change', _outdoorSidebar.onChange, true); });
    if (_outdoorSidebar.onMsg) window.removeEventListener('message', _outdoorSidebar.onMsg);
    _outdoorSidebar = null;
  }

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

    // Simulation 2D: swap the indoor vs OUTDOOR SIM V3 full-wave studio by environment.
    // Indoor = the 7th-floor plan (fw_studio2d); outdoor = base-station coverage over the
    // NoMa OSM city (fw_studio2d_outdoor). Lazy-load the outdoor iframe on first entry.
    const isOutdoor2d = window.appMode.environment === 'outdoor' && dim !== '3d';
    const inBar = document.getElementById('simFwBar'), inWrap = document.getElementById('simFwWrap');
    const outBar = document.getElementById('simFwBarOut'), outWrap = document.getElementById('simFwWrapOut');
    const outFrame = document.getElementById('simFwFrameOut'), outToggle = document.getElementById('simFwToggleOut');
    if (inBar) inBar.style.display = isOutdoor2d ? 'none' : '';
    if (inWrap && isOutdoor2d) inWrap.style.display = 'none';
    if (outBar) outBar.style.display = isOutdoor2d ? '' : 'none';
    if (!isOutdoor2d && outWrap) outWrap.style.display = 'none';
    if (isOutdoor2d && outFrame && !outFrame.getAttribute('src')) {   // auto-open on first outdoor entry
      outFrame.setAttribute('src', 'Simulation/fw_studio2d_outdoor.html');
      if (outWrap) outWrap.style.display = 'block';
      if (outToggle) outToggle.innerHTML = '▼ Outdoor Full-Wave (SIM V3 · base-station coverage)';
    }

    // Simulation 3D: same indoor/outdoor swap for the 3-D Full-Wave Studio. Indoor = fw_studio3d
    // (7th-floor voxel grid); outdoor = fw_studio3d_outdoor (NoMa city + known BS). Only relevant in 3D mode.
    const isOutdoor3d = window.appMode.environment === 'outdoor' && dim === '3d';
    const in3Bar = document.getElementById('simFw3dBar'), in3Wrap = document.getElementById('simFw3dWrap');
    const out3Bar = document.getElementById('simFw3dBarOut'), out3Wrap = document.getElementById('simFw3dWrapOut');
    if (in3Bar) in3Bar.style.display = isOutdoor3d ? 'none' : '';
    if (in3Wrap && isOutdoor3d) in3Wrap.style.display = 'none';
    if (out3Bar) out3Bar.style.display = isOutdoor3d ? '' : 'none';
    if (!isOutdoor3d && out3Wrap) out3Wrap.style.display = 'none';

    // Map Coverage: match the dimension (viewer3d wires these buttons).
    const btn = document.getElementById(dim === '3d' ? 'mapMode3dBtn' : 'mapMode2dBtn');
    if (btn) btn.click();

    // Outdoor mode: show the NoMa OSM outdoor view (2D Leaflet / 3D three.js — rectangular OSM
    // ground + base-station massing + walk points) in Map Coverage, replacing the indoor signal
    // map. OSM Buildings "3D city" and Unity C# were removed from this iframe; the iframe keeps
    // its own 2D/3D toggle and we sync it to the chosen dimension on entry.
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
        f.src = 'Map%20Coverage/outdoor_view.html?v=osm3d-three1';
        f.title = 'NoMa outdoor view';
        f.style.cssText = 'width:100%; height:100%; border:0; display:block;';
        oHost.appendChild(f);
      }
      const syncDim = () => { try { f.contentWindow.document.getElementById(dim === '3d' ? 'btn3d' : 'btn2d').click(); } catch (e) {} };
      if (f.contentWindow && f.contentWindow.document.readyState === 'complete') syncDim();
      else f.addEventListener('load', syncDim, { once: true });
      wireOutdoorSidebar(f);
    } else {
      unwireOutdoorSidebar();
    }

    // Outdoor mode: also swap the Time Elapse tab to the self-contained outdoor walk replay (iframe),
    // parallel to the Map Coverage swap above. #timeTab.outdoor-mode CSS hides the indoor playback; the
    // replay honours the same Coverage-sidebar filter (broadcastOutdoorFilter posts to it too).
    const timeTab = document.getElementById('timeTab');
    const tHost = document.getElementById('outdoorTimeHost');
    if (timeTab) timeTab.classList.toggle('outdoor-mode', outdoor);
    if (tHost && outdoor && !tHost.querySelector('iframe')) {
      const tf = document.createElement('iframe');
      tf.src = 'Time%20Elapse/outdoor_timelapse.html';
      tf.title = 'NoMa outdoor Time Elapse';
      tf.style.cssText = 'width:100%; height:1200px; border:0; display:block;';
      // Push the current sidebar filter once it's live (it may load after the map frame already broadcast).
      tf.addEventListener('load', () => broadcastOutdoorFilter(), { once: true });
      tHost.appendChild(tf);
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
    // Notify dashboard tabs (e.g. Statistics → Antenna Pattern) that Indoor/Outdoor flipped.
    window.dispatchEvent(new Event('appmodechange'));
  }

  bind('modeChange', () => showScreen('screenChooser'));

  function bind(id, fn) { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); }

  // Size the outdoor Time Elapse iframe to its own content (it reports height) so the tab shows the
  // whole replay without an inner scrollbar.
  window.addEventListener('message', (e) => {
    const d = e.data;
    if (!d || d.source !== 'outdoorTimeResize') return;
    const host = document.getElementById('outdoorTimeHost');
    const tf = host && host.querySelector('iframe');
    if (tf && Number.isFinite(d.height)) tf.style.height = Math.max(600, d.height) + 'px';
  });

  // Boot on the chooser.
  showScreen('screenChooser');

  // Deep-link / dataset-switch support: ?project=<id> boots straight into that project's
  // workspace. Loading a DIFFERENT dataset after the page already declared `const records`
  // can't reuse this global scope, so the project switch reloads with this param — a fresh
  // scope. Best-effort: needs the backend (or a built-in id) to resolve the manifest.
  (function bootFromQuery() {
    let pid = null;
    try { pid = new URLSearchParams(location.search).get('project'); } catch (e) { return; }
    if (!pid) return;
    const go = () => {
      const m = window.getProject && window.getProject(pid);
      if (!m) { console.warn('[workspace] ?project=' + pid + ' not found; staying on chooser.'); return; }
      window.appMode.environment = m.environment;
      window.appMode.dim = m.dim;
      window.appImport.project = m;
      enterWorkspace();
    };
    (window.refreshProjects ? window.refreshProjects() : Promise.resolve()).then(go, go);
  })();
})();
