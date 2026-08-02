/* simulation3d.js — the 3D Simulation tab (Indoor · 3D).
 *
 * A light-themed integration of the standalone antenna sandbox
 * (Physics Engine/3D Map Physics/SIM V1 3D/web/antenna_sandbox.html): construct
 * and place a Transmitter and a Receiver on the voxelized building (real
 * cannon-es mass/collision), pick a Waveform band, and view a first Static
 * field / Animated wavefront.
 *
 * Reuses the browser assets exported by SIM V1 3D (loaded as classic scripts
 * in Frontend_Data_Display.html):
 *   window.SIM3D_COLLISION       — voxel collision boxes + extent
 *   window.SIM3D_ANTENNA_CATALOG — antenna meshes + physics (mass, half-extents)
 *   window.SIM3D_ASSETS          — manifest (grid, materials) for the field calc
 *
 * The real geometry-aware wave propagation (reflection / refraction /
 * diffraction / absorption / scattering, GPU/ML) is deferred — the Static and
 * Animated views here are honest first-pass placeholders (an FSPL point cloud
 * and a kinematic wavefront) clearly labelled as such, ready to swap for the
 * engine_3d.py volume via export_web3.py. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as CANNON from 'cannon-es';

const COLLISION = window.SIM3D_COLLISION;
const CATALOG   = window.SIM3D_ANTENNA_CATALOG;
const MANIFEST  = (window.SIM3D_ASSETS && window.SIM3D_ASSETS.manifest_3d) || {};

// WiFi band presets (Indoor default). value = centre frequency in MHz.
const WIFI_BANDS = [
  { v: 2412, label: '2.4 GHz · ch 1 (2412 MHz)' },
  { v: 2437, label: '2.4 GHz · ch 6 (2437 MHz)' },
  { v: 2462, label: '2.4 GHz · ch 11 (2462 MHz)' },
  { v: 5180, label: '5 GHz · UNII-1 ch 36 (5180 MHz)' },
  { v: 5500, label: '5 GHz · UNII-2 ch 100 (5500 MHz)' },
  { v: 5745, label: '5 GHz · UNII-3 ch 149 (5745 MHz)' },
];

// ---- DOM handles ----
const viewport = document.getElementById('sim3dViewport');
const statusEl = document.getElementById('sim3dStatus');
const txType   = document.getElementById('tx3dType');
const txSwatch = document.getElementById('tx3dSwatch');
const txPlace  = document.getElementById('tx3dPlace');
const txList   = document.getElementById('tx3dList');
const rxType   = document.getElementById('rx3dType');
const rxPlace  = document.getElementById('rx3dPlace');
const rxList   = document.getElementById('rx3dList');
const wfBand   = document.getElementById('wf3dBand');
const runStatic = document.getElementById('run3dStatic');
const runAnim  = document.getElementById('run3dAnim');
const animCtl  = document.getElementById('anim3dControls');
const animPlay = document.getElementById('anim3dPlay');
const animScrub = document.getElementById('anim3dScrub');
const vizMode  = document.getElementById('viz3dMode');
const modeSel  = document.getElementById('mode3dSelect');
const modeNote = document.getElementById('mode3dNote');
// Tier-1 display controls (prediction plane · received-power coverage classes) and the
// viewport overlays they drive (context strip + point-probe readout).
const dispPlaneBtn  = document.getElementById('disp3dPlaneBtn');
const dispHeight    = document.getElementById('disp3dHeight');
const dispHeightVal = document.getElementById('disp3dHeightVal');
const dispEirp      = document.getElementById('disp3dEirp');
const dispScale     = document.getElementById('disp3dScale');
const dispClasses   = document.getElementById('disp3dClasses');
const dispThresh    = document.getElementById('disp3dThresh');
const dispThreshVal = document.getElementById('disp3dThreshVal');
const metaEl        = document.getElementById('sim3dMeta');
const probeEl       = document.getElementById('sim3dProbe');
const probeBody     = document.getElementById('sim3dProbeBody');
const probeClose    = document.getElementById('sim3dProbeClose');

function setStatus(msg) { if (statusEl) statusEl.textContent = msg || ''; }

// ---- Display state (Tier 1) ----
// How the solved volume is READ — never the physics. `plane`/`sliceY` cut a horizontal
// prediction plane; `eirpDbm` turns path loss into RSRP so `classes`/`fixed` can show
// received power in dBm the way WRAP's coverage view and the 2D sim do.
const DISPLAY = { plane: false, sliceY: null, eirpDbm: 20, scale: 'auto', classes: false, threshDbm: -85 };
const RSRP_LO = -100, RSRP_HI = -40;                 // fixed received-power display window (dBm)
const COV_GOOD = [0.13, 0.70, 0.29], COV_MARG = [0.98, 0.75, 0.18], COV_DEAD = [0.86, 0.28, 0.24];
let lastSolveMs = 0;                                  // wall time of the last field solve
let probeMarker = null;                               // little sphere at the probed voxel

// The voxelization target chosen on the Preprocess screen (window.appVoxel),
// surfaced here so the 2.5D/3D + cell choice is visible where the sim runs.
function voxTarget() {
  const V = window.appVoxel;
  return V ? ' · voxelization ' + (V.mode === '3d' ? '3D' : '2.5D') + ' @ ' + V.cell_m + ' m' : '';
}

// ---- populate rail selects (cheap; runs at module load, no scene needed) ----
function fillAntennaSelect(sel, swatchEl) {
  if (!CATALOG || !sel) return;
  CATALOG.kinds.forEach((k) => {
    const o = document.createElement('option');
    o.value = k; o.textContent = CATALOG.entries[k].label; sel.appendChild(o);
  });
  if (swatchEl) {
    const upd = () => {
      const c = CATALOG.entries[sel.value] && CATALOG.entries[sel.value].color;
      swatchEl.style.background = '#' + (c == null ? 0xffb020 : c).toString(16).padStart(6, '0');
    };
    sel.addEventListener('change', upd); upd();
  }
}
fillAntennaSelect(txType, txSwatch);
fillAntennaSelect(rxType, null);
if (wfBand) WIFI_BANDS.forEach((b) => {
  const o = document.createElement('option'); o.value = b.v; o.textContent = b.label; wfBand.appendChild(o);
});

// ---- three.js + cannon-es scene (lazy) ----
let initialized = false;
let renderer, scene, camera, controls, world;
let wallMesh = null, center = null, extent = null;
let wallMaterialC, antennaMaterialC;
let fieldObj = null;              // static field point cloud (THREE.Points)
let waveObj = null;               // animated wavefront shell (THREE.Mesh)
let armed = null;                 // 'tx' | 'rx' | null
const placed = [];                // { mesh, body, role, kind }
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const clock = new THREE.Clock();

function ensureInit() {
  if (initialized) return true;
  if (!COLLISION) { setStatus('3D collision assets not loaded (SIM3D/web/collision_3d.js).'); return false; }
  if (!viewport) return false;
  const w = viewport.clientWidth || 900, h = viewport.clientHeight || 640;
  try { renderer = new THREE.WebGPURenderer({ antialias: true }); }
  catch (e) { setStatus('3D rendering is unavailable in this browser.'); return false; }
  initialized = true;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.appendChild(renderer.domElement);
  renderer.domElement.addEventListener('pointerdown', onPointerDown);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef3f1);            // match viewer3d light bg

  extent = COLLISION.extent_m;                             // [X, Y_up, Z]
  center = new THREE.Vector3(extent[0] / 2, extent[1] / 2, extent[2] / 2);

  camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 2000);
  camera.position.set(center.x - extent[0] * 0.42, Math.max(extent[0], extent[2]) * 0.8, center.z + extent[2] * 1.4);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.update();

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8d9b96, 1.0));
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(center.x + extent[0] * 0.3, extent[1] * 4 + 10, center.z - extent[2] * 0.2);
  scene.add(key);

  const grid = new THREE.GridHelper(Math.max(extent[0], extent[2]) * 1.2, 40, 0xcdd8d3, 0xe1e8e4);
  grid.position.set(center.x, 0, center.z); scene.add(grid);

  world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.81, 0) });
  world.broadphase = new CANNON.SAPBroadphase(world);
  world.allowSleep = true;
  wallMaterialC = new CANNON.Material('wall');
  antennaMaterialC = new CANNON.Material('antenna');
  world.addContactMaterial(new CANNON.ContactMaterial(wallMaterialC, antennaMaterialC, { friction: 0.6, restitution: 0.1 }));

  // Building the scene is thousands of instanced boxes plus a static rigid body each, all
  // synchronous — the tab just sits there looking hung. Say what is happening, and hand
  // the browser a frame to paint that message on before the work starts.
  const nBoxes = COLLISION.boxes.length;
  setStatus('Building the 3D scene · ' + nBoxes.toLocaleString()
    + ' voxel boxes across ' + ((COLLISION.materials || []).length || 1)
    + ' materials · one moment…');
  // setTimeout, not requestAnimationFrame: rAF is throttled to zero in a backgrounded or
  // non-compositing tab, and deferring the build onto a callback that may never fire
  // would turn "slow to appear" into "never appears".
  const t0 = performance.now();
  setTimeout(() => {
    buildBuilding();
    buildMs = Math.round(performance.now() - t0);
    setStatus(readyStatus());
    refreshMeta();
  }, 0);
  window.addEventListener('resize', onResize);

  renderer.init().then(() => {
    renderer.setAnimationLoop(() => {
      if (viewport.offsetParent === null) return;          // skip when tab/panel hidden
      const dt = Math.min(clock.getDelta(), 0.05);
      world.step(1 / 60, dt, 5);
      for (const p of placed) { p.mesh.position.copy(p.body.position); p.mesh.quaternion.copy(p.body.quaternion); }
      if (waveObj && waveObj.userData.playing) advanceWave(dt);
      if (vectorState) advanceVectors(dt);
      if (interferenceState && interferenceState.animating) advanceInterference(dt);
      if (sweepState && sweepState.playing) advanceSweep(dt);
      controls.update();
      renderer.render(scene, camera);
    });
    // only claim ready once the geometry is actually up (the build is deferred)
    if (wallMeshes.length) setStatus(readyStatus());
  }).catch((err) => setStatus('3D backend failed to initialize: ' + (err && err.message || err)));
  return true;
}

// Per-class look. Concrete and the service core read as structure, drywall as
// partitions, glass as the envelope, furniture as soft clutter. Colours come from the
// exporter (which takes them from manifest_3d.json, shared with the 2D pipeline) so the
// floor plan and both simulators render the same building; only the surface treatment —
// opacity, roughness, metalness — is decided here.
const CLASS_STYLE = {
  1: { opacity: 0.40, roughness: 0.95, metalness: 0.00 },  // drywall_partition
  2: { opacity: 0.85, roughness: 0.95, metalness: 0.00 },  // concrete_masonry
  3: { opacity: 0.90, roughness: 0.55, metalness: 0.35 },  // core_service_area (metal)
  4: { opacity: 0.55, roughness: 0.85, metalness: 0.00 },  // furniture_clutter
  5: { opacity: 0.22, roughness: 0.08, metalness: 0.10 },  // exterior_glass
};
const DEFAULT_STYLE = { opacity: 0.55, roughness: 0.9, metalness: 0.04 };

// One InstancedMesh PER CLASS rather than one for everything. A single mesh forces one
// material on the whole building, which is why it used to render as undifferentiated
// grey — you cannot make glass translucent and concrete solid in the same draw call.
// Six draw calls is nothing next to being able to see what the RF engine is solving on.
let wallMeshes = [];
let buildMs = 0;

// What the tab says once the scene is up. The material breakdown is the quickest way to
// see whether the voxelization actually produced the building you expect.
function readyStatus() {
  const counts = wallMeshes.map((m) => {
    const mat = (COLLISION.materials || []).find((x) => x.id === m.userData.materialClass);
    return (mat ? mat.name.replace(/_/g, ' ') : 'class ' + m.userData.materialClass)
      + ' ' + m.count.toLocaleString();
  });
  return COLLISION.n_boxes.toLocaleString() + ' voxel boxes · '
    + extent.map((e) => e.toFixed(1)).join(' × ') + ' m' + voxTarget()
    + (buildMs ? ' · built in ' + buildMs + ' ms' : '')
    + (counts.length ? ' · ' + counts.join(', ') : '')
    + ' — click “Place on model”, then click a surface.';
}

function buildBuilding() {
  const boxes = COLLISION.boxes;
  const classes = COLLISION.classes || boxes.map(() => 1);
  const palette = {};
  for (const m of (COLLISION.materials || [])) {
    if (m.color) palette[m.id] = new THREE.Color(m.color);
  }

  // group box indices by class (skip -1: the synthetic ground pad is physics-only —
  // drawing a floor-sized slab would hide the whole building from above)
  const byClass = new Map();
  for (let i = 0; i < boxes.length; i++) {
    const c = classes[i];
    if (c < 0) continue;
    if (!byClass.has(c)) byClass.set(c, []);
    byClass.get(c).push(i);
  }

  const geo = new THREE.BoxGeometry(1, 1, 1);
  const dummy = new THREE.Object3D();
  wallMeshes = [];
  for (const [cls, idxs] of byClass) {
    const st = CLASS_STYLE[cls] || DEFAULT_STYLE;
    const mat = new THREE.MeshStandardMaterial({
      color: palette[cls] || new THREE.Color(0x9fb0bf),
      transparent: st.opacity < 1, opacity: st.opacity,
      roughness: st.roughness, metalness: st.metalness,
      depthWrite: st.opacity > 0.6,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, idxs.length);
    for (let k = 0; k < idxs.length; k++) {
      const b = boxes[idxs[k]];
      dummy.position.set(b[0], b[1], b[2]);
      dummy.scale.set(b[3] * 2, b[4] * 2, b[5] * 2);
      dummy.updateMatrix();
      mesh.setMatrixAt(k, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.materialClass = cls;
    // glass last so it composites over the structure behind it
    mesh.renderOrder = cls === 5 ? 2 : 1;
    scene.add(mesh);
    wallMeshes.push(mesh);
  }
  // `wallMesh` stays the raycast target for antenna placement. Point it at the opaque
  // structure (concrete/core/drywall) so clicking lands on a wall rather than on the
  // glass envelope in front of it.
  wallMesh = wallMeshes.find((m) => m.userData.materialClass === 2)
    || wallMeshes.find((m) => m.userData.materialClass === 1) || wallMeshes[0];

  // Physics: every box including the ground pad. Static bodies with SAPBroadphase +
  // allowSleep, so the per-frame cost is the broadphase insert, not n^2 pair checks.
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const body = new CANNON.Body({ mass: 0, material: wallMaterialC });
    body.addShape(new CANNON.Box(new CANNON.Vec3(b[3], b[4], b[5])));
    body.position.set(b[0], b[1], b[2]); world.addBody(body);
  }
}

function buildAntennaMesh(entry) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(entry.vertices.flat()), 3));
  g.setIndex(entry.faces.flat()); g.computeVertexNormals();
  const color = entry.color == null ? 0xffb020 : entry.color;
  const m = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.35,
    side: THREE.DoubleSide, emissive: color, emissiveIntensity: 0.14 });
  return new THREE.Mesh(g, m);
}

// ---- placement ----
function arm(role) {
  armed = (armed === role) ? null : role;
  if (txPlace) txPlace.classList.toggle('active', armed === 'tx');
  if (rxPlace) rxPlace.classList.toggle('active', armed === 'rx');
  if (renderer) renderer.domElement.style.cursor = armed ? 'crosshair' : 'default';
}
if (txPlace) txPlace.addEventListener('click', () => { if (ensureInit()) arm('tx'); });
if (rxPlace) rxPlace.addEventListener('click', () => { if (ensureInit()) arm('rx'); });

function onPointerDown(ev) {
  if (ev.button !== 0) return;
  if (!armed) { probeAt(ev); return; }   // not placing → probe the clicked cell instead
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  // Test every class mesh, not just one: the building is now several InstancedMeshes, so
  // raycasting a single one would make whole materials unclickable.
  const hit = raycaster.intersectObjects(wallMeshes.length ? wallMeshes : [wallMesh], false)[0];
  if (!hit) return;
  const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld).normalize();
  const pos = hit.point.clone().addScaledVector(normal, 0.4);
  const sel = (armed === 'tx') ? txType : rxType;
  const entry = CATALOG.entries[sel.value];
  const mesh = buildAntennaMesh(entry); mesh.position.copy(pos); scene.add(mesh);
  const phys = entry.physics;
  const body = new CANNON.Body({ mass: phys.mass_kg, material: antennaMaterialC });
  body.addShape(new CANNON.Box(new CANNON.Vec3(...phys.half_extents_m)), new CANNON.Vec3(...phys.center_offset_m));
  body.position.copy(pos); body.linearDamping = 0.15; body.angularDamping = 0.25;
  world.addBody(body);
  placed.push({ mesh, body, role: armed, kind: sel.value });
  refreshLists();
  if (armed === 'tx' && vizMode) runVizMode(vizMode.value);
}

function refreshLists() {
  const fmt = (arr) => arr.map((p) =>
    '<li>' + ((CATALOG.entries[p.kind] && CATALOG.entries[p.kind].label) || p.kind) + ' · ' + p.body.mass.toFixed(2) + ' kg</li>').join('')
    || '<li class="dim">none placed</li>';
  if (txList) txList.innerHTML = fmt(placed.filter((p) => p.role === 'tx'));
  if (rxList) rxList.innerHTML = fmt(placed.filter((p) => p.role === 'rx'));
}
refreshLists();

// ---- Static field: first-pass FSPL point cloud from the first transmitter ----
function firstTx() { return placed.find((p) => p.role === 'tx'); }

// viridis-ish ramp, t: 0 (weak) → 1 (strong), returns normalized rgb.
function ramp(t) {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(1, Math.max(0, t)) * (s.length - 1);
  const i = Math.floor(t), f = t - i, a = s[i], b = s[Math.min(i + 1, s.length - 1)];
  return [(a[0] + (b[0] - a[0]) * f) / 255, (a[1] + (b[1] - a[1]) * f) / 255, (a[2] + (b[2] - a[2]) * f) / 255];
}

// ---- geometry-aware path loss (3-D port of SIM/web/simulator_tab.js pathlossPhysics) ----
// This is the browser *analytic fallback* model (flat per-class loss_db + satObs);
// the real Fresnel/Airy + eikonal solve is engine_3d.py SceneV3 (surfaced later, P6).
const PHY        = MANIFEST.physics || {};
const CELL_M     = MANIFEST.cell_size_m || 0.3;
const STEP_CELL  = PHY.ray_step_cells != null ? PHY.ray_step_cells : 0.5;
const N_EXP      = PHY.n_exp != null ? PHY.n_exp : 2.0;          // Motley-Keenan
const D0_M       = PHY.d0_m != null ? PHY.d0_m : 1.0;
const FSPL_CONST = PHY.fspl_const_db != null ? PHY.fspl_const_db : -27.56;
const _OBS       = PHY.fallback_obstruction_db || {};           // moved here in P1
const SAT_T      = _OBS.obstruction_linear_db != null ? _OBS.obstruction_linear_db : 40.0;
const SAT_S      = _OBS.obstruction_sat_extra_db != null ? _OBS.obstruction_sat_extra_db : 50.0;
const MATS       = MANIFEST.materials || [];
const LOSS_DB    = MATS.map((m) => m.loss_db || 0);
const APM_DB     = MATS.map((m) => m.loss_per_m_db || 0);
const FREQS      = MANIFEST.freqs_mhz || [2442, 3500, 5500, 6125];
const FREQ_MULT  = MANIFEST.freq_loss_mult || {};

// saturating wall-loss (flat-dB diffraction stand-in). satObs(x) from simulator_tab.js:31.
function satObs(x) { return x <= SAT_T ? x : SAT_T + SAT_S * (1 - Math.exp(-(x - SAT_T) / SAT_S)); }

// per-band wall-loss multiplier by nearest manifest band — the WiFi presets in
// #wf3dBand (2412/5180/…) don't match freq_loss_mult keys (2442/3500/5500/6125).
function bandMult(fMHz) {
  let best = FREQS[0], bd = Infinity;
  for (const b of FREQS) { const d = Math.abs(b - fMHz); if (d < bd) { bd = d; best = b; } }
  const m = FREQ_MULT[String(Math.round(best))];
  return m != null ? m : 1.0;
}

// decode the int8 material grid once (base64 → Uint8Array), mirroring u8() in
// simulator_tab.js:21. C-order for grid_shape [NX,NY,NZ]; idx = (x*NY + y)*NZ + z.
let GRID3 = null, GDIMS = null;
function decodeGrid() {
  if (GRID3) return GRID3;
  const b64 = window.SIM3D_ASSETS && window.SIM3D_ASSETS.grid_b64;
  const dims = MANIFEST.grid_shape;
  if (!b64 || !dims) return null;
  const s = atob(b64), a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  GRID3 = a; GDIMS = dims;
  return GRID3;
}
let INSIDE3 = null;
function decodeInside() {
  if (INSIDE3) return INSIDE3;
  const b64 = window.SIM3D_ASSETS && window.SIM3D_ASSETS.inside_b64;
  if (!b64 || !MANIFEST.grid_shape) return null;
  const s = atob(b64), a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  INSIDE3 = a;
  return INSIDE3;
}
// The interior mask is addressed with the MANIFEST grid, and a precomputed volume is
// addressed with its own. Those are the same grid only while sim_assets_3d.js and
// web/volumes/ come from the same voxelization — they did not once (the assets were a
// re-voxelization behind), and indexing one array with the other's strides silently
// rejects almost every voxel instead of failing. So check, and skip the mask rather than
// mis-address it.
function insideMaskFor(dims) {
  const g = MANIFEST.grid_shape;
  if (!g || !dims || g[0] !== dims[0] || g[1] !== dims[1] || g[2] !== dims[2]) {
    if (!insideMaskFor._warned) {
      console.warn('[sim3d] interior mask is ' + (g || []).join('×') + ' but the volume is '
        + dims.join('×') + ' — mask ignored. Re-run SIM V1 3D/export_web3.py.');
      insideMaskFor._warned = true;
    }
    return null;
  }
  return decodeInside();
}
const clampi = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);

// ---- Precomputed full-physics volumes (real SceneV3 PL + eikonal T) ----
// Fetched lazily from SIM3D/web/volumes/ (index.json + float16 .bin). The coverage
// field (runStaticField) and the wavefront sweep read from window.SIM3D_VOLUME;
// coverage falls back to the P2 analytic march when no volume matches the Tx.
const VOL_TOL = 20;              // voxels — precomputed Tx are sparse; match generously
let volumeIndex = null;          // [] once fetched; entries = index.json sidecars
const volumeCache = {};          // txid -> decoded volume

// ---- Propagation modes (M3) ----
// The mode picks the SCENE and the SOURCE MODEL, so a volume solved in one mode is not
// interchangeable with another's: `o2i` has no transmitter inside the grid at all, and
// `outdoor` is a different grid entirely. The browser therefore filters the volume
// catalog by mode before it ever looks for a matching Tx. Physics lives in
// SIM V1 3D/modes_3d.py; this is only the selector and the explanation.
const MODE_INFO = {
  vacuum: { label: 'Vacuum / free space', source: 'point',
    note: 'All air — path loss must equal FSPL exactly. This is the invariant gate made visible: if the engine cannot reproduce free space, nothing else it says is trustworthy.' },
  indoor: { label: 'Indoor', source: 'point',
    note: 'The voxelized 7th floor with the transmitter inside. The production case — place a Tx on the model.' },
  o2i: { label: 'Indoor + Outdoor (O2I)', source: 'plane_wave',
    note: 'A macro cell 416 m away (Forte Hall, arriving from 237°) lights the facade; energy penetrates through 3GPP O2I loss and spreads inside. There is no transmitter to place — the source is outside the grid.' },
  outdoor: { label: 'Outdoor (city)', source: 'point',
    note: 'NoMa DC city block: buildings are one barrier class, streets are air. Ground reflection and terrain diffraction come from the same reflection/diffraction modules, on a scene whose walls are buildings.' },
};
function currentMode() { return (modeSel && modeSel.value) || 'indoor'; }
function modeIsPointSource() { return (MODE_INFO[currentMode()] || {}).source !== 'plane_wave'; }

// One half-float, decoded on read. The volumes are stored as float16 precisely to be
// compact, and eagerly expanding them to Float32Array threw that away: a 10-band
// production volume is 11.8 MB on disk and was becoming 23.5 MB of JS heap, doubled again
// by every prefetched neighbour. Decoding per read costs a few arithmetic ops in loops
// that already do far more work per voxel, and halves the resident footprint.
function h2f(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 5.960464477539063e-8 * f;      // 2^-24 * f
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
// Kept for callers that genuinely need a dense Float32Array; nothing on the hot path does.
function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = h2f(u16[i]);
  return out;
}

// ---- Browser-side LRU, mirroring the disk cache ----
// volumeCache/mechCache grew without bound: open six transmitters and every decode stayed
// resident until the tab died. Budget is in decoded bytes, evicting least-recently-used —
// same rule as cache_index.py, for the same reason.
const BROWSER_CACHE_BUDGET = 96 * 1024 * 1024;
let cacheClock = 0;
function cacheBytes() {
  let n = 0;
  for (const v of Object.values(volumeCache)) n += (v.pl.byteLength + v.t.byteLength);
  for (const c of Object.values(mechCache)) {
    n += c.pl.byteLength + c.tau.byteLength + (c.tauGeom ? c.tauGeom.byteLength : 0);
  }
  return n;
}
// `keep` is the volume being displayed and `keepMech` the channel just loaded — evicting
// either would make the very next read refetch what we just paid for.
function evictToBudget(keep, keepMech) {
  let guard = 64;
  while (cacheBytes() > BROWSER_CACHE_BUDGET && guard-- > 0) {
    let victim = null, oldest = Infinity, store = null;
    for (const [k, v] of Object.entries(mechCache)) {     // mechanism channels go first:
      if (k === keepMech) continue;                       // they are the easiest to refetch
      if (v.used < oldest) { oldest = v.used; victim = k; store = mechCache; }
    }
    if (!victim) {                                       // …then whole volumes
      for (const [k, v] of Object.entries(volumeCache)) {
        if (k === keep) continue;
        if (v.used < oldest) { oldest = v.used; victim = k; store = volumeCache; }
      }
    }
    if (!victim) break;                                  // nothing evictable left
    delete store[victim];
  }
}
// file:// blocks fetch, so the whole cached tier is unavailable there — and reporting
// that as "nothing cached" sends you off to run an export you have already run. Say which
// it is.
const ON_FILE_PROTOCOL = location.protocol === 'file:';
let volumeIndexError = null;
async function loadVolumeIndex() {
  if (volumeIndex !== null) return volumeIndex;
  if (ON_FILE_PROTOCOL) {
    volumeIndex = [];
    volumeIndexError = 'opened over file:// — the browser blocks fetch(), so precomputed '
      + 'volumes cannot load. Serve the folder over http (e.g. `python3 -m http.server 8777`) '
      + 'and open http://localhost:8777/Frontend_Data_Display.html';
    console.warn('[sim3d] ' + volumeIndexError);
    return volumeIndex;
  }
  try {
    const r = await fetch('SIM3D/web/volumes/index.json');
    volumeIndex = r.ok ? ((await r.json()).volumes || []) : [];
    if (!r.ok) volumeIndexError = 'index.json returned HTTP ' + r.status;
  } catch (e) {
    volumeIndex = [];
    volumeIndexError = 'could not fetch index.json — ' + e.message;
  }
  return volumeIndex;
}
// Volumes written before M3 carry no `mode`; they were all the indoor floor.
function volumesForMode(mode) {
  return (volumeIndex || []).filter((e) => (e.mode || 'indoor') === mode);
}
function matchVolume(txVox, mode) {
  const pool = volumesForMode(mode || currentMode());
  if (!pool.length) return null;
  // A plane-wave mode has no transmitter to match against — there is exactly one
  // solution per bearing, so take it rather than pretending to search. Same for a point
  // mode with no Tx placed yet: showing the mode's cached solve beats showing nothing.
  if (!modeIsPointSource() || !txVox) return pool[0];
  let best = null, bd = Infinity;
  for (const e of pool) {
    const d = Math.hypot(e.tx_vox[0] - txVox[0], e.tx_vox[1] - txVox[1], e.tx_vox[2] - txVox[2]);
    if (d < bd) { bd = d; best = e; }
  }
  return bd <= VOL_TOL ? best : null;
}
// setCurrent=false is the prefetch path: decode into the cache without stealing the
// volume the user is currently looking at.
async function loadVolume(entry, { setCurrent = true } = {}) {
  if (volumeCache[entry.txid]) {
    const hit = volumeCache[entry.txid];
    hit.used = ++cacheClock;
    if (setCurrent) window.SIM3D_VOLUME = hit;
    return hit;
  }
  const base = 'SIM3D/web/volumes/';
  const [plBuf, tBuf] = await Promise.all([
    fetch(base + entry.pl_file).then((r) => r.arrayBuffer()),
    fetch(base + entry.t_file).then((r) => r.arrayBuffer()),
  ]);
  // kept as float16; h2f() decodes per read (see the note on h2f)
  const vol = { pl: new Uint16Array(plBuf), t: new Uint16Array(tBuf),
    dims: entry.grid_shape, bands: entry.bands, tx_vox: entry.tx_vox,
    t_max_ns: entry.t_max_ns, entry, used: ++cacheClock };
  volumeCache[entry.txid] = vol;
  evictToBudget(entry.txid);
  if (setCurrent) { window.SIM3D_VOLUME = vol; prefetchNeighbours(entry); }
  return vol;
}
function volMatches(vol, txVox) {
  return !!(vol && txVox &&
    Math.hypot(vol.tx_vox[0] - txVox[0], vol.tx_vox[1] - txVox[1], vol.tx_vox[2] - txVox[2]) <= VOL_TOL);
}

// Warm the neighbours of the volume just loaded, so dragging the transmitter to the next
// cached position is instant rather than a 13 MB stall. Idle-time only, and never more
// than PREFETCH_MAX so a big cache does not turn one placement into a hundred fetches.
const PREFETCH_MAX = 3;
let prefetching = false;
function prefetchNeighbours(entry) {
  if (prefetching || !entry) return;
  // Never prefetch into an over-budget cache. Warming neighbours is a convenience; it
  // must not be what pushes the renderer over, and each volume is ~12 MB decoded.
  if (cacheBytes() > BROWSER_CACHE_BUDGET * 0.6) return;
  const pool = volumesForMode(entry.mode || 'indoor')
    .filter((e) => e.txid !== entry.txid && !volumeCache[e.txid]);
  if (!pool.length) return;
  const d = (e) => Math.hypot(e.tx_vox[0] - entry.tx_vox[0],
                              e.tx_vox[1] - entry.tx_vox[1],
                              e.tx_vox[2] - entry.tx_vox[2]);
  const want = pool.sort((a, b) => d(a) - d(b)).slice(0, PREFETCH_MAX);
  prefetching = true;
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 400));
  idle(() => {
    // sequential, not Promise.all: these are background fetches and must not contend
    // with whatever the user is actively waiting for. Re-check the budget between each,
    // since the user may have loaded mechanism channels in the meantime.
    want.reduce((p, e) => p.then(() => (cacheBytes() > BROWSER_CACHE_BUDGET * 0.6
      ? null : loadVolume(e, { setCurrent: false }).catch(() => {}))), Promise.resolve())
      .then(() => { prefetching = false; });
  });
}
function plAt(vol, band, x, y, z) { const NY = vol.dims[1], NZ = vol.dims[2]; return h2f(vol.pl[((band * vol.dims[0] + x) * NY + y) * NZ + z]); }
function tAt(vol, x, y, z) { const NY = vol.dims[1], NZ = vol.dims[2]; return h2f(vol.t[(x * NY + y) * NZ + z]); }
function nearestBandIndex(vol, fMHz) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < vol.bands.length; i++) { const d = Math.abs(vol.bands[i] - fMHz); if (d < bd) { bd = d; bi = i; } }
  return bi;
}
// ---- Resolution order: cached volume → DL surrogate → analytic fallback ----
// The analytic JS mirror is the guaranteed-correct floor, not a placeholder: the
// simulator has to work without the surrogate, so the surrogate can only ever be an
// accelerator. Tier is reported in the status line, because "which engine drew this"
// changes how much you should trust the picture.
const TIER = { CACHE: 'cached volume', SURROGATE: 'DL surrogate', ANALYTIC: 'analytic (in-browser)' };

// The 3-D surrogate is not trained yet (M4). The slot is real, but it deliberately
// refuses to guess: a network's input layout (channel order, normalization, blob sigma)
// is decided by the training run, and a wrong guess does not fail loudly — it produces a
// confident, wrong field. So the browser loads the model ONLY alongside a contract
// sidecar that the trainer writes, and otherwise falls straight through to analytic.
const SURROGATE = { session: null, spec: null, state: 'untried', note: '' };
async function tryLoadSurrogate() {
  if (SURROGATE.state !== 'untried') return SURROGATE;
  SURROGATE.state = 'loading';
  if (!window.ort || location.protocol === 'file:') {
    SURROGATE.state = 'unavailable';
    SURROGATE.note = window.ort ? 'needs http (file:// blocks fetch)' : 'onnxruntime-web not loaded';
    return SURROGATE;
  }
  try {
    const specRes = await fetch('SIM3D/web/pl_unet3d.json');
    if (!specRes.ok) throw new Error('no input contract (pl_unet3d.json)');
    SURROGATE.spec = await specRes.json();
    for (const eps of [['webgpu'], ['wasm']]) {
      try {
        SURROGATE.session = await ort.InferenceSession.create('SIM3D/web/pl_unet3d.onnx',
          { executionProviders: eps });
        SURROGATE.state = 'ready';
        SURROGATE.note = 'UNet3D (' + eps[0] + ')';
        return SURROGATE;
      } catch (e) { /* try the next execution provider */ }
    }
    throw new Error('model failed to load on every execution provider');
  } catch (e) {
    SURROGATE.state = 'unavailable';
    SURROGATE.note = 'not trained yet (M4) — ' + e.message;
  }
  return SURROGATE;
}

// background-load the volume matching txVox in the current mode; call onReady() when it arrives
function tryLoadVolume(txVox, onReady) {
  loadVolumeIndex().then(() => {
    const e = matchVolume(txVox);
    if (!e) { reportNoVolume(txVox); return; }
    if (!window.SIM3D_VOLUME || window.SIM3D_VOLUME.entry.txid !== e.txid
        || (window.SIM3D_VOLUME.entry.mode || 'indoor') !== (e.mode || 'indoor')) {
      loadVolume(e).then(onReady);
    } else if (onReady) { onReady(window.SIM3D_VOLUME); }
  });
}
// Two different failures wear the same "no volume" label, and the fix differs: either the
// mode has nothing cached (go solve one), or it has volumes but none near this
// transmitter (go move the Tx). Say which.
function reportNoVolume(txVox) {
  const m = currentMode(), label = (MODE_INFO[m] || {}).label;
  const pool = volumesForMode(m);
  const solveHint = ' — run: python "SIM V1 3D/export_pl_volume.py" --mode ' + m
    + ' --mechanisms path_loss,reflection,diffraction,scattering,refraction,absorption';
  if (volumeIndexError) {
    setStatus(label + ' · showing the analytic tier only — ' + volumeIndexError);
    return;
  }
  if (!pool.length) {
    setStatus(label + ' · nothing cached for this mode yet ('
      + (volumeIndex || []).length + ' cached in others)' + solveHint);
    return;
  }
  if (!txVox) { setStatus(label + ' · place a transmitter to pick a cached solve.'); return; }
  let bd = Infinity, near = null;
  for (const e of pool) {
    const d = Math.hypot(e.tx_vox[0] - txVox[0], e.tx_vox[1] - txVox[1], e.tx_vox[2] - txVox[2]);
    if (d < bd) { bd = d; near = e; }
  }
  setStatus(label + ' · ' + pool.length + ' cached solve' + (pool.length === 1 ? '' : 's')
    + ' in this mode, but the nearest (' + near.txid + ') is ' + Math.round(bd)
    + ' voxels away — beyond the ' + VOL_TOL + '-voxel match tolerance. Move the transmitter '
    + 'near it, or solve this position:' + solveHint + ' --tx '
    + txVox.join(' '));
}

// ---- Per-mechanism channels (m_<mech>_<txid>.bin + tau_<mech>_<txid>.bin) ----
// Written by precompute_volumes.py --mech-channels. Each mechanism carries its own dB
// volume and its own first-arrival time, so the browser can answer "how much of the
// signal here is reflected?" and play the mechanism time-lapse. Fetched LAZILY, one
// mechanism at a time: each is the same size as the total PL volume, so loading all of
// them up front would multiply the initial download by the mechanism count for nothing.
// Labels name the PHENOMENON, not the module. `Path_Loss_3D` is geometric spreading (S1)
// plus the per-crossing transmission stack (S2–S7); `Reflection_3D` is specular multipath;
// `Scattering_3D` is the diffuse tail. Reflection and diffuse scattering carry no S-number
// because the v1 spatial catalog (S1–S12) is a single-arrival model and lists them as v2
// forks — this engine implements them anyway, which is why the time-lapse can exist.
const MECHS = [
  { key: 'path_loss',   label: 'Spreading + transmission', short: 'Direct',
    color: [1.00, 0.93, 0.42] },                                            // direct — warm
  { key: 'reflection',  label: 'Specular reflection / multipath', short: 'Reflected',
    color: [0.32, 0.82, 0.94] },                                            // cyan
  { key: 'diffraction', label: 'UTD diffraction', short: 'Diffracted',
    color: [0.93, 0.45, 0.85] },                                            // magenta
  { key: 'scattering',  label: 'Diffuse scattering', short: 'Diffuse',
    color: [0.48, 0.88, 0.52] },                                            // green
  // Diagnostics: decompositions of the direct path, not summed into received power.
  // Refraction shows the through-wall transmission loss + refractive delay; absorption
  // shows where power is deposited into materials. Rendered as absolute maps, not shares.
  { key: 'refraction',  label: 'Refraction (transmitted field)', short: 'Refracted',
    color: [0.55, 0.75, 0.98], diagnostic: true },                         // light blue
  { key: 'absorption',  label: 'Absorption (deposited power)', short: 'Absorbed',
    color: [0.96, 0.55, 0.33], diagnostic: true },                         // warm orange
];
const MECH_BY_KEY = Object.fromEntries(MECHS.map((m) => [m.key, m]));

function mechEntry(vol, mech) {
  const m = vol && vol.entry && vol.entry.mechanisms;
  return (m && !Array.isArray(m) && m[mech]) || null;   // pre-v3 wrote a bare name list
}
function mechList(vol) { return MECHS.filter((m) => mechEntry(vol, m.key)); }

// Decoded channel cache, keyed txid|mech so switching mechanisms back and forth is free.
const mechCache = {};
async function loadMechanism(vol, mech) {
  const info = mechEntry(vol, mech);
  if (!info) return null;
  const key = vol.entry.txid + '|' + mech;
  if (mechCache[key]) { mechCache[key].used = ++cacheClock; return mechCache[key]; }
  const base = 'SIM3D/web/volumes/';
  const [plBuf, tauBuf, geomBuf] = await Promise.all([
    fetch(base + info.pl_file).then((r) => r.arrayBuffer()),
    fetch(base + info.tau_file).then((r) => r.arrayBuffer()),
    info.tau_geom_file ? fetch(base + info.tau_geom_file).then((r) => r.arrayBuffer()) : null,
  ]);
  const bands = vol.entry.mech_bands || vol.bands;
  const ch = { mech, pl: new Uint16Array(plBuf), tau: new Uint16Array(tauBuf),
    dims: vol.dims, bands, t_max_ns: info.t_max_ns, combine_as: info.combine_as,
    convention: info.tau_convention || 'geometric', info, used: ++cacheClock };
  // Second clock (path loss only): vacuum d/c, so every layer of the time-lapse can be
  // compared on one convention. See write_mechanism_channels' "TWO CLOCKS" note — the
  // eikonal charges the direct path for in-wall slowdown and the other mechanisms do not,
  // so mixing them makes the reflected front appear to beat line of sight.
  if (geomBuf) { ch.tauGeom = new Uint16Array(geomBuf); ch.tauGeomMax = info.tau_geom_max_ns; }
  mechCache[key] = ch;
  evictToBudget(vol.entry.txid, key);
  return ch;
}
// Same C-order (band, x, y, z) addressing as plAt — mechanism channels may carry a
// SUBSET of the total volume's bands, so they index with their own band list.
function chanAt(ch, band, x, y, z) { const NY = ch.dims[1], NZ = ch.dims[2]; return h2f(ch.pl[((band * ch.dims[0] + x) * NY + y) * NZ + z]); }
function tauAt(ch, x, y, z, useGeom) {
  const NY = ch.dims[1], NZ = ch.dims[2];
  const src = (useGeom && ch.tauGeom) ? ch.tauGeom : ch.tau;
  return h2f(src[(x * NY + y) * NZ + z]);
}
function chanBandIndex(ch, fMHz) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < ch.bands.length; i++) { const d = Math.abs(ch.bands[i] - fMHz); if (d < bd) { bd = d; bi = i; } }
  return bi;
}

// PL(dB) from Tx voxel to sample voxel: Motley-Keenan spreading + per-crossing
// wall loss (R3 runs count once) + Beer-Lambert furniture + satObs. Direct port
// of pathlossPhysics's inner loop (simulator_tab.js:42-55). L/Ap = dB × band mult.
function marchPL(tx, sx, sy, sz, L, Ap, f1) {
  const NX = GDIMS[0], NY = GDIMS[1], NZ = GDIMS[2];
  const dx = sx - tx[0], dy = sy - tx[1], dz = sz - tx[2];
  const distC = Math.hypot(dx, dy, dz);
  const dm = Math.max(distC * CELL_M, D0_M);
  const K = Math.max(1, Math.ceil(distC / STEP_CELL));
  let wall = 0, perM = 0, cur = -1;
  for (let k = 0; k <= K; k++) {
    const t = k / K;
    const xi = clampi(Math.round(tx[0] + t * dx), NX);
    const yi = clampi(Math.round(tx[1] + t * dy), NY);
    const zi = clampi(Math.round(tx[2] + t * dz), NZ);
    const c = GRID3[(xi * NY + yi) * NZ + zi];
    if (c !== cur) { cur = c; if (L[c] > 0) wall += L[c]; }   // R3: new run counts once
    perM += Ap[c];
  }
  wall += perM * (distC / K) * CELL_M;                        // furniture (dB/m × length)
  return f1 + 10 * N_EXP * Math.log10(dm) + satObs(wall);
}

// dB colour legend under the viewport (viridis weak→strong, dB end-labels).
let legendEl = null;
function ensureLegend() {
  if (legendEl) return legendEl;
  const wrap = viewport && viewport.parentElement;            // .sim3d-viewport-wrap
  if (!wrap) return null;
  legendEl = document.createElement('div');
  legendEl.id = 'sim3dLegend'; legendEl.className = 'sim3d-legend'; legendEl.hidden = true;
  wrap.appendChild(legendEl);
  return legendEl;
}
// The ramp always runs weak -> strong left to right, so the caller names the two ENDS
// rather than a min/max: a path-loss scale (high dB = weak, on the left) and a share
// scale (high dB = strong, on the right) both read correctly without inverting colours.
function showLegend(strongLabel, weakLabel, caption, unit) {
  const loMin = strongLabel, loMax = weakLabel;
  const el = ensureLegend();
  if (!el) return;
  const stops = [];
  for (let i = 0; i <= 6; i++) {
    const c = ramp(i / 6);
    stops.push('rgb(' + Math.round(c[0] * 255) + ',' + Math.round(c[1] * 255) + ',' + Math.round(c[2] * 255) + ') ' + Math.round(i / 6 * 100) + '%');
  }
  el.innerHTML = '<span class="lg-cap">' + (caption || 'path loss') + '</span>'
    + '<span class="lg-lab">' + loMax + '</span>'
    + '<span class="lg-bar" style="background:linear-gradient(90deg,' + stops.join(',') + ')"></span>'
    + '<span class="lg-lab">' + loMin + (unit === undefined ? ' dB' : unit) + '</span>';
  el.hidden = false;
}
// Legend for the mechanism time-lapse: a swatch per mechanism, since colour there is
// identity (which mechanism) rather than magnitude.
function showMechLegend(layers) {
  const el = ensureLegend();
  if (!el) return;
  el.innerHTML = '<span class="lg-cap">arrival</span>' + layers.map((L) => {
    const c = L.color;
    return '<span class="lg-lab" style="color:rgb(' + Math.round(c[0] * 200) + ','
      + Math.round(c[1] * 200) + ',' + Math.round(c[2] * 200) + ')">■ ' + (L.short || L.label) + '</span>';
  }).join('');
  el.hidden = false;
}
function hideLegend() { if (legendEl) legendEl.hidden = true; }

// ---- Coverage colouring (Tier 1) ----
// auto → the original path-loss ramp (unchanged). fixed/classes → received power
// (RSRP = EIRP − PL) shown in dBm, either on the fixed window or bucketed into
// good / marginal / dead. When a prediction plane is active `sliced` fills the whole
// plane (no weak-drop) so it reads as a continuous heatmap the way ProMan's does.
function coverageSample(pl, sliced) {
  if (DISPLAY.classes || DISPLAY.scale === 'fixed') {
    const rsrp = DISPLAY.eirpDbm - pl;
    if (DISPLAY.classes) {
      if (rsrp >= DISPLAY.threshDbm + 10) return { keep: true, rgb: COV_GOOD, mag: 0.9 };
      if (rsrp >= DISPLAY.threshDbm)      return { keep: true, rgb: COV_MARG, mag: 0.6 };
      if (!sliced) return { keep: false };
      return { keep: true, rgb: COV_DEAD, mag: 0.3 };
    }
    const t = (rsrp - RSRP_LO) / (RSRP_HI - RSRP_LO);
    if (!sliced && t < 0.12) return { keep: false };
    return { keep: true, rgb: ramp(t), mag: Math.min(1, Math.max(0, t)) };
  }
  const t = 1 - Math.min(1, Math.max(0, (pl - 45) / (130 - 45)));   // 1 = strong
  if (!sliced && t < 0.12) return { keep: false };
  return { keep: true, rgb: ramp(t), mag: t };
}
function coverageLegend() {
  if (DISPLAY.classes) { showClassLegend(DISPLAY.threshDbm); return; }
  if (DISPLAY.scale === 'fixed') { showLegend(RSRP_HI, RSRP_LO, 'received power', ' dBm'); return; }
  showLegend(45, 130, 'path loss');
}
function showClassLegend(thresh) {
  const el = ensureLegend(); if (!el) return;
  const sw = (rgb, txt) => '<span class="lg-lab"><span class="lg-sw" style="background:rgb('
    + rgb.map((c) => Math.round(c * 255)).join(',') + ')"></span>' + txt + '</span>';
  el.innerHTML = '<span class="lg-cap">coverage</span>'
    + sw(COV_GOOD, 'good ≥ ' + (thresh + 10))
    + sw(COV_MARG, 'marginal ≥ ' + thresh)
    + sw(COV_DEAD, 'dead < ' + thresh + ' dBm');
  el.hidden = false;
}

// Sample-grid bounds shared by the coverage and per-mechanism clouds. Full volume marches
// every height at the coarse step; a prediction plane samples one height on a finer grid.
function sampleBounds() {
  const sliced = DISPLAY.plane && DISPLAY.sliceY != null;
  const step = Math.max(extent[0], extent[2]) / 46;
  const g = sliced ? Math.max(extent[0], extent[2]) / 70 : step;
  return { sliced, g,
    yLo: sliced ? DISPLAY.sliceY : g * 0.5,
    yHi: sliced ? DISPLAY.sliceY + 1e-3 : extent[1],
    yInc: sliced ? 1e9 : g };
}

// ---- Context strip (WinProp status bar) ----
function refreshMeta() {
  if (!metaEl) return;
  const D = GDIMS || MANIFEST.grid_shape || null;
  const parts = [];
  if (D) parts.push('grid ' + D.join('×'));
  parts.push('cell ' + CELL_M.toFixed(2) + ' m');
  if (extent) parts.push(extent.map((e) => e.toFixed(1)).join('×') + ' m');
  parts.push('tier ' + (window.SIM3D_TIER || '—'));
  parts.push((DISPLAY.plane && DISPLAY.sliceY != null)
    ? 'plane y=' + DISPLAY.sliceY.toFixed(1) + ' m' : 'full volume');
  if (lastSolveMs) parts.push(lastSolveMs + ' ms');
  metaEl.textContent = parts.join('  ·  ');
  metaEl.hidden = false;
}

// ---- Point probe (WRAP Spectrum-Viewer style) ----
// Click a cell to read PL / RSRP / first-arrival delay + the dominant mechanism there.
// With a prediction plane active the pick is exact (ray ∩ plane); otherwise it hits the
// building, falling back to the mid-height plane.
function probeAt(ev) {
  if (!renderer || !camera) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  let p = null;
  if (DISPLAY.plane && DISPLAY.sliceY != null) {
    p = raycaster.ray.intersectPlane(
      new THREE.Plane(new THREE.Vector3(0, 1, 0), -DISPLAY.sliceY), new THREE.Vector3());
  } else {
    const hit = raycaster.intersectObjects(wallMeshes.length ? wallMeshes : [wallMesh], false)[0];
    if (hit) p = hit.point.clone();
    else p = raycaster.ray.intersectPlane(
      new THREE.Plane(new THREE.Vector3(0, 1, 0), -(center ? center.y : extent[1] / 2)), new THREE.Vector3());
  }
  if (p) showProbe(p); else hideProbe();
}
function showProbe(p) {
  if (!probeEl || !probeBody) return;
  const vol = window.SIM3D_VOLUME || null;
  const D = vol ? vol.dims : (GDIMS || MANIFEST.grid_shape);
  if (!D) return;
  const ix = clampi(Math.floor(p.x / CELL_M), D[0]);
  const iy = clampi(Math.floor(p.y / CELL_M), D[1]);
  const iz = clampi(Math.floor(p.z / CELL_M), D[2]);
  const freqMHz = Number(wfBand && wfBand.value) || 2437;
  let pl = null, tau = null;
  if (vol) {
    const bi = nearestBandIndex(vol, freqMHz);
    pl = plAt(vol, bi, ix, iy, iz);
    const tn = tAt(vol, ix, iy, iz);
    if (isFinite(tn) && tn < 65504) tau = tn;
  } else if (decodeGrid()) {
    const tx = firstTx(); const s = tx ? tx.body.position : center;
    const txv = [clampi(Math.floor(s.x / CELL_M), GDIMS[0]),
                 clampi(Math.floor(s.y / CELL_M), GDIMS[1]),
                 clampi(Math.floor(s.z / CELL_M), GDIMS[2])];
    const mult = bandMult(freqMHz);
    pl = marchPL(txv, ix, iy, iz, LOSS_DB.map((v) => v * mult), APM_DB.map((v) => v * mult),
      20 * Math.log10(freqMHz) + FSPL_CONST);
  } else {
    const s = (firstTx() || { body: { position: center } }).body.position;
    const d = Math.max(0.5, Math.hypot(p.x - s.x, p.y - s.y, p.z - s.z));
    pl = 20 * Math.log10(d) + 20 * Math.log10(freqMHz) - 27.55;
  }
  if (pl == null || !isFinite(pl)) { hideProbe(); return; }
  const rsrp = DISPLAY.eirpDbm - pl;
  // A cell buried in solid structure (or beyond the solved field) carries a huge, real loss
  // — 250+ dB. Reporting "-245 dBm RSRP" there is technically true but useless; say "no
  // signal" instead, and don't try to name a mechanism where none arrives.
  const noSignal = pl >= 180 || rsrp < -130;
  // Dominant mechanism among the channels already resident (no forced fetch): the one whose
  // own path loss is closest to the total — i.e. the biggest share of the power here. Apply
  // the same floor the cloud render uses (SHARE_DB_LO) so a mechanism that doesn't reach
  // this cell is not reported as "dominant".
  let dom = null;
  if (vol && !noSignal) {
    let best = -Infinity;
    for (const m of mechList(vol)) {
      if (m.diagnostic) continue;               // refraction/absorption are not arriving paths
      const ch = mechCache[vol.entry.txid + '|' + m.key];
      if (!ch) continue;
      const cbi = chanBandIndex(ch, freqMHz);
      const plM = chanAt(ch, cbi, ix, iy, iz);
      if (!isFinite(plM) || plM >= 65504) continue;
      const share = pl - plM;                 // ≤ 0 dB; 0 = this mechanism carries it all
      if (share < SHARE_DB_LO) continue;      // mechanism does not meaningfully reach here
      if (share > best) { best = share; dom = { short: m.short, share, color: m.color }; }
    }
  }
  const rows = ['<div class="pr-vx">voxel ' + ix + ',' + iy + ',' + iz + ' · y ' + p.y.toFixed(1) + ' m</div>'];
  if (noSignal) {
    rows.push('<div><b>no signal</b></div>');
    rows.push('<div class="pr-hint">' + pl.toFixed(0) + ' dB loss — inside structure or beyond the solved field</div>');
  } else {
    const cov = rsrp >= DISPLAY.threshDbm + 10 ? 'good' : rsrp >= DISPLAY.threshDbm ? 'marginal' : 'dead';
    rows.push('<div><b>' + pl.toFixed(1) + '</b> dB path loss</div>');
    rows.push('<div><b>' + rsrp.toFixed(0) + '</b> dBm RSRP · <span class="pr-cov pr-' + cov + '">' + cov + '</span></div>');
    if (tau != null) rows.push('<div>τ <b>' + tau.toFixed(0) + '</b> ns first arrival</div>');
    if (dom) rows.push('<div>dominant <span style="color:rgb('
      + dom.color.map((c) => Math.round(c * 200)).join(',') + ')">■</span> ' + dom.short
      + ' (' + dom.share.toFixed(1) + ' dB)</div>');
    else if (vol && mechList(vol).length) rows.push('<div class="pr-hint">open a mechanism view to load its share</div>');
  }
  probeBody.innerHTML = rows.join('');
  probeEl.hidden = false;
  if (!probeMarker) {
    probeMarker = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(extent[0], extent[2]) / 120, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xff3366 }));
    scene.add(probeMarker);
  }
  probeMarker.position.copy(p); probeMarker.visible = true;
}
function hideProbe() { if (probeEl) probeEl.hidden = true; if (probeMarker) probeMarker.visible = false; }
if (probeClose) probeClose.addEventListener('click', hideProbe);

// ---- Display-control wiring ----
function reRunViz() { if (vizMode) runVizMode(vizMode.value); }
function heightFromSlider() { return extent ? (Number(dispHeight.value) / 100) * extent[1] : null; }
function updateHeightLabel() {
  if (!dispHeightVal) return;
  dispHeightVal.textContent = (DISPLAY.plane && DISPLAY.sliceY != null)
    ? 'y = ' + DISPLAY.sliceY.toFixed(1) + ' m' : 'whole volume';
}
if (dispPlaneBtn) dispPlaneBtn.addEventListener('click', () => {
  DISPLAY.plane = !DISPLAY.plane;
  dispPlaneBtn.textContent = DISPLAY.plane ? 'Plane' : 'Full volume';
  dispPlaneBtn.classList.toggle('active', DISPLAY.plane);
  if (dispHeight) dispHeight.disabled = !DISPLAY.plane;
  DISPLAY.sliceY = DISPLAY.plane ? heightFromSlider() : null;
  updateHeightLabel(); reRunViz(); refreshMeta();
});
if (dispHeight) {
  dispHeight.addEventListener('input', () => { DISPLAY.sliceY = heightFromSlider(); updateHeightLabel(); refreshMeta(); });
  dispHeight.addEventListener('change', reRunViz);            // re-solve on release, not per pixel
}
if (dispEirp) dispEirp.addEventListener('change', () => { DISPLAY.eirpDbm = Number(dispEirp.value) || 20; reRunViz(); });
if (dispScale) dispScale.addEventListener('change', () => { DISPLAY.scale = dispScale.value; reRunViz(); });
if (dispClasses) dispClasses.addEventListener('change', () => { DISPLAY.classes = dispClasses.checked; reRunViz(); });
if (dispThresh) {
  dispThresh.addEventListener('input', () => {
    DISPLAY.threshDbm = Number(dispThresh.value);
    if (dispThreshVal) dispThreshVal.textContent = dispThresh.value;
  });
  dispThresh.addEventListener('change', reRunViz);
}

function runStaticField() {
  if (!ensureInit()) return;
  const _t0 = performance.now();
  const tx = firstTx();
  const src = tx ? tx.body.position : center;
  const freqMHz = Number(wfBand && wfBand.value) || 2437;
  disposeField();
  disposeLobes(); disposeVectors(); disposeInterference(); disposeSweep(); // field replaces others

  // Geometry-aware field: march Tx→sample through the material grid (analytic
  // Motley-Keenan). Falls back to free space if the grid asset is unavailable.
  const GRID = decodeGrid();
  const mult = bandMult(freqMHz);
  const L = LOSS_DB.map((v) => v * mult), Ap = APM_DB.map((v) => v * mult);
  const f1 = 20 * Math.log10(freqMHz) + FSPL_CONST;         // fspl at 1 m
  const txVox = GRID ? [clampi(Math.floor(src.x / CELL_M), GDIMS[0]),
                        clampi(Math.floor(src.y / CELL_M), GDIMS[1]),
                        clampi(Math.floor(src.z / CELL_M), GDIMS[2])] : null;

  // Prefer a precomputed full-physics PL volume when one matches the placed Tx;
  // otherwise render analytic now and upgrade in the background if one loads.
  const vol = volMatches(window.SIM3D_VOLUME, txVox) ? window.SIM3D_VOLUME : null;
  const bandIdx = vol ? nearestBandIndex(vol, freqMHz) : 0;
  // Tier 1 miss: try to upgrade in the background (cache first, then surrogate), and
  // render the analytic floor immediately so the user is never looking at nothing.
  if (!vol && txVox) {
    tryLoadVolume(txVox, () => { if (fieldObj) runStaticField(); });
    tryLoadSurrogate();
  }

  const { sliced, g, yLo, yHi, yInc } = sampleBounds();     // full volume, or one prediction plane
  const samples = [];                                       // [x, y, z, mag, r, g, b]
  for (let x = g * 0.5; x < extent[0]; x += g)
    for (let y = yLo; y < yHi; y += yInc)
      for (let z = g * 0.5; z < extent[2]; z += g) {
        let pl;
        if (vol) {
          pl = plAt(vol, bandIdx, clampi(Math.floor(x / CELL_M), vol.dims[0]),
            clampi(Math.floor(y / CELL_M), vol.dims[1]), clampi(Math.floor(z / CELL_M), vol.dims[2]));
        } else if (GRID) {
          pl = marchPL(txVox,
            clampi(Math.floor(x / CELL_M), GDIMS[0]),
            clampi(Math.floor(y / CELL_M), GDIMS[1]),
            clampi(Math.floor(z / CELL_M), GDIMS[2]), L, Ap, f1);
        } else {
          const dx = x - src.x, dy = y - src.y, dz = z - src.z;
          const d = Math.max(0.5, Math.sqrt(dx * dx + dy * dy + dz * dz));
          pl = 20 * Math.log10(d) + 20 * Math.log10(freqMHz) - 27.55;   // FSPL fallback
        }
        if (!isFinite(pl) || pl >= 65504) continue;
        const cs = coverageSample(pl, sliced);               // colour/keep per Display settings
        if (!cs.keep) continue;
        samples.push([x, y, z, cs.mag, cs.rgb[0], cs.rgb[1], cs.rgb[2]]);
      }
  // Instanced cubes (not THREE.Points — the WebGPU backend can't size points):
  // one small cube per sample, coloured/sized by the Display settings.
  const cube = new THREE.BoxGeometry(g * 0.42, g * 0.42, g * 0.42);
  const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false });
  fieldObj = new THREE.InstancedMesh(cube, mat, samples.length);
  const dummy = new THREE.Object3D(), col = new THREE.Color();
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    dummy.position.set(s[0], s[1], s[2]); dummy.scale.setScalar(0.5 + s[3] * 1.6); dummy.updateMatrix();
    fieldObj.setMatrixAt(i, dummy.matrix);
    col.setRGB(s[4], s[5], s[6]); fieldObj.setColorAt(i, col);
  }
  fieldObj.instanceMatrix.needsUpdate = true;
  if (fieldObj.instanceColor) fieldObj.instanceColor.needsUpdate = true;
  scene.add(fieldObj);
  coverageLegend();
  // Name the tier that actually produced this picture — cached volume, surrogate, or the
  // analytic floor. Which engine drew it changes how much it should be trusted.
  const tier = vol ? TIER.CACHE : (GRID ? TIER.ANALYTIC : TIER.ANALYTIC);
  const srcLabel = vol ? TIER.CACHE + ' — full-physics solve (SceneV3 eikonal/Fresnel)'
    : GRID ? TIER.ANALYTIC + ' — Motley-Keenan multiwall'
           : TIER.ANALYTIC + ' — FSPL only (material grid unavailable)';
  const fell = !vol && SURROGATE.state === 'unavailable'
    ? ' · surrogate tier skipped: ' + SURROGATE.note : '';
  setStatus('Static field · ' + srcLabel + ' @ ' + freqMHz + ' MHz · '
    + samples.length.toLocaleString() + ' samples'
    + (vol ? ' · curved eikonal shadows + in-wall lag.' : GRID ? ' · walls shadow the field.' : '.')
    + fell);
  window.SIM3D_TIER = tier;
  lastSolveMs = Math.round(performance.now() - _t0);
  refreshMeta();
}
function disposeField() {
  if (!fieldObj) return;
  scene.remove(fieldObj); fieldObj.geometry.dispose(); fieldObj.material.dispose(); fieldObj = null;
  hideLegend();
}

// ---- Per-mechanism field: one EM mechanism's own contribution ----
// Path loss is shown in absolute dB (it IS the level). The other mechanisms are shown as
// a CONTRIBUTION SHARE, because "reflection = 118 dB" on its own is not interpretable —
// what matters is how much of the signal standing at that voxel got there by reflecting.
// share = P_mech / P_total = 10^((PL_total - PL_mech)/10), both already in the cache.
//
// The share is coloured in dB, not percent. A mechanism that carries 0.1% of the power
// still has spatial structure worth seeing, and a linear 0–100% ramp collapses every such
// mechanism to a single empty colour — on the production scene that left 18 voxels on
// screen out of 588k. SHARE_DB_LO sets how far down the ramp reaches.
const SHARE_DB_LO = -50;      // dB below the total power at the same voxel
const SHARE_DB_HI = 0;        // this mechanism alone accounts for the whole level
function runMechanismField(mech) {
  if (!ensureInit()) return;
  const meta = MECH_BY_KEY[mech];
  const freqMHz = Number(wfBand && wfBand.value) || 2437;
  disposeField(); disposeLobes(); disposeVectors(); disposeInterference(); disposeSweep();

  const GRID = decodeGrid();
  const tx = firstTx();
  const src = tx ? tx.body.position : center;
  const D = GDIMS || MANIFEST.grid_shape;
  const txVox = D ? [clampi(Math.floor(src.x / CELL_M), D[0]),
                     clampi(Math.floor(src.y / CELL_M), D[1]),
                     clampi(Math.floor(src.z / CELL_M), D[2])] : null;

  const vol = window.SIM3D_VOLUME || null;
  if (!vol) {
    setStatus(meta.label + ' · loading the precomputed volume…');
    const done = () => { if (vizMode && vizMode.value === mech) runMechanismField(mech); };
    if (txVox) tryLoadVolume(txVox, done);
    else loadVolumeIndex().then(() => { const e = volumeIndex[0]; if (e) loadVolume(e).then(done); else noMechData(meta); });
    return;
  }
  if (!mechEntry(vol, mech)) { noMechData(meta, vol); return; }

  loadMechanism(vol, mech).then((ch) => {
    if (!ch || (vizMode && vizMode.value !== mech)) return;
    const _t0 = performance.now();
    const cbi = chanBandIndex(ch, freqMHz);          // channel bands may be a subset
    const tbi = nearestBandIndex(vol, ch.bands[cbi]); // matching band in the total volume
    // path loss and the diagnostics (refraction/absorption) are absolute maps; the true
    // additive mechanisms (reflection/diffraction/scattering) are shown as a share of total.
    const isShare = mech !== 'path_loss' && !(MECH_BY_KEY[mech] && MECH_BY_KEY[mech].diagnostic);
    const { sliced, g, yLo, yHi, yInc } = sampleBounds();  // full volume, or one prediction plane
    const inside = insideMaskFor(vol.dims);
    // Absorption is deposited IN materials, so its map lives in wall voxels, not the air
    // interior — it must NOT be masked to `inside` or nothing renders.
    const wallDiag = mech === 'absorption';
    const NY = vol.dims[1], NZ = vol.dims[2];
    const samples = [];
    const lossMin = 45, lossMax = 130;
    let maxShare = -Infinity;                        // dB, so 0 is a real value not a floor
    for (let x = g * 0.5; x < extent[0]; x += g)
      for (let y = yLo; y < yHi; y += yInc)
        for (let z = g * 0.5; z < extent[2]; z += g) {
          const ix = clampi(Math.floor(x / CELL_M), vol.dims[0]);
          const iy = clampi(Math.floor(y / CELL_M), NY);
          const iz = clampi(Math.floor(z / CELL_M), NZ);
          if (inside && !wallDiag && !inside[(ix * NY + iy) * NZ + iz]) continue;
          const plM = chanAt(ch, cbi, ix, iy, iz);
          if (!isFinite(plM) || plM >= 65504) continue;
          let t;
          if (isShare) {
            const plT = plAt(vol, tbi, ix, iy, iz);
            // >1 is physically possible where mechanisms interfere destructively; clamp
            // for display but keep the raw peak for the status line.
            const shareDb = plT - plM;
            if (shareDb > maxShare) maxShare = shareDb;
            if (shareDb < SHARE_DB_LO) continue;     // drop where this mechanism is absent
            t = (Math.min(shareDb, SHARE_DB_HI) - SHARE_DB_LO) / (SHARE_DB_HI - SHARE_DB_LO);
          } else {
            t = 1 - Math.min(1, Math.max(0, (plM - lossMin) / (lossMax - lossMin)));
            if (!sliced && t < 0.12) continue;       // a plane keeps its faint cells; the cloud drops them
          }
          samples.push([x, y, z, t]);
        }

    const cube = new THREE.BoxGeometry(g * 0.42, g * 0.42, g * 0.42);
    const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false });
    fieldObj = new THREE.InstancedMesh(cube, mat, samples.length);
    const dummy = new THREE.Object3D(), col = new THREE.Color();
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      dummy.position.set(s[0], s[1], s[2]); dummy.scale.setScalar(0.5 + s[3] * 1.6); dummy.updateMatrix();
      fieldObj.setMatrixAt(i, dummy.matrix);
      const c = ramp(s[3]); col.setRGB(c[0], c[1], c[2]); fieldObj.setColorAt(i, col);
    }
    fieldObj.instanceMatrix.needsUpdate = true;
    if (fieldObj.instanceColor) fieldObj.instanceColor.needsUpdate = true;
    scene.add(fieldObj);

    if (isShare) showLegend(SHARE_DB_HI, SHARE_DB_LO, meta.label + ' share', ' dB');
    else showLegend(lossMin, lossMax, 'path loss');
    setStatus(meta.label + ' · ' + (isShare
      ? 'share of the total power at each voxel, in dB (0 = this mechanism carries all of '
        + 'it; peak ' + maxShare.toFixed(1) + ' dB = ' + (Math.pow(10, maxShare / 10) * 100).toFixed(0) + ' %)'
      : 'absolute path loss, this mechanism alone')
      + ' @ ' + Math.round(ch.bands[cbi]) + ' MHz · ' + samples.length.toLocaleString() + ' voxels · '
      + (ch.combine_as === 'incoherent' ? 'summed as power (uncorrelated)' : 'coherent complex field')
      + ' · first arrival ≤ ' + ch.t_max_ns.toFixed(0) + ' ns.');
    window.SIM3D_TIER = TIER.CACHE;                  // per-mechanism views are always the cached solve
    lastSolveMs = Math.round(performance.now() - _t0);
    refreshMeta();
  });
}
// ---- Vacuum mode: the invariant gate, checked in the browser ----
// Not decorative. The vacuum volume is solved by the SAME engine as every other mode, so
// comparing it here against the closed form 20log10(d) + 20log10(f) - 27.55 is a genuine
// end-to-end check of the whole chain — solver, float16 export, index, JS decoder — and it
// is the one case where the right answer is known exactly.
function checkVacuum(vol, freqMHz) {
  const bi = nearestBandIndex(vol, freqMHz);
  const f = vol.bands[bi];
  const tx = vol.tx_vox;
  const NX = vol.dims[0], NY = vol.dims[1], NZ = vol.dims[2];
  let worst = 0, n = 0, worstAt = null, tauBad = 0;
  const d0 = 1.0;
  for (let ix = 0; ix < NX; ix += 3)
    for (let iy = 0; iy < NY; iy += 2)
      for (let iz = 0; iz < NZ; iz += 3) {
        const d = Math.hypot(ix - tx[0], iy - tx[1], iz - tx[2]) * CELL_M;
        const ana = 20 * Math.log10(Math.max(d, d0)) + 20 * Math.log10(f) - 27.55;
        const got = plAt(vol, bi, ix, iy, iz);
        if (!isFinite(got) || got >= 65504) continue;
        const e = Math.abs(got - ana);
        if (e > worst) { worst = e; worstAt = [ix, iy, iz]; }
        n++;
        // causality: the front cannot outrun c. One cell of slack for the eikonal's
        // source-contour offset (skfmm seeds on the source cell).
        const tNs = tAt(vol, ix, iy, iz);
        if (tNs < 65504 && tNs < (d / 299792458) * 1e9 - (CELL_M / 299792458) * 1e9 * 1.5) tauBad++;
      }
  // float16 storage is the floor on achievable agreement: half-precision spacing at
  // ~130 dB is 0.0625 dB, so anything under ~0.05 dB is exact to the wire format.
  return { band: f, samples: n, worstDb: worst, worstAt, tauViolations: tauBad,
           pass: worst < 0.1 && tauBad === 0 };
}
function runVacuumField() {
  if (!ensureInit()) return;
  const freqMHz = Number(wfBand && wfBand.value) || 2437;
  const vol = window.SIM3D_VOLUME;
  if (!vol || (vol.entry.mode || 'indoor') !== 'vacuum') {
    const tx = firstTx();
    const D = GDIMS || MANIFEST.grid_shape;
    const src = tx ? tx.body.position : null;
    const txVox = (src && D) ? [clampi(Math.floor(src.x / CELL_M), D[0]),
                               clampi(Math.floor(src.y / CELL_M), D[1]),
                               clampi(Math.floor(src.z / CELL_M), D[2])] : null;
    tryLoadVolume(txVox, () => { if (currentMode() === 'vacuum') runVacuumField(); });
    return;
  }
  runStaticField();                      // draw it with the normal coverage renderer
  const r = checkVacuum(vol, freqMHz);
  setStatus('Vacuum · ' + (r.pass ? 'INVARIANT GATE PASSED' : 'GATE FAILED')
    + ' · engine vs closed-form FSPL over ' + r.samples.toLocaleString() + ' voxels @ '
    + Math.round(r.band) + ' MHz: max |Δ| = ' + r.worstDb.toFixed(4) + ' dB'
    + (r.worstDb < 0.05 ? ' (float16 storage limit — exact to the wire format)' : '')
    + ' · causality violations: ' + r.tauViolations
    + (r.pass ? '' : ' · worst at voxel ' + (r.worstAt || []).join(',')));
}

function noMechData(meta, vol) {
  const have = vol ? mechList(vol).map((m) => m.label).join(', ') : '';
  setStatus(meta.label + ' · no mechanism channel in the cached volume'
    + (have ? ' (this Tx has: ' + have + ')' : '')
    + ' — re-export with `python "SIM V1 3D/export_pl_volume.py" --mechanisms path_loss,reflection,diffraction,scattering,refraction,absorption`.');
}

// ---- Radiation pattern: analytic gain lobes at each Tx (closed-form by kind) ----
// Local frame: main axis = +Y (matches the catalog rods). The catalog carries no
// gain data, so the lobe shape is keyed off the antenna kind; real per-kind
// patterns could later be exported from Antenna_Physics_3D.py into the catalog.
const PATTERN = {
  omni: { t: 'omni' }, monopole: { t: 'omni' }, whip: { t: 'omni' }, handheld: { t: 'omni' },
  dipole: { t: 'omni' }, discone: { t: 'omni' }, biconical: { t: 'omni' }, loop: { t: 'omni' },
  fractal: { t: 'omni' }, metamaterial: { t: 'omni' },
  inverted_f: { t: 'dir', n: 1.5, back: 0.3 }, pifa: { t: 'dir', n: 1.5, back: 0.3 },
  slot: { t: 'dir', n: 2, back: 0.25 }, patch: { t: 'dir', n: 2, back: 0.2 },
  panel: { t: 'dir', n: 3, back: 0.15 }, quad: { t: 'dir', n: 3, back: 0.2 },
  yagi: { t: 'dir', n: 4, back: 0.12 }, lpda: { t: 'dir', n: 4, back: 0.12 },
  helical: { t: 'dir', n: 5, back: 0.1 }, phased_array: { t: 'dir', n: 6, back: 0.1 },
  horn: { t: 'dir', n: 6, back: 0.08 }, dish: { t: 'dir', n: 8, back: 0.06 },
};

// normalized gain 0..1 for a unit direction in the antenna's local frame (+Y axis).
function gainNorm(kind, dx, dy, dz) {
  const p = PATTERN[kind] || { t: 'omni' };
  if (p.t === 'omni') return Math.max(0.03, Math.sqrt(Math.max(0, 1 - dy * dy)));  // donut around +Y
  const u = dy;                                                                     // cosθ from +Y
  const front = u > 0 ? Math.pow(u, p.n) : 0;
  const back = u < 0 ? p.back * u * u : 0;
  return Math.max(front, back, 0.03);
}

// a lobe mesh: SphereGeometry deformed so radius = gain, coloured by gain (viridis).
function gainMesh(kind, maxR) {
  const geo = new THREE.SphereGeometry(1, 48, 32);
  const pos = geo.attributes.position;
  const colArr = new Float32Array(pos.count * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    const g = gainNorm(kind, v.x, v.y, v.z), r = g * maxR + 0.01;
    pos.setXYZ(i, v.x * r, v.y * r, v.z * r);
    const c = ramp(g); colArr[i * 3] = c[0]; colArr[i * 3 + 1] = c[1]; colArr[i * 3 + 2] = c[2];
  }
  pos.needsUpdate = true;
  geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
  return new THREE.Mesh(geo, mat);
}

let lobeGroup = null;
function disposeLobes() {
  if (!lobeGroup) return;
  lobeGroup.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
  scene.remove(lobeGroup); lobeGroup = null;
}
function runRadiationPattern() {
  if (!ensureInit()) return;
  disposeLobes();
  disposeField(); disposeVectors(); disposeInterference(); disposeSweep(); // lobe replaces others
  const maxR = Math.max(2.5, Math.min(extent[0], extent[2]) * 0.14);
  lobeGroup = new THREE.Group();
  const txs = placed.filter((p) => p.role === 'tx');
  let label;
  if (txs.length) {
    for (const p of txs) {
      const lobe = gainMesh(p.kind, maxR);
      lobe.position.copy(p.body.position); lobe.quaternion.copy(p.mesh.quaternion);
      lobeGroup.add(lobe);
    }
    label = txs.length + ' transmitter' + (txs.length > 1 ? 's' : '');
  } else {
    const kind = txType ? txType.value : 'omni';
    const lobe = gainMesh(kind, maxR); lobe.position.copy(center);
    lobeGroup.add(lobe);
    label = 'preview · ' + ((CATALOG.entries[kind] && CATALOG.entries[kind].label) || kind);
  }
  scene.add(lobeGroup);
  hideLegend();
  setStatus('Radiation pattern · ' + label + ' · analytic gain lobe by antenna type (beam = yellow, nulls = purple).');
}

// ---- Field distribution: oscillating field-vector arrows (analytic amplitude) ----
// One arrow per interior lattice sample: direction = radial from the Tx, colour &
// base length = amplitude 10^(−PL/20) (reuses marchPL, so walls shadow it); the
// length pulses by cos(k·r − ωt) each frame → wavefronts sweep from the Tx. The
// per-instance data lives in flat typed arrays so the update can later move to a
// WebGPU/TSL compute pass (P13) unchanged.
let vectorState = null, vecTime = 0;
const _UP = new THREE.Vector3(0, 1, 0);

function disposeVectors() {
  if (!vectorState) return;
  scene.remove(vectorState.mesh);
  vectorState.mesh.geometry.dispose(); vectorState.mesh.material.dispose();
  vectorState = null;
}

function runFieldVectors() {
  if (!ensureInit()) return;
  disposeVectors(); disposeField(); disposeLobes(); disposeInterference(); disposeSweep(); // exclusive
  const tx = firstTx();
  const src = tx ? tx.body.position : center;
  const freqMHz = Number(wfBand && wfBand.value) || 2437;

  // amplitude setup — identical to runStaticField
  const GRID = decodeGrid();
  const mult = bandMult(freqMHz);
  const L = LOSS_DB.map((v) => v * mult), Ap = APM_DB.map((v) => v * mult);
  const f1 = 20 * Math.log10(freqMHz) + FSPL_CONST;
  const txVox = GRID ? [clampi(Math.floor(src.x / CELL_M), GDIMS[0]),
                        clampi(Math.floor(src.y / CELL_M), GDIMS[1]),
                        clampi(Math.floor(src.z / CELL_M), GDIMS[2])] : null;

  // interior lattice via inside_b64 (fallback: whole extent), strided to ~5k
  const inside = decodeInside();
  const dims = GDIMS || MANIFEST.grid_shape ||
    [Math.round(extent[0] / CELL_M), Math.round(extent[1] / CELL_M), Math.round(extent[2] / CELL_M)];
  const NX = dims[0], NY = dims[1], NZ = dims[2];
  const frac = inside ? 0.45 : 1.0;
  const sy = NY > 6 ? 2 : 1;
  const sxz = Math.max(2, Math.round(Math.sqrt(NX * NZ * (NY / sy) * frac / 5000)));
  const at = (x, y, z) => inside[(x * NY + y) * NZ + z];

  const lossMin = 45, lossMax = 130, maxLen = sxz * CELL_M * 0.8;
  const lamDisp = Math.min(6, Math.max(2.5, 6.5 - 4 * (freqMHz - 2400) / 3800));
  const kDisp = 2 * Math.PI / lamDisp;
  const basePos = [], quats = [], baseLen = [], phase0 = [], cols = [];
  const q = new THREE.Quaternion(), dir = new THREE.Vector3();
  for (let ix = 0; ix < NX; ix += sxz)
    for (let iy = 0; iy < NY; iy += sy)
      for (let iz = 0; iz < NZ; iz += sxz) {
        if (inside && !at(ix, iy, iz)) continue;
        const px = (ix + 0.5) * CELL_M, py = (iy + 0.5) * CELL_M, pz = (iz + 0.5) * CELL_M;
        dir.set(px - src.x, py - src.y, pz - src.z);
        const r = dir.length();
        if (r < CELL_M) continue;                          // skip the Tx cell
        const pl = GRID ? marchPL(txVox, ix, iy, iz, L, Ap, f1)
                        : 20 * Math.log10(Math.max(0.5, r)) + 20 * Math.log10(freqMHz) - 27.55;
        const t = 1 - Math.min(1, Math.max(0, (pl - lossMin) / (lossMax - lossMin)));
        if (t < 0.1) continue;                             // cull faint arrows
        dir.multiplyScalar(1 / r);
        q.setFromUnitVectors(_UP, dir);
        basePos.push(px, py, pz); quats.push(q.x, q.y, q.z, q.w);
        baseLen.push(maxLen * t); phase0.push(kDisp * r);
        const c = ramp(t); cols.push(c[0], c[1], c[2]);
      }

  const count = baseLen.length;
  const geo = new THREE.ConeGeometry(0.09, 1, 6).translate(0, 0.5, 0);  // base at 0, tip +Y
  const mesh = new THREE.InstancedMesh(geo,
    new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }), count);
  const col = new THREE.Color();
  for (let i = 0; i < count; i++) { col.setRGB(cols[i * 3], cols[i * 3 + 1], cols[i * 3 + 2]); mesh.setColorAt(i, col); }
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  vectorState = { mesh, count,
    basePos: new Float32Array(basePos), quat: new Float32Array(quats),
    baseLen: new Float32Array(baseLen), phase0: new Float32Array(phase0),
    omega: 2 * Math.PI * 0.7 };
  vecTime = 0;
  advanceVectors(0);                                       // set initial matrices
  scene.add(mesh);
  hideLegend();
  setStatus('Field distribution · ' + count.toLocaleString() + ' oscillating vectors @ ' + freqMHz +
    ' MHz · length/colour = amplitude, wavefronts sweep from the Tx (display λ ≈ ' + lamDisp.toFixed(1) + ' m).');
}

const _vdummy = new THREE.Object3D(), _vq = new THREE.Quaternion();
function advanceVectors(dt) {
  vecTime += dt;
  const V = vectorState, m = V.mesh;
  for (let i = 0; i < V.count; i++) {
    const s = 0.5 + 0.5 * Math.cos(V.phase0[i] - V.omega * vecTime);
    _vdummy.position.set(V.basePos[i * 3], V.basePos[i * 3 + 1], V.basePos[i * 3 + 2]);
    _vq.set(V.quat[i * 4], V.quat[i * 4 + 1], V.quat[i * 4 + 2], V.quat[i * 4 + 3]);
    _vdummy.quaternion.copy(_vq);
    _vdummy.scale.set(1, V.baseLen[i] * (0.2 + 0.8 * s), 1);
    _vdummy.updateMatrix();
    m.setMatrixAt(i, _vdummy.matrix);
  }
  m.instanceMatrix.needsUpdate = true;
}

// ---- Interference & multipath: per-voxel complex phasor sum (analytic) ----
// Σ_i A_i·e^{j(k·d_i+φ_i)} over all placed Tx (+ one nearest-wall image source
// when there is only one Tx). A_i = 10^(−PL_i/20) from marchPL, d_i the path
// length. Colour by combined loss −20·log10|Σ| (bright antinode, dark null). The
// per-voxel re/im live in flat arrays → WebGPU-compute-ready; the phase toggle
// shows the instantaneous field Re(Σ·e^{−jωt}).
function divRamp(t) {
  const s = [[25, 25, 80], [45, 100, 155], [70, 175, 165], [180, 215, 95], [255, 240, 130]];
  t = Math.min(1, Math.max(0, t)) * (s.length - 1);
  const i = Math.floor(t), f = t - i, a = s[i], b = s[Math.min(i + 1, s.length - 1)];
  return [(a[0] + (b[0] - a[0]) * f) / 255, (a[1] + (b[1] - a[1]) * f) / 255, (a[2] + (b[2] - a[2]) * f) / 255];
}

// nearest-wall image of a single Tx (rough multipath seed; full tree = P7).
function nearestWallImage(src, txVox) {
  const GRID = decodeGrid();
  if (!GRID) return null;
  const NX = GDIMS[0], NY = GDIMS[1], NZ = GDIMS[2];
  const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  let best = null;
  for (const d of dirs) {
    for (let step = 1; step <= 25; step++) {
      const vx = txVox[0] + d[0] * step, vy = txVox[1] + d[1] * step, vz = txVox[2] + d[2] * step;
      if (vx < 0 || vx >= NX || vy < 0 || vy >= NY || vz < 0 || vz >= NZ) break;
      if (GRID[(vx * NY + vy) * NZ + vz] > 0) { if (!best || step < best.step) best = { d, step }; break; }
    }
  }
  if (!best) return null;
  const wall = best.step * CELL_M;                          // distance to wall along the axis
  const world = { x: src.x + best.d[0] * 2 * wall, y: src.y + best.d[1] * 2 * wall, z: src.z + best.d[2] * 2 * wall };
  const iv = [clampi(Math.floor(world.x / CELL_M), NX), clampi(Math.floor(world.y / CELL_M), NY), clampi(Math.floor(world.z / CELL_M), NZ)];
  return { world, txVox: iv, phi: Math.PI, refl: 0.6 };     // reflection: −4.4 dB + sign flip
}

let interferenceState = null, intfTime = 0, phaseToggleEl = null;
function disposeInterference() {
  if (!interferenceState) return;
  scene.remove(interferenceState.mesh);
  interferenceState.mesh.geometry.dispose(); interferenceState.mesh.material.dispose();
  interferenceState = null;
  if (phaseToggleEl) phaseToggleEl.hidden = true;
}
function ensurePhaseToggle() {
  if (phaseToggleEl) return phaseToggleEl;
  const wrap = viewport && viewport.parentElement;
  if (!wrap) return null;
  phaseToggleEl = document.createElement('label');
  phaseToggleEl.id = 'sim3dPhaseToggle'; phaseToggleEl.className = 'sim3d-toggle'; phaseToggleEl.hidden = true;
  phaseToggleEl.innerHTML = '<input type="checkbox"> Animate phase';
  phaseToggleEl.querySelector('input').addEventListener('change', (e) => {
    if (!interferenceState) return;
    interferenceState.animating = e.target.checked;
    if (!e.target.checked) resetInterferenceColors();
  });
  wrap.appendChild(phaseToggleEl);
  return phaseToggleEl;
}
function resetInterferenceColors() {
  const S = interferenceState; if (!S) return;
  const col = new THREE.Color();
  for (let i = 0; i < S.count; i++) { col.setRGB(S.baseCol[i * 3], S.baseCol[i * 3 + 1], S.baseCol[i * 3 + 2]); S.mesh.setColorAt(i, col); }
  if (S.mesh.instanceColor) S.mesh.instanceColor.needsUpdate = true;
}

function runInterference() {
  if (!ensureInit()) return;
  disposeInterference(); disposeField(); disposeLobes(); disposeVectors(); disposeSweep();
  const freqMHz = Number(wfBand && wfBand.value) || 2437;
  const GRID = decodeGrid();
  const mult = bandMult(freqMHz);
  const L = LOSS_DB.map((v) => v * mult), Ap = APM_DB.map((v) => v * mult);
  const f1 = 20 * Math.log10(freqMHz) + FSPL_CONST;
  const toVox = (w) => GRID ? [clampi(Math.floor(w.x / CELL_M), GDIMS[0]), clampi(Math.floor(w.y / CELL_M), GDIMS[1]), clampi(Math.floor(w.z / CELL_M), GDIMS[2])] : [0, 0, 0];

  // sources = placed Tx (+ a nearest-wall image if fewer than 2)
  const sources = placed.filter((p) => p.role === 'tx').map((p) => {
    const w = { x: p.body.position.x, y: p.body.position.y, z: p.body.position.z };
    return { world: w, txVox: toVox(w), phi: 0, refl: 1 };
  });
  if (!sources.length) sources.push({ world: { x: center.x, y: center.y, z: center.z }, txVox: toVox(center), phi: 0, refl: 1 });
  if (sources.length < 2 && GRID) { const img = nearestWallImage(sources[0].world, sources[0].txVox); if (img) sources.push(img); }

  const lamDisp = Math.min(10, Math.max(5, 10 - 4 * (freqMHz - 2400) / 3800));
  const kDisp = 2 * Math.PI / lamDisp;

  // interior lattice (finer than the field vectors so fringes resolve)
  const inside = decodeInside();
  const dims = GDIMS || MANIFEST.grid_shape ||
    [Math.round(extent[0] / CELL_M), Math.round(extent[1] / CELL_M), Math.round(extent[2] / CELL_M)];
  const NX = dims[0], NY = dims[1], NZ = dims[2];
  const frac = inside ? 0.45 : 1.0;
  const sy = NY > 6 ? 2 : 1;
  const sxz = Math.max(2, Math.round(Math.sqrt(NX * NZ * (NY / sy) * frac / 6500)));
  const at = (x, y, z) => inside[(x * NY + y) * NZ + z];
  const minA = Math.pow(10, -130 / 20);                     // "no source reaches" floor

  const pos = [], reA = [], imA = [], magA = [], tA = [], cols = [];
  for (let ix = 0; ix < NX; ix += sxz)
    for (let iy = 0; iy < NY; iy += sy)
      for (let iz = 0; iz < NZ; iz += sxz) {
        if (inside && !at(ix, iy, iz)) continue;
        const px = (ix + 0.5) * CELL_M, py = (iy + 0.5) * CELL_M, pz = (iz + 0.5) * CELL_M;
        let re = 0, im = 0, sumA = 0;
        for (const s of sources) {
          const pl = GRID ? marchPL(s.txVox, ix, iy, iz, L, Ap, f1)
            : 20 * Math.log10(Math.max(0.5, Math.hypot(px - s.world.x, py - s.world.y, pz - s.world.z))) + 20 * Math.log10(freqMHz) - 27.55;
          const A = s.refl * Math.pow(10, -pl / 20);
          const d = Math.hypot(px - s.world.x, py - s.world.y, pz - s.world.z);
          const psi = kDisp * d + s.phi;
          re += A * Math.cos(psi); im += A * Math.sin(psi); sumA += A;
        }
        if (sumA < minA) continue;                          // keep nulls where sources are strong but cancel
        const mag = Math.hypot(re, im);
        const plEff = -20 * Math.log10(Math.max(mag, 1e-9));
        const t = Math.min(1, Math.max(0, 1 - (plEff - 45) / 85));
        pos.push(px, py, pz); reA.push(re); imA.push(im); magA.push(mag); tA.push(t);
        const c = divRamp(t); cols.push(c[0], c[1], c[2]);
      }

  const count = tA.length;
  const boxSz = Math.max(extent[0], extent[2]) / 46 * 0.5;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(boxSz, boxSz, boxSz),
    new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.62, depthWrite: false }), count);
  const dummy = new THREE.Object3D(), col = new THREE.Color();
  const baseCol = new Float32Array(cols);
  for (let i = 0; i < count; i++) {
    dummy.position.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    dummy.scale.setScalar(0.5 + 1.3 * tA[i]); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    col.setRGB(baseCol[i * 3], baseCol[i * 3 + 1], baseCol[i * 3 + 2]); mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
  interferenceState = { mesh, count, re: new Float32Array(reA), im: new Float32Array(imA),
    mag: new Float32Array(magA), baseCol, omega: 2 * Math.PI * 0.6, animating: false };
  ensurePhaseToggle(); if (phaseToggleEl) { phaseToggleEl.hidden = false; interferenceState.animating = phaseToggleEl.querySelector('input').checked; }
  hideLegend();
  setStatus('Interference & multipath · ' + sources.length + ' sources · ' + count.toLocaleString() +
    ' voxels · bright = constructive, dark = null (display λ ≈ ' + lamDisp.toFixed(1) + ' m).');
}

function advanceInterference(dt) {
  intfTime += dt;
  const S = interferenceState, ct = Math.cos(S.omega * intfTime), st = Math.sin(S.omega * intfTime);
  const col = new THREE.Color();
  for (let i = 0; i < S.count; i++) {
    const inst = S.re[i] * ct + S.im[i] * st;               // Re(Σ·e^{−jωt})
    const b = 0.2 + 0.8 * Math.min(1, Math.abs(inst) / Math.max(S.mag[i], 1e-9));
    col.setRGB(S.baseCol[i * 3] * b, S.baseCol[i * 3 + 1] * b, S.baseCol[i * 3 + 2] * b);
    S.mesh.setColorAt(i, col);
  }
  if (S.mesh.instanceColor) S.mesh.instanceColor.needsUpdate = true;
}

// ---- Wavefront sweep: animate an arrival-time volume by scrubbing a threshold τ ----
// Voxels with T ≤ τ are "lit"; the shell T ∈ [τ−Δτ, τ+Δτ] is the moving front. It
// bends around corners and lags inside walls because T is the eikonal first-arrival
// (no per-frame solve, no frame stack). Needs a precomputed T volume.
//
// The state holds a LIST of layers, all sharing one voxel lattice and one τ clock. With a
// single layer this is the eikonal sweep; with one layer per mechanism it is the mechanism
// time-lapse, where each front moves on its own tau channel — so the direct arrives first,
// the reflected after it, the diffracted after that and the diffuse halo last. Same clock,
// same scrub wiring, no second animation path.
let sweepState = null;
function disposeSweep() {
  if (!sweepState) return;
  for (const L of sweepState.layers) {
    scene.remove(L.mesh); L.mesh.geometry.dispose(); L.mesh.material.dispose();
  }
  sweepState = null;
  if (animCtl) animCtl.hidden = true;
}

// Voxel lattice shared by every sweep layer: subsampled to ~6k instances so the InstancedMesh
// stays cheap, and restricted to interior voxels. Returns { pos, idx } (idx = grid indices).
function sweepLattice(vol) {
  const inside = insideMaskFor(vol.dims);
  const NX = vol.dims[0], NY = vol.dims[1], NZ = vol.dims[2];
  const frac = inside ? 0.45 : 1.0, sy = NY > 6 ? 2 : 1;
  const sxz = Math.max(2, Math.round(Math.sqrt(NX * NZ * (NY / sy) * frac / 6000)));
  const pos = [], idx = [];
  for (let ix = 0; ix < NX; ix += sxz)
    for (let iy = 0; iy < NY; iy += sy)
      for (let iz = 0; iz < NZ; iz += sxz) {
        if (inside && !inside[(ix * NY + iy) * NZ + iz]) continue;
        pos.push((ix + 0.5) * CELL_M, (iy + 0.5) * CELL_M, (iz + 0.5) * CELL_M);
        idx.push(ix, iy, iz);
      }
  return { pos: new Float32Array(pos), idx: new Int32Array(idx), count: idx.length / 3 };
}
function makeSweepMesh(count) {
  const boxSz = Math.max(extent[0], extent[2]) / 46 * 0.6;
  return new THREE.InstancedMesh(new THREE.BoxGeometry(boxSz, boxSz, boxSz),
    new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.72, depthWrite: false }), count);
}

function runWavefrontSweep() {
  if (!ensureInit()) return;
  disposeSweep(); disposeField(); disposeLobes(); disposeVectors(); disposeInterference();
  if (waveObj) { scene.remove(waveObj); waveObj.geometry.dispose(); waveObj.material.dispose(); waveObj = null; }
  const tx = firstTx();
  const D = GDIMS || MANIFEST.grid_shape || [262, 11, 118];
  const txVox = tx ? [clampi(Math.floor(tx.body.position.x / CELL_M), D[0]),
                      clampi(Math.floor(tx.body.position.y / CELL_M), D[1]),
                      clampi(Math.floor(tx.body.position.z / CELL_M), D[2])] : null;
  const vol = window.SIM3D_VOLUME || null;                  // sweep uses any loaded T volume
  if (!vol) {
    setStatus('Wavefront sweep · loading precomputed T volume…');
    const done = () => { if (vizMode && vizMode.value === 'sweep') runWavefrontSweep(); };
    if (txVox) tryLoadVolume(txVox, done);
    else loadVolumeIndex().then(() => { const e = volumeIndex[0]; if (e) loadVolume(e).then(done); else setStatus('Wavefront sweep · no precomputed T volume found — run `python "SIM V1 3D/export_pl_volume.py"`.'); });
    return;
  }
  const lat = sweepLattice(vol);
  const tNs = new Float32Array(lat.count);
  let live = 0;
  for (let i = 0; i < lat.count; i++) {
    const T = tAt(vol, lat.idx[i * 3], lat.idx[i * 3 + 1], lat.idx[i * 3 + 2]);
    tNs[i] = (T < vol.t_max_ns + 1) ? T : Infinity;         // 65504 sentinel = unreachable
    if (isFinite(tNs[i])) live++;
  }
  const mesh = makeSweepMesh(lat.count);
  sweepState = { mode: 'eikonal', count: lat.count, basePos: lat.pos,
    layers: [{ key: 'eikonal', label: 'first arrival', color: null, tNs, mesh }],
    tMax: vol.t_max_ns, tau: 0, dTau: vol.t_max_ns * 0.06, playing: true };
  scene.add(mesh);
  startSweepUI();
  hideLegend();
  setStatus('Wavefront sweep · eikonal first-arrival T · ' + live.toLocaleString() +
    ' voxels · scrub / Play to sweep the front (bends around corners, lags in walls). T_max ≈ ' + vol.t_max_ns.toFixed(0) + ' ns.');
}

// ---- Mechanism time-lapse: every mechanism's front on one clock ----
// This is the payoff of the per-mechanism tau channels. Each mechanism gets its own
// InstancedMesh driven by its own first-arrival volume, all sharing the τ scrub, so the
// SEQUENCE is visible: direct, then specular reflections, then the diffracted field
// creeping around corners, then the diffuse halo filling in behind.
function runMechanismTimeLapse() {
  if (!ensureInit()) return;
  disposeSweep(); disposeField(); disposeLobes(); disposeVectors(); disposeInterference();
  if (waveObj) { scene.remove(waveObj); waveObj.geometry.dispose(); waveObj.material.dispose(); waveObj = null; }

  const tx = firstTx();
  const D = GDIMS || MANIFEST.grid_shape;
  const txVox = (tx && D) ? [clampi(Math.floor(tx.body.position.x / CELL_M), D[0]),
                             clampi(Math.floor(tx.body.position.y / CELL_M), D[1]),
                             clampi(Math.floor(tx.body.position.z / CELL_M), D[2])] : null;
  const vol = window.SIM3D_VOLUME || null;
  if (!vol) {
    setStatus('Mechanism time-lapse · loading the precomputed volume…');
    const done = () => { if (vizMode && vizMode.value === 'mechlapse') runMechanismTimeLapse(); };
    if (txVox) tryLoadVolume(txVox, done);
    else loadVolumeIndex().then(() => { const e = volumeIndex[0]; if (e) loadVolume(e).then(done); else setStatus('Mechanism time-lapse · no precomputed volume found.'); });
    return;
  }
  const want = mechList(vol);
  if (!want.length) { noMechData({ label: 'Mechanism time-lapse' }, vol); return; }

  setStatus('Mechanism time-lapse · loading ' + want.length + ' arrival-time channels…');
  Promise.all(want.map((m) => loadMechanism(vol, m.key))).then((chans) => {
    if (vizMode && vizMode.value !== 'mechlapse') return;
    const lat = sweepLattice(vol);
    const layers = [];
    let tMax = 0;
    for (let li = 0; li < chans.length; li++) {
      const ch = chans[li]; if (!ch) continue;
      const tNs = new Float32Array(lat.count);
      let live = 0;
      for (let i = 0; i < lat.count; i++) {
        // `true` = prefer the geometric clock, so all layers share one convention
        const T = tauAt(ch, lat.idx[i * 3], lat.idx[i * 3 + 1], lat.idx[i * 3 + 2], true);
        tNs[i] = (T < 65504 - 1) ? T : Infinity;
        if (isFinite(tNs[i])) { live++; if (tNs[i] > tMax) tMax = tNs[i]; }
      }
      if (!live) continue;                        // mechanism reaches nothing here
      const mesh = makeSweepMesh(lat.count);
      layers.push({ key: want[li].key, label: want[li].label, short: want[li].short,
        color: want[li].color, tNs, mesh, live });
      scene.add(mesh);
    }
    if (!layers.length) { noMechData({ label: 'Mechanism time-lapse' }, vol); return; }
    // Order by median arrival so the legend reads in the order the fronts actually appear.
    layers.sort((a, b) => medianFinite(a.tNs) - medianFinite(b.tNs));
    sweepState = { mode: 'mechanism', count: lat.count, basePos: lat.pos, layers,
      tMax: tMax || vol.t_max_ns, tau: 0, dTau: (tMax || vol.t_max_ns) * 0.05, playing: true };
    startSweepUI();
    showMechLegend(layers);
    const mixed = chans.some((c) => c && c.convention === 'eikonal' && !c.tauGeom);
    setStatus('Mechanism time-lapse · ' + layers.map((L) =>
      (L.short || L.label) + ' ' + medianFinite(L.tNs).toFixed(0) + ' ns').join(' → ')
      + ' (median arrival) · scrub / Play to watch the fronts arrive in sequence. '
      + (mixed
        ? 'NOTE: the direct path is on the eikonal clock (in-wall slowdown) and the others '
          + 'on vacuum path length, so the order understates the direct field — re-export '
          + 'to get the matched geometric clock.'
        : 'Clock: vacuum path length for every mechanism, so this is a like-for-like '
          + 'comparison; the Wavefront sweep view shows the true eikonal arrival with '
          + 'in-wall lag.'));
  });
}
function medianFinite(a) {
  const v = [];
  for (let i = 0; i < a.length; i++) if (isFinite(a[i])) v.push(a[i]);
  if (!v.length) return Infinity;
  v.sort((x, y) => x - y);
  return v[v.length >> 1];
}
function startSweepUI() {
  if (animCtl) animCtl.hidden = false;
  if (animPlay) animPlay.textContent = 'Pause';
  if (animScrub) animScrub.value = '0';
  setSweepTau(0);
}

function setSweepTau(tau) {
  const S = sweepState; if (!S) return;
  S.tau = tau;
  const lo = tau - S.dTau, hi = tau + S.dTau;
  for (const L of S.layers) {
    for (let i = 0; i < S.count; i++) {
      const T = L.tNs[i]; let scale, r, g, b;
      if (!(T <= hi)) { scale = 0.001; r = g = b = 0; }        // not yet reached (or ∞) → hidden
      else if (T >= lo) {                                      // the moving front → bright
        scale = 1.4;
        if (L.color) { r = L.color[0]; g = L.color[1]; b = L.color[2]; }
        else { r = 1.0; g = 0.95; b = 0.3; }
      } else {                                                 // already passed → dim trail
        scale = 0.55;
        if (L.color) { r = L.color[0] * 0.32; g = L.color[1] * 0.32; b = L.color[2] * 0.32; }
        else { const c = ramp(0.25 + 0.4 * (1 - T / S.tMax)); r = c[0] * 0.7; g = c[1] * 0.7; b = c[2] * 0.7; }
      }
      _sdummy.position.set(S.basePos[i * 3], S.basePos[i * 3 + 1], S.basePos[i * 3 + 2]);
      _sdummy.scale.setScalar(scale); _sdummy.updateMatrix(); L.mesh.setMatrixAt(i, _sdummy.matrix);
      _scol.setRGB(r, g, b); L.mesh.setColorAt(i, _scol);
    }
    L.mesh.instanceMatrix.needsUpdate = true;
    if (L.mesh.instanceColor) L.mesh.instanceColor.needsUpdate = true;
  }
}
const _sdummy = new THREE.Object3D(), _scol = new THREE.Color();
function advanceSweep(dt) {
  const S = sweepState;
  S.tau += dt * S.tMax * 0.5;                                 // ~2 s to cross the domain
  if (S.tau > S.tMax + S.dTau) S.tau = 0;
  if (animScrub) animScrub.value = String(Math.round(1000 * Math.min(1, S.tau / S.tMax)));
  setSweepTau(S.tau);
}

// ---- Animated wavefront: kinematic expanding shell from the transmitter ----
function runAnimWave() {
  if (!ensureInit()) return;
  disposeSweep();                                            // the kinematic wave replaces the sweep
  const tx = firstTx();
  const src = (tx ? tx.body.position : center).clone();
  if (waveObj) { scene.remove(waveObj); waveObj.geometry.dispose(); waveObj.material.dispose(); }
  const g = new THREE.SphereGeometry(1, 40, 24);
  const m = new THREE.MeshBasicMaterial({ color: 0x0a7f7a, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide });
  waveObj = new THREE.Mesh(g, m); waveObj.position.copy(src);
  waveObj.userData = { r: 0.001, rMax: Math.hypot(extent[0], extent[1], extent[2]), playing: true };
  scene.add(waveObj);
  if (animCtl) animCtl.hidden = false;
  if (animPlay) animPlay.textContent = 'Pause';
  setStatus('Animated wavefront · kinematic expansion from the transmitter (visual stand-in for the time-lapse solver).');
}
function advanceWave(dt) {
  const u = waveObj.userData;
  u.r += dt * u.rMax * 0.35;                                // sweep speed
  if (u.r > u.rMax) u.r = 0.001;                            // loop
  applyWave();
}
function applyWave() {
  const u = waveObj.userData;
  waveObj.scale.setScalar(Math.max(0.001, u.r));
  waveObj.material.opacity = 0.3 * (1 - u.r / u.rMax);
  if (animScrub) animScrub.value = String(Math.round(1000 * u.r / u.rMax));
}

// ---- rail wiring ----
if (runStatic) runStatic.addEventListener('click', runStaticField);
if (runAnim) runAnim.addEventListener('click', runAnimWave);
if (animPlay) animPlay.addEventListener('click', () => {
  if (sweepState) { sweepState.playing = !sweepState.playing; animPlay.textContent = sweepState.playing ? 'Pause' : 'Play'; return; }
  if (!waveObj) return;
  waveObj.userData.playing = !waveObj.userData.playing;
  animPlay.textContent = waveObj.userData.playing ? 'Pause' : 'Play';
});
if (animScrub) animScrub.addEventListener('input', () => {
  if (sweepState) { sweepState.playing = false; if (animPlay) animPlay.textContent = 'Play'; setSweepTau((Number(animScrub.value) / 1000) * sweepState.tMax); return; }
  if (!waveObj) return;
  const u = waveObj.userData; u.playing = false;
  if (animPlay) animPlay.textContent = 'Play';
  u.r = (Number(animScrub.value) / 1000) * u.rMax; applyWave();
});
// Central dispatch — every viz mode routes through here so the Tx-placement and band
// handlers below stay a one-liner instead of repeating the same if-chain three times.
function runVizMode(mode) {
  // Vacuum's coverage view IS the invariant gate, so it replaces the plain field render.
  if (currentMode() === 'vacuum' && (mode === 'coverage' || mode === 'path_loss')) {
    runVacuumField(); return;
  }
  if (mode === 'field') { runFieldVectors(); return; }
  if (mode === 'radiation') { runRadiationPattern(); return; }
  if (mode === 'interference') { runInterference(); return; }
  if (mode === 'coverage') { runStaticField(); return; }
  if (mode === 'sweep') { runWavefrontSweep(); return; }
  if (mode === 'mechlapse') { runMechanismTimeLapse(); return; }
  if (MECH_BY_KEY[mode]) { runMechanismField(mode); return; }
  disposeField(); disposeLobes(); disposeVectors(); disposeInterference(); disposeSweep();
  setStatus('Visualize: ' + (vizMode ? vizMode.options[vizMode.selectedIndex].text : mode) +
    ' — this mechanism module is not built yet (Refraction_3D.py / Absorption_3D.py are still stubs).');
}
if (vizMode) vizMode.addEventListener('change', () => runVizMode(vizMode.value));

// ---- Propagation mode ----
// Changing the mode invalidates the loaded volume (different scene, different source), so
// drop it and re-resolve rather than rendering one mode's physics under another's label.
function refreshModeNote() {
  if (!modeNote) return;
  const info = MODE_INFO[currentMode()] || {};
  const n = volumeIndex ? volumesForMode(currentMode()).length : null;
  modeNote.textContent = (info.note || '') + (n === null ? ''
    : '  ·  ' + (n ? n + ' cached volume' + (n === 1 ? '' : 's') + '.'
                   : 'nothing cached for this mode yet.'));
}
function applyMode() {
  refreshModeNote();
  hideProbe();                       // the probed point's values belong to the old scene
  // the Tx controls are meaningless when the source is a plane wave 416 m away
  const pt = modeIsPointSource();
  for (const el of [txPlace, txType]) if (el) el.disabled = !pt;
  // a volume from another mode is a different scene and a different source model, so it
  // must not survive the switch
  window.SIM3D_VOLUME = null;
  loadVolumeIndex().then(() => {
    refreshModeNote();
    if (vizMode) runVizMode(vizMode.value);
  });
}
if (modeSel) modeSel.addEventListener('change', applyMode);
if (modeNote) applyMode();
// changing the transmitter antenna type live-updates the lobe when in radiation mode
if (txType) txType.addEventListener('change', () => {
  if (vizMode && vizMode.value === 'radiation') runRadiationPattern();
});
// changing the band re-runs anything whose result depends on frequency (the oscillating
// field, the coverage map, and the per-mechanism views, which read a band channel)
if (wfBand) wfBand.addEventListener('change', () => {
  if (!vizMode) return;
  const m = vizMode.value;
  if (m === 'field' || m === 'interference' || m === 'coverage' || MECH_BY_KEY[m]) runVizMode(m);
});

function onResize() {
  if (!renderer || viewport.offsetParent === null) return;
  const w = viewport.clientWidth || 900, h = viewport.clientHeight || 640;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
}

// Init when the user opens the Simulation tab in 3D mode (the viewport must be
// visible so the canvas gets a real size).
const simBtn = document.getElementById('tabSimBtn');
if (simBtn) simBtn.addEventListener('click', () => {
  requestAnimationFrame(() => {
    if (window.appMode && window.appMode.dim === '3d' && viewport && viewport.offsetParent !== null) { ensureInit(); onResize(); }
  });
});

window.__sim3d = {
  ensureInit, onResize,
  // debug/verification handles (mirrors the sandbox's window.__debug)
  get scene() { return scene; }, get camera() { return camera; }, get renderer() { return renderer; },
  get wallMesh() { return wallMesh; }, get wallMeshes() { return wallMeshes; },
  get field() { return fieldObj; }, get wave() { return waveObj; },
  get lobe() { return lobeGroup; }, get vectors() { return vectorState; },
  get interference() { return interferenceState; }, get sweep() { return sweepState; },
  get volume() { return window.SIM3D_VOLUME || null; }, placed,
  // mechanism-channel handles (verification: which channels a cached Tx actually carries)
  runVizMode, loadVolumeIndex, loadVolume, loadMechanism, mechList, chanAt, tauAt,
  get mechChannels() { return mechCache; },
  get volumeIndex() { return volumeIndex; },
  // cache/offload handles: which tier drew the current field, and what is warm
  currentMode, volumesForMode, matchVolume, tryLoadSurrogate, prefetchNeighbours,
  get tier() { return window.SIM3D_TIER || null; },
  get surrogate() { return { state: SURROGATE.state, note: SURROGATE.note }; },
  get warmVolumes() { return Object.keys(volumeCache); },
  get cacheBytes() { return cacheBytes(); },
  get cacheBudget() { return BROWSER_CACHE_BUDGET; },
  get indexError() { return volumeIndexError; },
  // Tier-1 display handles (verification: prediction plane, coverage classes, probe)
  get display() { return DISPLAY; }, refreshMeta, probeAt,
  get lastSolveMs() { return lastSolveMs; },
};
