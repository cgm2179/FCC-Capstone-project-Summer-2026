// fw_solve3d.js — in-browser 3-D full-wave for the 3-D Full-Wave Studio. Two engines,
// mirroring fw_solve2d.js but over the voxel volume:
//
//   * onnxTiledPredict3d — fw_unet3d.onnx under onnxruntime-web, tiled in 24³ boxes over
//     the WHOLE floor. Bounded memory per tile, so it never crashes — just slow (minutes,
//     3-D convs + many tiles). Inert until you train + export the model (this is the Colab job).
//   * fdtdSolve3d — the sub-volume 3-D FDTD (fw_fdtd3d_worker.js). Whole-floor λ/10 would
//     freeze the tab, so we solve a few-metre box around the Tx — exactly the surrogate's box.
//
// Colormaps / dB / ETA formatting are shared verbatim with the 2-D studio.
export { cmap, cmapSigned, viridis, fmtEta } from './fw_solve2d.js';

export const C0 = 299792458.0;
const BARRIER = 3;
const tick = () => new Promise(r => setTimeout(r, 0));

// ---------------------------------------------------------------- ITU material EM (P.2040-3)
// Faithful JS port of Backend/Physics Engine/2D/SIM/physics/physics_v2.py (P2040 + permittivity +
// MATERIALS7) and 2D/SIM V3/physics/fullwave2d.py:damping_by_class, so the browser FDTD uses the
// SAME material physics as the validated Python engine (real εr → wave speed / refraction; Im(εr) →
// absorption). KEEP THESE TABLES IN SYNC WITH physics_v2.py.
const EPS0 = 8.8541878128e-12;
const P2040 = {                                        // eps' = a·f_GHz^b ; sigma = c·f_GHz^d
  vacuum:       { a: 1.00, b: 0.0, c: 0.0,    d: 0.0 },
  concrete:     { a: 5.24, b: 0.0, c: 0.0462, d: 0.7822 },
  plasterboard: { a: 2.73, b: 0.0, c: 0.0085, d: 0.9395 },
  wood:         { a: 1.99, b: 0.0, c: 0.0047, d: 1.0718 },
  glass:        { a: 6.31, b: 0.0, c: 0.0036, d: 1.3394 },
};
// 6-class indoor scheme (manifest ids 0..5). barrier (class 3) is a rigid reflector: it carries NO
// bulk speed/loss (held at u=0), exactly like physics_3d.speed_field masks it + damping_by_class skips
// it. Furniture is dilute clutter: FULL wood n for speed (physics_3d.refractive_index_by_class does not
// dilute n) but absorption scaled by fill_fraction (damping_by_class does).
const MATERIALS6 = [
  { id: 0, p2040: null,           per_metre: false },
  { id: 1, p2040: 'plasterboard', per_metre: false },
  { id: 2, p2040: 'concrete',     per_metre: false },
  { id: 3, p2040: 'concrete',     per_metre: false, barrier: true },
  { id: 4, p2040: 'wood',         per_metre: true, fill_fraction: 0.021 },
  { id: 5, p2040: 'glass',        per_metre: false },
];

// Complex relative permittivity eps' - j·eps'' at f (MHz). exp(+jωt) → lossy has positive `im` here
// (the magnitude of the negative imaginary part). Mirrors physics_v2.permittivity.
export function permittivity(mat, fMHz) {
  const p = P2040[mat], fG = fMHz / 1000.0;
  const epsRe = p.a * Math.pow(fG, p.b);
  const sigma = p.c * Math.pow(fG, p.d);
  const epsIm = sigma / (2 * Math.PI * (fG * 1e9) * EPS0);
  return { re: epsRe, im: epsIm };
}

// Per-class LUTs at f, indexed by class id:
//   n     real refractive index √εr'  (air / barrier → 1)
//   gamma damping rate π·f·tanδ [1/s] (air / barrier → 0; dilute clutter × fill_fraction)
//   alphaDbM spatial amplitude attenuation in dB/m: amp ~ e^{-γt}, t = L·n/C0 ⇒ α = 8.686·γ·n/C0
export function materialLuts(fMHz) {
  const N = MATERIALS6.length;
  const n = new Float64Array(N), gamma = new Float64Array(N), alphaDbM = new Float64Array(N);
  for (const m of MATERIALS6) {
    if (!m.p2040 || m.p2040 === 'vacuum' || m.barrier) { n[m.id] = 1; continue; }  // gamma/alpha stay 0
    const eps = permittivity(m.p2040, fMHz);
    const nRe = Math.sqrt((Math.hypot(eps.re, eps.im) + eps.re) / 2);     // Re(√εr) — matches physics_3d
    let g = Math.PI * (fMHz * 1e6) * (eps.im / Math.max(eps.re, 1e-6));   // π f tanδ
    if (m.per_metre && m.fill_fraction) g *= m.fill_fraction;
    n[m.id] = nRe; gamma[m.id] = g; alphaDbM[m.id] = 8.6858896 * g * nRe / C0;
  }
  return { n, gamma, alphaDbM };
}
// Per-class spatial attenuation (dB per metre) — the FDTD-consistent per-material loss used by the
// studio's CSV breakdown (NOT the analytic Motley-Keenan per-crossing model).
export const materialAlphaDbM = (fMHz) => materialLuts(fMHz).alphaDbM;

// ---------------------------------------------------------------- geometry helpers
export const wavelength = fMHz => C0 / (fMHz * 1e6);
export const cellFine = (fMHz, npw) => wavelength(fMHz) / npw;      // λ/npw target FDTD cell

// choose the coarse→fine upsample so the fine sub-volume stays under a cell budget
export function planFdtd3d(cellCoarse, fMHz, npw, halfCells, dimsCoarse, maxCells = 3.2e6) {
  const [X, Y, Z] = dimsCoarse;
  let up = Math.max(1, Math.round(cellCoarse / cellFine(fMHz, npw)));
  let hc = halfCells;
  const fineCells = (h, u) => (Math.min(2 * h + 1, X) * u) * (Y * u) * (Math.min(2 * h + 1, Z) * u);
  while (fineCells(hc, up) > maxCells && up > 1) up--;             // first coarsen the wave cell
  while (fineCells(hc, up) > maxCells && hc > 6) hc -= 2;          // then shrink the region
  return { up, halfCells: hc };
}

// extract a cube sub-volume around tx (full height Y), returns the coarse block + local tx + origin
export function extractSubvolume(grid, X, Y, Z, tx, halfCells) {
  const x0 = Math.max(0, tx.x - halfCells), x1 = Math.min(X, tx.x + halfCells + 1);
  const z0 = Math.max(0, tx.z - halfCells), z1 = Math.min(Z, tx.z + halfCells + 1);
  const sx = x1 - x0, sy = Y, sz = z1 - z0;
  const sub = new Uint8Array(sx * sy * sz);
  for (let x = 0; x < sx; x++) for (let y = 0; y < sy; y++) {
    const gBase = ((x0 + x) * Y + y) * Z, dBase = (x * sy + y) * sz;
    for (let z = 0; z < sz; z++) sub[dBase + z] = grid[gBase + (z0 + z)];
  }
  return { sub, dims: [sx, sy, sz], origin: [x0, 0, z0], txLocal: { x: tx.x - x0, y: tx.y, z: tx.z - z0 } };
}

// ---------------------------------------------------------------- FDTD (worker)
export function estimateFdtd3d(dimsCoarse, up, crossings = 1.4, safety = 0.4) {
  const [sx, sy, sz] = dimsCoarse, Xf = sx * up, Yf = sy * up, Zf = sz * up;
  const cells = Xf * Yf * Zf;
  const steps = Math.max(1, Math.round(crossings * Math.hypot(Xf, Yf, Zf) * Math.SQRT2 * 1.2247 / safety));
  return { cells, steps };
}
export function estimateFdtdSeconds3d(dimsCoarse, up, mcps) {
  const { cells, steps } = estimateFdtd3d(dimsCoarse, up);
  return (cells * steps) / (Math.max(mcps, 1) * 1e6);
}
// JS 3-D Mcell-updates/s on THIS device (a small stencil bench)
export function calibrateFdtd3dMcps() {
  const S = 40, N = S * S * S, YZ = S * S;
  const u = new Float32Array(N), up_ = new Float32Array(N), un = new Float32Array(N);
  for (let i = 0; i < N; i++) u[i] = Math.random() - 0.5;
  const iters = 4, t0 = performance.now();
  for (let it = 0; it < iters; it++) {
    for (let x = 1; x < S - 1; x++) for (let y = 1; y < S - 1; y++) {
      const b = x * YZ + y * S;
      for (let z = 1; z < S - 1; z++) {
        const i = b + z;
        un[i] = 2 * u[i] - up_[i] + 0.16 * (u[i - YZ] + u[i + YZ] + u[i - S] + u[i + S] + u[i - 1] + u[i + 1] - 6 * u[i]);
      }
    }
    up_.set(u); u.set(un);
  }
  return (N * iters) / ((performance.now() - t0) / 1000) / 1e6;
}

// run the sub-volume FDTD off-thread; resolves { re, im, dims:[sx,sy,sz], steps }
export function fdtdSolve3d(args, onProg) {
  // Material-aware (default): per-class LUTs (real εr → speed, Im(εr) → absorption) so the worker runs
  // the SAME damped, variable-speed leapfrog as the Python full-wave engine. Airy (materialAware:false):
  // omit the LUTs → the worker falls back to the lossless air-speed stand-in (free-space/FSPL; only the
  // class-3 core reflects).
  const msg = { ...args };
  if (args.materialAware !== false) { const { n, gamma } = materialLuts(args.fMHz); msg.nByClass = n; msg.gammaByClass = gamma; }
  return new Promise((resolve, reject) => {
    let wk;
    try { wk = new Worker(new URL('./fw_fdtd3d_worker.js', import.meta.url), { type: 'module' }); }
    catch (e) { wk = new Worker(new URL('./fw_fdtd3d_worker.js', import.meta.url)); }
    wk.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { if (onProg) onProg(m.f); }
      else if (m.type === 'done') { wk.terminate(); resolve(m); }
    };
    wk.onerror = (e) => { wk.terminate(); reject(new Error(e.message || 'FDTD worker error')); };
    wk.postMessage(msg);
  });
}

// ---------------------------------------------------------------- envelope dB (3-D)
// |U| → dB relative to the 99.5th-pct amplitude over OPEN cells (air=0 / furniture=4).
export function envelopeDb3d(re, im, grid) {
  const N = re.length, amp = new Float32Array(N), open = [];
  for (let i = 0; i < N; i++) {
    const a = Math.hypot(re[i], im[i]); amp[i] = a;
    const c = grid[i]; if ((c === 0 || c === 4) && a > 0) open.push(a);
  }
  open.sort((a, b) => a - b);
  const ref = open[Math.floor(open.length * 0.995)] || 1e-9;
  const db = new Float32Array(N);
  for (let i = 0; i < N; i++) db[i] = 20 * Math.log10(Math.max(amp[i], ref * 1e-6) / ref);
  return { db, ref };
}

// ---------------------------------------------------------------- ONNX (3-D, tiled)
// fw_unet3d.onnx — 18-ch V4 input (6 material one-hot + tx-blob + freq + log-dist + εr + σ
// + SDF + 3 normals + antenna |Jx|,|Jy|,|Jz|) → 2-ch phase-reduced complex field. Whole floor
// is tiled in 24³ boxes (X,Z multiples of 8; Y is the thin, un-pooled axis) and stitched.
export async function loadContract3d(url) {
  const j = await (await fetch(url, { cache: 'no-store' })).json();
  const inp = j.input || {}, out = j.output || {};
  return {
    sigma: inp.tx_blob_sigma_cells || 2.0, logdiv: inp.logdist_divisor || 3.0,
    epsrNorm: inp.epsr_norm || 6.0, sigmaLog: inp.sigma_log || 7.0, sdfNorm: inp.sdf_norm || 0.5,
    flo: inp.freq_log_lo_mhz || 600, fhi: inp.freq_log_hi_mhz || 6000,
    box: j.box_train || 24, stride: j.stride || 18, sizeMultiple: inp.size_multiple || 8,
    inChannels: j.in_channels || (inp.in_channels || 18),
    bands: j.bands || {}, reconstruct: out.reconstruct,
  };
}

// per-band ff + per-class (εr,σ). Falls back to a log-freq interpolation if the band is absent.
function bandParams(contract, fMHz) {
  const key = String(Math.round(fMHz));
  if (contract.bands[key]) return contract.bands[key];
  const lo = Math.log10(contract.flo), hi = Math.log10(contract.fhi);
  const ff = Math.max(0, Math.min(1, (Math.log10(fMHz) - lo) / (hi - lo)));
  const any = Object.values(contract.bands)[0];
  return { ff, class_eps: (any && any.class_eps) || [[1, 0], [2.7, .05], [5.2, .3], [1, 1e7], [2, .02], [6.3, .05]] };
}

// ---- Felzenszwalb & Huttenlocher separable squared-EDT (distance to nearest seed, in cells²) ----
function edt1d(f, n, d, v, z) {              // 1-D lower-envelope of parabolas; f in→out (len n)
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const dx = q - v[k]; d[q] = dx * dx + f[v[k]]; }
  for (let q = 0; q < n; q++) f[q] = d[q];
}
// sq Euclidean distance to nearest cell where seed(c)==true, over the (X,Y,Z) volume
function sqEDT(grid, X, Y, Z, seed) {
  const INF = 1e12, N = X * Y * Z, D = new Float64Array(N);
  const id = (x, y, z) => (x * Y + y) * Z + z;
  for (let i = 0; i < N; i++) D[i] = seed(grid[i]) ? 0 : INF;
  const mx = Math.max(X, Y, Z);
  const line = new Float64Array(mx), dd = new Float64Array(mx), v = new Int32Array(mx), z = new Float64Array(mx + 1);
  for (let y = 0; y < Y; y++) for (let z0 = 0; z0 < Z; z0++) {            // along X
    for (let x = 0; x < X; x++) line[x] = D[id(x, y, z0)];
    edt1d(line, X, dd, v, z);
    for (let x = 0; x < X; x++) D[id(x, y, z0)] = line[x];
  }
  for (let x = 0; x < X; x++) for (let z0 = 0; z0 < Z; z0++) {            // along Y
    for (let y = 0; y < Y; y++) line[y] = D[id(x, y, z0)];
    edt1d(line, Y, dd, v, z);
    for (let y = 0; y < Y; y++) D[id(x, y, z0)] = line[y];
  }
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) {              // along Z
    for (let z0 = 0; z0 < Z; z0++) line[z0] = D[id(x, y, z0)];
    edt1d(line, Z, dd, v, z);
    for (let z0 = 0; z0 < Z; z0++) D[id(x, y, z0)] = line[z0];
  }
  return D;
}
// signed distance (metres, +air/−solid) + unit air-ward normals — approximates the conformal
// SDF/normal channels from the binary class grid. Cached per grid (band-independent).
const _sdfCache = new WeakMap();
function sdfNormals(grid, X, Y, Z, cell) {
  const hit = _sdfCache.get(grid); if (hit) return hit;
  const dSolid = sqEDT(grid, X, Y, Z, c => c !== 0);   // in air: dist to nearest solid
  const dAir = sqEDT(grid, X, Y, Z, c => c === 0);      // in solid: dist to nearest air
  const N = X * Y * Z, sdf = new Float32Array(N);
  for (let i = 0; i < N; i++) sdf[i] = (grid[i] === 0 ? Math.sqrt(dSolid[i]) : -Math.sqrt(dAir[i])) * cell;
  const nx = new Float32Array(N), ny = new Float32Array(N), nz = new Float32Array(N);
  const id = (x, y, z) => (x * Y + y) * Z + z;
  for (let x = 1; x < X - 1; x++) for (let y = 1; y < Y - 1; y++) for (let z = 1; z < Z - 1; z++) {
    const i = id(x, y, z);
    let gx = sdf[id(x + 1, y, z)] - sdf[id(x - 1, y, z)];
    let gy = sdf[id(x, y + 1, z)] - sdf[id(x, y - 1, z)];
    let gz = sdf[id(x, y, z + 1)] - sdf[id(x, y, z - 1)];
    const m = Math.hypot(gx, gy, gz);
    if (m > 1e-6) { nx[i] = gx / m; ny[i] = gy / m; nz[i] = gz / m; }
  }
  const out = { sdf, nx, ny, nz }; _sdfCache.set(grid, out); return out;
}

function axStarts(D, box, stride) {
  const s = []; for (let x = 0; x < Math.max(1, D - box + 1); x += stride) s.push(x);
  const last = Math.max(0, D - box); if (s[s.length - 1] !== last) s.push(last); return s;
}

// Whole-floor (or near-region) complex U via the trained net, overlap-tiled.
// `opts.regionCells` restricts tiling to an XZ box of that half-width around the Tx (the near
// region where the sub-volume-trained net is valid) — the rest of `wsum` stays 0.
export async function onnxTiledPredict3d(session, grid, dims, cell, fMHz, tx, contract, onProg, opts = {}) {
  const [X, Y, Z] = dims, N = X * Y * Z, id = (x, y, z) => (x * Y + y) * Z + z;
  const { ff, class_eps } = bandParams(contract, fMHz);
  const sig2 = 2.0 * contract.sigma * contract.sigma, INV3 = 1 / Math.sqrt(3.0);
  const k = 2 * Math.PI / wavelength(fMHz);
  const { sdf, nx, ny, nz } = sdfNormals(grid, X, Y, Z, cell);
  const box = contract.box, stride = contract.stride;
  let xs = axStarts(X, box, stride); const ys = axStarts(Y, box, stride); let zs = axStarts(Z, box, stride);
  if (opts.regionCells != null) {                            // near-region only (hybrid): keep tiles overlapping tx±Rc
    const Rc = opts.regionCells, bxx = Math.min(box, X), bzz = Math.min(box, Z);
    xs = xs.filter(x0 => x0 <= tx.x + Rc && x0 + bxx >= tx.x - Rc);
    zs = zs.filter(z0 => z0 <= tx.z + Rc && z0 + bzz >= tx.z - Rc);
  }
  const nTiles = xs.length * ys.length * zs.length;
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const accRe = new Float32Array(N), accIm = new Float32Array(N), wsum = new Float32Array(N);
  let done = 0;
  for (const x0 of xs) for (const y0 of ys) for (const z0 of zs) {
    const bx = Math.min(box, X), by = Math.min(box, Y), bz = Math.min(box, Z), nb = bx * by * bz;
    const buf = new Float32Array(18 * nb);
    for (let i = 0; i < bx; i++) for (let j = 0; j < by; j++) for (let kk = 0; kk < bz; kk++) {
      const gx = x0 + i, gy = y0 + j, gz = z0 + kk, gi = id(gx, gy, gz);
      const off = (i * by + j) * bz + kk, C = c => c * nb + off;
      const cls = grid[gi]; if (cls < 6) buf[C(cls)] = 1;                       // 0..5 one-hot
      const dx = gx - tx.x, dy = gy - tx.y, dz = gz - tx.z, r2 = dx * dx + dy * dy + dz * dz;
      const blob = Math.exp(-r2 / sig2), dm = Math.sqrt(r2) * cell;
      const ce = class_eps[cls] || [1, 0];
      buf[C(6)] = blob;                                                          // tx blob
      buf[C(7)] = ff;                                                            // freq feature
      buf[C(8)] = Math.log10(Math.max(dm, 1.0)) / contract.logdiv;              // log distance
      buf[C(9)] = Math.max(0, Math.min(1.5, (ce[0] - 1.0) / contract.epsrNorm)); // εr eff
      buf[C(10)] = Math.max(0, Math.min(1.5, Math.log10(1.0 + Math.max(0, ce[1])) / contract.sigmaLog)); // σ eff
      buf[C(11)] = Math.max(-1, Math.min(1, sdf[gi] / contract.sdfNorm));       // SDF
      buf[C(12)] = nx[gi]; buf[C(13)] = ny[gi]; buf[C(14)] = nz[gi];            // normals
      buf[C(15)] = INV3 * blob; buf[C(16)] = INV3 * blob; buf[C(17)] = INV3 * blob; // omni |J|
    }
    const y = await session.run({ [inName]: new ort.Tensor('float32', buf, [1, 18, bx, by, bz]) });
    const pr = y[outName].data, np = bx * by * bz;
    for (let i = 0; i < bx; i++) for (let j = 0; j < by; j++) for (let kk = 0; kk < bz; kk++) {
      const gi = id(x0 + i, y0 + j, z0 + kk), off = (i * by + j) * bz + kk;
      accRe[gi] += pr[off]; accIm[gi] += pr[np + off]; wsum[gi] += 1;
    }
    if (onProg) onProg(++done / nTiles);
    await tick();
  }
  const re = new Float32Array(N), im = new Float32Array(N);                     // stitch + undo phase reduction
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) for (let z = 0; z < Z; z++) {
    const i = id(x, y, z), w = wsum[i] || 1e-9, rr = accRe[i] / w, ii = accIm[i] / w;
    const dg = Math.hypot(x - tx.x, y - tx.y, z - tx.z) * cell, th = k * dg, c = Math.cos(th), s = Math.sin(th);
    re[i] = rr * c + ii * s; im[i] = ii * c - rr * s;                           // U = Ũ·e^{-jkd}
  }
  return { re, im, wsum, dims: [X, Y, Z], tiles: nTiles };
}

// ---------------------------------------------------------------- eikonal far field (ray-marched)
// Per-material transmission loss (dB per 0.3-m solid voxel a ray crosses). class:
// 0 air · 1 drywall · 2 concrete/floor · 3 metal/core · 4 wood/furniture · 5 glass
const WALL_DB = [0, 2.5, 7.0, 30.0, 3.0, 3.0];
// Absolute-RSRP coverage over open cells: RSRP = Ptx + Gtx − FSPL(1m) − 10·n·log10(d) − Σ wall(ray).
// A 3-D DDA (Amanatides–Woo) marches Tx→voxel through the grid, summing wall loss (indoor shadows).
// opts.wallDb: optional per-class dB table (outdoor studio uses heavier concrete-core loss).
export function eikonalRsrp3d(grid, dims, cell, fMHz, tx, txDbm, opts = {}) {
  const [X, Y, Z] = dims, N = X * Y * Z, id = (x, y, z) => (x * Y + y) * Z + z;
  const n = opts.pathExp || 2.2, Gtx = opts.gainDbi ?? 2.0, capWall = opts.wallCap || 80.0;
  const fscale = Math.max(1.0, Math.pow(fMHz / 2000.0, 0.25));                  // walls lose a bit more at higher f
  const baseWall = opts.wallDb || WALL_DB;
  const wdb = baseWall.map(v => v * fscale);
  const fspl1 = 20 * Math.log10(fMHz) - 27.55;                                  // free-space PL at 1 m (dB)
  const db = new Float32Array(N).fill(-Infinity);
  const tcx = tx.x + 0.5, tcy = tx.y + 0.5, tcz = tx.z + 0.5;
  const yLo = opts.yLo != null ? opts.yLo : 0, yHi = opts.yHi != null ? opts.yHi : Y;  // outdoor: street band
  const xLo = opts.xLo != null ? opts.xLo : 0, xHi = opts.xHi != null ? opts.xHi : X;
  const zLo = opts.zLo != null ? opts.zLo : 0, zHi = opts.zHi != null ? opts.zHi : Z;
  // Optional panel pattern: boresightDeg from N (0=N,90=E), halfPowerDeg beamwidth → dB gain vs azimuth.
  const bore = opts.boresightDeg, hp = opts.halfPowerDeg || 65;
  const hasPanel = Number.isFinite(bore);
  for (let X0 = xLo; X0 < xHi; X0++) for (let Y0 = yLo; Y0 < yHi; Y0++) for (let Z0 = zLo; Z0 < zHi; Z0++) {
    const i = id(X0, Y0, Z0), c = grid[i]; if (!(c === 0 || c === 4)) continue;  // open cells only
    const dm = Math.hypot(X0 - tx.x, Y0 - tx.y, Z0 - tx.z) * cell;
    // --- march Tx→voxel, accumulate wall loss for solid voxels strictly between ---
    let wall = 0;
    let px = tcx, py = tcy, pz = tcz;
    let dx = (X0 + 0.5) - px, dy = (Y0 + 0.5) - py, dz = (Z0 + 0.5) - pz;
    const L = Math.hypot(dx, dy, dz) || 1e-6; dx /= L; dy /= L; dz /= L;
    let vx = tx.x, vy = tx.y, vz = tx.z;
    const sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
    const tDX = dx !== 0 ? Math.abs(1 / dx) : 1e30, tDY = dy !== 0 ? Math.abs(1 / dy) : 1e30, tDZ = dz !== 0 ? Math.abs(1 / dz) : 1e30;
    let tMX = dx !== 0 ? (((sx > 0 ? vx + 1 : vx) - px) / dx) : 1e30;
    let tMY = dy !== 0 ? (((sy > 0 ? vy + 1 : vy) - py) / dy) : 1e30;
    let tMZ = dz !== 0 ? (((sz > 0 ? vz + 1 : vz) - pz) / dz) : 1e30;
    let guard = 0;
    while (guard++ < 4 * (X + Y + Z)) {
      if (tMX <= tMY && tMX <= tMZ) { vx += sx; tMX += tDX; }
      else if (tMY <= tMZ) { vy += sy; tMY += tDY; }
      else { vz += sz; tMZ += tDZ; }
      if (vx === X0 && vy === Y0 && vz === Z0) break;
      if (vx < 0 || vy < 0 || vz < 0 || vx >= X || vy >= Y || vz >= Z) break;
      const cc = grid[id(vx, vy, vz)];
      if (cc !== 0 && cc !== 4) { wall += wdb[cc] || wdb[1]; if (wall > capWall) { wall = capWall; break; } }
    }
    const pl = fspl1 + 10 * n * Math.log10(Math.max(dm, 1.0));
    let gAnt = Gtx;
    if (hasPanel) {
      // Grid: +X ≈ west (render), +Z ≈ north. Azimuth from N, clockwise.
      const az = Math.atan2(-(X0 - tx.x), (Z0 - tx.z)) * 180 / Math.PI; // 0=N, + = E
      let dAz = ((az - bore + 540) % 360) - 180;
      const u = Math.max(0, Math.cos(dAz * Math.PI / 180));
      const nBeam = Math.log(0.5) / Math.log(Math.max(1e-6, Math.cos(hp * Math.PI / 360)));
      gAnt = Gtx + 10 * Math.log10(Math.max(1e-4, Math.pow(u, nBeam)));
    }
    db[i] = txDbm + gAnt - pl - Math.min(wall, capWall);
  }
  return db;
}

// ---------------------------------------------------------------- hybrid: ONNX (near) ⊕ eikonal (far)
// Blend radii (metres from the Tx): ≤ R1 = pure ONNX surrogate (near field, accurate); ≥ R2 = pure
// eikonal path-loss (far field, approximate); R1..R2 = linear dB blend. Exported so the studio's CSV
// method label + near-field warning stay in lock-step with the solver.
export const HYBRID_R1_M = 3.0, HYBRID_R2_M = 6.5;

// Absolute-RSRP whole-floor map: the sub-volume-trained net supplies the near-Tx structure; the
// eikonal supplies path loss + wall shadows everywhere else; blended in dB across a transition ring.
export async function hybridField3d(session, grid, dims, cell, fMHz, tx, txDbm, contract, onProg) {
  const [X, Y, Z] = dims, N = X * Y * Z, id = (x, y, z) => (x * Y + y) * Z + z;
  const R1 = HYBRID_R1_M, R2 = HYBRID_R2_M;                                     // metres: pure-ONNX / pure-eikonal
  const eik = eikonalRsrp3d(grid, dims, cell, fMHz, tx, txDbm);
  if (onProg) onProg(0.25);
  const Rc = Math.ceil(R2 / cell) + contract.box;                              // tile only the near box
  const near = await onnxTiledPredict3d(session, grid, dims, cell, fMHz, tx, contract,
    p => { if (onProg) onProg(0.25 + 0.6 * p); }, { regionCells: Rc });
  // ONNX |U| → relative dB; calibrate an offset so it meets the eikonal at the R1..R2 ring
  const onDb = new Float32Array(N).fill(-Infinity);
  let sOff = 0, nOff = 0;
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) for (let z = 0; z < Z; z++) {
    const i = id(x, y, z), c = grid[i]; if (!(c === 0 || c === 4)) continue;
    if (near.wsum[i] <= 0) continue;
    const a = Math.hypot(near.re[i], near.im[i]); if (a <= 0) continue;
    onDb[i] = 20 * Math.log10(a);
    const dm = Math.hypot(x - tx.x, y - tx.y, z - tx.z) * cell;
    if (dm >= R1 && dm <= R2 && isFinite(eik[i])) { sOff += eik[i] - onDb[i]; nOff++; }
  }
  const off = nOff ? sOff / nOff : 0;
  // blend
  const db = new Float32Array(N).fill(-Infinity), re = new Float32Array(N), im = new Float32Array(N);
  const k = 2 * Math.PI / wavelength(fMHz);
  let dbMax = -Infinity;
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) for (let z = 0; z < Z; z++) {
    const i = id(x, y, z), c = grid[i]; if (!(c === 0 || c === 4)) continue;
    const dm = Math.hypot(x - tx.x, y - tx.y, z - tx.z) * cell;
    const oa = isFinite(onDb[i]) ? onDb[i] + off : -Infinity, ea = eik[i];
    let val;
    if (dm <= R1 && isFinite(oa)) val = oa;
    else if (dm >= R2 || !isFinite(oa)) val = ea;
    else { const w = (R2 - dm) / (R2 - R1); val = w * oa + (1 - w) * ea; }      // linear dB blend
    db[i] = val; if (val > dbMax) dbMax = val;
  }
  for (let i = 0; i < N; i++) {                                                 // synth complex for the wave view
    if (!isFinite(db[i])) continue;
    const amp = Math.pow(10, (db[i] - dbMax) / 20);
    const x = (i / (Y * Z)) | 0, y = ((i / Z) | 0) % Y, z = i % Z;
    const th = k * Math.hypot(x - tx.x, y - tx.y, z - tx.z) * cell;
    re[i] = amp * Math.cos(th); im[i] = -amp * Math.sin(th);
  }
  if (onProg) onProg(1);
  return { re, im, db, dims: [X, Y, Z], tiles: near.tiles, dbMax, mode: 'hybrid' };
}
