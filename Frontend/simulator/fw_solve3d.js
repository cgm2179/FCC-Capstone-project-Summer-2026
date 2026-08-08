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
    wk.postMessage(args);
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
// Ready for fw_unet3d.onnx once trained. Contract mirrors fw_unet2d.json but 3-D (box³).
export async function loadContract3d(url) {
  const j = await (await fetch(url, { cache: 'no-store' })).json();
  const inp = j.input || {}, out = j.output || {};
  return {
    sigma: inp.tx_blob_sigma_cells || 2.0, logdiv: inp.logdist_divisor || 3.0,
    flo: inp.freq_log_lo_mhz || 600, fhi: inp.freq_log_hi_mhz || 6200,
    box: j.box_train || 24, sizeMultiple: inp.size_multiple || 8,
    inChannels: (j.in_channels && j.in_channels.length) || 18,
    trainedMhz: j.trained_mhz || j.bands_mhz || [], reconstruct: out.reconstruct,
  };
}
// (Full-volume 24³ tiled predict — implemented when the model + its exact channel featurization
// land from the Colab export. Kept as a clear stub so the ONNX toggle wires up now.)
export async function onnxTiledPredict3d() {
  throw new Error('fw_unet3d.onnx not available yet — train it in Colab, export, and drop it in SIM V1 3D/web/');
}
