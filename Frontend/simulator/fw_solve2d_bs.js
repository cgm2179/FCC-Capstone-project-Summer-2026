// fw_solve2d_bs.js — in-browser OUTDOOR 2-D full-wave solvers for the Outdoor
// Full-Wave Studio (SIM V3 base-station surrogate). Sibling of fw_solve2d.js
// (indoor), sharing its render + estimate helpers, but for the 10-channel
// base-station model (fw_bs.onnx) trained on the penetrable OSM city:
//
//   * a REGION is extracted around the picked base station and resampled to the
//     surrogate's fine grid (cell = lambda / npw), mirroring fw_bs_catalog.bs_region
//     (the coarse 2 m city raster is what the browser ships; the model needs the
//     fine grid). npw adapts to the region so the cell count stays in budget.
//   * onnxTiledPredictBs — fw_bs.onnx, tiled, with the 10th (directivity) channel
//     synthesized from the Tx boresight + antenna family (JS port of
//     antenna_patterns.pattern_gain / fw_bs_catalog.featurize_bs).
//   * fdtdSolvePenetrable — a damped-leapfrog FDTD whose per-cell speed (c/n) and
//     damping (gamma = gPerF * f) come from the baked material palette, so the
//     fallback matches the PENETRABLE physics the surrogate trained on (concrete /
//     glass / foliage attenuate, not just reflect) — mirror of fullwave2d.FullWaveScene.
//
// Both return { re, im } on the SAME fine region grid, so the page renders them
// identically (envelope |U| in dB, or the animated wavefront Re{U e^jwt}).

import {
  C0, cmap, cmapSigned, viridis, turbo, rdbu, envelopeDb, fmtEta,
  numTiles, calibrateFdtdMcps,
} from './fw_solve2d.js';

// re-export the shared render/estimate helpers so the studio imports one module
export { C0, cmap, cmapSigned, viridis, turbo, rdbu, envelopeDb, fmtEta, numTiles, calibrateFdtdMcps };

const tick = () => new Promise(r => setTimeout(r, 0));
const clamp05 = c => (c < 0 ? 0 : c > 5 ? 5 : c);
const nextMult = (n, m) => Math.ceil(n / m) * m;
const hann = (i, box) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / (box - 1)));
function tileStarts(n, box, stride) {
  if (n <= box) return [0];
  const xs = [];
  for (let x = 0; x + box <= n; x += stride) xs.push(x);
  xs.push(n - box);
  return [...new Set(xs)].sort((a, b) => a - b);
}

// ------------------------------------------------------------------ contract
// Read fw_bs.json — the 10-channel base-station contract, including the
// directivity-channel spec (panel HPBW / Am) the browser must synthesize.
export async function loadContractBs(url) {
  const j = await (await fetch(url, { cache: 'no-store' })).json();
  const inp = j.input, out = j.output, dir = inp.directivity || {};
  return {
    sigma: inp.tx_blob_sigma_cells,          // 2.0 cells
    logdiv: inp.logdist_divisor,             // 3.0
    flo: inp.freq_log_lo_mhz,                // 600
    fhi: inp.freq_log_hi_mhz,                // 6200
    box: j.box_train || 128,
    sizeMultiple: inp.size_multiple || 16,
    cin: inp.in_channels || 10,
    bands: j.bands_mhz || [],
    trainedMhz: j.trained_mhz || [],
    hpbw: dir.hpbw_deg || 65, am: dir.am_db || 25,     // panel default
    families: dir.families || {},
    reconstruct: out.reconstruct,
  };
}

const freqFeat = (fMHz, C) =>
  (Math.log10(fMHz) - Math.log10(C.flo)) / (Math.log10(C.fhi) - Math.log10(C.flo));

// per-family azimuth pattern (JS port of antenna_patterns.params/pattern_gain)
export function familyParams(C, kind) {
  const f = (C.families || {})[kind];
  return f ? [f.hpbw_deg, f.am_db] : [C.hpbw || 65, C.am || 25];
}
export function patternGain(hpbw, am, bearingDeg, boresightDeg) {
  let d = (((bearingDeg - boresightDeg + 180) % 360) + 360) % 360 - 180;
  const at = Math.min(12 * (d / hpbw) * (d / hpbw), am);
  return Math.pow(10, -at / 20);
}

// ------------------------------------------------------------------ region extraction
// Band-adaptive coverage extent (metres) — mirror of fw_bs_catalog.band_region_m.
export function bandRegionM(fMHz) {
  return fMHz < 900 ? 400 : fMHz < 2000 ? 250 : fMHz < 3000 ? 180 : 140;
}
// npw so (region/h)^2 ~ budget, clamped to the surrogate's trained range.
export function chooseNpw(regionM, wavelengthM, budget) {
  const npw = (Math.sqrt(budget) * wavelengthM) / regionM;
  return Math.max(1.5, Math.min(8, npw));
}

// Snap the source to air and, for DIRECTIONAL families, add a short rigid backplane
// behind it (opposite boresight) — mirror of fw_bs_catalog._place_source.
function placeSource(cl, H, W, txi, txj, boresightDeg, directional) {
  let ix = Math.max(2, Math.min(H - 3, Math.round(txi)));
  let iz = Math.max(2, Math.min(W - 3, Math.round(txj)));
  if (cl[ix * W + iz] !== 0) {
    let best = null, bd = 1e9;
    for (let r = 1; r < 90 && !best; r++)
      for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) {
        const a = ix + di, b = iz + dj;
        if (a < 0 || b < 0 || a >= H || b >= W) continue;
        if (cl[a * W + b] === 0) { const d = Math.abs(di) + Math.abs(dj); if (d < bd) { bd = d; best = [a, b]; } }
      }
    if (best) { ix = best[0]; iz = best[1]; }
  }
  if (directional) {
    const th = (boresightDeg * Math.PI) / 180, dX = -Math.sin(th), dZ = Math.cos(th);
    const bx0 = ix - Math.round(dX * 2), bz0 = iz - Math.round(dZ * 2), px = -dZ, pz = dX;
    for (let t = -6; t <= 6; t++) {
      const bx = Math.round(bx0 + px * t), bz = Math.round(bz0 + pz * t);
      if (bx >= 0 && bz >= 0 && bx < H && bz < W && (Math.abs(bx - ix) > 1 || Math.abs(bz - iz) > 1)) cl[bx * W + bz] = 3;
    }
  }
  return { i: ix, j: iz };
}

// Crop the coarse city raster around a station and nearest-upsample to the fine
// surrogate grid (cell = lambda/npw). `city` = window.NOMA_CITY_2D decoded (grid is
// Uint8, C-order (nx,nz) → grid[x*nz + z]). Returns the fine region + geo refs.
export function buildRegion(city, st, fMHz, opts = {}) {
  const cs = city.cell_m, nx = city.nx, nz = city.nz, grid = city.grid;
  const wl = C0 / (fMHz * 1e6);
  const regionM = opts.regionM || bandRegionM(fMHz);
  const budget = opts.budget || 3_000_000;
  const npw = opts.npw || chooseNpw(regionM, wl, budget);
  const h = wl / npw, zoom = cs / h;
  const halfC = Math.max(6, Math.round(regionM / cs / 2));
  const x0 = Math.max(0, Math.round(st.vx - halfC)), x1 = Math.min(nx, Math.round(st.vx + halfC));
  const z0 = Math.max(0, Math.round(st.vz - halfC)), z1 = Math.min(nz, Math.round(st.vz + halfC));
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, z1 - z0);
  const H = Math.max(16, Math.round(cw * zoom)), W = Math.max(16, Math.round(ch * zoom));
  const classes = new Uint8Array(H * W);
  for (let i = 0; i < H; i++) {
    const ci = Math.min(cw - 1, Math.floor(i / zoom)), gx = (x0 + ci) * nz;
    for (let j = 0; j < W; j++) {
      const cj = Math.min(ch - 1, Math.floor(j / zoom));
      classes[i * W + j] = grid[gx + (z0 + cj)];
    }
  }
  const tx = placeSource(classes, H, W, (st.vx - x0) * zoom, (st.vz - z0) * zoom,
                         st.boresight, opts.directional !== false);
  return { classes, H, W, h_m: h, cell_m: cs, tx, npw, zoom, x0, z0,
           extentM: [H * h, W * h], regionM, fMHz };
}

// ------------------------------------------------------------------ ONNX path
// (10,p0,p1) NCHW tile input: the 9 dataset_3d channels (fw_dataset._featurize) +
// a per-family directivity channel (fw_bs_catalog.featurize_bs, channel 9).
function buildTileInputBs(grid, H, W, x0, y0, b0, b1, p0, p1, tx, cell, ff, sig2, logdiv, boresight, hpbw, am) {
  const nT = p0 * p1, xin = new Float32Array(10 * nT);
  const blobOff = 6 * nT, ffOff = 7 * nT, logOff = 8 * nT, dirOff = 9 * nT, invLog = 1 / logdiv;
  const txli = tx.i - x0, txlj = tx.j - y0;
  for (let i = 0; i < p0; i++) {
    const rowBase = i * p1, inRow = i < b0, giW = (x0 + i) * W;
    const li = i - txli, li2 = li * li, dgi = (x0 + i) - tx.i, dgi2 = dgi * dgi;
    for (let j = 0; j < p1; j++) {
      const idx = rowBase + j;
      const cls = (inRow && j < b1) ? clamp05(grid[giW + (y0 + j)]) : 0;
      xin[cls * nT + idx] = 1;
      const lj = j - txlj;
      xin[blobOff + idx] = Math.exp(-(li2 + lj * lj) / sig2);
      xin[ffOff + idx] = ff;
      const dgj = (y0 + j) - tx.j, dm = Math.sqrt(dgi2 + dgj * dgj) * cell;
      xin[logOff + idx] = Math.log10(dm < 1 ? 1 : dm) * invLog;
      const bear = Math.atan2(-dgi, dgj) * 180 / Math.PI;         // to-cell bearing (deg)
      xin[dirOff + idx] = patternGain(hpbw, am, bear, boresight);
    }
  }
  return { xin, nT };
}

// Predict U over the fine region by Hann-feathered tiling; blend the phase-reduced
// field then reapply the global free-space ramp (mirror of onnxTiledPredict).
export async function onnxTiledPredictBs(session, region, C, st, onProg) {
  const grid = region.classes, H = region.H, W = region.W, cell = region.h_m, tx = region.tx;
  const fMHz = region.fMHz, [hpbw, am] = familyParams(C, st.kind || 'panel');
  const box = C.box, stride = Math.round(box * 0.75), sm = C.sizeMultiple;
  const ff = freqFeat(fMHz, C), sig2 = 2 * C.sigma * C.sigma;
  const k = (2 * Math.PI) / (C0 / (fMHz * 1e6));
  const xs = tileStarts(H, box, stride), ys = tileStarts(W, box, stride);
  const total = xs.length * ys.length;
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const accRe = new Float64Array(H * W), accIm = new Float64Array(H * W), wsum = new Float64Array(H * W);
  let done = 0;
  for (const x0 of xs) {
    for (const y0 of ys) {
      const b0 = Math.min(box, H - x0), b1 = Math.min(box, W - y0);
      const p0 = nextMult(b0, sm), p1 = nextMult(b1, sm);
      const { xin, nT } = buildTileInputBs(grid, H, W, x0, y0, b0, b1, p0, p1, tx, cell, ff, sig2, C.logdiv, st.boresight, hpbw, am);
      const feed = {}; feed[inName] = new ort.Tensor('float32', xin, [1, 10, p0, p1]);
      const y = (await session.run(feed))[outName].data;
      for (let i = 0; i < b0; i++) for (let j = 0; j < b1; j++) {
        const g = (x0 + i) * W + (y0 + j), t = i * p1 + j, w = hann(i, box) * hann(j, box) + 1e-3;
        accRe[g] += y[t] * w; accIm[g] += y[nT + t] * w; wsum[g] += w;
      }
      if (onProg) { onProg(++done / total); await tick(); }
    }
  }
  const re = new Float32Array(H * W), im = new Float32Array(H * W);
  for (let gi = 0; gi < H; gi++) for (let gj = 0; gj < W; gj++) {
    const g = gi * W + gj, ws = Math.max(wsum[g], 1e-9), ur = accRe[g] / ws, ui = accIm[g] / ws;
    const ph = -k * Math.hypot(gi - tx.i, gj - tx.j) * cell, c = Math.cos(ph), s = Math.sin(ph);
    re[g] = ur * c - ui * s; im[g] = ur * s + ui * c;
  }
  return { re, im };
}

export function estimateOnnxSecondsBs(region, C, perTileMs) {
  return (numTiles(region.H, region.W, C.box, Math.round(C.box * 0.75)) * perTileMs) / 1000;
}
export function onnxNumTiles(region, C) {
  return numTiles(region.H, region.W, C.box, Math.round(C.box * 0.75));
}
// Time one 10-channel tile on THIS device (featurize + run) for the up-front estimate.
export async function calibrateOnnxPerTileMsBs(session, C) {
  const box = C.box, inName = session.inputNames[0], grid = new Uint8Array(box * box);
  const tx = { i: box >> 1, j: box >> 1 };
  const one = () => {
    const { xin } = buildTileInputBs(grid, box, box, 0, 0, box, box, box, box, tx, 0.1, 0, 8, 3, 0, 65, 25);
    const feed = {}; feed[inName] = new ort.Tensor('float32', xin, [1, 10, box, box]);
    return session.run(feed);
  };
  await one();
  const N = 3, t0 = performance.now();
  for (let i = 0; i < N; i++) await one();
  return (performance.now() - t0) / N;
}

// ------------------------------------------------------------------ penetrable FDTD
// FDTD cost on the fine region (runs at the native region resolution; no upsample —
// the region is already cell = lambda/npw).
export function fdtdCostBs(region, opts = {}) {
  const { H, W, h_m: h } = region, crossings = opts.crossings || 1.6, safety = opts.safety || 0.4;
  const dt = (safety * h) / (C0 * Math.SQRT2), diag = Math.hypot(H, W) * h;
  return { cells: H * W, steps: Math.round((crossings * diag) / C0 / dt) };
}
// The bare-leapfrog calibration kernel (fw_solve2d.calibrateFdtdMcps) runs
// cache-resident and omits the per-cell coefficient/damping reads, the two full
// grid passes (sponge + DFT) and the tick() yields — so the real penetrable solve
// is several× slower and memory-bound. FDTD_EFFICIENCY maps the calibrated rate to
// the observed one so the "TIME TO CALCULATE GRID" warning is honest, not 6-9× low.
export const FDTD_EFFICIENCY = 0.11;
export function estimateFdtdSecondsBs(region, mcps, opts) {
  const { cells, steps } = fdtdCostBs(region, opts);
  return (cells * steps) / (Math.max(mcps, 1) * FDTD_EFFICIENCY * 1e6);
}

// Damped-leapfrog scalar FDTD with PER-CLASS speed (c/n) + damping (gamma = gPerF*f),
// from the baked physics palette. Mirror of fullwave2d.FullWaveScene.step:
//   u_next = (2u - (1-a)u_prev + coef*lap) / (1+a); rigid=0; *sponge.
// `phys` = window.NOMA_CITY_2D.physics { classId: {n, gPerF, rigid} }.
export async function fdtdSolvePenetrable(region, phys, opts, onProg) {
  const { classes: g, H, W, h_m: h, tx } = region, fMHz = region.fMHz;
  const { crossings = 1.6, safety = 0.4, periods = 2 } = opts || {};
  const f0 = fMHz * 1e6, w = 2 * Math.PI * f0, dt = (safety * h) / (C0 * Math.SQRT2);
  const cdt2ih2Air = ((C0 * dt) ** 2) / (h * h);
  const NCLS = 6, coef = new Float32Array(NCLS), aCls = new Float32Array(NCLS), rigidCls = new Uint8Array(NCLS);
  for (let c = 0; c < NCLS; c++) {
    const p = phys[c] || phys[String(c)] || { n: 1, gPerF: 0, rigid: false };
    coef[c] = cdt2ih2Air / (p.n * p.n); aCls[c] = p.gPerF * f0 * dt; rigidCls[c] = p.rigid ? 1 : 0;
  }
  const N = H * W;
  const u = new Float32Array(N), up_ = new Float32Array(N), un = new Float32Array(N);
  const cf = new Float32Array(N), inv1pa = new Float32Array(N), oma = new Float32Array(N), rig = new Uint8Array(N);
  for (let i = 0; i < N; i++) { const c = g[i] < NCLS ? g[i] : 0; cf[i] = coef[c]; inv1pa[i] = 1 / (1 + aCls[c]); oma[i] = 1 - aCls[c]; rig[i] = rigidCls[c]; }
  const { steps } = fdtdCostBs(region, opts);
  const sw = Math.max(8, Math.round(0.06 * Math.min(H, W))), damp = new Float32Array(N);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) { const d = Math.min(i, j, H - 1 - i, W - 1 - j); damp[i * W + j] = d >= sw ? 1 : (0.92 + 0.08 * (d / sw)); }
  const src = tx.i * W + tx.j;
  const periodSteps = Math.max(1, Math.round((1 / f0) / dt));
  const winStart = Math.max(Math.floor(steps * 0.5), steps - periods * periodSteps);
  const accRe = new Float64Array(N), accIm = new Float64Array(N);
  let nWin = 0;
  const rate = Math.max(1, Math.floor(steps / 60));
  for (let n = 0; n < steps; n++) {
    for (let i = 1; i < H - 1; i++) {
      const r = i * W;
      for (let j = 1; j < W - 1; j++) {
        const idx = r + j;
        if (rig[idx]) { un[idx] = 0; continue; }
        const lap = u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1] - 4 * u[idx];
        un[idx] = (2 * u[idx] - oma[idx] * up_[idx] + cf[idx] * lap) * inv1pa[idx];
      }
    }
    const t = (n + 1) * dt;
    un[src] += Math.sin(w * t);
    for (let i = 0; i < N; i++) { un[i] *= damp[i]; up_[i] = u[i] * damp[i]; }
    u.set(un);
    if (n + 1 >= winStart) { const ph = -w * t, c = Math.cos(ph), s = Math.sin(ph); for (let i = 0; i < N; i++) { accRe[i] += u[i] * c; accIm[i] += u[i] * s; } nWin++; }
    if (onProg && (n % rate === 0)) { onProg(n / steps); await tick(); }
  }
  const re = new Float32Array(N), im = new Float32Array(N), sc = 2 / Math.max(nWin, 1);
  for (let i = 0; i < N; i++) { re[i] = accRe[i] * sc; im[i] = accIm[i] * sc; }
  if (onProg) onProg(1);
  return { re, im, steps };
}
