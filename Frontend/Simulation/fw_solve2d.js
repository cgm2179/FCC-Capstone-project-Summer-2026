// fw_solve2d.js — in-browser indoor 2-D full-wave solvers for the Full-Wave
// Studio (2D). Two ways to produce the steady-state complex field U(x,y):
//
//   * onnxTiledPredict — fw_unet2d.onnx under onnxruntime-web, tiled over the
//     floor exactly like the Python fw_solver.onnx_tiled_predict. Milliseconds.
//   * fdtdSolve — a minimal scalar leapfrog FDTD (rigid barriers + sponge edges
//     + CW source + phasor DFT). This is the "regular FDTD solver" fallback the
//     product spec asks for when ONNX is unavailable. It is an APPROXIMATE,
//     air-speed stand-in (the exact ITU-material ground-truth FDTD is the Python
//     engine, Physics Engine/2D/SIM V3/run_wave2d.py --solver fdtd); it is much
//     slower than ONNX, which is exactly why we warn on the grid time first.
//
// Both return { re, im } as Float32Array(H*W) on the SAME native floor grid, so
// the page renders them identically: envelope |U| (dB) or animation Re{U e^jwt}.
//
// The 6-class material scheme (air, drywall, concrete, core=barrier, furniture,
// exterior) is shared by the 2-D floor plan (SIM/web/sim_assets.js) and the 3-D
// scene the model was trained on (SIM V1 3D/manifest_3d.json) — verified equal.

export const C0 = 299_792_458.0;
const BARRIER_CLASS = 3;               // core_service_area — rigid reflector
const tick = () => new Promise(r => setTimeout(r, 0));

// ------------------------------------------------------------------ contract
// Read fw_unet2d.json so the featurization can never drift from the model.
export async function loadContract(url) {
  const j = await (await fetch(url, { cache: 'no-store' })).json();
  const inp = j.input, out = j.output;
  return {
    sigma: inp.tx_blob_sigma_cells,          // 2.0 cells
    logdiv: inp.logdist_divisor,             // 3.0
    flo: inp.freq_log_lo_mhz,                // 600
    fhi: inp.freq_log_hi_mhz,                // 6200
    box: j.box_train || 128,
    sizeMultiple: inp.size_multiple || 16,
    bands: j.bands_mhz || [],
    trainedMhz: j.trained_mhz || [],       // numeric training bands (robust match)
    reconstruct: out.reconstruct,
  };
}

const freqFeat = (fMHz, C) =>
  (Math.log10(fMHz) - Math.log10(C.flo)) / (Math.log10(C.fhi) - Math.log10(C.flo));

// ------------------------------------------------------------------ ONNX path
function tileStarts(n, box, stride) {
  if (n <= box) return [0];
  const xs = [];
  for (let x = 0; x + box <= n; x += stride) xs.push(x);
  xs.push(n - box);
  return [...new Set(xs)].sort((a, b) => a - b);
}
const nextMult = (n, m) => Math.ceil(n / m) * m;
const hann = (i, box) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / (box - 1)));

export function numTiles(H, W, box, stride) {
  return tileStarts(H, box, stride).length * tileStarts(W, box, stride).length;
}

const clamp05 = c => (c < 0 ? 0 : c > 5 ? 5 : c);

// Build the (9,p0,p1) NCHW input for one tile: material one-hot (padded with
// air), Tx Gaussian blob (LOCAL tile coords), broadcast freq feature, global
// log-distance in metres. Hoisted + sqrt-not-hypot so it isn't the bottleneck.
function buildTileInput(grid, H, W, x0, y0, b0, b1, p0, p1, tx, cell, ff, sig2, logdiv) {
  const nT = p0 * p1, xin = new Float32Array(9 * nT);
  const blobOff = 6 * nT, ffOff = 7 * nT, logOff = 8 * nT, invLog = 1 / logdiv;
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
    }
  }
  return { xin, nT };
}

// Predict U over the whole grid by Hann-feathered tiling; blend on the smooth
// phase-reduced field then reapply the single global free-space ramp so the
// stitched field stays phase-coherent (mirror of fw_infer.tiled_predict).
export async function onnxTiledPredict(session, grid, H, W, cell, fMHz, tx, C, onProg) {
  const box = C.box, stride = Math.round(box * 0.75), sm = C.sizeMultiple;
  const ff = freqFeat(fMHz, C), sig2 = 2 * C.sigma * C.sigma;
  const k = (2 * Math.PI) / (C0 / (fMHz * 1e6));
  const xs = tileStarts(H, box, stride), ys = tileStarts(W, box, stride);
  const total = xs.length * ys.length;
  const inName = session.inputNames[0], outName = session.outputNames[0];

  const accRe = new Float64Array(H * W), accIm = new Float64Array(H * W);
  const wsum = new Float64Array(H * W);
  let done = 0;
  for (const x0 of xs) {
    for (const y0 of ys) {
      const b0 = Math.min(box, H - x0), b1 = Math.min(box, W - y0);
      const p0 = nextMult(b0, sm), p1 = nextMult(b1, sm);
      const { xin, nT } = buildTileInput(grid, H, W, x0, y0, b0, b1, p0, p1, tx, cell, ff, sig2, C.logdiv);
      const feed = {}; feed[inName] = new ort.Tensor('float32', xin, [1, 9, p0, p1]);
      const y = (await session.run(feed))[outName].data;    // (2,p0,p1)
      for (let i = 0; i < b0; i++) {
        for (let j = 0; j < b1; j++) {
          const g = (x0 + i) * W + (y0 + j), t = i * p1 + j;
          const w = hann(i, box) * hann(j, box) + 1e-3;
          accRe[g] += y[t] * w; accIm[g] += y[nT + t] * w; wsum[g] += w;
        }
      }
      if (onProg && (++done % 1 === 0)) { onProg(done / total); await tick(); }
    }
  }
  const re = new Float32Array(H * W), im = new Float32Array(H * W);
  for (let gi = 0; gi < H; gi++) {
    for (let gj = 0; gj < W; gj++) {
      const g = gi * W + gj, ws = Math.max(wsum[g], 1e-9);
      const ur = accRe[g] / ws, ui = accIm[g] / ws;
      const ph = -k * Math.hypot(gi - tx.i, gj - tx.j) * cell;   // global ramp
      const c = Math.cos(ph), s = Math.sin(ph);
      re[g] = ur * c - ui * s; im[g] = ur * s + ui * c;
    }
  }
  return { re, im };
}

export function estimateOnnxSeconds(H, W, C, perTileMs) {
  const box = C.box, stride = Math.round(box * 0.75);
  return (numTiles(H, W, box, stride) * perTileMs) / 1000;
}

// Time one box×box tile through the REAL path (featurize + run) so the up-front
// estimate reflects THIS device — featurization, not just inference, dominates.
export async function calibrateOnnxPerTileMs(session, C) {
  const box = C.box, inName = session.inputNames[0];
  const grid = new Uint8Array(box * box);                   // all-air stand-in
  const tx = { i: box >> 1, j: box >> 1 };
  const one = () => {
    const { xin } = buildTileInput(grid, box, box, 0, 0, box, box, box, box, tx, 0.1, 0, 3);
    const feed = {}; feed[inName] = new ort.Tensor('float32', xin, [1, 9, box, box]);
    return session.run(feed);
  };
  await one();                                              // warm up (JIT / GPU upload)
  const N = 3, t0 = performance.now();
  for (let i = 0; i < N; i++) await one();
  return (performance.now() - t0) / N;
}

// ------------------------------------------------------------------ FDTD path
// Nearest-neighbour upsample of the class grid to `up`× finer (so FDTD can hit a
// few cells per wavelength). tx scales with it.
function upsampleGrid(grid, H, W, up) {
  if (up <= 1) return { g: grid, H, W };
  const Hf = H * up, Wf = W * up, g = new Uint8Array(Hf * Wf);
  for (let i = 0; i < Hf; i++) {
    const si = (i / up) | 0;
    for (let j = 0; j < Wf; j++) g[i * Wf + j] = grid[si * W + ((j / up) | 0)];
  }
  return { g, H: Hf, W: Wf };
}

export function fdtdCells(H, W, cell, fMHz, up) {
  return H * up * W * up;
}
export function fdtdSteps(H, W, cell, fMHz, up, crossings = 1.4, safety = 0.4) {
  const h = cell / up, dt = (safety * h) / (C0 * Math.SQRT2);
  const diag = Math.hypot(H * up, W * up) * h;
  return Math.round((crossings * diag) / C0 / dt);
}

// Minimal scalar-wave leapfrog: u_tt = c^2 lap(u), air speed, class-3 rigid, an
// exponential sponge at the borders, a soft CW point source, and a running DFT
// over the last few periods to pull out the steady-state phasor U. Returns U on
// the NATIVE H×W grid (fine field sampled back down).
export async function fdtdSolve(grid0, H0, W0, cell, fMHz, tx0, opts, onProg) {
  const { up = 1, crossings = 1.4, safety = 0.4, periods = 2 } = opts || {};
  const { g, H, W } = upsampleGrid(grid0, H0, W0, up);
  const tx = { i: Math.round(tx0.i * up), j: Math.round(tx0.j * up) };
  const h = cell / up, f0 = fMHz * 1e6, w = 2 * Math.PI * f0;
  const dt = (safety * h) / (C0 * Math.SQRT2);
  const cdt2ih2 = (C0 * dt) ** 2 / (h * h);
  const steps = fdtdSteps(H0, W0, cell, fMHz, up, crossings, safety);

  const N = H * W;
  const u = new Float32Array(N), up_ = new Float32Array(N), un = new Float32Array(N);
  const rigid = new Uint8Array(N);
  for (let i = 0; i < N; i++) rigid[i] = g[i] === BARRIER_CLASS ? 1 : 0;

  // sponge: taper amplitude to ~0.92 over the outer `sw` cells (absorbing-ish)
  const sw = Math.max(8, Math.round(0.06 * Math.min(H, W)));
  const damp = new Float32Array(N);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const d = Math.min(i, j, H - 1 - i, W - 1 - j);
      damp[i * W + j] = d >= sw ? 1 : (0.92 + 0.08 * (d / sw));
    }
  }
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
        if (rigid[idx]) { un[idx] = 0; continue; }
        const lap = u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1] - 4 * u[idx];
        un[idx] = 2 * u[idx] - up_[idx] + cdt2ih2 * lap;
      }
    }
    const t = (n + 1) * dt;
    un[src] += Math.sin(w * t);                             // soft CW source
    for (let i = 0; i < N; i++) { un[i] *= damp[i]; up_[i] = u[i] * damp[i]; }
    u.set(un);
    if (n + 1 >= winStart) {
      const ph = -w * t, c = Math.cos(ph), s = Math.sin(ph);
      for (let i = 0; i < N; i++) { accRe[i] += u[i] * c; accIm[i] += u[i] * s; }
      nWin++;
    }
    if (onProg && (n % rate === 0)) { onProg(n / steps); await tick(); }
  }
  // U = (2/nWin) * acc ; sample the fine phasor back onto the native grid
  const re = new Float32Array(H0 * W0), im = new Float32Array(H0 * W0);
  const sc = 2 / Math.max(nWin, 1), off = (up / 2) | 0;
  for (let i = 0; i < H0; i++) {
    for (let j = 0; j < W0; j++) {
      const fi = Math.min(H - 1, i * up + off), fj = Math.min(W - 1, j * up + off);
      const fg = fi * W + fj, g0 = i * W0 + j;
      re[g0] = accRe[fg] * sc; im[g0] = accIm[fg] * sc;
    }
  }
  if (onProg) onProg(1);
  return { re, im, steps, dims: [H, W] };
}

// JS Mcell-updates/s so the FDTD grid warning reflects THIS device.
export function calibrateFdtdMcps() {
  const S = 256, N = S * S;
  const u = new Float32Array(N), up_ = new Float32Array(N), un = new Float32Array(N);
  for (let i = 0; i < N; i++) u[i] = Math.random() - 0.5;
  const iters = 12, t0 = performance.now();
  for (let it = 0; it < iters; it++) {
    for (let i = 1; i < S - 1; i++) {
      const r = i * S;
      for (let j = 1; j < S - 1; j++) {
        const idx = r + j;
        un[idx] = 2 * u[idx] - up_[idx] +
          0.25 * (u[idx - S] + u[idx + S] + u[idx - 1] + u[idx + 1] - 4 * u[idx]);
      }
    }
    up_.set(u); u.set(un);
  }
  return (N * iters) / ((performance.now() - t0) / 1000) / 1e6;
}
export function estimateFdtdSeconds(H, W, cell, fMHz, up, mcps) {
  const cells = fdtdCells(H, W, cell, fMHz, up);
  const steps = fdtdSteps(H, W, cell, fMHz, up);
  return (cells * steps) / (Math.max(mcps, 1) * 1e6);
}

export function fmtEta(s) {
  if (s == null) return '—';
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

// ------------------------------------------------------------------ render helpers
// U -> envelope dB (relative to the 99.5th-pct amplitude over non-wall cells).
export function envelopeDb(re, im, grid, H, W) {
  const N = H * W, amp = new Float32Array(N), open = [];
  for (let i = 0; i < N; i++) {
    const a = Math.hypot(re[i], im[i]); amp[i] = a;
    const c = grid[i];
    if ((c === 0 || c === 4) && a > 0) open.push(a);   // ref over open (air / furniture) cells
  }
  open.sort((a, b) => a - b);
  const ref = open[Math.floor(open.length * 0.995)] || 1e-9;
  const db = new Float32Array(N);
  for (let i = 0; i < N; i++) db[i] = 20 * Math.log10(Math.max(amp[i], ref * 1e-6) / ref);
  return { db, ref };
}

const VIRIDIS = ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c',
  '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'];
export function viridis(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(t), f = t - i, a = hex(VIRIDIS[i]), b = hex(VIRIDIS[Math.min(i + 1, 10)]);
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}
// diverging red/white/blue for the signed wavefront
export function rdbu(t) {                                   // t in [-1,1]
  t = Math.min(1, Math.max(-1, t));
  if (t >= 0) return [215 + (255 - 215) * (1 - t), (255 - 205 * t), (255 - 217 * t)];
  const a = -t; return [(247 - 178 * a), (247 - 141 * a), 247 * (1 - a) + 200 * a];
}
// jet / turbo "rainbow": blue → cyan → green → yellow → red
export function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const cl = v => Math.max(0, Math.min(1, v));
  return [cl(1.5 - Math.abs(4 * t - 3)) * 255, cl(1.5 - Math.abs(4 * t - 2)) * 255, cl(1.5 - Math.abs(4 * t - 1)) * 255];
}
// sequential dispatcher (t in [0,1]) — heatmap + colourbar
export function cmap(name, t) {
  if (name === 'rainbow') return turbo(t);
  if (name === 'rdbu') return rdbu(2 * t - 1);             // cool→warm (blue-white-red)
  return viridis(t);
}
// signed dispatcher (v in [-1,1]) — wavefront: diverging for rdbu, else sequential over [-1,1]
export function cmapSigned(name, v) {
  return name === 'rdbu' ? rdbu(v) : cmap(name, (v + 1) / 2);
}
function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
