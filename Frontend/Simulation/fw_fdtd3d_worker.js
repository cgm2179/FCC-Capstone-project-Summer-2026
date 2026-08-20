// fw_fdtd3d_worker.js — sub-volume 3-D scalar FDTD for the 3-D Full-Wave Studio.
//
// The 3-D twin of fw_solve2d.fdtdSolve, run OFF the main thread so the UI stays live.
// Nearest-upsamples the coarse material sub-volume (around the Tx) to λ/npw, then runs the damped,
// variable-speed leapfrog  u_tt + 2γ·u_t = c²·lap(u)  with per-class ITU materials (nByClass/gammaByClass
// from fw_solve3d.materialLuts): speed c = C0/√εr' → refraction/reflection, loss a = γ·dt → absorption.
// Class-3 stays a rigid reflector, plus a 6-face absorbing sponge + a soft CW point source. Pulls the
// steady-state phasor U with a running DFT over the last periods, then downsamples U back to the coarse
// sub-volume grid the page renders. (No per-class LUTs → falls back to a lossless air solve.)
//
// Whole-floor 3-D FDTD at λ/10 is millions of cells (would freeze the tab) — so the studio
// only ever hands this worker a few-metre sub-volume, exactly the box the surrogate is
// trained on. Posts {type:'progress',f} during the sweep and {type:'done',re,im,dims} at the end.
const C0 = 299792458.0, BARRIER = 3;               // 6-class: 3 = core_service_area (rigid reflector)

function upsample3d(grid, X, Y, Z, up) {
  if (up <= 1) return { g: grid, X, Y, Z };
  const Xf = X * up, Yf = Y * up, Zf = Z * up, g = new Uint8Array(Xf * Yf * Zf);
  for (let x = 0; x < Xf; x++) {
    const sx = (x / up) | 0;
    for (let y = 0; y < Yf; y++) {
      const sBase = (sx * Y + ((y / up) | 0)) * Z, dBase = (x * Yf + y) * Zf;
      for (let z = 0; z < Zf; z++) g[dBase + z] = grid[sBase + ((z / up) | 0)];
    }
  }
  return { g, X: Xf, Y: Yf, Z: Zf };
}

self.onmessage = (ev) => {
  const m = ev.data || {};
  const { grid, X0, Y0, Z0, cell, fMHz, tx0, up = 1, crossings = 1.4, safety = 0.4, periods = 2,
          nByClass = null, gammaByClass = null } = m;
  const { g, X, Y, Z } = upsample3d(grid, X0, Y0, Z0, up);
  const tx = { x: Math.round(tx0.x * up), y: Math.round(tx0.y * up), z: Math.round(tx0.z * up) };
  const h = cell / up, f0 = fMHz * 1e6, w = 2 * Math.PI * f0;
  const dt = (safety * h) / (C0 * Math.sqrt(3));                 // 3-D CFL (√3, vs √2 in 2-D)
  const cdt2ih2 = (C0 * dt) ** 2 / (h * h);
  const diag = Math.hypot(X, Y, Z) * h;
  const steps = Math.max(1, Math.round((crossings * diag) / C0 / dt));

  const YZ = Y * Z, N = X * YZ;
  const u = new Float32Array(N), up_ = new Float32Array(N), un = new Float32Array(N);
  const rigid = new Uint8Array(N);
  for (let i = 0; i < N; i++) rigid[i] = g[i] === BARRIER ? 1 : 0;

  // 6-face sponge: taper amplitude to ~0.92 over the outer `sw` cells (absorbing-ish)
  const sw = Math.max(6, Math.round(0.06 * Math.min(X, Y, Z)));
  const damp = new Float32Array(N);
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) {
    const base = (x * Y + y) * Z;
    for (let z = 0; z < Z; z++) {
      const d = Math.min(x, y, z, X - 1 - x, Y - 1 - y, Z - 1 - z);
      damp[base + z] = d >= sw ? 1 : (0.92 + 0.08 * (d / sw));
    }
  }
  let src = (tx.x * Y + tx.y) * Z + tx.z;
  if (rigid[src]) rigid[src] = 0;                               // never inject into a wall cell

  const periodSteps = Math.max(1, Math.round((1 / f0) / dt));
  const winStart = Math.max(Math.floor(steps * 0.5), steps - periods * periodSteps);
  const accRe = new Float64Array(N), accIm = new Float64Array(N);
  let nWin = 0;
  const rate = Math.max(1, Math.floor(steps / 60));

  // per-class material coefficients → per-cell damped, variable-speed leapfrog (matches Python fullwave2d):
  //   speed c = C0/n  ⇒  cdt2[cls] = (C0/n·dt)²/h² = cdt2ih2 / n²
  //   loss  a = γ·dt  ⇒  u_next = (2u − (1−a)·u_prev + cdt2·lap) / (1+a)
  // Unknown class / no LUT defaults to air (n=1, a=0) → the original lossless update, so this is safe.
  const NC = 256;
  const cdt2Cls = new Float64Array(NC).fill(cdt2ih2);
  const inv1paCls = new Float64Array(NC).fill(1);
  const oneMinusaCls = new Float64Array(NC).fill(1);
  if (nByClass || gammaByClass) {
    const L = Math.max(nByClass ? nByClass.length : 0, gammaByClass ? gammaByClass.length : 0);
    for (let c = 0; c < L; c++) {
      const ni = (nByClass && nByClass[c] > 0) ? nByClass[c] : 1;
      const a = (gammaByClass ? (gammaByClass[c] || 0) : 0) * dt;
      cdt2Cls[c] = cdt2ih2 / (ni * ni);
      inv1paCls[c] = 1 / (1 + a);
      oneMinusaCls[c] = 1 - a;
    }
  }

  for (let n = 0; n < steps; n++) {
    for (let x = 1; x < X - 1; x++) {
      const xB = x * YZ;
      for (let y = 1; y < Y - 1; y++) {
        const idx0 = xB + y * Z;
        for (let z = 1; z < Z - 1; z++) {
          const idx = idx0 + z;
          if (rigid[idx]) { un[idx] = 0; continue; }
          const cl = g[idx];                                    // material class → speed + absorption
          const lap = u[idx - YZ] + u[idx + YZ] + u[idx - Z] + u[idx + Z] + u[idx - 1] + u[idx + 1] - 6 * u[idx];
          un[idx] = (2 * u[idx] - oneMinusaCls[cl] * up_[idx] + cdt2Cls[cl] * lap) * inv1paCls[cl];
        }
      }
    }
    const t = (n + 1) * dt;
    un[src] += Math.sin(w * t);                                 // soft CW source
    for (let i = 0; i < N; i++) { un[i] *= damp[i]; up_[i] = u[i] * damp[i]; }
    u.set(un);
    if (n + 1 >= winStart) {
      const ph = -w * t, c = Math.cos(ph), s = Math.sin(ph);
      for (let i = 0; i < N; i++) { accRe[i] += u[i] * c; accIm[i] += u[i] * s; }
      nWin++;
    }
    if (n % rate === 0) self.postMessage({ type: 'progress', f: n / steps });
  }

  // U = (2/nWin)·acc — sample the fine phasor back onto the coarse sub-volume
  const re = new Float32Array(X0 * Y0 * Z0), im = new Float32Array(X0 * Y0 * Z0);
  const scale = 2 / Math.max(nWin, 1), off = (up / 2) | 0;
  for (let x = 0; x < X0; x++) for (let y = 0; y < Y0; y++) {
    const dBase = (x * Y0 + y) * Z0;
    for (let z = 0; z < Z0; z++) {
      const fx = Math.min(X - 1, x * up + off), fy = Math.min(Y - 1, y * up + off), fz = Math.min(Z - 1, z * up + off);
      const fg = (fx * Y + fy) * Z + fz, d = dBase + z;
      re[d] = accRe[fg] * scale; im[d] = accIm[fg] * scale;
    }
  }
  self.postMessage({ type: 'done', re, im, dims: [X0, Y0, Z0], steps, fine: [X, Y, Z] },
                   [re.buffer, im.buffer]);
};
