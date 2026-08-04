// sim_worker.js — the 3D Simulation's off-main-thread compute. Runs the two operations that used to
// freeze the UI (the whole codebase had NO workers): voxelizing an imported model, and the analytic
// per-voxel marchPL coverage solve. The main thread (simulation3d.js) keeps the cached-volume and
// surrogate tiers (bounded typed-array lookups) — only the analytic tier and voxelize come here.
//
// Module worker: both imports are RELATIVE and dependency-free (voxelize_browser's THREE wrapper takes
// THREE as a parameter, so the module itself imports nothing), so no import map is needed here.
//
// Protocol (main → worker):
//   {type:'voxelize', id, tris:Float32Array, opts, phys}   → {type:'voxelize', id, ok, vox}
//   {type:'setGrid',   id, grid:Uint8Array, dims, cellM, phys} → {type:'setGrid', id, ok}
//   {type:'solve', gen, kind:'wash'|'cloud', disp, params, phys} → {type:'solve', gen, kind, …buffers}
// The worker keeps the current scene grid RESIDENT (`state`) so `solve` payloads stay tiny.

import { voxelizeTriangles } from './voxelize_browser.js';
import { makeMarchPL } from './solve_core.js';

// Resident scene state. `grid`/`dims`/`cellM` identify the geometry the worker currently holds; the
// phys constants come from the manifest on the main thread and rarely change.
const state = { grid: null, dims: null, cellM: 1, nExp: 2, d0M: 1, stepCell: 0.5, satT: 40, satS: 50 };
const marchPL = makeMarchPL(() => state);

// viridis-ish ramp, t: 0 (weak) → 1 (strong). KEEP IN SYNC with `ramp` in simulation3d.js.
function ramp(t) {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(1, Math.max(0, t)) * (s.length - 1);
  const i = Math.floor(t), f = t - i, a = s[i], b = s[Math.min(i + 1, s.length - 1)];
  return [(a[0] + (b[0] - a[0]) * f) / 255, (a[1] + (b[1] - a[1]) * f) / 255, (a[2] + (b[2] - a[2]) * f) / 255];
}

// Colour/keep decision per Display settings. KEEP IN SYNC with `coverageSample` in simulation3d.js —
// here `disp` is passed explicitly (the main-thread copy reads the module DISPLAY/constants directly).
function coverageSample(disp, pl, sliced) {
  if (disp.classes || disp.scale === 'fixed') {
    const rsrp = disp.eirpDbm - pl;
    if (disp.classes) {
      if (rsrp >= disp.threshDbm + 10) return { keep: true, rgb: disp.covGood, mag: 0.9 };
      if (rsrp >= disp.threshDbm)      return { keep: true, rgb: disp.covMarg, mag: 0.6 };
      if (!sliced) return { keep: false };
      return { keep: true, rgb: disp.covDead, mag: 0.3 };
    }
    const t = (rsrp - disp.rsrpLo) / (disp.rsrpHi - disp.rsrpLo);
    if (!sliced && t < 0.12) return { keep: false };
    return { keep: true, rgb: ramp(t), mag: Math.min(1, Math.max(0, t)) };
  }
  const t = 1 - Math.min(1, Math.max(0, (pl - 45) / (130 - 45)));   // 1 = strong
  if (!sliced && t < 0.12) return { keep: false };
  return { keep: true, rgb: ramp(t), mag: t };
}

const clampi = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);

// WinProp-style smooth wash → an RGBA pixel buffer. Mirrors buildCoverageWash's loop in
// simulation3d.js (analytic branch of samplePlAt: world → voxel → marchPL). `dims`/`cellM` come from
// the resident state so the marchPL grid and the clamp bounds are always the same geometry.
function solveWash(disp, p) {
  const NX = state.dims[0], NY = state.dims[1], NZ = state.dims[2], cellM = state.cellM;
  const { nx, nz, ex, ez, yPlane, txVox, L, Ap, f1 } = p;
  const vy = clampi(Math.floor(yPlane / cellM), NY);
  const px = new Uint8ClampedArray(nx * nz * 4);
  let kept = 0;
  for (let jz = 0; jz < nz; jz++) {
    const z = ((jz + 0.5) / nz) * ez;
    const vz = clampi(Math.floor(z / cellM), NZ);
    for (let ix = 0; ix < nx; ix++) {
      const x = ((ix + 0.5) / nx) * ex;
      const vx = clampi(Math.floor(x / cellM), NX);
      const o = (jz * nx + ix) * 4;
      const pl = marchPL(txVox, vx, vy, vz, L, Ap, f1);
      if (!isFinite(pl) || pl >= 65504) { px[o + 3] = 0; continue; }
      const cs = coverageSample(disp, pl, true);
      if (!cs.keep) { px[o + 3] = 0; continue; }
      px[o]     = Math.round(cs.rgb[0] * 255);
      px[o + 1] = Math.round(cs.rgb[1] * 255);
      px[o + 2] = Math.round(cs.rgb[2] * 255);
      px[o + 3] = Math.round(40 + cs.mag * 180);
      kept++;
    }
  }
  return { pixels: px, nx, nz, kept };
}

// Instanced-cube point cloud → a flat [x,y,z,mag,r,g,b] Float32Array. Mirrors the triple loop in
// runStaticField (simulation3d.js), analytic branch.
function solveCloud(disp, p) {
  const NX = state.dims[0], NY = state.dims[1], NZ = state.dims[2], cellM = state.cellM;
  const { g, yLo, yHi, yInc, ex, ez, sliced } = p.bounds;
  const { txVox, L, Ap, f1 } = p;
  const out = [];
  for (let x = g * 0.5; x < ex; x += g) {
    const vx = clampi(Math.floor(x / cellM), NX);
    for (let y = yLo; y < yHi; y += yInc) {
      const vy = clampi(Math.floor(y / cellM), NY);
      for (let z = g * 0.5; z < ez; z += g) {
        const vz = clampi(Math.floor(z / cellM), NZ);
        const pl = marchPL(txVox, vx, vy, vz, L, Ap, f1);
        if (!isFinite(pl) || pl >= 65504) continue;
        const cs = coverageSample(disp, pl, sliced);
        if (!cs.keep) continue;
        out.push(x, y, z, cs.mag, cs.rgb[0], cs.rgb[1], cs.rgb[2]);
      }
    }
  }
  return { samples: new Float32Array(out), count: (out.length / 7) | 0 };
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'voxelize') {
      const vox = voxelizeTriangles(m.tris, m.opts);
      if (m.phys) Object.assign(state, m.phys);
      if (!vox) { self.postMessage({ type: 'voxelize', id: m.id, ok: false }); return; }
      // Keep the grid resident for subsequent solves; hand the MAIN thread its own copy (it needs
      // GRID3 for rendering / the probe / the marchPL fallback). Transferring a copy avoids neutering
      // the resident one.
      state.grid = vox.grid; state.dims = vox.dims; state.cellM = vox.cell_m;
      const gridCopy = vox.grid.slice();
      self.postMessage({ type: 'voxelize', id: m.id, ok: true, vox: {
        grid: gridCopy, inside: vox.inside, dims: vox.dims, cell_m: vox.cell_m, origin: vox.origin,
        extent: vox.extent, n_solid: vox.n_solid, n_inside: vox.n_inside,
      } }, [gridCopy.buffer, vox.inside.buffer]);
      return;
    }
    if (m.type === 'setGrid') {
      state.grid = m.grid; state.dims = m.dims; state.cellM = m.cellM;
      if (m.phys) Object.assign(state, m.phys);
      self.postMessage({ type: 'setGrid', id: m.id, ok: true });
      return;
    }
    if (m.type === 'solve') {
      if (m.phys) Object.assign(state, m.phys);
      if (!state.grid) { self.postMessage({ type: 'solve', gen: m.gen, ok: false }); return; }
      if (m.kind === 'wash') {
        const r = solveWash(m.disp, m.params);
        self.postMessage({ type: 'solve', gen: m.gen, ok: true, kind: 'wash',
          pixels: r.pixels, nx: r.nx, nz: r.nz, kept: r.kept, yPlane: m.params.yPlane },
          [r.pixels.buffer]);
      } else {
        const r = solveCloud(m.disp, m.params);
        self.postMessage({ type: 'solve', gen: m.gen, ok: true, kind: 'cloud',
          samples: r.samples, count: r.count }, [r.samples.buffer]);
      }
      return;
    }
  } catch (err) {
    self.postMessage({ type: (m && m.type) || 'error', id: m && m.id, gen: m && m.gen, ok: false,
      error: String(err && err.message || err) });
  }
};
