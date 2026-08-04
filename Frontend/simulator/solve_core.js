// solve_core.js — the analytic path-loss KERNEL, shared verbatim by the main thread
// (Frontend/simulator/simulation3d.js) and the solve worker (Frontend/simulator/sim_worker.js).
//
// This is the single source of truth for the physics: `marchPL` is a direct port of the 2-D
// `pathlossPhysics` inner loop (SIM/web/simulator_tab.js) — Motley-Keenan spreading + per-crossing
// wall loss (each new run counts once) + Beer-Lambert furniture + a saturating obstruction cap.
//
// It is factored as `makeMarchPL(getCtx)` so the caller supplies the grid + constants lazily: the
// main thread passes a getter that reads the live module globals (so an imported scene with a fresh
// GRID3/CELL_M is picked up with no rebuild), and the worker passes a getter that reads its resident
// scene state. `getCtx()` is invoked once per marchPL call (not per ray step), so it is cheap.
//
// Pure module: NO imports, no THREE, no DOM — safe to `import` inside a Web Worker with no import map.

const clampi = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);

// getCtx() → { grid:Uint8Array, dims:[NX,NY,NZ], cellM, nExp, d0M, stepCell, satT, satS }
// Returns marchPL(tx, sx, sy, sz, L, Ap, f1) → path loss in dB from Tx voxel `tx` to sample voxel.
// L / Ap are per-material dB and dB/m arrays (already band-scaled by the caller); f1 = FSPL at 1 m.
export function makeMarchPL(getCtx) {
  return function marchPL(tx, sx, sy, sz, L, Ap, f1) {
    const { grid, dims, cellM, nExp, d0M, stepCell, satT, satS } = getCtx();
    const NX = dims[0], NY = dims[1], NZ = dims[2];
    const dx = sx - tx[0], dy = sy - tx[1], dz = sz - tx[2];
    const distC = Math.hypot(dx, dy, dz);
    const dm = Math.max(distC * cellM, d0M);
    const K = Math.max(1, Math.ceil(distC / stepCell));
    let wall = 0, perM = 0, cur = -1;
    for (let k = 0; k <= K; k++) {
      const t = k / K;
      const xi = clampi(Math.round(tx[0] + t * dx), NX);
      const yi = clampi(Math.round(tx[1] + t * dy), NY);
      const zi = clampi(Math.round(tx[2] + t * dz), NZ);
      const c = grid[(xi * NY + yi) * NZ + zi];
      if (c !== cur) { cur = c; if (L[c] > 0) wall += L[c]; }   // R3: new run counts once
      perM += Ap[c];
    }
    wall += perM * (distC / K) * cellM;                          // furniture (dB/m × length)
    // saturating wall-loss (flat-dB diffraction stand-in). satObs(x) from simulator_tab.js:31.
    const obs = wall <= satT ? wall : satT + satS * (1 - Math.exp(-(wall - satT) / satS));
    return f1 + 10 * nExp * Math.log10(dm) + obs;
  };
}
