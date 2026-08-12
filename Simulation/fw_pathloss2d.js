/* fw_pathloss2d.js — non-AI/ML analytic path-loss solvers on the SIM V3 class grid.
 *
 * These are the deterministic / statistical models from the capstone, offered alongside the
 * full-wave FDTD and the ONNX surrogate in fw_studio2d.html. Each solver takes the material
 * class grid + a transmitter cell and returns { re, im, db } compatible with the studio's
 * heatmap render (db = RELATIVE received field in dB, normalised so the peak near the Tx is 0;
 * re = 10^(db/20), im = 0 — there is no wave phase, so the "Wavefront" view is flat for these).
 *
 *   FSPL          free-space loss only, 20·log10(4π·d/λ) — LOS baseline, ignores walls.
 *   Motley-Keenan FSPL + a fixed loss per wall/floor CROSSING along the LOS ray (contiguous
 *                 runs of a lossy material count once), the dominant indoor NLOS mechanism.
 *   Eikonal       shortest-PATH loss: a fast-marching / Dijkstra effective distance where walls
 *                 have high slowness, so the field routes AROUND obstructions (shadowing).
 *   Ray-trace     Motley-Keenan direct ray ⊕ first-order specular reflections off the plan's
 *                 bounding walls, combined in linear power (a simplified SBR tier).
 *
 * matLoss maps a material class id → dB per crossing (from the manifest's loss_db). Air (0) and
 * furniture (4) are transparent by SOLID() convention in the studio; here any class with
 * matLoss[c] > 0 is a "wall".
 */
(function () {
  const C_MHZ_M = 299.792458;                 // λ(m) = C_MHZ_M / fMHz

  function fspl_db(d, lam) { return 20 * Math.log10((4 * Math.PI * Math.max(d, 1e-6)) / lam); }

  // Normalise a raw received-dB field (higher = stronger) to peak 0 and build re/im for the render.
  function finish(recv, grid, H, W) {
    let mx = -Infinity;
    for (let g = 0; g < recv.length; g++) if ((grid[g] === 0 || grid[g] === 4) && recv[g] > mx) mx = recv[g];
    if (!isFinite(mx)) mx = 0;
    const db = new Float32Array(recv.length), re = new Float32Array(recv.length), im = new Float32Array(recv.length);
    for (let g = 0; g < recv.length; g++) { const v = recv[g] - mx; db[g] = v; re[g] = Math.pow(10, v / 20); }
    return { re, im, db };
  }

  // dB lost crossing walls along the ray (i0,j0)->(i1,j1); contiguous run of a lossy class = 1 crossing.
  function wallLossAlong(grid, H, W, i0, j0, i1, j1, matLoss) {
    const di = i1 - i0, dj = j1 - j0, n = Math.max(Math.abs(di), Math.abs(dj));
    if (n === 0) return 0;
    let loss = 0, prev = 0;
    for (let s = 1; s <= n; s++) {
      const i = Math.round(i0 + di * s / n), j = Math.round(j0 + dj * s / n);
      const c = grid[i * W + j], L = matLoss[c] || 0;
      if (L > 0 && c !== prev) loss += L;     // entering a new lossy run
      prev = L > 0 ? c : 0;
    }
    return loss;
  }

  function fspl(grid, H, W, cell, fMHz, tx) {
    const lam = C_MHZ_M / fMHz, recv = new Float32Array(H * W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const d = Math.hypot((j - tx.j) * cell, (i - tx.i) * cell);
      recv[i * W + j] = -fspl_db(d < cell ? cell : d, lam);
    }
    return finish(recv, grid, H, W);
  }

  function motley(grid, H, W, cell, fMHz, tx, matLoss) {
    const lam = C_MHZ_M / fMHz, recv = new Float32Array(H * W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      let d = Math.hypot((j - tx.j) * cell, (i - tx.i) * cell); if (d < cell) d = cell;
      const pl = fspl_db(d, lam) + wallLossAlong(grid, H, W, tx.i, tx.j, i, j, matLoss);
      recv[i * W + j] = -pl;
    }
    return finish(recv, grid, H, W);
  }

  // Dijkstra effective distance: air cells cost their euclidean step; lossy cells cost much more,
  // so the shortest path bends around walls. PL = FSPL(effective distance). 8-connected, binary heap.
  function eikonal(grid, H, W, cell, fMHz, tx, matLoss) {
    const lam = C_MHZ_M / fMHz, N = H * W, D = new Float32Array(N).fill(Infinity);
    const slo = (c) => (matLoss[c] ? 1 + matLoss[c] * 0.6 : 1);   // wall dB → slowness multiplier
    const heap = [], hpos = new Int32Array(N).fill(-1);           // min-heap of node ids by D
    const swap = (a, b) => { const t = heap[a]; heap[a] = heap[b]; heap[b] = t; hpos[heap[a]] = a; hpos[heap[b]] = b; };
    const up = (k) => { while (k > 0) { const p = (k - 1) >> 1; if (D[heap[p]] <= D[heap[k]]) break; swap(p, k); k = p; } };
    const down = (k) => { for (;;) { let l = 2 * k + 1, r = l + 1, m = k; if (l < heap.length && D[heap[l]] < D[heap[m]]) m = l; if (r < heap.length && D[heap[r]] < D[heap[m]]) m = r; if (m === k) break; swap(m, k); k = m; } };
    const push = (n) => { heap.push(n); hpos[n] = heap.length - 1; up(heap.length - 1); };
    const pop = () => { const top = heap[0], last = heap.pop(); hpos[top] = -1; if (heap.length) { heap[0] = last; hpos[last] = 0; down(0); } return top; };
    const s0 = tx.i * W + tx.j; D[s0] = 0; push(s0);
    const NB = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1], [-1, -1, 1.41421], [-1, 1, 1.41421], [1, -1, 1.41421], [1, 1, 1.41421]];
    while (heap.length) {
      const u = pop(), ui = (u / W) | 0, uj = u - ui * W, du = D[u];
      for (const [di, dj, w] of NB) {
        const vi = ui + di, vj = uj + dj; if (vi < 0 || vj < 0 || vi >= H || vj >= W) continue;
        const v = vi * W + vj, nd = du + w * cell * 0.5 * (slo(grid[u]) + slo(grid[v]));
        if (nd < D[v]) { D[v] = nd; if (hpos[v] < 0) push(v); else up(hpos[v]); }
      }
    }
    const recv = new Float32Array(N);
    for (let g = 0; g < N; g++) { const d = D[g] === Infinity ? 1e6 : Math.max(D[g], cell); recv[g] = -fspl_db(d, lam); }
    return finish(recv, grid, H, W);
  }

  // Ray-trace (lite): direct Motley-Keenan ray ⊕ first-order reflections off the 4 plan edges,
  // summed in linear power. Captures the "extra" energy from specular bounces near walls.
  function raytrace(grid, H, W, cell, fMHz, tx, matLoss) {
    const lam = C_MHZ_M / fMHz, recv = new Float32Array(H * W);
    const mirrors = [ { i: -tx.i, j: tx.j }, { i: 2 * (H - 1) - tx.i, j: tx.j },
                      { i: tx.i, j: -tx.j }, { i: tx.i, j: 2 * (W - 1) - tx.j } ];
    const REFLECT_DB = 6;   // per-bounce loss on top of the extra path length
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      let d = Math.hypot((j - tx.j) * cell, (i - tx.i) * cell); if (d < cell) d = cell;
      let p = Math.pow(10, -(fspl_db(d, lam) + wallLossAlong(grid, H, W, tx.i, tx.j, i, j, matLoss)) / 10);
      for (const m of mirrors) {
        let dr = Math.hypot((j - m.j) * cell, (i - m.i) * cell); if (dr < cell) dr = cell;
        p += Math.pow(10, -(fspl_db(dr, lam) + REFLECT_DB) / 10);
      }
      recv[i * W + j] = 10 * Math.log10(p);
    }
    return finish(recv, grid, H, W);
  }

  window.PL2D = { fspl, motley, eikonal, raytrace,
                  SOLVERS: ['fspl', 'motley', 'eikonal', 'raytrace'] };
})();
