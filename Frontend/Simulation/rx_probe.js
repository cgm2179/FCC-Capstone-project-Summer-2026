/* rx_probe.js — receiver probe + per-cell CSV export for the SIM V3 studios.
 *
 * Loaded as a plain <script> (like fw_pathloss2d.js) → exposes window.RxProbe. DOM-free math
 * (buildAnchor / metricsAt / buildCsv) plus a small downloadCsv() UI helper, shared by the indoor
 * 2D/3D and outdoor 2D studios so the receiver readout + "export all cells" behave identically.
 *
 * WHAT IT COMPUTES  — "field-derived" path loss / RSRP, per the capstone's energy→power→loss chain.
 *   The studios solve a steady-state field U(x,y[,z]); |U| = hypot(re, im) is the field amplitude
 *   (RELATIVE — the source amplitude is arbitrary). Power density S ∝ |U|². To turn the relative
 *   field into an ABSOLUTE path loss we anchor it to free space near the Tx:
 *
 *     • relative field:  pick the open cell nearest a reference distance d0 (≈1 m) from the Tx; call
 *       its amplitude A_ref at distance d_ref. Then for any cell g with amplitude A:
 *           PL(g)   = FSPL(d_ref, λ) + 20·log10(A_ref / A)          [dB, absolute]
 *           RSRP(g) = txDbm + Gt + Gr − PL(g)                       [dBm]   (Gt=Gr=0 dBi default)
 *       So the near-Tx cell sits at free-space loss and every other cell adds exactly the extra
 *       loss the simulation itself computed (walls, shadowing, diffraction).
 *     • absolute field (field.absolute — the eikonal / 3D-hybrid path already emits RSRP in dBm):
 *           RSRP(g) = db(g);   PL(g) = txDbm − RSRP(g).
 *
 *   FSPL(d, λ) = 20·log10(4π·d/λ),  λ = C0/f  — matches fw_pathloss2d.js:fspl_db.
 */
(function () {
  const C0 = 299792458.0;                 // m/s
  const C_MHZ_M = 299.792458;             // λ(m) = C_MHZ_M / fMHz

  const lambda = (fMHz) => C_MHZ_M / fMHz;
  const fsplDb = (d, lam) => 20 * Math.log10((4 * Math.PI * Math.max(d, 1e-6)) / lam);

  // Build the free-space anchor for a solved field. Callers stay geometry-agnostic by passing:
  //   nCells, ampOf(g) → |U|, isOpen(g) → carries the field (air/furniture), distOf(g) → metres to Tx.
  // For an absolute field (db already in dBm) no anchor search is needed.
  function buildAnchor({ field, fMHz, txDbm, gainDb = 0, nCells, ampOf, isOpen, distOf, d0 = 1.0 }) {
    const lam = lambda(fMHz);
    if (field && field.absolute) return { absolute: true, lam, fMHz, txDbm, gainDb };
    let A_ref = 0, d_ref = 0, best = Infinity;
    for (let g = 0; g < nCells; g++) {
      if (!isOpen(g)) continue;
      const A = ampOf(g); if (!(A > 0)) continue;
      const d = Math.max(distOf(g), 1e-6), score = Math.abs(d - d0);
      if (score < best) { best = score; A_ref = A; d_ref = d; }
    }
    if (!(A_ref > 0)) { A_ref = 1; d_ref = d0; }     // degenerate (nothing solved) — harmless fallback
    return { absolute: false, lam, fMHz, txDbm, gainDb, A_ref, d_ref, PL_ref: fsplDb(d_ref, lam) };
  }

  // Metrics for one cell. sample carries the geometry the studio already knows:
  //   { A:|U|, db, d, dx, dy, dz, world:[Rx,Ry,Rz], material }.  Returns numbers ('' → blank in CSV).
  function metricsAt(anchor, s) {
    const g = anchor.gainDb || 0;
    const plFree = fsplDb(s.d, anchor.lam);     // free-space floor — PL can't beat free space (kills the
    let PL, RSRP;                               // near-source FDTD spike that would give PL<0 / RSRP>EIRP)
    if (anchor.absolute) {                      // db is already RSRP in dBm
      let rsrp = isFinite(s.db) ? s.db + g : NaN;
      if (isFinite(rsrp)) rsrp = Math.min(rsrp, anchor.txDbm + g - plFree);
      RSRP = rsrp;
      PL = isFinite(RSRP) ? anchor.txDbm - (RSRP - g) : NaN;
    } else {
      const Aclamp = Math.max(s.A, anchor.A_ref * 1e-6);
      PL = Math.max(anchor.PL_ref + 20 * Math.log10(anchor.A_ref / Aclamp), plFree);
      RSRP = anchor.txDbm + g - PL;
    }
    return { U: s.A, db: s.db, d: s.d, dx: s.dx, dy: s.dy, dz: s.dz, PL, RSRP,
             world: s.world, material: s.material };
  }

  // --- CSV over every cell. Leading 9 columns are the requested spec; the rest are cheap diagnostics.
  const num = (v, dp) => (v == null || !isFinite(v)) ? '' : v.toFixed(dp);
  const sci = (v) => (v == null || !isFinite(v)) ? '' : v.toExponential(4);
  const cell = (s) => { s = String(s == null ? '' : s); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };

  // opts: { anchor, nCells, sampleOf(g) → {A,db,d,dx,dy,dz,world,material,i,j,k?,extra?}, has3D, extraCols }
  //   extraCols: string[] of trailing header names; sampleOf returns a matching `extra` array (e.g. lon/lat).
  function buildCsv({ anchor, nCells, sampleOf, has3D = false, extraCols = null }) {
    const cols = ['Rx_m', 'Ry_m', 'Rz_m', 'U_abs', 'PathLoss_dB', 'RSRP_dBm',
      'tx_minus_rx_m', 'ty_minus_ry_m', 'tz_minus_rz_m', 'dist_m', 'field_dB', 'material', 'cell_i', 'cell_j'];
    if (has3D) cols.push('cell_k');
    if (extraCols) cols.push(...extraCols);
    const lines = [cols.join(',')];
    for (let g = 0; g < nCells; g++) {
      const s = sampleOf(g); if (!s) continue;
      const m = metricsAt(anchor, s), w = s.world;
      const row = [num(w[0], 3), num(w[1], 3), num(w[2], 3), sci(m.U), num(m.PL, 2), num(m.RSRP, 2),
        num(s.dx, 3), num(s.dy, 3), num(s.dz, 3), num(s.d, 3), num(m.db, 2), cell(s.material), s.i, s.j];
      if (has3D) row.push(s.k);
      if (extraCols && s.extra) for (const v of s.extra) row.push(typeof v === 'number' ? num(v, 6) : cell(v));
      lines.push(row.join(','));
    }
    return lines.join('\n');
  }

  function downloadCsv(filename, text) {
    const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
  }

  window.RxProbe = { C0, lambda, fsplDb, buildAnchor, metricsAt, buildCsv, downloadCsv };
})();
