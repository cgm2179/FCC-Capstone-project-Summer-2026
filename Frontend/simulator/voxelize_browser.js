// voxelize_browser.js — rasterize a 3-D model into the engine's material grid, in the browser.
//
// Phase B: lets the Simulation tab solve on an IMPORTED model, not just the fixed pre-baked
// scene. The analytic tier (marchPL) and the smooth surface both read a Uint8Array material
// grid in C-order [NX,NY,NZ] with z fastest (idx = (x*NY+y)*NZ+z) — this produces exactly that,
// plus an interior mask and the world extent.
//
// SPEED NOTE: a voxelized imported model is a NEW scene, so it CANNOT reuse the cached volumes
// or a scene-locked DL surrogate — the sim runs on the slower analytic tier until a surrogate
// is trained (the xy/yz/zx-plane surrogate plan). This delivers capability, not speed.

// ---- core (pure; node-testable): world-space triangles → material grid + inside mask ----
// `tris` is a flat Float32Array of world coords, 9 floats per triangle (v0.xyz, v1.xyz, v2.xyz).
// Imported models come in ARBITRARY units (a GLB may be thousands of "units" wide for an ~80 m
// building), so the grid is fitted to `resolution` voxels on the longest axis (robust to any
// scale, and caps memory). `real_longest_m` sets the true metres-per-voxel for the physics; if
// omitted, `cell_m` (metres) is used directly (assumes the model is already in metres).
export function voxelizeTriangles(tris, { resolution = 160, real_longest_m = null,
    cell_m = null, bounds_m = null, barrierClass = 2, pad = 1 } = {}) {
  const nt = (tris.length / 9) | 0;
  if (nt < 1) return null;

  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3)
    for (let j = 0; j < 3; j++) { const v = tris[i + j]; if (v < mn[j]) mn[j] = v; if (v > mx[j]) mx[j] = v; }
  const size = [Math.max(mx[0] - mn[0], 1e-9), Math.max(mx[1] - mn[1], 1e-9), Math.max(mx[2] - mn[2], 1e-9)];
  const longest = Math.max(size[0], size[1], size[2]);

  // The whole rasterizer maps a MODEL-space point p to a voxel index via g = floor(p*cf + off[axis]),
  // and `half` is the surface-sampling step in MODEL units — computed once here for whichever mode is
  // active, so the loop below is identical for both. cellM is metres/voxel (isotropic → physics + all
  // world↔voxel conversions in simulation3d.js stay unchanged).
  let NX, NY, NZ, cellM, cf, ox, oy, oz, half, origin;
  if (bounds_m) {
    // Sandbox-bounds mode (user X/Y/Z): the domain is a real-world box of BX×BY×BZ metres. Isotropic
    // voxels sized so the LONGEST box axis gets `resolution` cells; the model is UNIFORMLY scaled to
    // fit inside (contain — never distorted) and centred, so its proportions are preserved and any
    // extra box on an axis is air margin.
    const BX = Math.max(bounds_m[0] || 1, 1e-3), BY = Math.max(bounds_m[1] || 1, 1e-3), BZ = Math.max(bounds_m[2] || 1, 1e-3);
    cellM = Math.max(BX, BY, BZ) / resolution;
    NX = Math.max(2, Math.round(BX / cellM) + 1);
    NY = Math.max(2, Math.round(BY / cellM) + 1);
    NZ = Math.max(2, Math.round(BZ / cellM) + 1);
    const s = Math.min(BX / size[0], BY / size[1], BZ / size[2]);          // model units → metres (contain)
    const mcx = (mn[0] + mx[0]) / 2, mcy = (mn[1] + mx[1]) / 2, mcz = (mn[2] + mx[2]) / 2;
    cf = s / cellM;                                                        // model units → voxels
    ox = (NX * cellM / 2 - mcx * s) / cellM;                               // centre the model in the box
    oy = (NY * cellM / 2 - mcy * s) / cellM;
    oz = (NZ * cellM / 2 - mcz * s) / cellM;
    half = 0.5 * cellM / s;                                                // half-voxel, in model units
    origin = [0, 0, 0];
  } else {
    // Longest-axis mode (legacy): fit ~`resolution` voxels on the model's longest axis; real_longest_m
    // sets metres/voxel; the extent follows the model's own proportions.
    const nativeCell = (cell_m != null) ? cell_m : longest / resolution;
    const nAcross = longest / nativeCell;
    cellM = real_longest_m ? (real_longest_m / nAcross) : nativeCell;
    origin = [mn[0] - pad * nativeCell, mn[1] - pad * nativeCell, mn[2] - pad * nativeCell];
    NX = Math.max(2, Math.ceil(size[0] / nativeCell) + 2 * pad + 1);
    NY = Math.max(2, Math.ceil(size[1] / nativeCell) + 2 * pad + 1);
    NZ = Math.max(2, Math.ceil(size[2] / nativeCell) + 2 * pad + 1);
    cf = 1 / nativeCell;
    ox = -origin[0] / nativeCell; oy = -origin[1] / nativeCell; oz = -origin[2] / nativeCell;
    half = nativeCell * 0.5;
  }
  const grid = new Uint8Array(NX * NY * NZ);            // 0 = air; barrierClass = solid
  const idx = (x, y, z) => (x * NY + y) * NZ + z;
  const clamp = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);

  // Surface voxelization by dense barycentric sampling at ~half-cell spacing along the two
  // edges — conservative enough that thin walls are not missed at this resolution.
  for (let t = 0; t < nt; t++) {
    const o = t * 9;
    const ax = tris[o],     ay = tris[o + 1], az = tris[o + 2];
    const bx = tris[o + 3], by = tris[o + 4], bz = tris[o + 5];
    const cx = tris[o + 6], cy = tris[o + 7], cz = tris[o + 8];
    const n1 = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay, bz - az) / half));
    const n2 = Math.max(1, Math.ceil(Math.hypot(cx - ax, cy - ay, cz - az) / half));
    for (let i = 0; i <= n1; i++) {
      const u = i / n1;
      for (let k = 0; k <= n2; k++) {
        const v = k / n2;
        if (u + v > 1) continue;
        const gx = clamp(Math.floor((ax + u * (bx - ax) + v * (cx - ax)) * cf + ox), NX);
        const gy = clamp(Math.floor((ay + u * (by - ay) + v * (cy - ay)) * cf + oy), NY);
        const gz = clamp(Math.floor((az + u * (bz - az) + v * (cz - az)) * cf + oz), NZ);
        grid[idx(gx, gy, gz)] = barrierClass;
      }
    }
  }

  const inside = floodInside(grid, NX, NY, NZ, idx);
  let nSolid = 0, nInside = 0;
  for (let i = 0; i < grid.length; i++) { if (grid[i]) nSolid++; if (inside[i]) nInside++; }
  return { grid, dims: [NX, NY, NZ], cell_m: cellM, origin,
    extent: [NX * cellM, NY * cellM, NZ * cellM], inside, n_solid: nSolid, n_inside: nInside };
}

// Interior mask: flood-fill air inward from the grid boundary → exterior air; interior air is
// what the flood never reaches (enclosed by the shell). Non-watertight models leak (few interior
// voxels) — acceptable for v1; the analytic tier does not require the mask.
function floodInside(grid, NX, NY, NZ, idx) {
  const ext = new Uint8Array(grid.length);
  const stack = [];
  const push = (x, y, z) => { const i = idx(x, y, z); if (grid[i] === 0 && !ext[i]) { ext[i] = 1; stack.push(x, y, z); } };
  for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) { push(x, y, 0); push(x, y, NZ - 1); }
  for (let x = 0; x < NX; x++) for (let z = 0; z < NZ; z++) { push(x, 0, z); push(x, NY - 1, z); }
  for (let y = 0; y < NY; y++) for (let z = 0; z < NZ; z++) { push(0, y, z); push(NX - 1, y, z); }
  while (stack.length) {
    const z = stack.pop(), y = stack.pop(), x = stack.pop();
    if (x > 0) push(x - 1, y, z); if (x < NX - 1) push(x + 1, y, z);
    if (y > 0) push(x, y - 1, z); if (y < NY - 1) push(x, y + 1, z);
    if (z > 0) push(x, y, z - 1); if (z < NZ - 1) push(x, y, z + 1);
  }
  const inside = new Uint8Array(grid.length);
  for (let i = 0; i < grid.length; i++) inside[i] = (grid[i] === 0 && !ext[i]) ? 1 : 0;
  return inside;
}

// ---- THREE wrapper (browser): extract world-space triangles from an Object3D ----
export function objectToTriangles(THREE, object) {
  const out = [];
  object.updateMatrixWorld(true);
  const v = new THREE.Vector3();
  object.traverse((o) => {
    if (!o.isMesh || !o.geometry || !o.geometry.attributes.position) return;
    const pos = o.geometry.attributes.position, m = o.matrixWorld, index = o.geometry.index;
    const emit = (i) => { v.fromBufferAttribute(pos, i).applyMatrix4(m); out.push(v.x, v.y, v.z); };
    if (index) for (let i = 0; i < index.count; i++) emit(index.getX(i));
    else for (let i = 0; i < pos.count; i++) emit(i);
  });
  return new Float32Array(out);
}
