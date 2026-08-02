// smooth_surface.js — smooth iso-surface from a voxel occupancy grid.
//
// The building renders as one cube per voxel (blocky). This rebuilds a SMOOTH surface from
// the same grid so the slice view reads as a clean model, not a block cloud — and, because it
// runs on the decoded material grid client-side, it works for the fixed scene AND any future
// imported grid (Phase B). Cache-safe: pure rendering, never touches the physics.
//
// Method: Naive Surface Nets (S. Lysenko; dual-contouring family) rather than raw marching
// cubes — one vertex per surface cell placed at the centroid of its edge crossings, which is
// smoother on binary voxel data and needs no 256-case triangle table. The occupancy is lightly
// box-blurred first so the 0.5 isosurface is smooth, not staircased.

// ---- surface-nets edge tables (standard init; public-domain algorithm) ----
const CUBE_EDGES = new Int32Array(24);
const EDGE_TABLE = new Int32Array(256);
(function initTables() {
  let k = 0;
  for (let i = 0; i < 8; ++i) {
    for (let j = 1; j <= 4; j <<= 1) {
      const p = i ^ j;
      if (i <= p) { CUBE_EDGES[k++] = i; CUBE_EDGES[k++] = p; }
    }
  }
  for (let i = 0; i < 256; ++i) {
    let em = 0;
    for (let j = 0; j < 24; j += 2) {
      const a = !!(i & (1 << CUBE_EDGES[j]));
      const b = !!(i & (1 << CUBE_EDGES[j + 1]));
      em |= (a !== b) ? (1 << (j >> 1)) : 0;
    }
    EDGE_TABLE[i] = em;
  }
})();

// Naive Surface Nets on a scalar field `data` (< 0 = inside the surface), dims [dx,dy,dz],
// x fastest (data[x + dx*(y + dy*z)]). Returns { vertices:[[x,y,z],…], faces:[[a,b,c,d],…] }
// in grid coordinates.
function surfaceNets(data, dims) {
  const vertices = [], faces = [];
  const R = [1, dims[0] + 1, (dims[0] + 1) * (dims[1] + 1)];
  const grid = new Float32Array(8);
  let buf_no = 1;
  const buffer = new Int32Array(R[2] * 2);
  let n = 0;
  const x = [0, 0, 0];
  for (x[2] = 0; x[2] < dims[2] - 1; ++x[2], n += dims[0], buf_no ^= 1, R[2] = -R[2]) {
    let m = 1 + (dims[0] + 1) * (1 + buf_no * (dims[1] + 1));
    for (x[1] = 0; x[1] < dims[1] - 1; ++x[1], ++n, m += 2)
      for (x[0] = 0; x[0] < dims[0] - 1; ++x[0], ++n, ++m) {
        let mask = 0, g = 0, idx = n;
        for (let k = 0; k < 2; ++k, idx += dims[0] * (dims[1] - 2))
          for (let j = 0; j < 2; ++j, idx += dims[0] - 2)
            for (let i = 0; i < 2; ++i, ++g, ++idx) {
              const p = data[idx];
              grid[g] = p;
              mask |= (p < 0) ? (1 << g) : 0;
            }
        if (mask === 0 || mask === 0xff) continue;    // cell fully inside or fully outside
        const edge_mask = EDGE_TABLE[mask];
        const v = [0, 0, 0];
        let e_count = 0;
        for (let i = 0; i < 12; ++i) {
          if (!(edge_mask & (1 << i))) continue;
          ++e_count;
          const e0 = CUBE_EDGES[i << 1], e1 = CUBE_EDGES[(i << 1) + 1];
          const g0 = grid[e0], g1 = grid[e1];
          let t = g0 - g1;
          if (Math.abs(t) > 1e-6) t = g0 / t; else continue;
          for (let j = 0, kk = 1; j < 3; ++j, kk <<= 1) {
            const a = e0 & kk, b = e1 & kk;
            if (a !== b) v[j] += a ? 1.0 - t : t; else v[j] += a ? 1.0 : 0.0;
          }
        }
        const s = 1.0 / e_count;
        for (let i = 0; i < 3; ++i) v[i] = x[i] + s * v[i];
        buffer[m] = vertices.length;
        vertices.push(v);
        for (let i = 0; i < 3; ++i) {
          if (!(edge_mask & (1 << i))) continue;
          const iu = (i + 1) % 3, iv = (i + 2) % 3;
          if (x[iu] === 0 || x[iv] === 0) continue;
          const du = R[iu], dv = R[iv];
          if (mask & 1) faces.push([buffer[m], buffer[m - du], buffer[m - du - dv], buffer[m - dv]]);
          else faces.push([buffer[m], buffer[m - dv], buffer[m - du - dv], buffer[m - du]]);
        }
      }
  }
  return { vertices, faces };
}

// Separable box blur (radius r) on the x-fastest field — smooths the isosurface.
// Ping-pongs two work buffers (never mutates the caller's `src`).
function boxBlur3(src, NX, NY, NZ, r) {
  if (r <= 0) return src.slice();
  const at = (x, y, z) => x + NX * (y + NY * z);
  let a = src.slice(), b = new Float32Array(src.length);
  for (const axis of [0, 1, 2]) {
    for (let z = 0; z < NZ; ++z) for (let y = 0; y < NY; ++y) for (let x = 0; x < NX; ++x) {
      let s = 0, c = 0;
      for (let k = -r; k <= r; ++k) {
        const xx = axis === 0 ? x + k : x, yy = axis === 1 ? y + k : y, zz = axis === 2 ? z + k : z;
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) { s += a[at(xx, yy, zz)]; ++c; }
      }
      b[at(x, y, z)] = s / c;
    }
    const t = a; a = b; b = t;         // swap; reuse the old read buffer as the next write buffer
  }
  return a;
}

// Build a smooth surface for one material class.
// `grid` is the decoded material grid (Uint8Array) in C-order [NX,NY,NZ] with z fastest
// (idx = (x*NY+y)*NZ+z, matching plAt). Returns { positions:Float32Array, indices:Uint32Array }
// in world metres, or null if the class has no surface.
export function smoothSurfaceForClass(grid, NX, NY, NZ, cls, cellM, blur = 1) {
  const N = NX * NY * NZ;
  const occ = new Float32Array(N);                       // x-fastest layout for surface nets
  for (let z = 0; z < NZ; ++z)
    for (let y = 0; y < NY; ++y)
      for (let x = 0; x < NX; ++x)
        occ[x + NX * (y + NY * z)] = (grid[(x * NY + y) * NZ + z] === cls) ? 1 : 0;
  const data = boxBlur3(occ, NX, NY, NZ, blur);
  for (let i = 0; i < N; ++i) data[i] = 0.5 - data[i];   // < 0 where occupancy > 0.5 (inside)

  const { vertices, faces } = surfaceNets(data, [NX, NY, NZ]);
  if (!vertices.length) return null;

  const positions = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; ++i) {
    positions[i * 3]     = vertices[i][0] * cellM;
    positions[i * 3 + 1] = vertices[i][1] * cellM;
    positions[i * 3 + 2] = vertices[i][2] * cellM;
  }
  const indices = new Uint32Array(faces.length * 6);     // quads → two triangles
  for (let i = 0; i < faces.length; ++i) {
    const f = faces[i], o = i * 6;
    indices[o] = f[0]; indices[o + 1] = f[1]; indices[o + 2] = f[2];
    indices[o + 3] = f[0]; indices[o + 4] = f[2]; indices[o + 5] = f[3];
  }
  return { positions, indices };
}
