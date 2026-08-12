// georef.js — the floor-plan georeference transform chain.
//
//   lon/lat  ⇄  floor-plan pixels  →  local ENU metres  →  (voxel grid)  →  model world
//
// Replaces viewer3d.js's rubber-sheet (min/max-normalising a lon/lat walk onto the floor
// rectangle, which silently dropped the −7.33° map rotation and mis-registered any track
// that didn't fill the exact plan bbox). The affines come from
// Physics Engine/2D/STEP_1/floorplan_meta.json, surfaced to the browser as
// window.FLOORPLAN_META (Data/floorplan_meta.js).
//
// The pure functions take a `meta` argument (default window.FLOORPLAN_META) so they are
// unit-testable in node. `available()` guards callers when the asset is absent.

const WIN_META = (typeof window !== 'undefined' && window.FLOORPLAN_META) || null;

export function available(meta = WIN_META) {
  return !!(meta && meta.affine_px_to_lonlat && meta.affine_px_to_local_m);
}

// affine_px_to_lonlat = [[a,b,c],[d,e,f]] maps (px, py) → (lon, lat):
//   lon = a·px + b·py + c ,  lat = d·px + e·py + f
export function pxToLonlat(px, py, meta = WIN_META) {
  const A = meta.affine_px_to_lonlat;
  return { lon: A[0][0] * px + A[0][1] * py + A[0][2],
           lat: A[1][0] * px + A[1][1] * py + A[1][2] };
}

// Inverse of the 2×2 linear part (with rotation) — this is what fixes the registration.
export function lonlatToPx(lon, lat, meta = WIN_META) {
  const A = meta.affine_px_to_lonlat;
  const a = A[0][0], b = A[0][1], c = A[0][2], d = A[1][0], e = A[1][1], f = A[1][2];
  const det = a * e - b * d;
  const X = lon - c, Y = lat - f;
  return { px: (e * X - b * Y) / det, py: (-d * X + a * Y) / det };
}

// px → local ENU metres (physical distances; the mercator affine is inflated by ~1/cos φ).
export function pxToLocalM(px, py, meta = WIN_META) {
  const A = meta.affine_px_to_local_m;
  return { mx: A[0][0] * px + A[0][1] * py + A[0][2],
           my: A[1][0] * px + A[1][1] * py + A[1][2] };
}

// px → voxel indices. NOTE: registration_3d.json stores only a SCALAR px→vox scale
// (≈0.2264) with no offset/rotation, and it does not reconcile the two grid axes
// (NX 262 ≈ 1150·0.2264 ✓ but NZ 132 ≠ 515·0.2264). So this is the documented scale only,
// pending an explicit px_to_vox affine derived from inside_mask.npy. simulation3d.js is
// voxel-native and does not use this; the viewer overlay uses lonlatToPx + its own
// px→world mapping.
export function localMToVox(px, py, meta = WIN_META) {
  const s = meta.px_to_vox_scale || { x: 1, z: 1 };
  return { vx: px * s.x, vz: py * s.z };
}

// voxel → model world metres (isotropic cell size; matches simulation3d.js CELL_M).
export function voxToWorld(vx, vy, vz, meta = WIN_META) {
  const c = (meta && meta.cell_size_m) || 0.3;
  return { x: vx * c, y: vy * c, z: vz * c };
}
