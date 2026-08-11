// preprocess_cad.mjs — the FAITHFUL preprocess step: voxelize the ORIGINAL, full-resolution CAD into
// the material grid that governs the physics, so the RF solve never depends on a decimated mesh.
//
//   node preprocess_cad.mjs --in model.glb --bounds 60 15 40 --out ../Data/models --name sandbox_v1
//
// Emits, into <out>/:
//   <name>.vox.json     — the PHYSICS artifact: the full-CAD voxel grid (grid_b64 + inside_b64 + dims +
//                         cell_m + extent). Compact (~1-2 MB) and exact at the grid resolution. The
//                         Simulation tab loads this and solves on it directly (no in-browser voxelize).
//   <name>.display.glb  — a DISPLAY-ONLY decimated mesh, Draco-compressed, with the simplify error
//                         PINNED UNDER the cell size (so even if it were voxelized it would match). It
//                         never feeds the physics — the grid does.
//
// Input is a .glb (incl. Draco). For a .obj, first: npx obj2gltf -i model.obj -o model.glb  (copy the
// .obj WITHOUT its .mtl to skip textures — irrelevant to RF). Node deps: @gltf-transform/core,
// @gltf-transform/extensions, @gltf-transform/functions, draco3dgltf (npm i -D those).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { flatten } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';
import { voxelizeTriangles } from './voxelize_browser.js';
import { writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { basename, join } from 'node:path';

function arg(name, n = 1) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return null;
  return n === 1 ? process.argv[i + 1] : process.argv.slice(i + 1, i + 1 + n);
}
const IN = arg('in');
const OUT = arg('out') || '.';
const NAME = arg('name') || basename(IN || 'model').replace(/\.[^.]+$/, '');
const RES = Number(arg('resolution')) || 160;
const BOUNDS = (arg('bounds', 3) || []).map(Number).filter((v) => v > 0);
if (!IN || BOUNDS.length !== 3) {
  console.error('usage: node preprocess_cad.mjs --in model.glb --bounds X Y Z [--out dir] [--name n] [--resolution 160]');
  process.exit(1);
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'draco3d.encoder': await draco3d.createEncoderModule(),
});

console.log('[1/3] reading ' + IN + ' (full resolution) …');
const doc = await io.read(IN);
await doc.transform(flatten());                       // bake node transforms → world-space geometry
const out = [];
for (const mesh of doc.getRoot().listMeshes())
  for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION'); if (!pos) continue;
    const P = pos.getArray(), idx = prim.getIndices();
    if (idx) { const I = idx.getArray(); for (let i = 0; i < I.length; i++) { const b = I[i] * 3; out.push(P[b], P[b + 1], P[b + 2]); } }
    else { for (let i = 0; i < P.length; i++) out.push(P[i]); }
  }
const tris = new Float32Array(out);
console.log('      ' + (tris.length / 9).toLocaleString() + ' triangles');

console.log('[2/3] voxelizing the ORIGINAL mesh → grid (the physics artifact) …');
const vox = voxelizeTriangles(tris, { resolution: RES, bounds_m: BOUNDS, barrierClass: 2, pad: 1 });
if (!vox) { console.error('voxelization produced an empty grid'); process.exit(1); }
const b64 = (u8) => Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString('base64');
const voxJson = {
  name: NAME, source: basename(IN), created: new Date().toISOString(),
  dims: vox.dims, cell_m: vox.cell_m, extent: vox.extent, bounds_m: BOUNDS, n_solid: vox.n_solid,
  grid_b64: b64(vox.grid), inside_b64: b64(vox.inside),
};
const voxPath = join(OUT, NAME + '.vox.json');
writeFileSync(voxPath, JSON.stringify(voxJson));
console.log('      grid ' + vox.dims.join('×') + ' · cell ' + vox.cell_m.toFixed(3) + ' m · '
  + vox.n_solid.toLocaleString() + ' solid → ' + voxPath);

// Display-only decimation: error pinned to a fraction of the cell so surfaces never move a full voxel.
// (This mesh is never voxelized by the app — the grid above is. It is a cosmetic, faithful stand-in.)
const cell = vox.cell_m, err = (cell / 3) / Math.max(...BOUNDS);   // relative meshopt error < cell/3
const glbPath = join(OUT, NAME + '.display.glb');
const cmd = 'npx --yes @gltf-transform/cli optimize "' + IN + '" "' + glbPath
  + '" --compress draco --simplify-error ' + err.toFixed(5);
console.log('[3/3] display mesh (decimation error < cell/3 = ' + err.toFixed(5) + ' rel) …');
try {
  execSync(cmd, { stdio: 'ignore' });
  console.log('      → ' + glbPath);
} catch (e) {
  console.log('      (skipped — run it yourself:)\n      ' + cmd);
}
console.log('\nDone. Load ' + NAME + '.vox.json in the Simulation tab → physics solves on the full-CAD grid.');
