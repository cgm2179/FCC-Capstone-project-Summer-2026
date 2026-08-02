#!/usr/bin/env node
/*
 * convert_obj_to_glb.mjs — turn a heavy OBJ into a small Draco-compressed GLB.
 *
 * WHY
 * The 7th-floor OBJ is 329 MB of ASCII with 3.28 M triangles and 65,845 `usemtl`
 * switches — but only 253 distinct materials. Loading it in the browser is slow for two
 * separate reasons: the ASCII has to be downloaded and then PARSED synchronously (the
 * frozen tab), and OBJLoader makes one draw group per `usemtl` run, so the scene ends up
 * with ~65 k groups. Converting to glTF fixes both at once:
 *
 *   obj2gltf       OBJ -> GLB. Deduplicates materials by name, so 65,845 groups collapse
 *                  to 253 primitives (~260x fewer draw calls) and the geometry becomes a
 *                  binary buffer — no ASCII parse in the browser at all.
 *   gltf-pipeline  Draco-compress the geometry. 3.28 M triangles go from a ~100 MB binary
 *                  buffer to a few MB, decoded on the GPU by three's DRACOLoader.
 *
 * Net: a 329 MB OBJ becomes a ~10-20 MB GLB that loads in a second or two.
 *
 * TOOLCHAIN
 * All via `npx` so nothing is added to the repo's dependencies. Three stages, chained
 * through a temp dir:
 *   1. obj2gltf         OBJ -> glTF (separate resources)
 *   2. strip textures   remove images/textures/samplers and every material's texture refs
 *   3. gltf-transform   MERGE primitives by material, then Draco-compress -> final GLB
 *
 * WHY gltf-transform AND NOT gltf-pipeline
 * obj2gltf preserves the OBJ's group structure — the 7th floor comes out as ~50,000
 * meshes / 54,377 primitives, one per `usemtl` run. gltf-pipeline Dracos each of those
 * but does NOT merge them, so the GLB stays ~112 MB (all JSON overhead for 54 k
 * primitives) and still costs 54 k draw calls — barely better than the OBJ. gltf-transform
 * `optimize` runs dedup + join + weld, which collapses the 54,377 primitives to ~130
 * (one per distinct material), THEN Dracos: 112 MB -> ~10 MB, 54,377 draw calls -> ~130.
 * `--simplify false` keeps every triangle (lossless merge); `--palette false` keeps
 * materials distinct so the viewer's per-material hover/RF-attenuation tooltip still works.
 *
 * WHY STRIP TEXTURES
 * glTF only allows JPEG and PNG images, but real CAD exports carry .tif and .bmp maps
 * (this one has 70 textures, several of them .tif/.bmp). gltf-pipeline dies on them with
 * "Image data does not have valid header". For a massing/context model behind an RF
 * overlay the flat material base colours are what carry the read, not the surface photos,
 * so dropping textures makes the tool robust on ANY messy OBJ, shrinks the output, and
 * loses nothing that matters here. `--keep-textures` opts out (only safe if every map is
 * already JPEG/PNG).
 *
 * USAGE
 *   node Data/models/convert_obj_to_glb.mjs <input.obj> [output.glb] [--keep-textures]
 *   # default output: <input dir>/<input basename>.glb
 *
 * If Draco encoding runs out of memory on an even larger model, the meshopt path
 *   npx gltfpack -i plain.glb -o packed.glb -cc
 * also merges primitives and compresses; three then needs MeshoptDecoder instead of
 * DRACOLoader. Draco is the default here because it is what the viewer is wired for.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, statSync, mkdtempSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

const args = process.argv.slice(2);
const keepTextures = args.includes('--keep-textures');
const positional = args.filter((a) => !a.startsWith('--'));
const inPath = positional[0];
if (!inPath || !existsSync(inPath)) {
  console.error('usage: node convert_obj_to_glb.mjs <input.obj> [output.glb] [--keep-textures]');
  console.error(inPath ? `  not found: ${inPath}` : '  (no input given)');
  process.exit(2);
}
const outPath = positional[1]
  || join(dirname(resolve(inPath)), basename(inPath).replace(/\.obj$/i, '') + '.glb');

const mb = (n) => (n / 1048576).toFixed(1) + ' MB';
const inMB = mb(statSync(inPath).size);
console.log(`converting ${inPath} (${inMB})  ->  ${outPath}`
  + (keepTextures ? '  [keeping textures]' : '  [stripping textures]') + '\n');

function run(label, cmd, cmdArgs) {
  console.log(label);
  const t0 = Date.now();
  const r = spawnSync(cmd, cmdArgs, { stdio: 'inherit' });
  if (r.error) { console.error(`  ${cmd} failed: ${r.error.message}`); process.exit(3); }
  if (r.status) process.exit(r.status);
  console.log(`  …done in ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
}

// Remove every image/texture reference from a glTF JSON so gltf-pipeline never tries to
// read a .tif/.bmp map (which glTF cannot hold). Material base colours survive, so the
// model keeps its per-material look. Returns how many materials were de-textured.
function stripTextures(gltfPath) {
  const g = JSON.parse(readFileSync(gltfPath, 'utf8'));
  let touched = 0;
  for (const m of g.materials || []) {
    const pbr = m.pbrMetallicRoughness;
    let hit = false;
    if (pbr) {
      if (pbr.baseColorTexture) { delete pbr.baseColorTexture; hit = true; }
      if (pbr.metallicRoughnessTexture) { delete pbr.metallicRoughnessTexture; hit = true; }
    }
    for (const k of ['normalTexture', 'occlusionTexture', 'emissiveTexture']) {
      if (m[k]) { delete m[k]; hit = true; }
    }
    if (hit) touched++;
  }
  delete g.images;
  delete g.textures;
  delete g.samplers;
  writeFileSync(gltfPath, JSON.stringify(g));
  return touched;
}

const tmp = mkdtempSync(join(tmpdir(), 'glbconv-'));
const midGltf = join(tmp, 'model.gltf');   // separate resources (.bin + textures alongside)
try {
  run('[1/3] obj2gltf — OBJ -> glTF (merges the 65,845 groups by material)',
    'npx', ['--yes', 'obj2gltf@3', '-i', inPath, '-o', midGltf, '-s', '--checkTransparency']);

  if (!keepTextures) {
    console.log('[2/3] stripping textures (glTF allows only JPEG/PNG; CAD maps include .tif/.bmp)');
    const n = stripTextures(midGltf);
    console.log(`  de-textured ${n} materials; base colours kept\n`);
  } else {
    console.log('[2/3] keeping textures (--keep-textures)\n');
  }

  run('[3/3] gltf-transform — merge primitives by material, then Draco-compress',
    'npx', ['--yes', '@gltf-transform/cli@4', 'optimize', midGltf, outPath,
      '--compress', 'draco', '--simplify', 'false', '--palette', 'false',
      '--texture-compress', 'false']);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

if (!existsSync(outPath)) {
  console.error('  conversion produced no output file');
  process.exit(4);
}
const outMB = statSync(outPath).size;
console.log(`wrote ${outPath} (${mb(outMB)}) from ${inMB} OBJ`);
console.log(`  ${(statSync(inPath).size / statSync(outPath).size).toFixed(0)}x smaller · `
  + 'binary (no ASCII parse) · materials pre-merged · Draco geometry');
