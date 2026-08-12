/* Built-in "Default" project — Indoor · 3D (the 7th-floor CAD model + voxel physics).
 *
 * dashboard.js still boots on the indoor base data (the Map Coverage tab keeps its 2D
 * heatmap under the 3D toggle). The CAD model is loaded by Frontend/2d-3d/viewer3d.js;
 * the 3D full-wave studio uses the SIM3D/web assets (eagerly loaded in the HTML). The
 * `model` / `sim3d` / `elevation` fields document those sources and seed the 3D import +
 * elevation controls added in a later phase. */
(function () {
  window.PROJECTS = window.PROJECTS || {};
  window.PROJECTS['indoor-3d-default'] = {
    id: 'indoor-3d-default',
    label: 'Default (Indoor · 3D)',
    environment: 'indoor',
    dim: '3d',
    builtin: true,
    ai_ml: true,   // surrogate (ONNX) valid — trained on THIS default dataset
    data: window.PROJECT_DATA_INDOOR_BASE,
    model: { src: 'Data/models/Indoor 7th floor v2.obj', label: 'Indoor 7th floor v2.obj' },
    sim3d: [
      'SIM3D/web/sim_assets_3d.js',
      'SIM3D/web/collision_3d.js',
      'SIM3D/web/antenna_catalog_3d.js',
    ],
    // Elevation: the 7th floor sits at street level in a 6-storey massing; ceiling 4.58 m
    // (from Data/floorplan_meta.js / SIM3D manifest). mode 'floor' = a single elevated slab.
    elevation: { mode: 'floor', floor_z_m: 0, ceiling_height_m: 4.58 },
  };
})();
