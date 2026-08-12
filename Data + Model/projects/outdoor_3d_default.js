/* Built-in "Default" project — Outdoor · 3D (the NoMa DC render + city voxel physics).
 *
 * dashboard.js boots on the indoor base data as in every mode; the outdoor 3D studio
 * (Frontend/simulator/fw_studio3d_outdoor.html) + SIM3D outdoor assets provide the city.
 * `model` / `sim3d` / `iframes` / `elevation` document those sources for the 3D import +
 * elevation controls added in a later phase. */
(function () {
  window.PROJECTS = window.PROJECTS || {};
  window.PROJECTS['outdoor-3d-default'] = {
    id: 'outdoor-3d-default',
    label: 'Default (Outdoor · 3D)',
    environment: 'outdoor',
    dim: '3d',
    builtin: true,
    ai_ml: true,   // surrogate (ONNX) valid — trained on THIS default dataset
    data: window.PROJECT_DATA_INDOOR_BASE,
    model: {
      src: 'Data/models/NoMa_DC/Actual_Outdoor_Sim_3D_Render_NoMa_Washington_DC_3D.obj',
      label: 'NoMa DC render',
    },
    sim3d: [
      'SIM3D/web/sim_assets_outdoor_3d.js',
      'SIM3D/web/collision_outdoor_3d.js',
      'SIM3D/web/antenna_catalog_3d.js',
    ],
    iframes: { sim3d: 'Frontend/simulator/fw_studio3d_outdoor.html' },
    // Outdoor city buildings carry their own OSM heights; no single floor elevation.
    elevation: { mode: 'none' },
  };
})();
