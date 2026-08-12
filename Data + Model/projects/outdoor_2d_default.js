/* Built-in "Default" project — Outdoor · 2D (the 7/24 NoMa DC walk on OSM).
 *
 * The dashboard still boots on the indoor base data (it initializes `records` etc. in
 * every mode); the outdoor experience is layered on top by landing.js — it swaps the
 * Map Coverage / Time Elapse panes to the OSM iframes below and lazily loads the outdoor
 * walk. `iframes` / `lazy` document those sources (landing.js currently hardcodes them;
 * they migrate to reading this manifest in a later phase). */
(function () {
  window.PROJECTS = window.PROJECTS || {};
  window.PROJECTS['outdoor-2d-default'] = {
    id: 'outdoor-2d-default',
    label: 'Default (Outdoor · 2D)',
    environment: 'outdoor',
    dim: '2d',
    builtin: true,
    ai_ml: true,   // surrogate (ONNX) valid — trained on THIS default dataset
    data: window.PROJECT_DATA_INDOOR_BASE,
    iframes: {
      map:   'Frontend/osm3d/outdoor_view.html',
      time:  'Frontend/osm3d/outdoor_timelapse.html',
      sim2d: 'Frontend/simulator/fw_studio2d_outdoor.html',
    },
    // Layered lazily by the iframes / dashboard.js (window.OUTDOOR_WALK), not at boot.
    lazy: [
      { role: 'outdoorWalk',       src: 'Data/outdoor_walk.js' },        // window.OUTDOOR_WALK
      { role: 'outdoorTimeseries', src: 'Data/outdoor_timeseries.js' },
      { role: 'nomaClutter',       src: 'Data/noma_clutter.js' },
      { role: 'nomaCity2d',        src: 'Data/noma_city2d.js' },
    ],
  };
})();
