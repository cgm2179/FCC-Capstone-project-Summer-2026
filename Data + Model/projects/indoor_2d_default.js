/* Built-in "Default" project — Indoor · 2D (the 7/7 indoor walk on the 7th-floor plan).
 * The base dataset the dashboard boots from. See registry.js for PROJECT_DATA_INDOOR_BASE. */
(function () {
  window.PROJECTS = window.PROJECTS || {};
  window.PROJECTS['indoor-2d-default'] = {
    id: 'indoor-2d-default',
    label: 'Default (Indoor · 2D)',
    environment: 'indoor',
    dim: '2d',
    builtin: true,
    ai_ml: true,   // surrogate (ONNX) valid — trained on THIS default dataset
    data: window.PROJECT_DATA_INDOOR_BASE,
  };
})();
