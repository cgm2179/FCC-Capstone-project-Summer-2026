/* aiml_gate.js — disable a Simulation studio's AI/ML solver when the parent passes ?aiml=0.
 *
 * A CREATED project carries ai_ml:false (its ONNX surrogate was trained on the DEFAULT dataset
 * only, so it's invalid here); project_loader.simStudioSrc() then appends ?aiml=0 to the studio
 * iframe URL. This drop-in disables every solver button marked data-aiml="1", activates the first
 * remaining (physics) solver, and shows a note. Runs on `load` so the studio's own segmented-
 * control wiring is already in place (clicking the fallback triggers its handler + solver state). */
(function () {
  if (new URLSearchParams(location.search).get('aiml') !== '0') return;
  window.addEventListener('load', function () {
    var seg = document.getElementById('solverSeg');
    if (!seg) return;
    var aiml = seg.querySelectorAll('button[data-aiml="1"]');
    if (!aiml.length) return;
    aiml.forEach(function (b) {
      b.disabled = true; b.classList.remove('on');
      b.style.opacity = 0.5; b.style.cursor = 'not-allowed';
      b.title = 'Surrogate trained on the default dataset — retrain per docs/ai-ml to enable for this project.';
    });
    if (!seg.querySelector('button.on:not([data-aiml="1"])')) {
      var fb = seg.querySelector('button:not([data-aiml="1"]):not([disabled])');
      if (fb) { fb.classList.add('on'); fb.click(); }        // activate a physics solver
    }
    if (!seg.parentNode.querySelector('.aiml-note')) {
      var note = document.createElement('div');
      note.className = 'note aiml-note'; note.style.color = '#b0771f';
      note.textContent = 'AI/ML surrogate disabled for this created project — using the physics solvers (see docs/ai-ml).';
      seg.parentNode.insertBefore(note, seg.nextSibling);
    }
  });
})();
