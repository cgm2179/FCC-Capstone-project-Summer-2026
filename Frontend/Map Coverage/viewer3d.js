/* 3D Map-Coverage viewer (three.js) + RF coverage overlay — extracted
 * verbatim from the inline <script type="module"> of Frontend_Data_Display.html.
 * Stays a MODULE (uses import for three/OrbitControls). Reaches dashboard
 * state only through window.filtered / window.filteredTimeseries / window.idwGrid. */
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import * as georef from './georef.js';   // lon/lat → floor-plan pixel affine (rotation-correct)

    // Available floor models, newest first. To add an iteration, drop the
    // export folder in and add one line here (or just use "Load model file…"
    // for a one-off). URLs are encoded because the folders contain spaces.
    const MODELS = [
      // Full detailed 7th floor as a Draco-compressed GLB: converted once from the 329 MB
      // OBJ (3.28M tris, 65,845 usemtl groups) with Data/models/convert_obj_to_glb.mjs.
      // Binary, materials pre-merged, ~1-2 s to load instead of the OBJ's minute-long
      // download-then-parse freeze. This is the default; the OBJ iterations stay selectable.
      { name: '7th floor · full detail (fast GLB)',
        url: 'Data/models/7th_floor_full.glb' },
      // Second full-detail SketchUp export (3ff8432c…), converted the same way (328.8 MB OBJ,
      // 3.28M tris → 10.8 MB Draco GLB via convert_obj_to_glb.mjs).
      { name: '7th floor · full detail v2 (GLB)',
        url: 'Data/models/7th_floor_3ff8432c.glb' },
      { name: 'v2 · First Render (materials, OBJ)',
        url: 'Data/models/Indoor%207th%20floor%20v2%20First%20Render.obj/6afecb6b-b2f0-47e9-8488-d48cd283b15a.obj' },
      { name: 'v2 · Test 2 (OBJ)',
        url: 'Data/models/Indoor%207th%20floor%20v2.obj-HTML%20Test%202/12d74bc1-4044-458b-ae88-3c28b951399f.obj' },
      { name: 'v2 · Test 1 (OBJ)',
        url: 'Data/models/Indoor%207th%20floor%20v2.obj-HTML%20Test%201/8681a299-6227-4f6a-a55c-d149f0e525ff.obj' },
    ];
    const DEFAULT_MODEL_URL = MODELS[0].url;

    const wrap      = document.getElementById('plot3dWrap');
    const container = document.getElementById('plot3d');
    const matTip    = document.getElementById('matTip');
    const statusEl  = document.getElementById('viewer3dStatus');
    const nameEl    = document.getElementById('viewer3dName');
    const hintEl    = document.getElementById('mapModeHint');
    const plot2d    = document.getElementById('plot');
    const mode2dBtn = document.getElementById('mapMode2dBtn');
    const mode3dBtn = document.getElementById('mapMode3dBtn');
    const fileInput = document.getElementById('model3dFile');
    const resetBtn  = document.getElementById('view3dReset');
    const wireBtn   = document.getElementById('view3dWire');
    const upBtn     = document.getElementById('view3dUp');
    const modelSelect = document.getElementById('modelSelect');

    let renderer, scene, camera, controls, headlight;
    let currentModel = null;
    let initialized = false;
    let wireframe = false;
    // RF coverage contour overlay state.
    let rfActive = false;         // is the overlay switched on?
    let rfMesh = null;            // the textured contour plane (added to scene)
    let rfPoints = null;          // optional measurement-dot cloud
    let floorInfo = null;         // { minX,maxX,minZ,maxZ,topY,height } in world coords
    // Indoor 3D real-world context: the georeferenced OSM street map on the ground, the 7th
    // floor lifted onto a 6-storey building massing that stands on it, and the surveyed base
    // stations around it — so the model reads as one floor of a real building on the real map
    // rather than floating in space. showOsmGround / showShell toggle the map and the massing.
    let contextGroup = null, showShell = true, showOsmGround = true;
    let lastGroundY = 0;          // street level of the current building context (for framing)
    let lastUpm = 1, lastFloor7Y = 0;         // real-world scale + 7th-floor slab Y (for the slice)
    let modelOpacity = 1, modelClipH = 4.5;   // model display: opacity (0–1) + slice height (m)
    // One shared world-space clip plane (normal points down) = the horizontal "slice"; its
    // constant is the cut height. Assigned once to every model material, so moving the slider
    // just changes the constant and re-clips. constant 1e9 ⇒ above the model ⇒ nothing clipped.
    const modelClipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1e9);

    function setStatus(msg, isError) {
      statusEl.textContent = msg || '';
      statusEl.classList.toggle('error', !!isError);
    }

    // ---- load-progress bar ----
    // A big OBJ used to give no feedback at all — the tab just sat there while the file
    // downloaded and then parsed, and it read as a crash. This is a slim bar injected
    // under the status line (no HTML/CSS churn): a real fill during download, then an
    // indeterminate "parsing" stripe for the one phase that genuinely blocks the main
    // thread and cannot be sped up in the browser (synchronous OBJLoader.parse).
    let progressEl = null, progressBar = null;
    function ensureProgress() {
      if (progressEl || !statusEl || !statusEl.parentNode) return progressEl;
      progressEl = document.createElement('div');
      progressEl.style.cssText = 'height:4px;margin-top:4px;border-radius:2px;'
        + 'background:rgba(0,0,0,0.10);overflow:hidden;display:none';
      progressBar = document.createElement('div');
      progressBar.style.cssText = 'height:100%;width:0;border-radius:2px;'
        + 'background:#0a7f7a;transition:width .15s linear';
      progressEl.appendChild(progressBar);
      statusEl.parentNode.insertBefore(progressEl, statusEl.nextSibling);
      // one keyframe for the indeterminate parse stripe, injected once
      if (!document.getElementById('v3d-prog-kf')) {
        const st = document.createElement('style');
        st.id = 'v3d-prog-kf';
        st.textContent = '@keyframes v3dprog{0%{transform:translateX(-100%)}'
          + '100%{transform:translateX(400%)}}';
        document.head.appendChild(st);
      }
      return progressEl;
    }
    const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB';
    function setProgress(loaded, total) {
      ensureProgress();
      if (!progressEl) return;
      progressEl.style.display = 'block';
      progressBar.style.animation = '';
      const pct = total ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
      progressBar.style.width = (total ? pct : 8) + '%';
    }
    // Indeterminate stripe for the synchronous parse — a determinate bar would lie, since
    // nothing reports progress during OBJLoader.parse.
    function setParsing() {
      ensureProgress();
      if (!progressEl) return;
      progressEl.style.display = 'block';
      progressBar.style.width = '30%';
      progressBar.style.animation = 'v3dprog 1s linear infinite';
    }
    function hideProgress() {
      if (progressEl) { progressEl.style.display = 'none'; progressBar.style.animation = ''; }
    }

    // A three.js ProgressEvent handler for `name`. Drives the bar during download and, on
    // reaching 100 %, flips to the parse notice so the freeze is labelled rather than
    // silent. `isObj` controls that notice — binary formats (GLB) do not stall on parse.
    function progressHandler(name, isObj) {
      return (e) => {
        if (!e || !e.lengthComputable) { setProgress(0, 0); return; }
        setProgress(e.loaded, e.total);
        if (e.loaded >= e.total) {
          if (isObj) { setStatus('Parsing ' + fmtMB(e.total)
            + ' — large models may pause the tab briefly…'); setParsing(); }
          else setStatus('Decoding ' + name + ' …');
        } else {
          setStatus('Loading ' + name + ' · ' + fmtMB(e.loaded) + ' / ' + fmtMB(e.total));
        }
      };
    }

    // Lazily build the scene, lights, grid and render loop on first 3D view so
    // three.js + the model are never downloaded unless the user asks for them.
    function initScene() {
      if (initialized) return true;
      const w = container.clientWidth || 900;
      const h = container.clientHeight || 760;
      try {
        renderer = new THREE.WebGPURenderer({ antialias: true });
      } catch (e) {
        setStatus('3D rendering is unavailable in this browser, so the model cannot render.', true);
        return false;
      }
      initialized = true;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.localClippingEnabled = true;   // per-material slice clipping (model only)
      container.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0xeef3f1);

      camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 500000);
      camera.position.set(300, 300, 300);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.screenSpacePanning = true;

      scene.add(new THREE.HemisphereLight(0xffffff, 0x8d9b96, 1.1));
      scene.add(new THREE.AmbientLight(0xffffff, 0.45));
      const key = new THREE.DirectionalLight(0xffffff, 1.7);
      key.position.set(1, 2, 1.5);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.6);
      fill.position.set(-1.2, 1, -1);
      scene.add(fill);
      // Headlight follows the camera (updated in the loop) so whichever side
      // the user orbits to is always lit — keeps dark/under-lit exports legible.
      headlight = new THREE.DirectionalLight(0xffffff, 0.85);
      scene.add(headlight);

      addGrid(2000);

      setupHover();
      window.addEventListener('resize', onResize);

      // WebGPU initialises its backend asynchronously (and transparently falls
      // back to a WebGL2 backend when WebGPU isn't available). Everything above
      // is built synchronously, so initScene() still returns true immediately;
      // we only start painting — and prefilter the environment map, which needs
      // a live backend — once init() resolves. Model loading kicked off by the
      // caller runs in parallel and is on screen by the first painted frame.
      renderer.init().then(() => {
        // Image-based lighting so PBR materials (glass, brushed steel) get real
        // reflections. Best-effort: if unsupported, materials still render.
        import('three/addons/environments/RoomEnvironment.js').then(({ RoomEnvironment }) => {
          const pmrem = new THREE.PMREMGenerator(renderer);
          scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        }).catch(() => {});

        renderer.setAnimationLoop(() => {
          // Only paint when the 3D pane is actually on screen (map tab active +
          // 3D mode). offsetParent is null whenever an ancestor is display:none.
          if (wrap.offsetParent === null) return;
          controls.update();
          if (headlight) headlight.position.copy(camera.position); // track the camera
          renderer.render(scene, camera);
        });
      }).catch((err) => {
        setStatus('3D backend failed to initialize (needs WebGPU or WebGL2): '
          + (err && err.message || err), true);
      });
      return true;
    }

    // Hover tooltip: raycast the building model under the cursor and show what
    // the polygon's material maps to (Glass / Stainless steel / BLANK) plus its
    // RF attenuation. Only the CAD model is tested — never the RF overlay.
    const hoverRay = new THREE.Raycaster();
    const hoverNDC = new THREE.Vector2();
    function setupHover() {
      const el = renderer.domElement;
      el.addEventListener('pointermove', (e) => {
        if (!currentModel) { matTip.style.display = 'none'; return; }
        const r = el.getBoundingClientRect();
        hoverNDC.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        hoverNDC.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        hoverRay.setFromCamera(hoverNDC, camera);
        const hits = hoverRay.intersectObject(currentModel, true);
        if (!hits.length) { matTip.style.display = 'none'; return; }
        const h = hits[0];
        let mat = h.object.material;
        if (Array.isArray(mat)) mat = mat[(h.face && h.face.materialIndex) || 0] || mat[0];
        const rawName = (mat && mat.name) || '';
        const info = materialInfo(rawName);
        matTip.innerHTML = `<b>${info.label}</b> · ${info.atten.toFixed(2)} dB`
          + `<br><span class="mt-raw">${rawName || '—'}</span>`;
        matTip.style.display = 'block';
        matTip.style.left = (e.clientX - r.left + 14) + 'px';
        matTip.style.top  = (e.clientY - r.top + 14) + 'px';
      });
      el.addEventListener('pointerleave', () => { matTip.style.display = 'none'; });
    }

    function addGrid(size) {
      const old = scene.getObjectByName('__grid');
      if (old) { scene.remove(old); old.geometry.dispose(); old.material.dispose(); }
      const grid = new THREE.GridHelper(size, 40, 0xbfcdc8, 0xdae4e0);
      grid.name = '__grid';
      scene.add(grid);
    }

    function onResize() {
      if (!renderer) return;
      const w = container.clientWidth || 900;
      const h = container.clientHeight || 760;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }

    // Post-load cleanup so exports of varying quality all render legibly:
    //  • compute vertex normals when the file has none — lit materials render
    //    solid black without normals (common with bare extrude/CAD exports);
    //  • force double-sided so SketchUp's one-sided faces don't vanish;
    //  • undo any fully-transparent materials some exporters emit.
    function normalizeModel(object) {
      object.traverse((o) => {
        if (!o.isMesh) return;
        if (o.geometry && !o.geometry.getAttribute('normal')) {
          o.geometry.computeVertexNormals();
        }
        if (!o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => {
          m.side = THREE.DoubleSide;
          if (m.transparent && m.opacity === 0) { m.opacity = 1; m.transparent = false; }
        });
      });
    }

    // ---- Material library: SketchUp material name -> physical look + RF
    // attenuation. EDIT THESE dB VALUES to your measured/spec figures. Any
    // material that matches nothing falls through to BLANK / 0.00 dB.
    const MATERIAL_LIB = [
      { test: /glass/i,                 label: 'Glass',           atten: 4.0 },
      { test: /steel|stainless|metal/i, label: 'Stainless steel', atten: 30.0 },
    ];
    const BLANK_MAT = { label: 'BLANK', atten: 0.00 };
    function materialInfo(name) {
      if (name) for (const e of MATERIAL_LIB) if (e.test.test(name)) return e;
      return BLANK_MAT;
    }

    // Give the recognised SketchUp materials a real look: glass becomes
    // transmissive (see-through, reflective), stainless steel becomes metallic
    // (keeping its brushed texture). Everything else keeps its assigned colour.
    // Material .name is preserved so the hover lookup still works.
    function applyRealMaterials(object) {
      object.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        const upgrade = (m) => {
          const name = (m && m.name) || '';
          try {
            if (/glass/i.test(name)) {
              const g = new THREE.MeshPhysicalMaterial({
                color: (m.color && m.color.clone()) || new THREE.Color(0xbcd2df),
                metalness: 0.0, roughness: 0.06, transmission: 0.82, thickness: 4,
                ior: 1.5, transparent: true, opacity: 1, side: THREE.DoubleSide,
                envMapIntensity: 1.0 });
              g.name = name; m.dispose && m.dispose(); return g;
            }
            if (/steel|stainless|metal/i.test(name)) {
              const s = new THREE.MeshStandardMaterial({
                map: m.map || null, color: m.map ? new THREE.Color(0xffffff)
                  : ((m.color && m.color.clone()) || new THREE.Color(0xc2c8cc)),
                metalness: 0.92, roughness: 0.34, side: THREE.DoubleSide, envMapIntensity: 1.0 });
              s.name = name; return s;   // keep the texture, so don't dispose m.map
            }
          } catch (e) { /* fall through to original */ }
          return m;
        };
        o.material = Array.isArray(o.material) ? o.material.map(upgrade) : upgrade(o.material);
      });
    }

    // Wireframe: the WebGPU backend ignores material.wireframe, so instead overlay
    // an EdgesGeometry LineSegments on each mesh (cached in userData.__edges) and
    // dim the solid — a reversible, WebGPU-safe see-through wireframe.
    function applyWireframe(object, on) {
      object.traverse((o) => {
        if (!o.isMesh || !o.geometry) return;
        const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
        if (on) {
          if (!o.userData.__edges) {
            const edges = new THREE.LineSegments(
              new THREE.EdgesGeometry(o.geometry, 30),
              new THREE.LineBasicMaterial({ color: 0x1f2a2a, transparent: true, opacity: 0.85 }));
            edges.renderOrder = 4;
            o.add(edges);                                  // child inherits the mesh transform
            o.userData.__edges = edges;
          }
          mats.forEach((m) => {
            if (m.userData.__wfSaved == null) {
              m.userData.__wfSaved = { transparent: m.transparent, opacity: m.opacity };
              m.transparent = true; m.opacity = Math.min(m.opacity, 0.12); m.needsUpdate = true;
            }
          });
        } else {
          if (o.userData.__edges) {
            o.remove(o.userData.__edges);
            o.userData.__edges.geometry.dispose();
            o.userData.__edges.material.dispose();
            o.userData.__edges = null;
          }
          mats.forEach((m) => {
            if (m.userData.__wfSaved) {
              m.transparent = m.userData.__wfSaved.transparent;
              m.opacity = m.userData.__wfSaved.opacity;
              m.userData.__wfSaved = null; m.needsUpdate = true;
            }
          });
        }
      });
    }

    // Model display (Opacity + Slice). WebGPU (this three build) does NOT pick up in-place
    // changes to material.opacity / transparent / clippingPlanes — only a *fresh* material
    // rebuilds the render pipeline. So we keep each mesh's pristine base material and re-clone
    // from it on every change, baking in the opacity and the shared slice clip plane. Same-
    // config clones reuse the cached shader (just a new uniform), so it stays cheap; throttled
    // via requestModelDisplay so dragging a slider doesn't churn every frame.
    function applyModelDisplay(object) {
      if (!object) return;
      const clip = modelClipH < 4.5;
      updateModelClip();
      object.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        if (o.userData.__wfSaved != null) return;          // wireframe owns the material
        if (!o.userData.__baseMat) o.userData.__baseMat = o.material;   // pristine original(s)
        const base = o.userData.__baseMat, baseArr = Array.isArray(base) ? base : [base];
        if (o.userData.__dispMat) {                         // dispose the previous clones
          (Array.isArray(o.userData.__dispMat) ? o.userData.__dispMat : [o.userData.__dispMat]).forEach((m) => m.dispose());
        }
        const fresh = baseArr.map((bm) => {
          const m = bm.clone();
          const baseOp = (bm.opacity == null ? 1 : bm.opacity);
          m.opacity = baseOp * modelOpacity;
          m.transparent = bm.transparent || modelOpacity < 0.999;
          m.depthWrite = m.opacity >= 0.999;
          m.side = THREE.DoubleSide;                        // so a slice shows the interior faces
          m.clippingPlanes = clip ? [modelClipPlane] : null;
          m.clipShadows = true;
          return m;
        });
        o.material = Array.isArray(base) ? fresh : fresh[0];
        o.userData.__dispMat = Array.isArray(base) ? fresh : fresh[0];
      });
    }

    // Position the slice plane (metres above the 7th-floor slab) + update the label. At full
    // height the plane sits above the model so nothing is clipped (constant 1e9).
    function updateModelClip() {
      const full = modelClipH >= 4.5;
      modelClipPlane.constant = full ? 1e9 : (lastFloor7Y + modelClipH * lastUpm);
      const lbl = document.getElementById('view3dSliceVal');
      if (lbl) lbl.textContent = full ? 'Slice: off' : `Slice: ${modelClipH.toFixed(1)} m`;
    }

    // Throttle the rebuild so dragging a slider stays smooth (≤ ~1 rebuild / 90 ms + trailing).
    let __dispPending = false, __dispLast = 0;
    function requestModelDisplay() {
      if (!currentModel) return;
      const now = performance.now();
      if (now - __dispLast > 90) { __dispLast = now; applyModelDisplay(currentModel); }
      else if (!__dispPending) {
        __dispPending = true;
        setTimeout(() => { __dispPending = false; __dispLast = performance.now(); applyModelDisplay(currentModel); }, 90);
      }
    }

    function disposeModel(object) {
      object.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        // dispose the rendered material(s) and the pristine base kept for the display controls
        const all = [];
        if (o.material) all.push(...(Array.isArray(o.material) ? o.material : [o.material]));
        if (o.userData && o.userData.__baseMat) all.push(...(Array.isArray(o.userData.__baseMat) ? o.userData.__baseMat : [o.userData.__baseMat]));
        all.forEach((m) => {
          if (!m || !m.dispose) return;
          for (const k in m) { const v = m[k]; if (v && v.isTexture) v.dispose(); }
          m.dispose();
        });
      });
    }

    // Recenter the model over the origin (base on the grid) and frame the
    // camera to its bounding box. Idempotent — safe to call on reset / flip.
    function frameObject(object) {
      object.position.set(0, 0, 0);
      object.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(object);
      if (box.isEmpty()) { setStatus('Model loaded but contains no geometry.', true); return; }
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      object.position.set(-center.x, -box.min.y, -center.z);

      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      camera.near = Math.max(0.01, maxDim / 1000);
      camera.far = maxDim * 100;
      const d = maxDim * 1.5;
      camera.position.set(d * 0.9, d * 0.75, d * 0.9);
      camera.updateProjectionMatrix();
      controls.target.set(0, size.y * 0.4, 0);
      controls.maxDistance = maxDim * 10;
      controls.update();
      addGrid(Math.ceil(maxDim * 1.4));

      setStatus(`Loaded · bounding box ${size.x.toFixed(0)} × ${size.y.toFixed(0)} × ${size.z.toFixed(0)} units · `
        + 'drag to orbit · scroll to zoom · right-drag to pan.');
    }

    // Lay a floor plan flat (three.js is Y-up). Up-axis conventions vary by
    // exporter — SketchUp writes .dae as Z-up (ColladaLoader fixes it) but
    // remaps .obj to Y-up, while STL/3DS carry no convention at all. Rather
    // than guess per format, detect it from the geometry: a floor's vertical
    // axis is its *thinnest* extent, so rotate whichever axis that is up to Y.
    // The "Flip up-axis" button overrides this if a model isn't floor-shaped.
    function autoOrientUp(object) {
      object.rotation.set(0, 0, 0);
      object.updateMatrixWorld(true);
      const s = new THREE.Box3().setFromObject(object).getSize(new THREE.Vector3());
      const min = Math.min(s.x, s.y, s.z);
      if (min === s.z)      object.rotation.x = -Math.PI / 2; // Z-up  -> Y-up
      else if (min === s.x) object.rotation.z =  Math.PI / 2; // X-up  -> Y-up
      // else already Y-up (thinnest axis is Y) — leave as-is.
    }

    function placeModel(object, label) {
      hideProgress();               // the model is here — download/parse both done
      if (currentModel) { scene.remove(currentModel); disposeModel(currentModel); }
      clearRF();
      normalizeModel(object);
      applyRealMaterials(object);   // glass/steel -> physical look; names kept for hover
      autoOrientUp(object);
      applyWireframe(object, wireframe);
      scene.add(object);
      currentModel = object;
      frameObject(object);
      updateFloorInfo(object);
      addBuildingContext(object);
      frameToBuilding(object);      // re-frame to include the 6-storey drop to the street
      applyModelDisplay(object);    // opacity + slice clip (needs the scale set above)
      if (rfActive) buildRFContour();
      if (label) nameEl.textContent = label;
    }

    // After the building context exists, frame the whole 7-storey building (street → 7th-floor
    // roof) instead of just the thin floor slab, and open up the zoom range so the user can
    // pull back to the base stations and the outdoor-walk extent on the expanded OSM ground.
    function frameToBuilding(object) {
      if (!floorInfo || !camera || !controls) return;
      const box = new THREE.Box3().setFromObject(object);
      const cx = (floorInfo.minX + floorInfo.maxX) / 2, cz = (floorInfo.minZ + floorInfo.maxZ) / 2;
      const w = floorInfo.maxX - floorInfo.minX, d = floorInfo.maxZ - floorInfo.minZ;
      const cy = (lastGroundY + box.max.y) / 2, h = box.max.y - lastGroundY;
      const span = Math.max(w, d, h) || 1;
      const dist = span * 1.5;
      camera.near = Math.max(0.01, span / 1000);
      camera.far = span * 400;                       // OSM ground + base stations extend far
      camera.position.set(cx + dist * 0.85, cy + dist * 0.7, cz + dist * 0.85);
      camera.updateProjectionMatrix();
      controls.target.set(cx, cy, cz);
      controls.maxDistance = span * 60;              // room to reach the walk boundary
      controls.update();
    }

    // Locate the floor in world space so the RF overlay registers to it.
    // When the model carries the 1150×515 floor-plan image on a large textured
    // mesh (Test 1/2), that mesh's X-Z bounds are the exact px/py frame — use
    // it. Otherwise (e.g. First Render has only a small steel texture, no floor
    // image) fall back to the whole model's footprint, which is the same frame
    // since every export traces the same floor plan.
    function updateFloorInfo(model) {
      // frameObject just repositioned the model; flush world matrices so the
      // bounding boxes below are in the recentred coordinates actually shown.
      model.updateMatrixWorld(true);
      const full = new THREE.Box3().setFromObject(model);
      const footprint = (full.max.x - full.min.x) * (full.max.z - full.min.z) || 1;
      const height = (full.max.y - full.min.y) || 1;

      let floorMesh = null, maxArea = 0;
      model.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        if (!mats.some((m) => m && m.map)) return;
        const b = new THREE.Box3().setFromObject(o);
        const area = (b.max.x - b.min.x) * (b.max.z - b.min.z);
        if (area > maxArea) { maxArea = area; floorMesh = o; }
      });

      // Only trust a textured mesh as "the floor" if it spans most of the plan.
      if (floorMesh && maxArea >= footprint * 0.4) {
        const b = new THREE.Box3().setFromObject(floorMesh);
        floorInfo = { minX: b.min.x, maxX: b.max.x, minZ: b.min.z, maxZ: b.max.z,
                      topY: b.max.y, height };
      } else {
        floorInfo = { minX: full.min.x, maxX: full.max.x, minZ: full.min.z, maxZ: full.max.z,
                      topY: full.min.y + Math.max(2, height * 0.02), height };
      }
    }

    function clearBuildingContext() {
      if (!contextGroup) return;
      scene.remove(contextGroup);
      contextGroup.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
        mats.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
      });
      contextGroup = null;
    }

    // Set the 7th-floor model in real-world context: it sits on top of 6 storeys of building
    // (4.5 m each), which stand on the OSM street map (addOsmGround, dropped to street level),
    // with the surveyed base stations placed around it as boxes at their real building height
    // topped by a red antenna dot. Heights use model-units-per-metre derived from the
    // georeference; all horizontal placement goes through px2x/py2z (Pseudo-Mercator), so
    // everything shares the one rectangular grid. "Building" (showShell) toggles the massing.
    function addBuildingContext(model) {
      clearBuildingContext();
      if (!scene || !floorInfo) return;
      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model);
      if (!isFinite(box.min.y) || box.isEmpty()) return;
      const g = new THREE.Group(); g.name = '__buildingContext';

      // Model units per metre, from the georeference. Horizontal is exact (plan px → model via
      // floorInfo); height reuses the horizontal average — the SketchUp export is ~uniform
      // (~57 u/m here, so the ~249-unit-tall model ≈ one 4.3 m storey).
      const mpp = (window.FLOORPLAN_META || {}).meters_per_px || 0.0679;
      const fw = floorInfo.maxX - floorInfo.minX, fd = floorInfo.maxZ - floorInfo.minZ;
      const upmX = fw / (FLOOR_W * mpp), upmZ = fd / (FLOOR_H * mpp), upm = (upmX + upmZ) / 2;

      // 7th floor on top of 6 storeys standing on the street.
      const FLOOR_M = 4.5, FLOORS_UNDER = 6;
      const floor7Y = box.min.y;                              // 7th-floor slab (model base)
      const groundY = floor7Y - FLOORS_UNDER * FLOOR_M * upm; // street / OSM level
      lastGroundY = groundY; lastUpm = upm; lastFloor7Y = floor7Y;
      const cx = (floorInfo.minX + floorInfo.maxX) / 2, cz = (floorInfo.minZ + floorInfo.maxZ) / 2;

      if (showOsmGround) addOsmGround(g, groundY);

      if (showShell) {                                        // the 6 storeys under the 7th floor
        const h = floor7Y - groundY;
        const massing = new THREE.Mesh(
          new THREE.BoxGeometry(fw, h, fd),
          new THREE.MeshStandardMaterial({ color: 0xaeb9c4, transparent: true, opacity: 0.28,
            roughness: 0.92, depthWrite: false }));
        massing.position.set(cx, (groundY + floor7Y) / 2, cz);
        g.add(massing);
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(fw, h, fd)),
          new THREE.LineBasicMaterial({ color: 0x6b8299, transparent: true, opacity: 0.35 }));
        edges.position.copy(massing.position); g.add(edges);
        for (let fl = 1; fl < FLOORS_UNDER; fl++) {           // faint storey outlines
          const line = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(fw, 0.0001, fd)),
            new THREE.LineBasicMaterial({ color: 0x6b8299, transparent: true, opacity: 0.16 }));
          line.position.set(cx, groundY + fl * FLOOR_M * upm, cz); g.add(line);
        }
      }

      if (showOsmGround) addBaseStations(g, groundY, upm, upmX, upmZ);

      scene.add(g);
      contextGroup = g;
    }

    // OSM basemap ground on the SAME plane as the 7th-floor model box, georeferenced by the
    // floor-plan GCP affine (EPSG:3857 / Pseudo-Mercator) — so the box sits on the real map
    // and each side lines up with its street (L St NE etc.). Live OSM tiles are warped into
    // the plan's pixel frame: the whole px→mercator→tile-pixel chain is affine, so a single
    // ctx.setTransform per tile bakes in the ~-7.33° map rotation (same idea as the committed
    // 2D static raster, but wider and live). Axis-aligned to the model like the RF overlay
    // (px2x/py2z), so no plane yaw is needed. Falls back to the embedded INDOOR_BASEMAP
    // footprint raster if the affine is missing or the tiles can't load (offline). Other
    // buildings / elevation are intentionally ignored — this is just a ground reference.
    function addOsmGround(g, groundY) {
      if (!floorInfo) return;
      const yGround = (groundY != null) ? groundY : floorInfo.topY + 0.05;   // street level
      const mpp = (window.FLOORPLAN_META || {}).meters_per_px || 0.0679;

      // Drape a texture over the plan-px rectangle [pxMin,pxMax]×[pyMin,pyMax] on the floor.
      const placePlane = (tex, pxMin, pxMax, pyMin, pyMax) => {
        if (g.parent !== scene) { if (tex && tex.dispose) tex.dispose(); return; }
        const old = g.getObjectByName('__osmGround');
        if (old) { g.remove(old); old.geometry.dispose(); if (old.material.map) old.material.map.dispose(); old.material.dispose(); }
        const w = px2x(pxMax) - px2x(pxMin), d = py2z(pyMax) - py2z(pyMin);
        const mesh = new THREE.Mesh(
          new THREE.PlaneGeometry(w, d),
          new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.96, depthWrite: false }));
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set((px2x(pxMin) + px2x(pxMax)) / 2, yGround, (py2z(pyMin) + py2z(pyMax)) / 2);
        mesh.renderOrder = 1;
        mesh.name = '__osmGround';
        g.add(mesh);
      };

      const useEmbedded = () => {
        if (!(window.INDOOR_BASEMAP && window.INDOOR_BASEMAP.source)) return;
        const tex = new THREE.TextureLoader().load(window.INDOOR_BASEMAP.source);
        tex.colorSpace = THREE.SRGBColorSpace;
        placePlane(tex, 0, FLOOR_W, 0, FLOOR_H);             // raster is exactly the plan frame
      };

      const A = (window.FLOORPLAN_META || {}).affine_px_to_mercator;
      if (!A) { useEmbedded(); return; }

      // Extent (plan px): the plan frame ∪ the whole outdoor walk (matches Outdoor 3D's crop).
      const { pxMin, pxMax, pyMin, pyMax } = osmGroundExtent();
      const merc = (px, py) => [A[0][0]*px + A[0][1]*py + A[0][2], A[1][0]*px + A[1][1]*py + A[1][2]];
      const cs = [merc(pxMin, pyMin), merc(pxMax, pyMin), merc(pxMax, pyMax), merc(pxMin, pyMax)];
      const mxMin = Math.min(...cs.map(p => p[0])), mxMax = Math.max(...cs.map(p => p[0]));
      const myMin = Math.min(...cs.map(p => p[1])), myMax = Math.max(...cs.map(p => p[1]));

      const R = 20037508.342789244, worldPx = (z) => 256 * Math.pow(2, z);
      const gW = (z) => (mxMax - mxMin) / (2 * R) * worldPx(z);
      const gH = (z) => (myMax - myMin) / (2 * R) * worldPx(z);
      // Highest zoom that keeps the mosaic sane (≤ ~2400 px a side and ≤ 63 tiles) — as the
      // extent grows to the walk boundary this steps the zoom out automatically.
      let z = 19;
      while (z > 12 && (gW(z) > 2400 || gH(z) > 2400 ||
                        (Math.ceil(gW(z) / 256) + 1) * (Math.ceil(gH(z) / 256) + 1) > 63)) z--;
      const WP = worldPx(z), TM = 2 * R / WP;                // mercator metres / tile pixel
      const merc2gx = (mx) => (mx + R) / (2 * R) * WP;
      const merc2gy = (my) => (R - my) / (2 * R) * WP;
      const nTiles = Math.pow(2, z);
      const tx0 = Math.floor(merc2gx(mxMin) / 256), tx1 = Math.floor(merc2gx(mxMax) / 256);
      const ty0 = Math.floor(merc2gy(myMax) / 256), ty1 = Math.floor(merc2gy(myMin) / 256);

      // Output canvas = the extent rectangle (aspect-preserving); wider extent → bigger canvas.
      const CW = Math.min(4096, Math.max(2048, Math.round((pxMax - pxMin) * mpp / 0.4)));
      const CH = Math.max(1, Math.round(CW * (pyMax - pyMin) / (pxMax - pxMin)));
      const canvas = document.createElement('canvas'); canvas.width = CW; canvas.height = CH;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#e9edf0'; ctx.fillRect(0, 0, CW, CH);

      // merc → canvas-pixel affine M (via the inverse of the px→merc 2×2, then plan→canvas).
      const a = A[0][0], b = A[0][1], cc = A[1][0], dd = A[1][1], e = A[0][2], f = A[1][2];
      const det = a * dd - b * cc || 1e-9;
      const ia = dd / det, ib = -b / det, ic = -cc / det, idd = a / det;
      const sX = CW / (pxMax - pxMin), sY = CH / (pyMax - pyMin);
      const M = {
        a: sX * ia, c: sX * ib, e: sX * (-ia * e - ib * f - pxMin),
        b: sY * ic, d: sY * idd, f: sY * (-ic * e - idd * f - pyMin)
      };

      // Build the tile job list, then fetch PROGRESSIVELY and THROTTLED. OSM's tile server
      // rate-limits big bursts (fetching a whole ~50-tile mosaic at once 429s a block of them
      // and leaves the ground sliced), so cap concurrency and retry each tile a couple of
      // times. The plane is placed on the first tile that lands and refreshed as the rest
      // arrive, so the map fills in to a full rectangle instead of blocking on the full set.
      const jobs = [];
      for (let tx = tx0; tx <= tx1; tx++) for (let ty = ty0; ty <= ty1; ty++) {
        if (ty < 0 || ty >= nTiles) continue;
        jobs.push({ nx: ((tx % nTiles) + nTiles) % nTiles, ty,
                    mx0: tx * 256 * TM - R, my0: R - ty * 256 * TM });
      }
      let tex = null, any = false, remaining = jobs.length, idx = 0, active = 0, done = false;
      const subd = ['a', 'b', 'c'], MAX_ACTIVE = 5;
      const finish = () => { if (done) return; done = true; if (!any) useEmbedded(); };
      const showProgress = () => {
        if (!tex) {
          tex = new THREE.CanvasTexture(canvas);
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
          placePlane(tex, pxMin, pxMax, pyMin, pyMax);
        } else tex.needsUpdate = true;
      };
      const drawTile = (img, j) => {   // tile-image (u,v) → canvas affine = M ∘ (tile-px → merc)
        ctx.setTransform(M.a * TM, M.b * TM, -M.c * TM, -M.d * TM,
                         M.a * j.mx0 + M.c * j.my0 + M.e, M.b * j.mx0 + M.d * j.my0 + M.f);
        ctx.drawImage(img, -0.5, -0.5, 257, 257);            // slight bleed hides seams
      };
      const startJob = (j, attempt) => {
        const img = new Image(); img.crossOrigin = 'anonymous';
        img.onload = () => { any = true; drawTile(img, j); showProgress(); active--; if (--remaining === 0) finish(); pump(); };
        img.onerror = () => {
          if (attempt < 2) { setTimeout(() => startJob(j, attempt + 1), 350 + 250 * attempt); return; }
          active--; if (--remaining === 0) finish(); pump();
        };
        img.src = `https://${subd[(j.nx + j.ty) % 3]}.tile.openstreetmap.org/${z}/${j.nx}/${j.ty}.png`;
      };
      const pump = () => { while (active < MAX_ACTIVE && idx < jobs.length) { active++; startJob(jobs[idx++], 0); } };
      if (jobs.length === 0) finish(); else pump();
    }

    // Rectangle (plan px) the OSM ground should cover: the floor plan ∪ the whole outdoor walk
    // + 5% pad — matching the Outdoor-3D view's walkBounds() so BOTH 3D basemaps crop to the
    // same size. Base stations are still PLACED (addBaseStations) but are NOT counted in the
    // extent: the far towers (Gonzaga, JHU-APL) would otherwise blow the crop well past the
    // walk. Memoized — depends only on the static datasets.
    let _osmExtent = null;
    function osmGroundExtent() {
      if (_osmExtent) return _osmExtent;
      let pxMin = 0, pxMax = FLOOR_W, pyMin = 0, pyMax = FLOOR_H;
      const grow = (px, py) => {
        if (!Number.isFinite(px) || !Number.isFinite(py)) return;
        if (px < pxMin) pxMin = px; if (px > pxMax) pxMax = px;
        if (py < pyMin) pyMin = py; if (py > pyMax) pyMax = py;
      };
      if (georef.available()) {
        const W = window.OUTDOOR_WALK;
        if (W && Array.isArray(W.pts)) for (const r of W.pts) { const p = georef.lonlatToPx(r[0], r[1]); grow(p.px, p.py); }
      }
      const mX = (pxMax - pxMin) * 0.05, mY = (pyMax - pyMin) * 0.05;
      _osmExtent = { pxMin: pxMin - mX, pxMax: pxMax + mX, pyMin: pyMin - mY, pyMax: pyMax + mY };
      return _osmExtent;
    }

    // The surveyed base stations, each a box at its real building height (building_height_m)
    // topped by a red antenna dot — the 3D read of the red pins on the 2D Leaflet map. Placed
    // by lon/lat → px (georef) → px2x/py2z, i.e. the same Pseudo-Mercator grid as the OSM
    // ground, so a station lands on its real rooftop, standing on the shared street level.
    function addBaseStations(g, groundY, upm, upmX, upmZ) {
      const list = window.BASE_STATIONS || [];
      if (!list.length || !georef.available()) return;
      const FP_M = 16;                                        // ~16 m footprint per site
      for (const b of list) {
        if (!Number.isFinite(b.lat) || !Number.isFinite(b.lon)) continue;
        const p = georef.lonlatToPx(b.lon, b.lat);
        const x = px2x(p.px), z = py2z(p.py);
        const live = b.status === 'live';
        const hU = Math.max((Number.isFinite(b.building_height_m) ? b.building_height_m : 20) * upm, 1);
        const bldg = new THREE.Mesh(
          new THREE.BoxGeometry(FP_M * upmX, hU, FP_M * upmZ),
          new THREE.MeshStandardMaterial({ color: live ? 0x8f9ba8 : 0xbfb4ab,
            transparent: true, opacity: 0.55, roughness: 0.9 }));
        bldg.position.set(x, groundY + hU / 2, z);
        bldg.name = '__baseStation';
        g.add(bldg);
        const r = Math.max(4.5 * upm, 1.5);
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(r, 20, 16),
          new THREE.MeshStandardMaterial({ color: live ? 0xc0392b : 0x7f8c8d, roughness: 0.5 }));
        dot.position.set(x, groundY + hU + r, z);
        dot.name = '__baseStation';
        g.add(dot);
      }
    }

    const MODEL_EXTS = ['dae', 'obj', 'glb', 'gltf', 'fbx', '3ds', 'stl', 'kmz'];
    const extOf = (name) => name.split('?')[0].split('.').pop().toLowerCase();

    // Dynamically import the right loader for `ext` and return a uniform
    // load(url, onLoad(Object3D), onError) wrapper. `onProgress` (a three ProgressEvent
    // handler) is threaded into every loader so large models show a real download bar.
    async function makeLoader(ext, manager, onProgress) {
      switch (ext) {
        case 'dae': {
          const { ColladaLoader } = await import('three/addons/loaders/ColladaLoader.js');
          const l = new ColladaLoader(manager);
          return (url, ok, err) => l.load(url, (r) => ok(r.scene), onProgress, err);
        }
        case 'glb':
        case 'gltf': {
          const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
          const { DRACOLoader } = await import('three/addons/loaders/DRACOLoader.js');
          const l = new GLTFLoader(manager);
          // Decode Draco-compressed geometry (our converter's output, and any Draco GLB a
          // user drops in). The decoder wasm comes from the same pinned three@0.185.1 on
          // jsdelivr the importmap already uses — no new origin. Draco is what packs the
          // 7th floor's 3.28M triangles from a ~100MB buffer down to a few MB.
          const draco = new DRACOLoader()
            .setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.185.1/examples/jsm/libs/draco/');
          l.setDRACOLoader(draco);
          return (url, ok, err) => l.load(url,
            (r) => { draco.dispose(); ok(r.scene); },
            onProgress, (e) => { draco.dispose(); err(e); });
        }
        case 'fbx': {
          const { FBXLoader } = await import('three/addons/loaders/FBXLoader.js');
          const l = new FBXLoader(manager);
          return (url, ok, err) => l.load(url, ok, onProgress, err);
        }
        case '3ds': {
          const { TDSLoader } = await import('three/addons/loaders/TDSLoader.js');
          const l = new TDSLoader(manager);
          return (url, ok, err) => l.load(url, ok, onProgress, err);
        }
        case 'stl': {
          const { STLLoader } = await import('three/addons/loaders/STLLoader.js');
          const l = new STLLoader(manager);
          return (url, ok, err) => l.load(url, (geo) => {
            geo.computeVertexNormals();
            ok(new THREE.Mesh(geo, new THREE.MeshStandardMaterial(
              { color: 0xb7c3be, roughness: 0.92, metalness: 0.0 })));
          }, onProgress, err);
        }
        case 'kmz': {
          const { KMZLoader } = await import('three/addons/loaders/KMZLoader.js');
          const l = new KMZLoader(manager);
          return (url, ok, err) => l.load(url, (r) => ok(r.scene), onProgress, err);
        }
        case 'obj': {
          const { OBJLoader } = await import('three/addons/loaders/OBJLoader.js');
          const l = new OBJLoader(manager);
          return (url, ok, err) => l.load(url, ok, onProgress, err);
        }
        default:
          throw new Error('.' + ext + ' is not web-renderable — export Collada (.dae) or glTF (.glb) instead.');
      }
    }

    function loadErr(name) {
      return (err) => {
        hideProgress();
        console.error('[3d] load failed:', err);
        const detail = (err && err.message) ? err.message
          : 'fetch/parse error. If you opened this page as a file:// URL, serve it over http '
            + '(e.g. `python3 -m http.server`) or use “Load model file…”.';
        setStatus('Could not load ' + name + ' — ' + detail, true);
      };
    }

    // Default / network path: load a model by relative URL.
    async function loadModelURL(url) {
      const ext = extOf(url);
      const name = decodeURIComponent(url.split('/').pop());
      const dir = url.substring(0, url.lastIndexOf('/') + 1);
      setStatus('Loading ' + name + ' …');
      const onProg = progressHandler(name, ext === 'obj');
      try {
        const manager = new THREE.LoadingManager();
        if (ext === 'obj') {
          // Load the sibling .mtl so materials + textures come through; the
          // MTL's map_Kd paths resolve relative to the model's directory.
          const { OBJLoader } = await import('three/addons/loaders/OBJLoader.js');
          const { MTLLoader } = await import('three/addons/loaders/MTLLoader.js');
          const mtlName = name.replace(/\.obj$/i, '.mtl');
          const loadObj = (materials) =>
            new OBJLoader(manager).setMaterials(materials).load(
              url, (obj) => placeModel(obj, name), onProg, loadErr(name));
          new MTLLoader(manager).setPath(dir).load(
            encodeURIComponent(mtlName),
            (m) => { m.preload(); loadObj(m); },
            undefined,
            () => new OBJLoader(manager).load(   // no .mtl → geometry only
              url, (obj) => placeModel(obj, name), onProg, loadErr(name)));
          return;
        }
        const load = await makeLoader(ext, manager, onProg);
        load(url, (obj) => placeModel(obj, name), loadErr(name));
      } catch (e) { loadErr(name)(e); }
    }

    // File-picker path: works even from file:// since everything resolves to
    // object URLs. A URL modifier remaps texture/mtl references (by basename)
    // to the companion files the user selected alongside the model.
    async function loadFromFiles(fileList) {
      const files = Array.from(fileList);
      if (!files.length) return;
      const map = new Map();
      files.forEach((f) => map.set(f.name, URL.createObjectURL(f)));

      const modelFile = files.find((f) => MODEL_EXTS.includes(extOf(f.name)));
      if (!modelFile) {
        setStatus('No supported model in selection — include a .dae, .obj, .glb/.gltf, .fbx, .3ds, .stl or .kmz.', true);
        return;
      }
      const ext = extOf(modelFile.name);
      const manager = new THREE.LoadingManager();
      manager.setURLModifier((url) => {
        const base = decodeURIComponent(url).split('/').pop().split('\\').pop();
        return map.has(base) ? map.get(base) : url;
      });

      setStatus('Loading ' + modelFile.name + ' …');
      const onProg = progressHandler(modelFile.name, ext === 'obj');
      const done = (obj) => {
        placeModel(obj, modelFile.name);
        setTimeout(() => map.forEach((u) => URL.revokeObjectURL(u)), 15000);
      };
      try {
        if (ext === 'obj') {
          // Pair the OBJ with its .mtl (and thus textures) if one was selected.
          const { OBJLoader } = await import('three/addons/loaders/OBJLoader.js');
          const objLoader = new OBJLoader(manager);
          const mtl = files.find((f) => extOf(f.name) === 'mtl');
          if (mtl) {
            const { MTLLoader } = await import('three/addons/loaders/MTLLoader.js');
            new MTLLoader(manager).load(map.get(mtl.name), (materials) => {
              materials.preload();
              objLoader.setMaterials(materials);
              objLoader.load(map.get(modelFile.name), done, onProg, loadErr(modelFile.name));
            }, undefined, loadErr(mtl.name));
          } else {
            objLoader.load(map.get(modelFile.name), done, onProg, loadErr(modelFile.name));
          }
        } else {
          const load = await makeLoader(ext, manager, onProg);
          load(map.get(modelFile.name), done, loadErr(modelFile.name));
        }
      } catch (e) { loadErr(modelFile.name)(e); }
    }

    function setMode(mode) {
      const is3d = mode === '3d';
      mode2dBtn.classList.toggle('active', !is3d);
      mode3dBtn.classList.toggle('active', is3d);
      plot2d.style.display = is3d ? 'none' : '';
      // Explicit 'block' (not '') — the base CSS rule is display:none, so an
      // empty inline value would fall back to hidden and give the canvas 0 size.
      wrap.style.display = is3d ? 'block' : 'none';
      hintEl.textContent = is3d ? '7th-floor CAD model — drag to orbit'
                                : 'Floor-plan signal heatmap';
      if (is3d) {
        if (initScene() && !currentModel) loadModelURL(DEFAULT_MODEL_URL);
        requestAnimationFrame(onResize);
      } else if (window.Plotly) {
        // Returning to 2D: let Plotly reclaim the full width.
        try { window.Plotly.Plots.resize(plot2d); } catch (e) {}
      }
    }

    mode2dBtn.addEventListener('click', () => setMode('2d'));
    mode3dBtn.addEventListener('click', () => setMode('3d'));
    fileInput.addEventListener('change', (e) => {
      if (initScene()) loadFromFiles(e.target.files);
      e.target.value = '';
    });
    // Model switcher: populate from MODELS and load the chosen one.
    MODELS.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.url; opt.textContent = m.name;
      modelSelect.appendChild(opt);
    });
    modelSelect.value = DEFAULT_MODEL_URL;
    modelSelect.addEventListener('change', () => {
      if (initScene()) loadModelURL(modelSelect.value);
    });
    resetBtn.addEventListener('click', () => { if (currentModel) frameObject(currentModel); });
    const shellCbx = document.getElementById('view3dShell');
    const osmGroundCbx = document.getElementById('view3dOsmGround');
    if (shellCbx) shellCbx.addEventListener('change', () => { showShell = shellCbx.checked; if (currentModel) addBuildingContext(currentModel); });
    if (osmGroundCbx) osmGroundCbx.addEventListener('change', () => { showOsmGround = osmGroundCbx.checked; if (currentModel) addBuildingContext(currentModel); });
    const opacityEl = document.getElementById('view3dOpacity');
    if (opacityEl) opacityEl.addEventListener('input', () => {
      modelOpacity = Number(opacityEl.value) / 100;
      requestModelDisplay();                     // rebuild materials (WebGPU won't mutate live)
    });
    const sliceEl = document.getElementById('view3dSlice');
    if (sliceEl) sliceEl.addEventListener('input', () => {
      modelClipH = Number(sliceEl.value) / 10;   // slider is decimetres (0–45 → 0–4.5 m)
      updateModelClip();                         // live label + plane position
      requestModelDisplay();                     // rebuild so the clip actually applies
    });
    wireBtn.addEventListener('click', () => {
      wireframe = !wireframe;
      wireBtn.classList.toggle('active', wireframe);
      if (currentModel) applyWireframe(currentModel, wireframe);
    });
    upBtn.addEventListener('click', () => {
      if (!currentModel) return;
      currentModel.rotation.x -= Math.PI / 2;   // step 90°; cycles through orientations
      frameObject(currentModel);
    });

    // ---- RF coverage overlay: Heat Map or traced Path ------------------
    // Two display modes over one registered coordinate frame (floorInfo):
    //   • Heat Map — an interpolated field, drawn flat (viridis/jet bands +
    //                iso-lines) or extruded into a 3D "signal terrain".
    //   • Path     — the walk-test route as floating dots + a connecting line,
    //                placed at each point's elevation (default 3 ft AGL of the
    //                3D model when the data has no elevation column).
    // Data comes from the built-in walk (window.filtered / window.idwGrid /
    // window.filteredTimeseries) or from an imported CSV (importedPoints).
    const RF_UNITS = { rsrp: 'dBm', rsrq: 'dB', cinr: 'dB', rssi: 'dBm' };
    // Colour ramps (low → high). Viridis for analysis; "jet" mimics the
    // iBwave/SpectraNet RSRP look (blue weak → green/yellow → red/magenta hot).
    const CMAPS = {
      viridis: [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
                [39,173,129],[92,200,99],[170,220,50],[253,231,37]],
      jet:     [[0,0,140],[0,110,240],[0,220,220],[60,210,70],[240,235,0],
                [255,140,0],[230,20,20],[255,0,190]]
    };
    let rfCmap = 'viridis';
    function cmap(t) {
      const s0 = CMAPS[rfCmap];
      t = Math.max(0, Math.min(1, t));
      const s = t * (s0.length - 1), i = Math.floor(s), f = s - i;
      const a = s0[i], b = s0[Math.min(i + 1, s0.length - 1)];
      return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
    }

    const rfBar        = document.getElementById('rfBar');
    const rfContourBtn = document.getElementById('rfContourBtn');
    const rfModeBtn    = document.getElementById('rfModeBtn');
    const rfCmapBtn    = document.getElementById('rfCmapBtn');
    const rfStyleBtn   = document.getElementById('rfStyleBtn');
    const rfHeightEl   = document.getElementById('rfHeight');
    const rfOpacityEl  = document.getElementById('rfOpacity');
    const rfLinesEl    = document.getElementById('rfLines');
    const rfPointsEl   = document.getElementById('rfPoints');
    const rfMetricLabel = document.getElementById('rfMetricLabel');
    const rfLegend      = document.getElementById('rfLegend');
    const rfNoteEl      = document.getElementById('rfNote');
    const rfCsvEl       = document.getElementById('rfCsv');
    const rfClearImport = document.getElementById('rfClearImport');

    let rfMode = 'heatmap';       // 'heatmap' | 'path'
    let rfStyle = 'terrain';      // heatmap sub-style: 'terrain' | 'flat'
    let importedPoints = null;    // [{x01,y01,v,t,elevFt}] from CSV, or null
    let importUnit = '';
    // Cached field + selection so height/style/opacity tweaks don't recompute IDW.
    let rfGrid = null, rfData = null, rfVmin = 0, rfVmax = 1, rfMetric = 'rsrp';
    const RF_BANDS = 14;

    const FLOOR_W = 1150, FLOOR_H = 515;                 // floor-plan pixel frame
    const IN_PER_FT = 12, DEFAULT_AGL_FT = 3;            // model units are inches
    const px2x = (px) => floorInfo.minX + (px / FLOOR_W) * (floorInfo.maxX - floorInfo.minX);
    const py2z = (py) => floorInfo.minZ + (py / FLOOR_H) * (floorInfo.maxZ - floorInfo.minZ);
    const clampT = (t) => Math.max(0, Math.min(1, t));
    const terrainAmp = () => (Number(rfHeightEl.value) / 100) * floorInfo.height * 1.6;
    const flatLift   = () => (Number(rfHeightEl.value) / 100) * floorInfo.height * 0.15;
    const baseY      = () => floorInfo.topY + 0.6;       // just above the floor slab
    // Path point height: elevation (ft AGL) above the floor, +/- a slider lift
    // (slider at its 34 default = no extra lift, so dots sit at true 3 ft).
    const pathY = (elevFt) => {
      const ft = (elevFt == null || !isFinite(elevFt)) ? DEFAULT_AGL_FT : elevFt;
      const lift = (Number(rfHeightEl.value) / 100 - 0.34) * floorInfo.height;
      return Math.max(floorInfo.topY + 2, floorInfo.topY + ft * IN_PER_FT + lift);
    };

    function legendCss(name) {
      const s = CMAPS[name];
      return 'linear-gradient(to right,' + s.map((c, i) =>
        `rgb(${c[0]|0},${c[1]|0},${c[2]|0}) ${(i/(s.length-1)*100).toFixed(0)}%`).join(',') + ')';
    }
    function updateLegend() { rfLegend.style.background = legendCss(rfCmap); }

    // Which data + colour range is active: imported CSV takes over from the
    // built-in walk when present.
    function rfSelection() {
      if (importedPoints) {
        const vals = importedPoints.map(p => p.v).filter(Number.isFinite).sort((a, b) => a - b);
        const q = (f) => vals.length ? vals[Math.min(vals.length-1, Math.round((vals.length-1)*f))] : 0;
        return { key: 'v', vmin: q(0.05), vmax: q(0.95), unit: importUnit,
                 label: 'Imported (' + importedPoints.length + ')' };
      }
      const m = document.getElementById('metric').value;
      return { key: m, vmin: parseFloat(document.getElementById('vmin').value),
               vmax: parseFloat(document.getElementById('vmax').value),
               unit: RF_UNITS[m] || '', label: m.toUpperCase() };
    }
    // Point set for the heat-map interpolation ({px,py,<key>}).
    function heatData(key) {
      if (importedPoints) return importedPoints.map(p => ({ px: p.x01*FLOOR_W, py: p.y01*FLOOR_H, v: p.v }));
      return window.filtered(key);
    }
    // Ordered points for the traced path.
    function pathRows() {
      if (importedPoints) {
        return importedPoints.slice().sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
      }
      const m = document.getElementById('metric').value;
      const rows = (typeof window.filteredTimeseries === 'function' ? window.filteredTimeseries(m) : [])
        .slice().sort((a, b) => a.t_sec - b.t_sec);
      return rows.map(r => ({ x01: r.px/FLOOR_W, y01: r.py/FLOOR_H, v: r[m], t: r.t_sec,
                              elevFt: (r.elev_ft ?? null) }));
    }

    function clearRF() {
      for (const obj of [rfMesh, rfPoints]) {
        if (!obj) continue;
        scene.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) { if (obj.material.map) obj.material.map.dispose(); obj.material.dispose(); }
      }
      rfMesh = null; rfPoints = null;
    }

    function contourCanvas(grid, isolines) {
      const ny = grid.z.length, nx = grid.z[0].length;
      const cv = document.createElement('canvas'); cv.width = nx; cv.height = ny;
      const ctx = cv.getContext('2d');
      const img = ctx.createImageData(nx, ny);
      const span = (rfVmax - rfVmin) || 1;
      const idx = new Int16Array(nx * ny);
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const v = grid.z[y][x];
        idx[y*nx+x] = (v == null || !isFinite(v))
          ? -1 : Math.max(0, Math.min(RF_BANDS - 1, Math.floor((v - rfVmin) / span * RF_BANDS)));
      }
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const p = y*nx+x, o = p*4, bi = idx[p];
        if (bi < 0) { img.data[o+3] = 0; continue; }
        let [r, g, b] = cmap((bi + 0.5) / RF_BANDS);
        if (isolines) {
          const right = x+1 < nx ? idx[p+1] : bi, down = y+1 < ny ? idx[p+nx] : bi;
          if ((right >= 0 && right !== bi) || (down >= 0 && down !== bi)) { r*=0.3; g*=0.3; b*=0.3; }
        }
        img.data[o] = r; img.data[o+1] = g; img.data[o+2] = b; img.data[o+3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      return cv;
    }

    // Full rebuild: re-read the selection, (re)interpolate for heat map, redraw.
    function buildRFContour() {
      if (!scene || !floorInfo) return;
      const sel = rfSelection();
      rfMetric = sel.key; rfVmin = sel.vmin; rfVmax = sel.vmax;
      rfMetricLabel.textContent = sel.label;
      updateLegend();
      if (rfMode === 'path') { rfGrid = null; renderRF(); return; }
      rfData = heatData(rfMetric);
      rfGrid = window.idwGrid(rfData, rfMetric, FLOOR_W, FLOOR_H, 168, 76);
      if (!rfGrid) { clearRF(); rfNoteEl.textContent = 'not enough points'; return; }
      rfNoteEl.textContent =
        `${rfVmin.toFixed(1)} … ${rfVmax.toFixed(1)} ${sel.unit} · ${rfData.length} pts`;
      renderRF();
    }

    // Redraw from cached state (no IDW) — height / style / band / colormap / mode.
    function renderRF() {
      clearRF();
      if (rfMode === 'path') { buildPath(); return; }
      if (!rfGrid) return;
      if (rfStyle === 'terrain') buildTerrain(); else buildFlat();
      if (rfPointsEl.checked) buildRFPoints();
    }

    function buildFlat() {
      const tex = new THREE.CanvasTexture(contourCanvas(rfGrid, rfLinesEl.checked));
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
      const w = floorInfo.maxX - floorInfo.minX, d = floorInfo.maxZ - floorInfo.minZ;
      rfMesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({
        map: tex, transparent: true, opacity: Number(rfOpacityEl.value) / 100,
        depthWrite: false, side: THREE.DoubleSide }));
      rfMesh.rotation.x = -Math.PI / 2;
      rfMesh.position.set((floorInfo.minX + floorInfo.maxX) / 2, baseY() + flatLift(),
                          (floorInfo.minZ + floorInfo.maxZ) / 2);
      rfMesh.renderOrder = 2;
      scene.add(rfMesh);
    }

    function buildTerrain() {
      const ny = rfGrid.z.length, nx = rfGrid.z[0].length;
      const span = (rfVmax - rfVmin) || 1, amp = terrainAmp(), y0 = baseY();
      const banded = rfLinesEl.checked;
      const pos = [], col = [], idx = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const v = rfGrid.z[j][i];
        const t = (v == null || !isFinite(v)) ? 0 : clampT((v - rfVmin) / span);
        pos.push(px2x(i / (nx - 1) * FLOOR_W), y0 + amp * t, py2z(j / (ny - 1) * FLOOR_H));
        const tc = banded ? (Math.floor(t * RF_BANDS) + 0.5) / RF_BANDS : t;
        const [r, g, b] = cmap(tc);
        col.push(r/255, g/255, b/255);
      }
      for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
        const a = j*nx + i, b = a + 1, c = a + nx, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
      geo.setIndex(idx);
      geo.computeVertexNormals();
      rfMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.82, metalness: 0.0, side: THREE.DoubleSide,
        transparent: true, opacity: Number(rfOpacityEl.value) / 100 }));
      rfMesh.renderOrder = 2;
      scene.add(rfMesh);
    }

    // Heat-map dots: raw measurements floating at their own signal height.
    function buildRFPoints() {
      const span = (rfVmax - rfVmin) || 1, amp = terrainAmp(), y0 = baseY();
      // Instanced cubes, not THREE.Points — the WebGPU backend can't size points.
      const size = (floorInfo.maxX - floorInfo.minX) * 0.012;
      const mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: Number(rfOpacityEl.value) / 100 }),
        rfData.length);
      const dummy = new THREE.Object3D(), col = new THREE.Color();
      for (let i = 0; i < rfData.length; i++) {
        const r = rfData[i], t = clampT((r[rfMetric] - rfVmin) / span);
        dummy.position.set(px2x(r.px), y0 + amp * t + 2.0, py2z(r.py));
        dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
        const [cr, cg, cb] = cmap(t); col.setRGB(cr/255, cg/255, cb/255); mesh.setColorAt(i, col);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.renderOrder = 3;
      rfPoints = mesh;
      scene.add(rfPoints);
    }

    // Traced path: floating dots along the collection route, connected by a
    // colour-graded line, placed at each point's elevation (3 ft AGL default).
    function buildPath() {
      const rows = pathRows();
      if (!rows.length) { rfNoteEl.textContent = 'no path points for this selection'; return; }
      const span = (rfVmax - rfVmin) || 1;
      const linePos = [], lineCol = [], dotPos = [], dotCol = [];
      let withElev = 0;
      for (const r of rows) {
        const x = px2x(r.x01 * FLOOR_W), z = py2z(r.y01 * FLOOR_H), y = pathY(r.elevFt);
        if (r.elevFt != null && isFinite(r.elevFt)) withElev++;
        const [cr, cg, cb] = cmap(Number.isFinite(r.v) ? clampT((r.v - rfVmin) / span) : 0);
        linePos.push(x, y, z); lineCol.push(cr/255, cg/255, cb/255);
        dotPos.push(x, y, z);  dotCol.push(cr/255, cg/255, cb/255);
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
      lg.setAttribute('color', new THREE.Float32BufferAttribute(lineCol, 3));
      rfMesh = new THREE.Line(lg, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: Number(rfOpacityEl.value) / 100 }));
      rfMesh.renderOrder = 2;
      scene.add(rfMesh);
      if (rfPointsEl.checked) {
        // Instanced cubes, not THREE.Points (WebGPU can't size points).
        const size = (floorInfo.maxX - floorInfo.minX) * 0.014;
        const n = dotPos.length / 3;
        const mesh = new THREE.InstancedMesh(
          new THREE.BoxGeometry(size, size, size),
          new THREE.MeshBasicMaterial({ transparent: true, opacity: Number(rfOpacityEl.value) / 100 }), n);
        const dummy = new THREE.Object3D(), col = new THREE.Color();
        for (let i = 0; i < n; i++) {
          dummy.position.set(dotPos[i*3], dotPos[i*3+1], dotPos[i*3+2]);
          dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
          col.setRGB(dotCol[i*3], dotCol[i*3+1], dotCol[i*3+2]); mesh.setColorAt(i, col);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        mesh.renderOrder = 3;
        rfPoints = mesh;
        scene.add(rfPoints);
      }
      const sel = rfSelection();
      const elevNote = withElev ? `${withElev}/${rows.length} w/ elevation` : `${DEFAULT_AGL_FT} ft AGL default`;
      rfNoteEl.textContent = `path · ${rows.length} pts · ${rfVmin.toFixed(1)}…${rfVmax.toFixed(1)} ${sel.unit} · ${elevNote}`;
    }

    function setRFActive(on) {
      rfActive = on;
      rfContourBtn.textContent = 'RF contour: ' + (on ? 'on' : 'off');
      rfContourBtn.classList.toggle('active', on);
      rfBar.style.display = on ? 'flex' : 'none';
      if (on) { if (initScene()) buildRFContour(); }
      else clearRF();
    }
    function setRFMode(mode) {
      rfMode = mode;
      rfModeBtn.textContent = 'Mode: ' + (mode === 'path' ? 'Path' : 'Heat Map');
      rfModeBtn.classList.toggle('active', mode === 'path');
      rfBar.classList.toggle('mode-path', mode === 'path');
      if (mode === 'path') rfPointsEl.checked = true;   // dots are the point of a path
      if (rfActive) buildRFContour();
    }
    function setRFStyle(style) {
      rfStyle = style;
      rfStyleBtn.textContent = style === 'terrain' ? '3D terrain' : 'Flat contour';
      rfStyleBtn.classList.toggle('active', style === 'terrain');
      if (rfActive) renderRF();
    }
    function setRFCmap(name) {
      rfCmap = name;
      rfCmapBtn.textContent = 'Colormap: ' + (name === 'jet' ? 'RF (jet)' : 'Viridis');
      updateLegend();
      if (rfActive) renderRF();
    }
    setRFStyle('terrain'); setRFMode('heatmap'); updateLegend();

    // ---- CSV import: points with or without elevation -------------------
    // Auto-detects columns: px/py (floor-plan pixels) or x/y|lon/lat
    // (min-max fitted to the floor); a value column (rsrp/dbm/value/…); and
    // optional elevation (ft) and time/order columns.
    function parseRFCsv(text) {
      const lines = text.replace(/^﻿/, '').trim().split(/\r?\n/).filter((l) => l.length);
      if (lines.length < 2) return { points: [], unit: '' };
      const H = lines[0].split(',').map((s) => s.trim().toLowerCase().replace(/^["']|["']$/g, ''));
      const body = lines.slice(1).map((l) => l.split(','));
      const sub = (subs) => H.findIndex((h) => subs.some((s) => h.includes(s)));
      const iPx = H.indexOf('px'), iPy = H.indexOf('py');
      const pixel = iPx >= 0 && iPy >= 0;
      const iX = pixel ? iPx : sub(['lon', 'east', 'x']);
      const iY = pixel ? iPy : sub(['lat', 'north', 'y']);
      let iV = sub(['rsrp', 'dbm', 'value', 'signal', 'rsrq', 'cinr', 'rssi']);
      if (iV < 0) for (let k = H.length - 1; k >= 0; k--) {           // fallback: last numeric column
        if (body.every((r) => r[k] !== undefined && r[k].trim() !== '' && isFinite(+r[k]))) { iV = k; break; }
      }
      const iE = sub(['elev', 'height', 'alt', 'agl', 'floor_z']);
      const iT = sub(['t_sec', 'timestamp', 'time', 'sec', 'order', 'seq', 'index']);
      const unit = iV >= 0 && /rsrp|dbm|rssi/.test(H[iV]) ? 'dBm'
                 : iV >= 0 && /rsrq|cinr|s[in]nr/.test(H[iV]) ? 'dB' : '';
      if (iX < 0 || iY < 0) throw new Error('need px/py or x/y (or lon/lat) columns');
      const raw = body.map((r, i) => ({
        x: +r[iX], y: +r[iY], v: iV >= 0 ? +r[iV] : NaN,
        e: iE >= 0 ? +r[iE] : null, t: iT >= 0 ? +r[iT] : i
      })).filter((o) => isFinite(o.x) && isFinite(o.y));
      if (!raw.length) return { points: [], unit };
      let pts;
      if (pixel) {
        pts = raw.map((o) => ({ x01: o.x/FLOOR_W, y01: o.y/FLOOR_H, v: o.v, t: o.t,
                                elevFt: (o.e != null && isFinite(o.e)) ? o.e : null }));
      } else if (/lon/.test(H[iX]) && /lat/.test(H[iY]) && georef.available()) {
        // geographic lon/lat → floor-plan pixels via the real affine, which CARRIES the
        // −7.33° map rotation. The old min/max rubber-sheet below dropped the rotation and
        // stretched the walk's own extent onto the plan bbox, mis-registering any track
        // that didn't exactly fill it. This is what an M5 scanner overlay needs.
        pts = raw.map((o) => {
          const p = georef.lonlatToPx(o.x, o.y);
          return { x01: p.px / FLOOR_W, y01: p.py / FLOOR_H, v: o.v, t: o.t,
                   elevFt: (o.e != null && isFinite(o.e)) ? o.e : null };
        });
      } else {                                                       // projected x/y: min-max fit (legacy)
        const xs = raw.map((o) => o.x), ys = raw.map((o) => o.y);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        pts = raw.map((o) => ({ x01: x1 > x0 ? (o.x-x0)/(x1-x0) : 0.5, y01: y1 > y0 ? (o.y-y0)/(y1-y0) : 0.5,
                                v: o.v, t: o.t, elevFt: (o.e != null && isFinite(o.e)) ? o.e : null }));
      }
      return { points: pts, unit };
    }

    rfContourBtn.addEventListener('click', () => setRFActive(!rfActive));
    rfModeBtn.addEventListener('click', () => setRFMode(rfMode === 'path' ? 'heatmap' : 'path'));
    rfCmapBtn.addEventListener('click', () => setRFCmap(rfCmap === 'viridis' ? 'jet' : 'viridis'));
    rfStyleBtn.addEventListener('click', () => setRFStyle(rfStyle === 'terrain' ? 'flat' : 'terrain'));
    rfHeightEl.addEventListener('input', () => { if (rfActive) renderRF(); });
    rfOpacityEl.addEventListener('input', () => {
      const o = Number(rfOpacityEl.value) / 100;
      if (rfMesh) rfMesh.material.opacity = o;
      if (rfPoints) rfPoints.material.opacity = o;
    });
    rfLinesEl.addEventListener('change', () => { if (rfActive) renderRF(); });
    rfPointsEl.addEventListener('change', () => { if (rfActive) renderRF(); });
    rfCsvEl.addEventListener('change', (e) => {
      const f = e.target.files[0]; e.target.value = '';
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = parseRFCsv(String(reader.result));
          if (!parsed.points.length) { rfNoteEl.textContent = 'CSV: no usable rows'; return; }
          importedPoints = parsed.points; importUnit = parsed.unit;
          rfClearImport.style.display = '';
          if (!rfActive) setRFActive(true); else buildRFContour();
        } catch (err) { console.error('[rf csv]', err); rfNoteEl.textContent = 'CSV error: ' + err.message; }
      };
      reader.readAsText(f);
    });
    rfClearImport.addEventListener('click', () => {
      importedPoints = null; importUnit = ''; rfClearImport.style.display = 'none';
      if (rfActive) buildRFContour();
    });

    // Keep the built-in overlay in sync with the sidebar (ignored once a CSV
    // is imported, since that supplies its own field).
    ['metric', 'network', 'band', 'pci', 'channel', 'vmin', 'vmax'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', () => { if (rfActive) buildRFContour(); });
    });

    // ---- Public hooks for the app router (landing.js) -------------------
    // Lets the Import / Preprocess screens drive this viewer without reaching
    // into its internals: switch mode, load an imported model, validate a CSV,
    // and push a CSV into the coverage overlay (reusing parseRFCsv above).
    window.__viewer3d = {
      setMode,
      loadFiles(files) { if (files && files.length && initScene()) loadFromFiles(files); },
      parseCsvText(text) {
        try { return parseRFCsv(String(text)); }
        catch (e) { return { points: [], unit: '', error: e && e.message || String(e) }; }
      },
      importCsvText(text) {
        try {
          const parsed = parseRFCsv(String(text));
          if (!parsed.points.length) return { ok: false, msg: 'no usable rows' };
          importedPoints = parsed.points; importUnit = parsed.unit;
          if (rfClearImport) rfClearImport.style.display = '';
          if (!rfActive) setRFActive(true); else buildRFContour();
          return { ok: true, count: parsed.points.length, unit: parsed.unit };
        } catch (e) { return { ok: false, msg: e && e.message || String(e) }; }
      },
    };
