/* project_loader.js — the project/dataset layer that decouples the workspace from any
 * single dataset. (classic script; shares window.*, loaded before landing.js.)
 *
 * A PROJECT manifest (Data/projects/*_default.js now; the Flask server later) NAMES the
 * data files a workspace boots from:
 *     { id, label, environment, dim, data: [ {role, src, optional?}, ... ], ... }
 *
 * loadProject() injects each data <script> IN ORDER (awaiting each — the dashboard reads
 * bare consts `records` / `timeseriesRecords` / `floorPlanImage` at init, so they must
 * exist first), then injects dashboard.js + spectrum.js exactly ONCE. Built-in "Default"
 * projects point at the committed Data/*.js, so the default workspace is unchanged.
 *
 * Generalizes the inject-and-await pattern already used in dashboard.js for the lazy
 * outdoor-walk load. The dashboard (the "features") reads whatever project loaded (the
 * "data"), so new workspace features are dataset-agnostic by construction. */
(function () {
  var DASHBOARD_SRC = 'Map Coverage/dashboard.js?v=osm-align2';
  var SPECTRUM_SRC  = 'Statistics/spectrum.js';

  window.PROJECTS = window.PROJECTS || {};
  window.PROJECT_API_BASE = window.PROJECT_API_BASE || '/api';

  var _script = {};    // src -> Promise (dedupe: a const/data file must execute at most once)
  var _booted = false; // dashboard.js/spectrum.js wired exactly once per page
  var _active = null;
  var _apiOk = null;   // null=unknown, true/false after refreshProjects()

  function injectScript(src) {
    if (_script[src]) return _script[src];
    _script[src] = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = function () { resolve(src); };
      s.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(s);
    });
    return _script[src];
  }

  // Every known project (built-in now; server-provided projects merge in a later phase).
  function allProjects() {
    var P = window.PROJECTS || {};
    return Object.keys(P).map(function (k) { return P[k]; });
  }

  // Dropdown helper — "get the projects (and their file manifests) available for a mode".
  window.getProjects = function (environment, dim) {
    return allProjects().filter(function (p) {
      return (!environment || p.environment === environment) && (!dim || p.dim === dim);
    });
  };

  window.getProject = function (id) { return (window.PROJECTS || {})[id] || null; };

  /* Pull projects from the Flask backend (built-in defaults + user-created) and merge them
   * into window.PROJECTS, so getProjects() sees created/saved projects too. Best-effort: if
   * the server isn't running (e.g. plain Live Preview / a static host), it resolves to the
   * committed built-ins so the Default workspace still works — only create/save need the
   * server. The dropdown calls this before populating; enterWorkspace never depends on it. */
  window.refreshProjects = function () {
    return fetch(window.PROJECT_API_BASE + '/projects', { headers: { Accept: 'application/json' } })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (list) {
        (list || []).forEach(function (p) { if (p && p.id) window.PROJECTS[p.id] = p; });
        _apiOk = true;
        return allProjects();
      })
      .catch(function (e) {
        _apiOk = false;
        console.info('[project] backend not reachable (' + e.message + '); using built-in projects.');
        return allProjects();
      });
  };
  window.projectApiReachable = function () { return _apiOk; };

  window.resolveDefaultProject = function (environment, dim) {
    return window.getProject(environment + '-' + dim + '-default');
  };

  window.activeProject = function () { return _active; };
  window.workspaceBooted = function () { return _booted; };

  /* Inject the project's data files (in order), then boot the dashboard once. Returns a
   * Promise that resolves when the workspace is ready to have applyModeToWorkspace() run.
   * A non-optional data file that 404s rejects; optional ones warn and continue. */
  window.loadProject = function (manifest, opts) {
    opts = opts || {};
    _active = manifest || null;
    var chain = Promise.resolve();
    var data = (manifest && manifest.data) || [];
    data.forEach(function (d) {
      chain = chain.then(function () {
        if (typeof opts.onProgress === 'function') opts.onProgress(d.role, d.src);
        return injectScript(d.src).catch(function (err) {
          if (d.optional) { console.warn('[project] optional data absent:', d.src); return; }
          throw err;
        });
      });
    });
    return chain.then(function () {
      if (_booted) return;
      _booted = true;
      return injectScript(DASHBOARD_SRC).then(function () { return injectScript(SPECTRUM_SRC); });
    });
  };
})();
