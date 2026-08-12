#!/usr/bin/env python3
"""app.py — RF Simulation Studio project backend (Flask).

The "functional backend" behind the Import step: it stores projects (named datasets)
and, in a later phase, turns raw uploads (CSV folders, floor plan + TAB, 3D models) into
a project by shelling out to the existing rasterize/bake scripts. It serves the static
site AND the /api on ONE origin so everything is same-origin.

Run:
    python Backend/server/app.py            # http://127.0.0.1:8000
    python Backend/server/app.py --port 8000 --host 127.0.0.1
    python Backend/server/app.py --self-test

Endpoints (this phase):
    GET  /api/health                       -> {ok: true, ...}
    GET  /api/projects?environment=&dim=   -> [manifest, ...]   (built-ins + created)
    GET  /api/projects/<id>                -> manifest | 404
    GET  /api/data/<id>/<role>             -> the data file for that project role
    GET  /                                 -> Frontend_Data_Display.html
    GET  /<path>                           -> static file from the repo root

The four built-in "Default" projects are defined here (mirroring Data/projects/registry.js)
and point at the committed Data/*.js, so the default workspace needs no on-disk copy. Created
projects live under Backend/server/projects/<id>/ (manifest.json + data/ + inputs/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

# Backend/server/app.py -> parents[0]=server, [1]=Backend, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = Path(__file__).resolve().parent / "projects"   # created projects (on disk)
INDEX_HTML = "Frontend_Data_Display.html"

# --- Built-in defaults (single source with Data/projects/registry.js; keep in sync) -------
# The indoor walk dataset the dashboard boots from in every mode. The two `optional` files
# are git-ignored (generated locally); the loader/dashboard degrade gracefully if absent.
INDOOR_BASE = [
    {"role": "records",      "src": "Data/records_data.js"},
    {"role": "timeseries",   "src": "Data/timeseries_data.js"},
    {"role": "floorplan",    "src": "Data/floorplan_image.js"},
    {"role": "meta",         "src": "Data/floorplan_meta.js"},
    {"role": "baseStations", "src": "Data/base_stations.js"},
    {"role": "outdoorWalk",  "src": "Data/outdoor_walk.js", "optional": True},
    {"role": "basemap",      "src": "Data/basemap_image.js", "optional": True},
    {"role": "crossval",     "src": "Data/pl2d_crossval.js", "optional": True},
]


def _builtin(pid: str, label: str, env: str, dim: str, extra: dict | None = None) -> dict:
    m = {"id": pid, "label": label, "environment": env, "dim": dim,
         "builtin": True, "ai_ml": True, "data": INDOOR_BASE}
    if extra:
        m.update(extra)
    return m


BUILTINS: dict[str, dict] = {
    "indoor-2d-default": _builtin(
        "indoor-2d-default", "Default (Indoor · 2D)", "indoor", "2d"),
    "outdoor-2d-default": _builtin(
        "outdoor-2d-default", "Default (Outdoor · 2D)", "outdoor", "2d",
        {"iframes": {"map": "Frontend/osm3d/outdoor_view.html",
                     "time": "Frontend/osm3d/outdoor_timelapse.html",
                     "sim2d": "Frontend/simulator/fw_studio2d_outdoor.html"}}),
    "indoor-3d-default": _builtin(
        "indoor-3d-default", "Default (Indoor · 3D)", "indoor", "3d",
        {"model": {"src": "Data/models/Indoor 7th floor v2.obj",
                   "label": "Indoor 7th floor v2.obj"},
         "elevation": {"mode": "floor", "floor_z_m": 0, "ceiling_height_m": 4.58}}),
    "outdoor-3d-default": _builtin(
        "outdoor-3d-default", "Default (Outdoor · 3D)", "outdoor", "3d",
        {"model": {"src": "Data/models/NoMa_DC/Actual_Outdoor_Sim_3D_Render_NoMa_Washington_DC_3D.obj",
                   "label": "NoMa DC render"},
         "iframes": {"sim3d": "Frontend/simulator/fw_studio3d_outdoor.html"},
         "elevation": {"mode": "none"}}),
}


# --- Created-project store (on disk) ------------------------------------------------------
def list_created() -> list[dict]:
    """Every user-created project — one Backend/server/projects/<id>/manifest.json each."""
    out: list[dict] = []
    if not PROJECTS_DIR.exists():
        return out
    for d in sorted(PROJECTS_DIR.iterdir()):
        mf = d / "manifest.json"
        if mf.is_file():
            try:
                out.append(json.loads(mf.read_text()))
            except (OSError, json.JSONDecodeError) as e:
                print(f"[server] skipping unreadable manifest {mf}: {e}", file=sys.stderr)
    return out


def all_projects() -> list[dict]:
    """Built-in defaults first, then created projects (created may override a built-in id)."""
    merged: dict[str, dict] = dict(BUILTINS)
    for p in list_created():
        if p.get("id"):
            merged[p["id"]] = p
    return list(merged.values())


def get_project(pid: str) -> dict | None:
    for p in all_projects():
        if p.get("id") == pid:
            return p
    return None


def resolve_data_path(project: dict, role: str) -> Path | None:
    """Absolute path to a project's data file for `role`. Repo-relative `src` -> repo file;
    a "stored:<name>" src -> the created project's own data/ dir."""
    entry = next((d for d in project.get("data", []) if d.get("role") == role), None)
    if not entry:
        return None
    src = entry["src"]
    if src.startswith("stored:"):
        return PROJECTS_DIR / project["id"] / "data" / src[len("stored:"):]
    return REPO_ROOT / src


# --- Flask app ----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB cap on uploaded CSV folders/models


@app.after_request
def _cors(resp):
    # Local dev tool: allow the API to be read cross-origin (e.g. from a Live Preview tab).
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/health")
def health():
    return jsonify(ok=True, repo_root=str(REPO_ROOT),
                   builtins=list(BUILTINS), created=[p["id"] for p in list_created()])


def serialize(project: dict, base_url: str) -> dict:
    """Manifest as the browser loader should see it. Created projects get each data `src`
    rewritten to an absolute /api/data/<id>/<role> URL (so scripts load even when the page is
    served from a different origin); built-in srcs stay repo-relative to the page origin."""
    if project.get("builtin"):
        return project
    out = dict(project)
    out["data"] = [{**d, "src": f"{base_url}api/data/{project['id']}/{d['role']}"}
                   for d in project.get("data", [])]
    return out


@app.get("/api/projects")
def api_projects():
    env = request.args.get("environment")
    dim = request.args.get("dim")
    items = [serialize(p, request.host_url) for p in all_projects()
             if (not env or p.get("environment") == env)
             and (not dim or p.get("dim") == dim)]
    return jsonify(items)


@app.get("/api/projects/<pid>")
def api_project(pid: str):
    p = get_project(pid)
    if not p:
        abort(404)
    return jsonify(serialize(p, request.host_url))


@app.get("/api/data/<pid>/<role>")
def api_data(pid: str, role: str):
    p = get_project(pid)
    if not p:
        abort(404)
    path = resolve_data_path(p, role)
    if not path or not path.is_file():
        abort(404)
    return send_from_directory(path.parent, path.name)


# --- Create / validate (multipart uploads) ------------------------------------------------
import re as _re            # noqa: E402
import shutil as _shutil    # noqa: E402
import tempfile as _tempfile  # noqa: E402
import uuid as _uuid        # noqa: E402


def _mode_from_form() -> dict:
    return {"environment": request.form.get("environment"), "dim": request.form.get("dim")}


def _opts_from_form() -> dict:
    try:
        return json.loads(request.form.get("opts") or "{}")
    except json.JSONDecodeError:
        return {}


def _save_uploads(dest: Path) -> dict:
    """Save multipart files into dest/<role>/ and return {role: [Path, ...]}. Roles: csv
    (many), image, tab, model. secure_filename guards against path traversal."""
    files: dict[str, list] = {}
    for role in ("csv", "image", "tab", "model"):
        got = request.files.getlist(role) + request.files.getlist(role + "[]")
        for f in got:
            if not f or not f.filename:
                continue
            sub = dest / role
            sub.mkdir(parents=True, exist_ok=True)
            # Keep the descriptive basename — build_all_bands_data.py decodes the band and the
            # "Blind Scan" filter FROM the filename (e.g. "…EB 13… Blind Scan.CSV"), so
            # secure_filename's underscoring would break ingestion. Basename-only + separator
            # scrub guards against path traversal (Path().name drops any directory components).
            base = Path(f.filename).name or (role + ".bin")
            base = base.replace("/", "_").replace("\\", "_")
            if base in (".", ".."):
                base = role + ".bin"
            p = sub / base
            f.save(str(p))
            files.setdefault(role, []).append(p)
    return files


def _slug(s: str) -> str:
    s = _re.sub(r"[^a-z0-9]+", "-", (s or "project").lower()).strip("-")
    return s or "project"


@app.post("/api/validate")
def api_validate():
    import validate as _validate
    tmp = Path(_tempfile.mkdtemp(prefix="rfval_"))
    try:
        files = _save_uploads(tmp)
        return jsonify(_validate.validate(_mode_from_form(), files, _opts_from_form()))
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/projects")
def api_create_project():
    import pipeline as _pipeline
    import validate as _validate
    mode, opts = _mode_from_form(), _opts_from_form()
    label = request.form.get("label") or \
        f"{(mode['environment'] or '').title()} · {(mode['dim'] or '').upper()} project"
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    pid = f"{_slug(label)}-{_uuid.uuid4().hex[:6]}"
    files = _save_uploads(PROJECTS_DIR / pid / "inputs")
    report = _validate.validate(mode, files, opts)
    if not report.get("ok"):
        _shutil.rmtree(PROJECTS_DIR / pid, ignore_errors=True)
        return jsonify(error="validation failed", report=report), 400
    try:
        manifest = _pipeline.create(PROJECTS_DIR, pid, mode, label, files, opts)
    except _pipeline.PipelineError as e:
        _shutil.rmtree(PROJECTS_DIR / pid, ignore_errors=True)
        return jsonify(error=str(e)), 501
    except Exception as e:  # noqa: BLE001 — surface any pipeline failure to the UI
        _shutil.rmtree(PROJECTS_DIR / pid, ignore_errors=True)
        return jsonify(error=f"pipeline error: {e}"), 500
    return jsonify(serialize(manifest, request.host_url)), 201


@app.delete("/api/projects/<pid>")
def api_delete_project(pid: str):
    if pid in BUILTINS:
        return jsonify(error="cannot delete a built-in project"), 400
    d = PROJECTS_DIR / pid
    if not d.exists():
        abort(404)
    _shutil.rmtree(d, ignore_errors=True)
    return jsonify(ok=True)


# --- Versioned workspace SAVE (git-backed, on the workspace-store branch — never main) -----
@app.post("/api/projects/<pid>/save")
def api_save_project(pid: str):
    """Commit the current project as a new version on the workspace-store branch. Local commit
    only — publishing to GitHub is the separate /api/workspace-store/publish step."""
    import workspace_store as ws
    if pid in BUILTINS:
        return jsonify(error="built-in projects are already committed; nothing to save"), 400
    try:
        v = ws.save_version(pid)
    except ws.StoreError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, saved=bool(v), version=v)


@app.get("/api/projects/<pid>/versions")
def api_project_versions(pid: str):
    import workspace_store as ws
    return jsonify(ws.list_versions(pid))


@app.post("/api/projects/<pid>/restore")
def api_restore_project(pid: str):
    import workspace_store as ws
    sha = (request.get_json(silent=True) or {}).get("sha") or request.form.get("sha")
    if not sha:
        return jsonify(error="missing 'sha'"), 400
    try:
        ws.restore(pid, sha)
    except ws.StoreError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


@app.post("/api/workspace-store/publish")
def api_publish_store():
    """Push the workspace-store branch to origin so saved workspaces survive a fresh clone.
    Outward-facing: the caller (a user clicking Publish) authorizes it."""
    import workspace_store as ws
    try:
        ws.push()
    except ws.StoreError as e:
        return jsonify(error=str(e)), 502
    return jsonify(ok=True, branch=ws.BRANCH)


# --- Static site (repo root) --------------------------------------------------------------
# Explicit /api/* routes above take precedence over this catch-all in Flask's URL map.
@app.get("/")
def index():
    return send_from_directory(REPO_ROOT, INDEX_HTML)


@app.get("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(REPO_ROOT, filename)


# --- self-test ----------------------------------------------------------------------------
def self_test() -> int:
    ok = True
    if not (REPO_ROOT / INDEX_HTML).is_file():
        print(f"FAIL: {INDEX_HTML} not found under {REPO_ROOT}"); ok = False
    if len(BUILTINS) != 4:
        print(f"FAIL: expected 4 built-ins, got {len(BUILTINS)}"); ok = False
    for pid, m in BUILTINS.items():
        for d in m["data"]:
            path = resolve_data_path(m, d["role"])
            missing = not (path and path.is_file())
            if missing and not d.get("optional"):
                print(f"FAIL: {pid} required data '{d['role']}' missing: {path}"); ok = False
            elif missing:
                print(f"warn: {pid} optional data '{d['role']}' absent (ok): {d['src']}")
    with app.test_client() as c:
        if c.get("/api/projects").status_code != 200: print("FAIL: GET /api/projects"); ok = False
        if len(c.get("/api/projects?environment=indoor&dim=2d").get_json()) < 1:
            print("FAIL: indoor/2d filter returned nothing"); ok = False
        if c.get("/api/projects/nope").status_code != 404: print("FAIL: missing project not 404"); ok = False
        if c.get("/api/data/indoor-2d-default/records").status_code != 200:
            print("FAIL: GET /api/data records"); ok = False
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def stress_test() -> int:
    """Integrity gate: creating projects must NEVER touch the committed default data. Runs a
    real create and asserts git sees no change under Data / Data + Model / ARCHIVE, and that the
    project's outputs live only under the (git-ignored) project store."""
    import glob
    import subprocess
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pipeline

    def dirty_data() -> set:
        p = subprocess.run(["git", "status", "--porcelain", "--",
                            "Data", "Data + Model", "ARCHIVE"],
                           cwd=str(REPO_ROOT), capture_output=True, text=True)
        return {ln for ln in p.stdout.splitlines() if ln.strip()}

    base = REPO_ROOT / "FCC_Walk_Outdoor_Indoor_Full" / "Outdoor_Walk_Test_7-24" / "ARCHIVE" / "raw_walk_data"
    csvs = [Path(c) for c in sorted(glob.glob(str(base / "**" / "*Blind Scan*.CSV"), recursive=True))[:6]]
    if not csvs:
        print("stress-test: no sample CSVs found; SKIP")
        return 0

    before = dirty_data()
    store = Path(tempfile.mkdtemp(prefix="rfstress_"))
    ok = True
    try:
        pipeline.create(store, "stress-outdoor", {"environment": "outdoor", "dim": "2d"},
                        "stress", {"csv": csvs}, {})
        new = dirty_data() - before
        if new:
            ok = False
            print("FAIL: create modified tracked default data:")
            for x in new:
                print("   ", x)
        else:
            print("  ✓ create touched no tracked default data (Data / Data + Model / ARCHIVE)")
        if (store / "stress-outdoor" / "data" / "outdoor_walk.js").is_file():
            print("  ✓ project outputs written only under the project store")
        else:
            ok = False
            print("FAIL: expected output not produced under the project store")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL: create raised: {e}")
    finally:
        _shutil.rmtree(store, ignore_errors=True)
    print("stress-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def fake_project_test() -> int:
    """Fabricate a SYNTHETIC walk-test project (fake CSVs + PNG + .TAB) and run it through
    validate + pipeline.create, asserting a valid project — exercising the whole preprocess path
    on fakes. Cleans up after itself; never touches committed data."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_fake_walktest as fake
    import pipeline
    import validate as _validate

    src = Path(tempfile.mkdtemp(prefix="rffake_"))
    store = Path(tempfile.mkdtemp(prefix="rffakestore_"))
    ok = True
    try:
        gen = fake.generate(src / "inputs")
        files = {"csv": gen["csv"], "image": [gen["image"]], "tab": [gen["tab"]]}
        mode = {"environment": "indoor", "dim": "2d"}
        rep = _validate.validate(mode, files, {"cell_m": 0.3})
        g = rep.get("grid") or {}
        print(f"  validate ok: {rep.get('ok')} · grid ≈ {g.get('cols')}×{g.get('rows')} @ {g.get('cell_m')} m"
              f" · columns: {len((rep.get('csv') or {}).get('columns') or [])}")
        ok = bool(rep.get("ok"))
        m = pipeline.create(store, "fake-walktest", mode, "Fake Walk Test", files, {})
        recs = store / "fake-walktest" / "data" / "records_data.js"
        good = recs.is_file() and recs.stat().st_size > 100
        print(f"  create records_data.js: {'ok, ' + format(recs.stat().st_size, ',') + ' bytes' if good else 'MISSING/empty'}"
              f" · ai_ml={m['ai_ml']} · roles={[x['role'] for x in m['data']]}")
        ok = ok and good
    except Exception as e:  # noqa: BLE001
        ok = False
        import traceback
        traceback.print_exc()
    finally:
        _shutil.rmtree(src, ignore_errors=True)
        _shutil.rmtree(store, ignore_errors=True)
    print("fake-project:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def load_env() -> None:
    """Best-effort read of Backend/server/.env (KEY=VALUE lines) into os.environ. No key is
    required for the app; this is only for optional providers (see .env.example)."""
    import os
    env = Path(__file__).resolve().parent / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="RF Simulation Studio project backend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--self-test", action="store_true", help="run checks and exit")
    ap.add_argument("--stress-test", action="store_true",
                    help="create a project and assert the default data is untouched, then exit")
    ap.add_argument("--fake-project", action="store_true",
                    help="fabricate a synthetic walk-test project (fake CSV/PNG/TAB) + validate/create it")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.stress_test:
        return stress_test()
    if args.fake_project:
        return fake_project_test()
    load_env()
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import workspace_store as ws
        n = ws.materialize()   # restore SAVEd workspaces from the workspace-store branch, if any
        if n:
            print(f"workspace-store: materialized {n} saved project(s)")
    except Exception as e:  # noqa: BLE001 — never let the store block startup
        print(f"workspace-store: skipped ({e})")
    print(f"RF Simulation Studio → http://{args.host}:{args.port}  (serving {REPO_ROOT})")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
