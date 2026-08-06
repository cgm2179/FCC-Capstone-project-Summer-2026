// fdtd_progress.js — drive the #simProgress bar from a SIM V3 full-wave run.
//
// The Python full-wave (FDTD) engine runs on the CPU and writes a progress.json
// (perf_v3.ProgressWriter): {fraction, eta_s, mcells_per_s, step, steps, stage,
// done}. The frontend is a static site (python3 -m http.server, no backend), so
// the browser simply POLLS that file over fetch — no server, no websocket.
//
// Usage:
//   import { pollFdtdProgress, estimateLine } from './fdtd_progress.js';
//   const stop = pollFdtdProgress('/Physics%20Engine/2D/SIM%20V3/out/<run>/progress.json');
// or opt-in on any page hosting the simulator: append ?fdtd_progress=<url>.

function fmtEta(s) {
  if (s == null) return '—';
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

// Up-front estimate string (mirrors perf_v3.estimate_seconds).
export function estimateLine(cells, steps, mcps) {
  const s = (cells * steps) / (mcps * 1e6);
  return `est ~${fmtEta(s)} on this CPU (${Math.round(mcps)} Mcell/s)`;
}

// Ensure the #simProgress bar + a status span exist; create them if the host
// page (e.g. a bare simulator) never rendered them. Reuses simulator.css ids,
// with inline fallbacks so it also works standalone.
export function mountFdtdProgress(anchor) {
  let bar = document.getElementById('simProgress');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'simProgress';
    bar.appendChild(Object.assign(document.createElement('div'),
      { id: 'simProgressFill' }));
    const host = anchor || document.getElementById('sim3dStatus') || document.body;
    if (host.parentNode) host.parentNode.insertBefore(bar, host.nextSibling);
    else host.appendChild(bar);
  }
  // inline fallbacks (harmless if simulator.css already styles these)
  Object.assign(bar.style, {
    display: 'block', height: '8px', minWidth: '160px', width: bar.style.width || '260px',
    background: 'rgba(128,128,128,.25)', borderRadius: '4px', overflow: 'hidden',
  });
  const fill = document.getElementById('simProgressFill');
  Object.assign(fill.style, {
    height: '100%', width: fill.style.width || '0%',
    background: 'var(--accent, #3b82f6)',   // honours the host theme, blue fallback
    transition: 'width .2s linear',
  });
  let status = document.getElementById('fdtdStatus');
  if (!status) {
    status = document.createElement('span');
    status.id = 'fdtdStatus';
    status.style.cssText = 'font:12px/1.5 system-ui,sans-serif;opacity:.85;margin-left:8px;';
    bar.after(status);
  }
  return { bar, fill, status };
}

// Poll progress.json until done. Returns a stop() function.
export function pollFdtdProgress(url, opts = {}) {
  const { intervalMs = 500, onDone, anchor } = opts;
  const ui = mountFdtdProgress(anchor);
  let stopped = false;

  async function tick() {
    if (stopped) return;
    try {
      const bust = (url.includes('?') ? '&' : '?') + 't=' + Date.now();
      const r = await fetch(url + bust, { cache: 'no-store' });
      if (r.ok) {
        const p = await r.json();
        const pct = Math.round((p.fraction || 0) * 100);
        ui.fill.style.width = pct + '%';
        const rate = p.mcells_per_s ? `${Math.round(p.mcells_per_s)} Mcell/s` : '';
        ui.status.textContent = p.done
          ? `full-wave (FDTD) complete — ${rate}`
          : `full-wave (FDTD) ${p.stage || ''} · ${pct}% · ETA ${fmtEta(p.eta_s)} · ${rate}`;
        if (p.done) { stopped = true; if (onDone) onDone(p); return; }
      }
    } catch (e) { /* not started yet / transient — keep polling */ }
    setTimeout(tick, intervalMs);
  }
  tick();
  return () => { stopped = true; };
}

// Zero-impact auto-start: append ?fdtd_progress=<progress.json url> to any page.
try {
  const q = new URLSearchParams(location.search).get('fdtd_progress');
  if (q) pollFdtdProgress(q);
} catch (e) { /* no location (module test) */ }
