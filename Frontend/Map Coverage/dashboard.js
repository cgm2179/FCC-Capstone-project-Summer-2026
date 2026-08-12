/* Indoor Walk-Test dashboard — extracted verbatim from the inline
 * <script> block of Frontend_Data_Display.html (Plotly map, histogram,
 * time-elapsed playback, tab control).
 *
 * MUST remain a CLASSIC script (not type=module): its top-level function
 * declarations (filtered, filteredTimeseries, idwGrid, ...) attach to
 * window, which the 3D viewer module (viewer3d.js) reads. Loads AFTER
 * records_data.js / floorplan_image.js / timeseries_data.js (shared
 * global scope) and AFTER the DOM markup (top-level code calls getElementById).
 */
    // Physical signal limits — used only as hard floors/caps; computeDefaultRange
    // sets the actual color scale from the selected band's own q10..q90, so every
    // band (incl. weak C-band n77/n78 down to ~-136 dBm) auto-scales on its own.
    const ranges = {"rsrp": {"vmin": -140.0, "vmax": -44.0, "unit": "dBm"}, "rsrq": {"vmin": -43.0, "vmax": 20.0, "unit": "dB"}, "cinr": {"vmin": -23.0, "vmax": 40.0, "unit": "dB"}, "rssi": {"vmin": -110.0, "vmax": 0.0, "unit": "dBm"}};
    const units = { rsrp: 'dBm', rsrq: 'dB', cinr: 'dB', rssi: 'dBm' };
    const metricLabels = { rsrp: 'RSRP (dBm)', rsrq: 'RSRQ (dB)', cinr: 'CINR (dB)', rssi: 'RSSI (dBm)' };
    const plotEl = document.getElementById('plot');
    const metricEl = document.getElementById('metric');
    const logTimeEl = document.getElementById('logTime');
    const coverageEl = document.getElementById('coverage');
    const networkEl = document.getElementById('network');
    const bandEl = document.getElementById('band');
    const pciEl = document.getElementById('pci');
    const channelEl = document.getElementById('channel');
    const vminEl = document.getElementById('vmin');
    const vmaxEl = document.getElementById('vmax');
    const sizeEl = document.getElementById('size');
    const statsEl = document.getElementById('stats');
    // OSM under-layer controls (PR #25) live in the map-modes row, which the workspace
    // router (re)builds after this script's top-level runs — so query them fresh at call
    // time (load-time refs go stale) and listen via delegation on document (below).
    const osmBasemapMode = () => { const el = document.getElementById('osmBasemap'); return el ? el.value : 'off'; };
    const planOpacity = () => { const el = document.getElementById('floorOpacity'); return el ? Number(el.value) / 100 : 1; };
    // 2D O2I cross-validation overlay (PR #26): selector value + lazy population.
    const crossvalKey = () => { const el = document.getElementById('crossvalSelect'); return el ? el.value : ''; };
    function populateCrossval() {
      const sel = document.getElementById('crossvalSelect');
      if (!sel || sel.dataset.filled === '1') return;
      for (const k of (window.PL2D_CROSSVAL_KEYS || [])) {
        const e = (window.PL2D_CROSSVAL || {})[k] || {};
        sel.add(new Option(e.label || k, k));
      }
      sel.dataset.filled = '1';
    }
    const tabMapBtn = document.getElementById('tabMapBtn');
    const tabHistogramBtn = document.getElementById('tabHistogramBtn');
    const tabTimeBtn = document.getElementById('tabTimeBtn');
    const mapTabEl = document.getElementById('mapTab');
    const histTabEl = document.getElementById('histTab');
    const timeTabEl = document.getElementById('timeTab');
    const timeSliderEl = document.getElementById('timeSlider');
    const playBtnEl = document.getElementById('playBtn');
    const playSpeedEl = document.getElementById('playSpeed');
    const timeLabelEl = document.getElementById('timeLabel');
    const clockFormatEl = document.getElementById('clockFormat');
    const timeJumpInputEl = document.getElementById('timeJumpInput');
    const timeJumpBtnEl = document.getElementById('timeJumpBtn');
    const timeStatsEl = document.getElementById('timeStats');
    const histMetricEl = document.getElementById('histMetric');
    const histTimeEl = document.getElementById('histTime');
    const histDistPaneEl = document.getElementById('histDistPane');
    const histAntPaneEl = document.getElementById('histAntPane');
    const histSubDistBtn = document.getElementById('histSubDistBtn');
    const histSubAntBtn = document.getElementById('histSubAntBtn');
    const antSectorEl = document.getElementById('antSector');
    const antBinsEl = document.getElementById('antBins');
    const antDistNormEl = document.getElementById('antDistNorm');
    const antPatternPlotEl = document.getElementById('antPatternPlot');
    const antPatternStatsEl = document.getElementById('antPatternStats');
    const antPatternMapEl = document.getElementById('antPatternMap');
    const antMapPointsEl = document.getElementById('antMapPoints');
    let histSubTab = 'dist';   // 'dist' | 'antenna'
    let outdoorWalkPromise = null;
    let antMap = null, antMapLayer = null;   // Leaflet map + dynamic overlay for the pattern-on-OSM view
    const playbackMapEl = document.getElementById('playbackMap');
    const playbackHistogramEl = document.getElementById('playbackHistogram');
    const playbackLineEl = document.getElementById('playbackLine');
    const lineAggregationEl = document.getElementById('lineAggregation');
    const lineRegressionEl = document.getElementById('lineRegression');
    const playbackTableBodyEl = document.getElementById('playbackTableBody');

    const tsRecords = (typeof timeseriesRecords !== 'undefined') ? timeseriesRecords : [];
    const testStartTime = (typeof timeseriesStartTime !== 'undefined') ? new Date(timeseriesStartTime) : null;
    const maxTSec = tsRecords.reduce((m, r) => Number.isFinite(r.t_sec) ? Math.max(m, r.t_sec) : m, 0);
    timeSliderEl.max = String(Math.ceil(maxTSec));


    let currentTab = 'map';
    let playTimer = null;
    let plotKind = null;   // 'cart' | 'mapbox' | 'leaflet' — Map Coverage surface, to purge on switch (PR #25)
    let leafletMap = null, leafletDataLayer = null, leafletFloorOverlay = null, leafletLegend = null;

    // Compact Viridis ramp for the Leaflet live-OSM markers (PR #25) — Plotly owns its own.
    const VIRIDIS = ['#440154','#482878','#3e4a89','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
    function _hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n>>16)&255, (n>>8)&255, n&255]; }
    function viridis(t) {
      t = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
      const x = t * (VIRIDIS.length - 1), i = Math.floor(x), f = x - i;
      if (i >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];
      const a = _hexRgb(VIRIDIS[i]), b = _hexRgb(VIRIDIS[i + 1]);
      return `rgb(${Math.round(a[0]+f*(b[0]-a[0]))},${Math.round(a[1]+f*(b[1]-a[1]))},${Math.round(a[2]+f*(b[2]-a[2]))})`;
    }

    function uniqueSorted(values) {
      return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
    }

    // Network -> Band -> PCI -> Channel cascade: each *Options() function is
    // scoped by the selections above it, so a dropdown only ever offers
    // values that actually occur given the current parent filters.
    function networkOptions() {
      return uniqueSorted(records.map(r => r.network).filter(v => typeof v === 'string' && v.length));
    }

    function bandOptions(networkValue) {
      const scoped = records.filter(r => networkValue === 'all' || r.network === networkValue);
      return uniqueSorted(scoped.map(r => r.band).filter(v => Number.isFinite(v)));
    }

    function pciOptions(networkValue, bandValue) {
      const scoped = records.filter(r =>
        (networkValue === 'all' || r.network === networkValue) &&
        (bandValue === 'all' || String(r.band) === bandValue)
      );
      return uniqueSorted(scoped.map(r => r.pci).filter(v => Number.isFinite(v)));
    }

    function channelOptions(networkValue, bandValue, pciValue) {
      const scoped = records.filter(r =>
        (networkValue === 'all' || r.network === networkValue) &&
        (bandValue === 'all' || String(r.band) === bandValue) &&
        (pciValue === 'all' || String(r.pci) === pciValue)
      );
      return uniqueSorted(scoped.map(r => r.freq).filter(v => Number.isFinite(v)));
    }

    // Tallies how many records share each value of `key`, used to show the
    // "(n)" point counts alongside every dropdown option.
    function countBy(data, key) {
      const counts = {};
      for (const r of data) {
        const v = r[key];
        if (v === undefined || v === null || v === '') continue;
        if (typeof v !== 'string' && !Number.isFinite(v)) continue;
        counts[v] = (counts[v] || 0) + 1;
      }
      return counts;
    }

    function populateNetworks() {
      const current = networkEl.value || 'all';
      networkEl.innerHTML = '';

      const counts = countBy(records, 'network');

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = `All (${records.length})`;
      networkEl.appendChild(allOpt);

      for (const net of networkOptions()) {
        const opt = document.createElement('option');
        opt.value = net;
        opt.textContent = `${net} (${counts[net] || 0})`;
        networkEl.appendChild(opt);
      }

      networkEl.value = [...networkEl.options].some(o => o.value === current) ? current : 'all';
    }

    function populateBands() {
      const current = bandEl.value || 'all';
      bandEl.innerHTML = '';

      const networkValue = networkEl.value || 'all';
      const scoped = records.filter(r => networkValue === 'all' || r.network === networkValue);
      const counts = countBy(scoped, 'band');

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = `All (${scoped.length})`;
      bandEl.appendChild(allOpt);

      for (const b of bandOptions(networkValue)) {
        const opt = document.createElement('option');
        opt.value = String(b);
        opt.textContent = `Band ${b} (${counts[b] || 0})`;
        bandEl.appendChild(opt);
      }

      bandEl.value = [...bandEl.options].some(o => o.value === current) ? current : 'all';
    }

    function populatePcis() {
      const current = pciEl.value || 'all';
      pciEl.innerHTML = '';

      const networkValue = networkEl.value || 'all';
      const bandValue = bandEl.value || 'all';
      const scoped = records.filter(r =>
        (networkValue === 'all' || r.network === networkValue) &&
        (bandValue === 'all' || String(r.band) === bandValue)
      );
      const counts = countBy(scoped, 'pci');

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = `All (${scoped.length})`;
      pciEl.appendChild(allOpt);

      for (const pci of pciOptions(networkValue, bandValue)) {
        const opt = document.createElement('option');
        opt.value = String(pci);
        opt.textContent = `PCI ${pci} (${counts[pci] || 0})`;
        pciEl.appendChild(opt);
      }

      pciEl.value = [...pciEl.options].some(o => o.value === current) ? current : 'all';
    }

    function populateChannels() {
      const current = channelEl.value || 'all';
      channelEl.innerHTML = '';

      const networkValue = networkEl.value || 'all';
      const bandValue = bandEl.value || 'all';
      const pciValue = pciEl.value || 'all';
      const scoped = records.filter(r =>
        (networkValue === 'all' || r.network === networkValue) &&
        (bandValue === 'all' || String(r.band) === bandValue) &&
        (pciValue === 'all' || String(r.pci) === pciValue)
      );
      const counts = countBy(scoped, 'freq');

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = `All (${scoped.length})`;
      channelEl.appendChild(allOpt);

      for (const ch of channelOptions(networkValue, bandValue, pciValue)) {
        const opt = document.createElement('option');
        opt.value = String(ch);
        opt.textContent = `${ch} MHz (${counts[ch] || 0})`;
        channelEl.appendChild(opt);
      }

      channelEl.value = [...channelEl.options].some(o => o.value === current) ? current : 'all';
    }

    function updateMetricCounts() {
      const selectedNetwork = networkEl.value || 'all';
      const selectedBand = bandEl.value || 'all';
      const selectedPci = pciEl.value || 'all';
      const selectedChannel = channelEl.value || 'all';
      const scoped = records.filter(r =>
        (selectedNetwork === 'all' || r.network === selectedNetwork) &&
        (selectedBand === 'all' || String(r.band) === selectedBand) &&
        (selectedPci === 'all' || String(r.pci) === selectedPci) &&
        (selectedChannel === 'all' || String(r.freq) === selectedChannel)
      );
      for (const opt of metricEl.options) {
        const key = opt.value;
        const count = scoped.filter(r => Number.isFinite(r[key])).length;
        opt.textContent = `${metricLabels[key] || key} (${count})`;
      }
    }

    // Two parallel datasets feed this dashboard: `records` (one row per
    // coordinate, averaged across repeat measurements — used by the default
    // Map Coverage view) and `tsRecords` (one row per raw measurement, with
    // elapsed time — used by the Histogram tab and Time Elapsed playback).
    function filtered(metric) {
      const selectedNetwork = networkEl.value || 'all';
      const selectedBand = bandEl.value || 'all';
      const selectedPci = pciEl.value || 'all';
      const selectedChannel = channelEl.value || 'all';
      return records.filter(r =>
        Number.isFinite(r.px) &&
        Number.isFinite(r.py) &&
        Number.isFinite(r[metric]) &&
        (selectedNetwork === 'all' || r.network === selectedNetwork) &&
        (selectedBand === 'all' || String(r.band) === selectedBand) &&
        (selectedPci === 'all' || String(r.pci) === selectedPci) &&
        (selectedChannel === 'all' || String(r.freq) === selectedChannel)
      );
    }

    function scopedTimeseries() {
      const selectedNetwork = networkEl.value || 'all';
      const selectedBand = bandEl.value || 'all';
      const selectedPci = pciEl.value || 'all';
      const selectedChannel = channelEl.value || 'all';
      return tsRecords.filter(r =>
        (selectedNetwork === 'all' || r.network === selectedNetwork) &&
        (selectedBand === 'all' || String(r.band) === selectedBand) &&
        (selectedPci === 'all' || String(r.pci) === selectedPci) &&
        (selectedChannel === 'all' || String(r.freq) === selectedChannel)
      );
    }

    function filteredTimeseries(metric, upToSec) {
      return scopedTimeseries().filter(r =>
        Number.isFinite(r.px) &&
        Number.isFinite(r.py) &&
        Number.isFinite(r[metric]) &&
        (upToSec === undefined || r.t_sec <= upToSec)
      );
    }

    // Timeseries records captured up to a given elapsed second — the basis
    // for everything on the Time Elapsed Playback tab (its own mini map,
    // histogram, and points table all read from this single cutoff, which is
    // what keeps them in sync with each other as the slider moves).
    function timeScoped(upToSec) {
      return scopedTimeseries().filter(r => Number.isFinite(r.t_sec) && r.t_sec <= upToSec);
    }

    // Ordinary least-squares fit of y on x — used to draw a trend line over
    // the metric-vs-time scatter that recomputes (and visibly re-slopes) as
    // more measurements come into scope during playback.
    function linearRegression(xs, ys) {
      const n = xs.length;
      if (n < 2) return null;
      const meanX = xs.reduce((a, b) => a + b, 0) / n;
      const meanY = ys.reduce((a, b) => a + b, 0) / n;
      let num = 0;
      let den = 0;
      for (let i = 0; i < n; i++) {
        num += (xs[i] - meanX) * (ys[i] - meanY);
        den += (xs[i] - meanX) ** 2;
      }
      if (den === 0) return null;
      const slope = num / den;
      return { slope, intercept: meanY - slope * meanX };
    }

    // Gaussian elimination with partial pivoting — used by polynomialRegression
    // to solve the normal equations (X^T X) c = X^T y.
    function solveLinearSystem(A, b) {
      const n = b.length;
      const M = A.map((row, i) => [...row, b[i]]);
      for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let r = col + 1; r < n; r++) {
          if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
        }
        if (Math.abs(M[pivot][col]) < 1e-10) return null;
        [M[col], M[pivot]] = [M[pivot], M[col]];
        for (let r = 0; r < n; r++) {
          if (r === col) continue;
          const factor = M[r][col] / M[col][col];
          for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
        }
      }
      return M.map((row, i) => row[n] / row[i]);
    }

    // Least-squares fit of a degree-N polynomial (default quadratic) via the
    // normal equations over a Vandermonde design matrix.
    function polynomialRegression(xs, ys, degree = 2) {
      const n = xs.length;
      const size = degree + 1;
      if (n < size) return null;
      const XtX = Array.from({ length: size }, () => new Array(size).fill(0));
      const Xty = new Array(size).fill(0);
      for (let i = 0; i < n; i++) {
        const powers = [1];
        for (let k = 1; k <= 2 * degree; k++) powers.push(powers[k - 1] * xs[i]);
        for (let r = 0; r < size; r++) {
          Xty[r] += powers[r] * ys[i];
          for (let c = 0; c < size; c++) XtX[r][c] += powers[r + c];
        }
      }
      const coeffs = solveLinearSystem(XtX, Xty);
      if (!coeffs) return null;
      return { predict: (x) => coeffs.reduce((sum, c, k) => sum + c * Math.pow(x, k), 0) };
    }

    // Exponential fit y = a*e^(b*x), done via log-linear regression on
    // ln(y - shift). Signal metrics (dBm/dB) are frequently negative, so
    // values are shifted to be strictly positive before the log transform
    // and the shift is added back when predicting.
    function exponentialRegression(xs, ys) {
      if (xs.length < 2) return null;
      const yMin = Math.min(...ys);
      const shift = yMin <= 0 ? yMin - 1 : 0;
      const logYs = ys.map(y => Math.log(y - shift));
      const fit = linearRegression(xs, logYs);
      if (!fit) return null;
      const a = Math.exp(fit.intercept);
      const b = fit.slope;
      return { predict: (x) => a * Math.exp(b * x) + shift };
    }

    // Dispatches to the selected regression kind; all three return the same
    // { predict(x) } shape so the caller doesn't need to branch again.
    function fitRegression(kind, xs, ys) {
      if (kind === 'linear') {
        const fit = linearRegression(xs, ys);
        return fit ? { predict: (x) => fit.intercept + fit.slope * x } : null;
      }
      if (kind === 'exponential') return exponentialRegression(xs, ys);
      if (kind === 'polynomial') return polynomialRegression(xs, ys, 2);
      return null;
    }

    // Builds the line-chart series for the Playback tab at the selected
    // granularity: raw chronological measurements, or averaged into fixed
    // N-minute buckets (x = bucket midpoint in minutes).
    function buildLineSeries(captured, metric, aggregation) {
      const withMetric = captured
        .filter(r => Number.isFinite(r.t_sec) && Number.isFinite(r[metric]))
        .sort((a, b) => a.t_sec - b.t_sec);

      if (aggregation === 'point') {
        return { x: withMetric.map(r => r.t_sec / 60), y: withMetric.map(r => r[metric]), n: withMetric.length };
      }

      const bucketSec = Number(aggregation) * 60;
      const buckets = new Map();
      for (const r of withMetric) {
        const idx = Math.floor(r.t_sec / bucketSec);
        if (!buckets.has(idx)) buckets.set(idx, []);
        buckets.get(idx).push(r[metric]);
      }
      const idxs = [...buckets.keys()].sort((a, b) => a - b);
      const x = idxs.map(i => (i * bucketSec + bucketSec / 2) / 60);
      const y = idxs.map(i => {
        const vals = buckets.get(i);
        return vals.reduce((sum, v) => sum + v, 0) / vals.length;
      });
      return { x, y, n: withMetric.length };
    }

    function metricValues(data, metric) {
      return data.map(r => r[metric]).filter(v => Number.isFinite(v));
    }

    function quantile(values, q) {
      if (!values.length) return NaN;
      const sorted = [...values].sort((a, b) => a - b);
      const pos = (sorted.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
      }
      return sorted[base];
    }

    // Auto-scale the color range to the 10th-90th percentile of the current
    // selection, clamped to the metric's hard physical bounds (`ranges`) so
    // a skewed filter can't blow out the color scale.
    function computeDefaultRange(data, metric) {
      const vals = metricValues(data, metric);
      if (!vals.length) {
        return ranges[metric];
      }
      const q10 = quantile(vals, 0.10);
      const q90 = quantile(vals, 0.90);
      const hard = ranges[metric];
      let vmin = Math.max(q10, hard.vmin);
      let vmax = Math.min(q90, hard.vmax);
      if (!(vmin < vmax)) {
        vmin = Math.min(...vals);
        vmax = Math.max(...vals);
      }
      return { vmin, vmax };
    }

    // Inverse-distance-weighted interpolation: builds an nx*ny grid over the
    // floor plan and estimates each cell as a distance-weighted average of
    // nearby points, powering the "Gradient Coverage" heatmap mode.
    function idwGrid(data, metric, width, height, nx = 110, ny = 80, power = 2.0) {
      if (data.length < 3) return null;
      const xVals = Array.from({ length: nx }, (_, i) => (i * width) / (nx - 1));
      const yVals = Array.from({ length: ny }, (_, i) => (i * height) / (ny - 1));
      const z = [];

      for (let yi = 0; yi < ny; yi++) {
        const row = [];
        const y = yVals[yi];
        for (let xi = 0; xi < nx; xi++) {
          const x = xVals[xi];
          let num = 0;
          let den = 0;
          let exact = null;

          for (const p of data) {
            const v = p[metric];
            if (!Number.isFinite(v)) continue;
            const dx = x - p.px;
            const dy = y - p.py;
            const d2 = dx * dx + dy * dy;
            if (d2 < 1e-8) {
              exact = v;
              break;
            }
            const w = 1.0 / Math.pow(d2, power / 2.0);
            num += w * v;
            den += w;
          }

          if (exact !== null) {
            row.push(exact);
          } else if (den > 0) {
            row.push(num / den);
          } else {
            row.push(null);
          }
        }
        z.push(row);
      }

      return { x: xVals, y: yVals, z };
    }

    function hoverTemplate(metric) {
      const u = units[metric] || '';
      return [
        `<b>${metric.toUpperCase()}</b>: %{{marker.color:.2f}} ${u}`,
        'Lat: %{customdata[0]:.6f}',
        'Lon: %{customdata[1]:.6f}',
        'Samples: %{customdata[2]}',
        'PCI: %{customdata[3]}',
        'Freq: %{customdata[4]}',
        '<extra></extra>'
      ].join('<br>');
    }

    function updateStats(data, metric) {
      const vals = metricValues(data, metric);
      const n = vals.length;
      const min = n ? Math.min(...vals).toFixed(2) : 'NA';
      const max = n ? Math.max(...vals).toFixed(2) : 'NA';
      const avg = n ? (vals.reduce((a,b) => a+b, 0) / n).toFixed(2) : 'NA';
      const networkText = networkEl.value === 'all' ? 'All' : networkEl.value;
      const bandText = bandEl.value === 'all' ? 'All' : `Band ${bandEl.value}`;
      const pciText = pciEl.value === 'all' ? 'All' : `PCI ${pciEl.value}`;
      const channelText = channelEl.value === 'all' ? 'All' : `${channelEl.value} MHz`;
      const modeText = coverageEl.value;
      statsEl.innerHTML = `
        <div>Points: <b>${n}</b></div>
        <div>Metric: <b>${metric.toUpperCase()}</b></div>
        <div>Mode: <b>${modeText}</b></div>
        <div>Network / Band / PCI: <b>${networkText} / ${bandText} / ${pciText}</b></div>
        <div>Channel: <b>${channelText}</b></div>
        <div>Min / Max: <b>${min} / ${max}</b></div>
        <div>Mean: <b>${avg}</b></div>
      `;
    }

    function applyDefaults(metric, data) {
      const d = computeDefaultRange(data, metric);
      vminEl.value = d.vmin.toFixed(2);
      vmaxEl.value = d.vmax.toFixed(2);
    }

    // Renders the Map Coverage tab. When Time Elapsed Playback is off this
    // plots the aggregated `records`; when it's on it plots `tsRecords` up
    // to the current slider position instead, so points appear in the order
    // they were actually walked.
    // Map Coverage always plots the whole (aggregated) dataset — it does not
    // respond to the Time Elapsed Playback slider. The time-scoped view lives
    // entirely on the Time Elapsed Playback tab (see renderPlayback below).
    function render() {
      populateCrossval();
      const cvk = crossvalKey();
      if (cvk && window.PL2D_CROSSVAL && window.PL2D_CROSSVAL[cvk]) { renderCrossval(cvk); return; }
      const cs = document.getElementById('crossvalStats'); if (cs) cs.textContent = '';
      if (osmBasemapMode() === 'live') { renderLeaflet(); return; }
      // non-live: Plotly surface owns #plot; hide the Leaflet surface (unless in 3D CAD mode)
      const in3D = !!(document.getElementById('mapMode3dBtn') &&
                      document.getElementById('mapMode3dBtn').classList.contains('active'));
      const lmDiv = document.getElementById('leafletMap');
      if (lmDiv) lmDiv.style.display = 'none';
      const plotDiv = document.getElementById('plot');
      if (plotDiv && !in3D) plotDiv.style.display = '';
      const metric = metricEl.value;
      const data = filtered(metric);
      const cmin = Number(vminEl.value);
      const cmax = Number(vmaxEl.value);
      const size = Number(sizeEl.value);
      const mode = coverageEl.value;

      const traces = [];

      if (mode === 'gradient') {
        const grid = idwGrid(data, metric, 1150, 515);
        if (grid) {
          traces.push({
            type: 'heatmap',
            x: grid.x,
            y: grid.y,
            z: grid.z,
            colorscale: 'Viridis',
            zmin: cmin,
            zmax: cmax,
            opacity: 0.72,
            colorbar: {
              title: `${metric.toUpperCase()} (${units[metric] || ''})`,
              len: 0.8,
              thickness: 16
            },
            hovertemplate: `${metric.toUpperCase()}: %{z:.2f} ${units[metric] || ''}<extra></extra>`
          });
        }
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          x: data.map(r => r.px),
          y: data.map(r => r.py),
          customdata: data.map(r => [r.latitude, r.longitude, (r.n_samples !== undefined ? r.n_samples : 1), r.pci, r.freq]),
          marker: { size: Math.max(4, Math.floor(size / 3)), color: '#111', opacity: 0.35 },
          hovertemplate: hoverTemplate(metric),
          showlegend: false
        });
      } else {
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          x: data.map(r => r.px),
          y: data.map(r => r.py),
          customdata: data.map(r => [r.latitude, r.longitude, (r.n_samples !== undefined ? r.n_samples : 1), r.pci, r.freq]),
          marker: {
            size,
            color: metricValues(data, metric),
            colorscale: 'Viridis',
            cmin,
            cmax,
            colorbar: {
              title: `${metric.toUpperCase()} (${units[metric] || ''})`,
              len: 0.8,
              thickness: 16
            },
            line: {color: '#111', width: 0.35},
            opacity: 0.88
          },
          hovertemplate: hoverTemplate(metric)
        });
      }

      const layout = {
        margin: {l: 12, r: 16, t: 40, b: 12},
        title: {
          text: `${(networkEl.value === 'all' ? 'All Networks' : networkEl.value)} · ${(bandEl.value === 'all' ? 'All Bands' : ('Band ' + bandEl.value))} · ${(pciEl.value === 'all' ? 'All PCI' : ('PCI ' + pciEl.value))} · ${(channelEl.value === 'all' ? 'All Channels' : (channelEl.value + ' MHz'))} · ${metric.toUpperCase()}`,
          x: 0.02
        },
        xaxis: {visible: false, range: [0, 1150] },
        yaxis: {visible: false, range: [515, 0], scaleanchor: 'x'},
        images: basemapImages(),
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
      };

      if (plotKind === 'mapbox') Plotly.purge(plotEl);   // cartesian<->mapbox needs a clean slate
      plotKind = 'cart';
      Plotly.react(plotEl, traces, layout, {responsive: true, displaylogo: false});
      updateStats(data, metric);
    }

    // Layout images for Map Coverage: optional OSM basemap below the (opacity-faded)
    // floor plan. INDOOR_BASEMAP is warped to the exact 1150x515 plan frame, so it
    // drops straight in. Absent asset -> just the plan (unchanged behaviour).
    function basemapImages() {
      const fp = Object.assign({}, floorPlanImage, { opacity: planOpacity() });
      if (osmBasemapMode() === 'static' && window.INDOOR_BASEMAP) {
        return [{
          source: window.INDOOR_BASEMAP.source,
          xref: 'x', yref: 'y', x: 0, y: 0, sizex: 1150, sizey: 515,
          xanchor: 'left', yanchor: 'top', sizing: 'stretch', layer: 'below'
        }, fp];
      }
      return [fp];
    }

    // Floor-plan pixel (px,py) -> [lon,lat] via the plan's georeference affine. This is
    // the EPSG:3857 (Web Mercator / "Pseudo-Mercator") solution fitted from the .TAB GCPs,
    // so a point transformed this way lands at the building's true real-world spot — unlike
    // the record's own latitude/longitude, which is raw indoor GPS (mean ~76 m, max ~260 m
    // off here, useless for placing points inside an 80 m building). Same math as
    // georef.pxToLonlat, inlined (dashboard.js is a classic script, can't import the ESM).
    function pxToLonLat(px, py) {
      const A = (window.FLOORPLAN_META || {}).affine_px_to_lonlat;
      if (!A || !Number.isFinite(px) || !Number.isFinite(py)) return null;
      return [A[0][0] * px + A[0][1] * py + A[0][2],
              A[1][0] * px + A[1][1] * py + A[1][2]];
    }

    // The 4 floor-plan corners in lon/lat (TL, TR, BR, BL) — a rotated quad, since the
    // plan is -7.33 deg off north.
    function planCornersLonLat() {
      if (!(window.FLOORPLAN_META || {}).affine_px_to_lonlat) return null;
      const W = 1150, H = 515;
      return [pxToLonLat(0, 0), pxToLonLat(W, 0), pxToLonLat(W, H), pxToLonLat(0, H)];
    }

    // Live OSM path: Plotly Mapbox (open-street-map, no token). Floor plan draped on the
    // 4 lon/lat corners; RSRP from record lat/long; base-station pins + Forte Hall bearing
    // line from window.BASE_STATIONS. Separate from render() because the pixel-space plot
    // and the mapbox plot are different Plotly worlds.
    function renderMapbox() {
      const metric = metricEl.value;
      const data = filtered(metric);
      const cmin = Number(vminEl.value);
      const cmax = Number(vmaxEl.value);
      const size = Number(sizeEl.value);
      const mode = coverageEl.value;
      const corners = planCornersLonLat();
      // Georeference the points (px,py -> lon/lat) rather than trusting raw indoor GPS —
      // same reasoning as renderLeaflet. Falls back to the record lon/lat only if the plan
      // has no affine.
      const geo = data.map(r => pxToLonLat(r.px, r.py) || [r.longitude, r.latitude]);
      const lls = corners || geo;
      const center = {
        lon: lls.reduce((a, c) => a + c[0], 0) / lls.length,
        lat: lls.reduce((a, c) => a + c[1], 0) / lls.length
      };
      const op = planOpacity();

      const traces = [];
      if (mode === 'gradient') {
        traces.push({
          type: 'densitymapbox',
          lon: geo.map(g => g[0]), lat: geo.map(g => g[1]),
          z: metricValues(data, metric),
          radius: Math.max(8, size * 2), colorscale: 'Viridis',
          zmin: cmin, zmax: cmax, opacity: 0.75,
          colorbar: { title: `${metric.toUpperCase()} (${units[metric] || ''})`, len: 0.8, thickness: 16 },
          hovertemplate: `${metric.toUpperCase()}: %{z:.2f}<extra></extra>`
        });
      } else {
        traces.push({
          type: 'scattermapbox', mode: 'markers',
          lon: geo.map(g => g[0]), lat: geo.map(g => g[1]),
          marker: {
            size: Math.max(6, Math.floor(size / 2)), color: metricValues(data, metric),
            colorscale: 'Viridis', cmin, cmax, opacity: 0.9,
            colorbar: { title: `${metric.toUpperCase()} (${units[metric] || ''})`, len: 0.8, thickness: 16 }
          },
          hovertemplate: `${metric.toUpperCase()}: %{marker.color:.2f}<extra></extra>`
        });
      }

      const bs = (window.BASE_STATIONS || []).filter(s => s.lat != null);
      if (bs.length) {
        traces.push({
          type: 'scattermapbox', mode: 'markers+text',
          lon: bs.map(s => s.lon), lat: bs.map(s => s.lat),
          text: bs.map(s => s.name || s.site_id), textposition: 'top right',
          marker: { size: 13, color: bs.map(s => s.status === 'live' ? '#c0392b' : '#7f8c8d') },
          hovertext: bs.map(s => `${s.name}<br>${(s.sectors || []).length} sectors · ${s.distance_m} m · arrival ${s.bearing_deg}°`),
          hoverinfo: 'text', showlegend: false
        });
        const fh = bs.find(s => s.site_id === 'BS7_forte_hall');
        if (fh) {
          traces.push({
            type: 'scattermapbox', mode: 'lines',
            lon: [fh.lon, center.lon], lat: [fh.lat, center.lat],
            line: { width: 2, color: '#c0392b' }, opacity: 0.6,
            hoverinfo: 'skip', showlegend: false
          });
        }
      }

      const layout = {
        margin: { l: 0, r: 0, t: 40, b: 0 },
        title: {
          text: `${(networkEl.value === 'all' ? 'All Networks' : networkEl.value)} · ${(bandEl.value === 'all' ? 'All Bands' : ('Band ' + bandEl.value))} · ${(pciEl.value === 'all' ? 'All PCI' : ('PCI ' + pciEl.value))} · ${metric.toUpperCase()} · OSM live`,
          x: 0.02
        },
        mapbox: {
          style: 'open-street-map', center, zoom: 16.4,
          layers: corners ? [{
            sourcetype: 'image', source: floorPlanImage.source,
            coordinates: corners, below: 'traces', opacity: op
          }] : []
        },
        paper_bgcolor: '#ffffff'
      };

      if (plotKind === 'cart') Plotly.purge(plotEl);
      plotKind = 'mapbox';
      Plotly.react(plotEl, traces, layout, { responsive: true, displaylogo: false });
      updateStats(data, metric);
    }

    // Live OSM via Leaflet with muted gray CartoDB Positron tiles (PR #25) — the RF data
    // reads more clearly on gray than on the colourful OSM raster. Floor plan draped as a
    // rotated image overlay; RSRP as Viridis circle markers; base-station pins + Forte Hall
    // bearing line. Falls back to the Plotly-mapbox path if Leaflet didn't load.
    function renderLeaflet() {
      if (typeof L === 'undefined') { renderMapbox(); return; }
      const in3D = !!(document.getElementById('mapMode3dBtn') &&
                      document.getElementById('mapMode3dBtn').classList.contains('active'));
      const lmDiv = document.getElementById('leafletMap');
      const plotDiv = document.getElementById('plot');
      if (in3D) { if (lmDiv) lmDiv.style.display = 'none'; return; }   // 3D CAD view owns the surface
      if (plotDiv) plotDiv.style.display = 'none';
      if (lmDiv) lmDiv.style.display = 'block';
      if (plotKind === 'cart' || plotKind === 'mapbox') Plotly.purge(plotEl);
      plotKind = 'leaflet';

      const metric = metricEl.value;
      const data = filtered(metric);
      const cmin = Number(vminEl.value), cmax = Number(vmaxEl.value);
      const size = Number(sizeEl.value);
      const op = planOpacity();
      const corners = planCornersLonLat();     // [TL,TR,BR,BL] as [lon,lat]
      const cLat = corners ? (corners[0][1] + corners[2][1]) / 2 : (data[0] && data[0].latitude) || 38.9036;
      const cLon = corners ? (corners[0][0] + corners[2][0]) / 2 : (data[0] && data[0].longitude) || -77.0074;

      if (!leafletMap) {
        leafletMap = L.map('leafletMap', { preferCanvas: true }).setView([cLat, cLon], 17);
        // Muted CARTO Positron reads best under the RF data, but that CDN isn't always
        // reachable — if its tiles fail, swap once to standard OpenStreetMap so the streets
        // (e.g. L St NE, so you can tell which side of the building faces the road) still
        // show instead of a blank grey pane.
        const base = L.tileLayer('https://{s}.basemap.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
          subdomains: 'abcd', maxZoom: 20,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(leafletMap);
        let tilesSwapped = false;
        base.on('tileerror', () => {
          if (tilesSwapped) return;
          tilesSwapped = true;
          leafletMap.removeLayer(base);
          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            subdomains: 'abc', maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          }).addTo(leafletMap);
        });
        leafletDataLayer = L.layerGroup().addTo(leafletMap);
      }

      // floor-plan drape (rotated overlay if the plugin is present; else skip — data still shows)
      if (leafletFloorOverlay) { leafletMap.removeLayer(leafletFloorOverlay); leafletFloorOverlay = null; }
      if (corners && floorPlanImage && floorPlanImage.source && L.imageOverlay && L.imageOverlay.rotated) {
        leafletFloorOverlay = L.imageOverlay.rotated(
          floorPlanImage.source,
          L.latLng(corners[0][1], corners[0][0]),   // top-left
          L.latLng(corners[1][1], corners[1][0]),   // top-right
          L.latLng(corners[3][1], corners[3][0]),   // bottom-left
          { opacity: op, interactive: false }
        ).addTo(leafletMap);
      }

      // RF points — placed by the floor-plan georeference (px,py -> lon/lat), NOT by the
      // record's raw GPS. Indoor GPS here is ~76 m off on average, which would spray the
      // walk across several city blocks; the pixel positions are drawing-exact, so routed
      // through the Web-Mercator affine they sit precisely on the draped plan and align to
      // the OSM streets underneath.
      leafletDataLayer.clearLayers();
      const span = (cmax - cmin) || 1;
      const vals = metricValues(data, metric);
      const rad = Math.max(3, Math.floor(size / 3));
      for (let i = 0; i < data.length; i++) {
        const r = data[i], v = vals[i];
        const ll = pxToLonLat(r.px, r.py);
        if (!ll) continue;
        L.circleMarker([ll[1], ll[0]], {
          radius: rad, fillColor: viridis((v - cmin) / span),
          color: '#111', weight: 0.3, fillOpacity: 0.9
        }).bindPopup(`${metric.toUpperCase()}: ${Number.isFinite(v) ? v.toFixed(2) : '—'} ${units[metric] || ''}`
          + `<br>PCI ${r.pci ?? '—'} · ${r.network || ''} B${r.band ?? '—'}`)
          .addTo(leafletDataLayer);
      }

      // base-station pins + Forte Hall bearing line
      const bs = (window.BASE_STATIONS || []).filter(s => s.lat != null);
      for (const s of bs) {
        L.circleMarker([s.lat, s.lon], {
          radius: 7, fillColor: s.status === 'live' ? '#c0392b' : '#7f8c8d',
          color: '#fff', weight: 1.5, fillOpacity: 1
        }).bindTooltip(`${s.name} · ${(s.sectors || []).length} sectors · ${s.distance_m} m · arr ${s.bearing_deg}°`,
          { direction: 'top' }).addTo(leafletDataLayer);
      }
      const fh = bs.find(s => s.site_id === 'BS7_forte_hall');
      if (fh) {
        L.polyline([[fh.lat, fh.lon], [cLat, cLon]],
          { color: '#c0392b', weight: 2, opacity: 0.6, dashArray: '6 4' }).addTo(leafletDataLayer);
      }

      if (corners) {
        leafletMap.fitBounds(L.latLngBounds(corners.map(c => [c[1], c[0]])).pad(0.6));
      }

      // legend
      if (leafletLegend) leafletMap.removeControl(leafletLegend);
      leafletLegend = L.control({ position: 'bottomright' });
      leafletLegend.onAdd = () => {
        const d = L.DomUtil.create('div');
        d.style.cssText = 'background:#fff;padding:6px 8px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px sans-serif;width:140px;';
        d.innerHTML = `${metric.toUpperCase()} (${units[metric] || ''})`
          + `<div style="height:10px;margin:4px 0;background:linear-gradient(90deg,${VIRIDIS.join(',')});"></div>`
          + `<div style="display:flex;justify-content:space-between;"><span>${cmin}</span><span>${cmax}</span></div>`;
        return d;
      };
      leafletLegend.addTo(leafletMap);

      leafletMap.invalidateSize();
      updateStats(data, metric);
    }

    // PR #26: overlay the 2D O2I predicted RSRP heatmap (viridis, from export_web.py) under
    // the measured points coloured by residual (measured − predicted, level-calibrated).
    function renderCrossval(key) {
      const e = window.PL2D_CROSSVAL[key];
      const lmDiv = document.getElementById('leafletMap'); if (lmDiv) lmDiv.style.display = 'none';
      const plotDiv = document.getElementById('plot'); if (plotDiv) plotDiv.style.display = '';
      if (plotKind === 'mapbox' || plotKind === 'leaflet') Plotly.purge(plotEl);
      plotKind = 'cart';
      const pts = e.points || [];
      const absr = pts.map(p => Math.abs(p.resid)).sort((a, b) => a - b);
      const lim = Math.max(6, absr.length ? absr[Math.floor(absr.length * 0.95)] : 6);
      const traces = [{
        type: 'scattergl', mode: 'markers',
        x: pts.map(p => p.px), y: pts.map(p => p.py),
        marker: {
          size: 10, color: pts.map(p => p.resid), colorscale: 'RdBu', reversescale: true,
          cmin: -lim, cmax: lim, cmid: 0, line: { color: '#222', width: 0.5 },
          colorbar: { title: 'measured − predicted (dB)', len: 0.8, thickness: 16 }
        },
        text: pts.map(p => `measured ${p.meas} · predicted ${p.pred_cal} · resid ${p.resid} dB`),
        hovertemplate: '%{text}<extra></extra>', showlegend: false
      }];
      const pred = {
        source: e.rsrp_png, xref: 'x', yref: 'y', x: 0, y: 0, sizex: 1150, sizey: 515,
        xanchor: 'left', yanchor: 'top', sizing: 'stretch', layer: 'below', opacity: 0.85
      };
      const fp = Object.assign({}, floorPlanImage, { opacity: 0.30 });
      const sign = e.level_cal_db >= 0 ? '+' : '';
      const layout = {
        margin: { l: 12, r: 16, t: 54, b: 12 },
        title: {
          text: `Predicted 2D O2I vs measured — ${e.label}`
            + `<br><sub>ρ ${e.spearman_rho} · RMSE(level-cal) ${e.rmse_after_cal_db} dB · `
            + `level ${sign}${e.level_cal_db} dB · n ${e.n} · measured mean ${e.measured_mean} dBm · `
            + `viridis = predicted RSRP, dots = residual</sub>`,
          x: 0.02, font: { size: 13 }
        },
        xaxis: { visible: false, range: [0, 1150] },
        yaxis: { visible: false, range: [515, 0], scaleanchor: 'x' },
        images: [pred, fp],
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff'
      };
      Plotly.react(plotEl, traces, layout, { responsive: true, displaylogo: false });
      const stats = document.getElementById('crossvalStats');
      if (stats) stats.textContent = `ρ ${e.spearman_rho} · RMSE ${e.rmse_after_cal_db} dB · n ${e.n}`;
    }

    // m:ss elapsed-duration label (e.g. "2:05"), independent of clock format.
    function formatDuration(sec) {
      if (!Number.isFinite(sec)) return '0:00';
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}:${String(s).padStart(2, '0')}`;
    }

    // testStartTime is stored as UTC so it round-trips exactly regardless of
    // the viewer's timezone; timeZone: 'UTC' below makes toLocaleTimeString
    // render the raw recorded time-of-day rather than shifting it.
    function formatClockTime(sec) {
      if (!testStartTime || !Number.isFinite(sec)) return '';
      const t = new Date(testStartTime.getTime() + sec * 1000);
      const is24 = clockFormatEl.value === '24';
      return t.toLocaleTimeString('en-US', {
        hour12: !is24,
        hour: is24 ? '2-digit' : 'numeric',
        minute: '2-digit',
        second: '2-digit',
        timeZone: 'UTC'
      });
    }

    function updateTimeLabel() {
      const elapsed = Number(timeSliderEl.value);
      const clockText = formatClockTime(elapsed);
      const durationText = `${formatDuration(elapsed)} / ${formatDuration(maxTSec)}`;
      timeLabelEl.textContent = clockText ? `${clockText} · ${durationText}` : durationText;
    }

    // Histogram tab: always draws from `tsRecords` (raw per-measurement data)
    // rather than the position-averaged `records`, since a distribution over
    // averaged points would understate how many samples were actually taken.
    // Histogram tab always plots the whole dataset too — stationary, just
    // like Map Coverage. The animated, time-scoped version of these charts
    // lives on the Time Elapsed Playback tab (see renderPlayback below).
    function renderHistograms() {
      const metric = metricEl.value;
      const scoped = scopedTimeseries();
      const metricVals = metricValues(scoped, metric);

      Plotly.react(histMetricEl, [{
        type: 'histogram',
        x: metricVals,
        marker: { color: '#0a7f7a' },
        nbinsx: 40,
        hovertemplate: `${metric.toUpperCase()}: %{x:.1f} ${units[metric] || ''}<br>Count: %{y}<extra></extra>`
      }], {
        title: { text: `${metric.toUpperCase()} Distribution (${metricVals.length} measurements)` },
        xaxis: { title: { text: `${metric.toUpperCase()} (${units[metric] || ''})` } },
        yaxis: { title: { text: 'Measurement count' } },
        margin: { t: 40, l: 55, r: 16, b: 45 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        bargap: 0.05
      }, { responsive: true, displaylogo: false });

      // How the selected metric trends over the walk: raw per-measurement
      // points, a per-minute average line, and a regression trend line, so
      // signal drift/dead spots show up against elapsed time. All three
      // recompute from `scoped` above, so during playback they grow point by
      // point (and the regression slope updates) as elapsed time advances.
      const withMetric = scoped.filter(r => Number.isFinite(r.t_sec) && Number.isFinite(r[metric]));
      const minuteBuckets = new Map();
      for (const r of withMetric) {
        const minute = Math.floor(r.t_sec / 60);
        if (!minuteBuckets.has(minute)) minuteBuckets.set(minute, []);
        minuteBuckets.get(minute).push(r[metric]);
      }
      const minutes = [...minuteBuckets.keys()].sort((a, b) => a - b);
      const minuteAvgs = minutes.map(m => {
        const vals = minuteBuckets.get(m);
        return vals.reduce((sum, v) => sum + v, 0) / vals.length;
      });

      const xsMinutes = withMetric.map(r => r.t_sec / 60);
      const ysMetric = withMetric.map(r => r[metric]);
      const regression = linearRegression(xsMinutes, ysMetric);

      const timeTraces = [
        {
          type: 'scattergl',
          mode: 'markers',
          x: xsMinutes,
          y: ysMetric,
          marker: { size: 4, color: '#b7c4c1', opacity: 0.6 },
          name: 'Measurements',
          hovertemplate: `Minute %{x:.1f}<br>${metric.toUpperCase()}: %{y:.1f} ${units[metric] || ''}<extra></extra>`
        },
        {
          type: 'scatter',
          mode: 'lines+markers',
          x: minutes,
          y: minuteAvgs,
          line: { color: '#3d6fae', width: 2 },
          marker: { size: 5, color: '#3d6fae' },
          name: 'Per-minute average',
          hovertemplate: `Minute %{x}<br>Avg ${metric.toUpperCase()}: %{y:.1f} ${units[metric] || ''}<extra></extra>`
        }
      ];

      if (regression && xsMinutes.length) {
        const xMin = Math.min(...xsMinutes);
        const xMax = Math.max(...xsMinutes);
        timeTraces.push({
          type: 'scatter',
          mode: 'lines',
          x: [xMin, xMax],
          y: [regression.intercept + regression.slope * xMin, regression.intercept + regression.slope * xMax],
          line: { color: '#c0392b', width: 2, dash: 'dash' },
          name: 'Trend (regression)',
          hoverinfo: 'skip'
        });
      }

      Plotly.react(histTimeEl, timeTraces, {
        title: { text: `${metric.toUpperCase()} Over Elapsed Time (${withMetric.length} measurements)` },
        xaxis: { title: { text: 'Elapsed time (minutes)' }, type: logTimeEl.checked ? 'log' : 'linear' },
        yaxis: { title: { text: `${metric.toUpperCase()} (${units[metric] || ''})` } },
        margin: { t: 40, l: 55, r: 16, b: 45 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        legend: { orientation: 'h', y: -0.18 }
      }, { responsive: true, displaylogo: false });
    }

    // ---- Antenna Pattern (Statistics sub-tab) --------------------------------
    // Measured polar pattern per surveyed base-station sector: metric vs. azimuth
    // from the Tx to each walk point, plus a binned mean envelope. Indoor uses
    // floor-plan px→lon/lat (never raw indoor GPS); Outdoor uses OUTDOOR_WALK.

    function isOutdoorWorkspace() {
      return !!(window.appMode && window.appMode.environment === 'outdoor');
    }

    // Inline of georef.js pxToLonlat — dashboard.js is a classic script and cannot
    // import the ES module. Affines come from window.FLOORPLAN_META.
    function pxToLonlatInline(px, py) {
      const meta = window.FLOORPLAN_META;
      if (!meta || !meta.affine_px_to_lonlat) return null;
      const A = meta.affine_px_to_lonlat;
      return {
        lon: A[0][0] * px + A[0][1] * py + A[0][2],
        lat: A[1][0] * px + A[1][1] * py + A[1][2]
      };
    }

    function bearingDeg(lat1, lon1, lat2, lon2) {
      const φ1 = lat1 * Math.PI / 180, φ2 = lat2 * Math.PI / 180;
      const Δλ = (lon2 - lon1) * Math.PI / 180;
      const y = Math.sin(Δλ) * Math.cos(φ2);
      const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
      return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
    }

    function haversineM(lat1, lon1, lat2, lon2) {
      const R = 6371000;
      const φ1 = lat1 * Math.PI / 180, φ2 = lat2 * Math.PI / 180;
      const Δφ = (lat2 - lat1) * Math.PI / 180;
      const Δλ = (lon2 - lon1) * Math.PI / 180;
      const a = Math.sin(Δφ / 2) ** 2 + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
      return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
    }

    function compassLabel(deg) {
      const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
      return dirs[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
    }

    // Lazy-load Data/outdoor_walk.js if somehow absent (main HTML already loads it
    // for the 3D OSM ground — this is the defensive path used when that script
    // tag is stripped or blocked).
    function ensureOutdoorWalk() {
      if (window.OUTDOOR_WALK && Array.isArray(window.OUTDOOR_WALK.pts)) {
        return Promise.resolve(window.OUTDOOR_WALK);
      }
      if (outdoorWalkPromise) return outdoorWalkPromise;
      outdoorWalkPromise = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = 'Data/outdoor_walk.js';
        s.onload = () => {
          if (window.OUTDOOR_WALK) resolve(window.OUTDOOR_WALK);
          else reject(new Error('OUTDOOR_WALK missing after load'));
        };
        s.onerror = () => reject(new Error('Failed to load outdoor_walk.js'));
        document.head.appendChild(s);
      });
      return outdoorWalkPromise;
    }

    function unpackOutdoorWalk(W) {
      return (W.pts || []).map(r => ({
        lon: r[0], lat: r[1], rsrp: r[2], pci: r[3],
        op: W.ops[r[4]], network: W.nets[r[5]], band: W.bands[r[6]], ch: W.chs[r[7]]
      }));
    }

    // Unique (site, PCI) sectors with a known transmitter lat/lon. Point counts
    // come from the active environment so empty sectors stay out of the picker.
    function listAntennaSectors(pointCounts) {
      const out = [];
      for (const s of (window.BASE_STATIONS || [])) {
        if (s.lat == null || s.lon == null) continue;
        const seen = new Set();
        for (const se of (s.sectors || [])) {
          const pci = se.pci;
          if (pci == null || seen.has(pci)) continue;
          seen.add(pci);
          const n = pointCounts.get(Number(pci)) || 0;
          if (n <= 0) continue;
          out.push({
            key: `${s.site_id}::${pci}`,
            site_id: s.site_id,
            name: s.name || s.site_id,
            lat: s.lat,
            lon: s.lon,
            pci: Number(pci),
            carrier: se.carrier || '',
            n
          });
        }
      }
      out.sort((a, b) => b.n - a.n || a.name.localeCompare(b.name) || a.pci - b.pci);
      return out;
    }

    function populateAntSectors(sectors) {
      const prev = antSectorEl.value;
      antSectorEl.innerHTML = '';
      if (!sectors.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No base-station sectors in this walk';
        antSectorEl.appendChild(opt);
        return;
      }
      for (const s of sectors) {
        const opt = document.createElement('option');
        opt.value = s.key;
        const carrier = s.carrier ? `${s.carrier} · ` : '';
        opt.textContent = `${s.name} · PCI ${s.pci} (${carrier}${s.n} pts)`;
        antSectorEl.appendChild(opt);
      }
      antSectorEl.value = [...antSectorEl.options].some(o => o.value === prev) ? prev : sectors[0].key;
    }

    function indoorLonLat(r) {
      if (Number.isFinite(r.px) && Number.isFinite(r.py)) {
        const ll = pxToLonlatInline(r.px, r.py);
        if (ll && Number.isFinite(ll.lat) && Number.isFinite(ll.lon)) return ll;
      }
      // Fallback only if georef is unavailable — raw indoor GPS is known-bad.
      if (Number.isFinite(r.latitude) && Number.isFinite(r.longitude)) {
        return { lat: r.latitude, lon: r.longitude };
      }
      return null;
    }

    function collectAntennaSamples(sector, metric) {
      const outdoor = isOutdoorWorkspace();
      const selectedNetwork = networkEl.value || 'all';
      const selectedBand = bandEl.value || 'all';
      const selectedChannel = channelEl.value || 'all';
      const samples = [];

      if (outdoor) {
        const W = window.OUTDOOR_WALK;
        if (!W) return { samples, note: 'Outdoor walk not loaded.' };
        // Outdoor walk bake carries RSRP only.
        if (metric !== 'rsrp') {
          return { samples, note: `Outdoor walk has RSRP only — switch the Metric sidebar to RSRP (asked for ${metric.toUpperCase()}).` };
        }
        for (const p of unpackOutdoorWalk(W)) {
          if (Number(p.pci) !== sector.pci) continue;
          if (selectedNetwork !== 'all' && p.network !== selectedNetwork) continue;
          // Outdoor bands are strings like "B13" / "n77"; sidebar band is numeric.
          if (selectedBand !== 'all') {
            const bandNum = String(p.band).replace(/^[A-Za-z]+/, '');
            if (bandNum !== selectedBand) continue;
          }
          if (selectedChannel !== 'all' && String(p.ch) !== selectedChannel) continue;
          if (!Number.isFinite(p.lat) || !Number.isFinite(p.lon) || !Number.isFinite(p.rsrp)) continue;
          const az = bearingDeg(sector.lat, sector.lon, p.lat, p.lon);
          const dist = haversineM(sector.lat, sector.lon, p.lat, p.lon);
          if (!(dist > 1)) continue;
          samples.push({ az, dist, value: p.rsrp });
        }
        return { samples, note: null };
      }

      for (const r of tsRecords) {
        if (Number(r.pci) !== sector.pci) continue;
        if (selectedNetwork !== 'all' && r.network !== selectedNetwork) continue;
        if (selectedBand !== 'all' && String(r.band) !== selectedBand) continue;
        if (selectedChannel !== 'all' && String(r.freq) !== selectedChannel) continue;
        if (!Number.isFinite(r[metric])) continue;
        const ll = indoorLonLat(r);
        if (!ll) continue;
        const az = bearingDeg(sector.lat, sector.lon, ll.lat, ll.lon);
        const dist = haversineM(sector.lat, sector.lon, ll.lat, ll.lon);
        if (!(dist > 1)) continue;
        samples.push({ az, dist, value: r[metric] });
      }
      return { samples, note: null };
    }

    function binAzimuthMeans(samples, nBins, valueOf) {
      const bins = Array.from({ length: nBins }, () => []);
      const width = 360 / nBins;
      for (const s of samples) {
        const i = Math.min(nBins - 1, Math.floor(s.az / width));
        bins[i].push(valueOf(s));
      }
      const theta = [];
      const r = [];
      const counts = [];
      for (let i = 0; i < nBins; i++) {
        const vals = bins[i];
        if (!vals.length) continue;
        theta.push((i + 0.5) * width);
        r.push(vals.reduce((a, b) => a + b, 0) / vals.length);
        counts.push(vals.length);
      }
      // Close the envelope for a continuous polar ring when we have ≥2 bins.
      if (theta.length >= 2) {
        theta.push(theta[0] + 360);
        r.push(r[0]);
        counts.push(counts[0]);
      }
      return { theta, r, counts };
    }

    function setHistSubTab(sub) {
      histSubTab = sub;
      const isAnt = sub === 'antenna';
      if (histDistPaneEl) histDistPaneEl.style.display = isAnt ? 'none' : '';
      if (histAntPaneEl) histAntPaneEl.style.display = isAnt ? '' : 'none';
      if (histSubDistBtn) histSubDistBtn.classList.toggle('active', !isAnt);
      if (histSubAntBtn) histSubAntBtn.classList.toggle('active', isAnt);
      if (currentTab === 'histogram') refresh();
    }

    // Forward geodesic: the point `distM` metres from (lat,lon) along compass `bearing`°.
    function destPoint(lat, lon, bearing, distM) {
      const R = 6371000, d = distM / R, br = bearing * Math.PI / 180;
      const φ1 = lat * Math.PI / 180, λ1 = lon * Math.PI / 180;
      const φ2 = Math.asin(Math.sin(φ1) * Math.cos(d) + Math.cos(φ1) * Math.sin(d) * Math.cos(br));
      const λ2 = λ1 + Math.atan2(Math.sin(br) * Math.sin(d) * Math.cos(φ1), Math.cos(d) - Math.sin(φ1) * Math.sin(φ2));
      return [φ2 * 180 / Math.PI, λ2 * 180 / Math.PI];
    }
    function antMapClear() { if (antMapLayer) antMapLayer.clearLayers(); }
    function pctl(arr, p) {
      if (!arr.length) return 0;
      const s = [...arr].sort((a, b) => a - b);
      return s[Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))))];
    }

    // Draw the measured pattern on OSM: the base station, the binned-mean lobe at real bearings/scale,
    // a dashed peak-lobe spike, and a translucent −3 dB main-lobe wedge. `values` is parallel to `samples`
    // (both from renderAntennaPattern). The lobe is a directional-gain shape, not a coverage boundary.
    function renderAntennaMap(ctx) {
      if (!antPatternMapEl || typeof L === 'undefined' || !L.map) return;
      const { sector, envelope, peakAz, peakVal, samples, values, range, nBins } = ctx;

      if (!antMap) {
        // SVG renderer (not canvas): the overlay is a handful of vector shapes, and SVG repaints on
        // every update without waiting on requestAnimationFrame — robust even where rAF is throttled.
        antMap = L.map(antPatternMapEl, { zoomControl: true, renderer: L.svg() });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
          { attribution: '© OpenStreetMap · © CARTO', maxZoom: 20, subdomains: 'abcd' }).addTo(antMap);
        antMapLayer = L.layerGroup().addTo(antMap);
      }
      antMapClear();
      if (!sector) { return; }
      if (!samples || !samples.length) { antMap.setView([sector.lat, sector.lon], 15); return; }

      // Outer radius follows the measurement spread so the lobe sits over the walk; the radius per
      // bearing is the binned mean, normalised (weakest→inner, strongest→outer).
      const dists = samples.map(s => s.dist);
      const rOuter = Math.max(80, Math.min(1100, pctl(dists, 0.78)));
      const rInner = 0.22 * rOuter;
      const env = { theta: envelope.theta.slice(), r: envelope.r.slice() };
      if (env.theta.length >= 2 && Math.abs((env.theta[env.theta.length - 1] % 360) - (env.theta[0] % 360)) < 1e-6) {
        env.theta.pop(); env.r.pop();     // drop the closing duplicate binAzimuthMeans appends
      }
      let rMin = Math.min(...env.r), rMax = Math.max(...env.r);
      if (!(rMax > rMin)) rMax = rMin + 1;
      const radiusFor = v => rInner + (v - rMin) / (rMax - rMin) * (rOuter - rInner);
      const bs = [sector.lat, sector.lon];

      // 1) the lobe (pattern outline) — polygon when it spans enough directions, else an arc back to Tx.
      const lobe = env.theta.map((t, i) => destPoint(sector.lat, sector.lon, t, radiusFor(env.r[i])));
      if (lobe.length >= 3) {
        L.polygon(lobe, { color: '#c0392b', weight: 2, fillColor: '#c0392b', fillOpacity: 0.10 }).addTo(antMapLayer);
      } else if (lobe.length >= 1) {
        L.polyline([bs].concat(lobe).concat([bs]), { color: '#c0392b', weight: 2, dashArray: '4,4' }).addTo(antMapLayer);
      }

      // 2) optional measured points, reconstructed from (az,dist) about the Tx, coloured by value.
      if (antMapPointsEl && antMapPointsEl.checked && range) {
        const step = Math.max(1, Math.ceil(samples.length / 1500));
        const span = (range.vmax > range.vmin) ? (range.vmax - range.vmin) : 1;
        for (let i = 0; i < samples.length; i += step) {
          const ll = destPoint(sector.lat, sector.lon, samples[i].az, samples[i].dist);
          L.circleMarker(ll, { radius: 2.6, stroke: false, fillColor: viridis((values[i] - range.vmin) / span), fillOpacity: 0.7 }).addTo(antMapLayer);
        }
      }

      // 3) peak-lobe spike (dashed) + a −3 dB main-lobe wedge.
      if (peakAz != null) {
        const halfBin = 180 / Math.max(12, nBins || 36);   // half a bin's angular width
        const thresh = peakVal - 3;
        let loOff = -halfBin, hiOff = halfBin;
        for (let i = 0; i < env.theta.length; i++) {
          if (env.r[i] < thresh) continue;
          const off = ((env.theta[i] - peakAz + 540) % 360) - 180;   // [-180,180]
          if (Math.abs(off) > 90) continue;                          // ignore back lobes
          loOff = Math.min(loOff, off - halfBin);
          hiOff = Math.max(hiOff, off + halfBin);
        }
        const arc = [];
        const stepA = Math.max(2, (hiOff - loOff) / 24);
        for (let a = loOff; a <= hiOff + 1e-6; a += stepA) arc.push(destPoint(sector.lat, sector.lon, peakAz + a, rOuter));
        L.polygon([bs].concat(arc), { color: '#e67e22', weight: 1, fillColor: '#f1c40f', fillOpacity: 0.18 }).addTo(antMapLayer);

        const tip = destPoint(sector.lat, sector.lon, peakAz, rOuter * 1.06);
        L.polyline([bs, tip], { color: '#c0392b', weight: 2.5, dashArray: '7,6' }).addTo(antMapLayer);
        L.marker(tip, { icon: L.divIcon({ className: '', html: '<span class="ant-peak-label">' + peakAz.toFixed(0) + '° ' + compassLabel(peakAz) + '</span>', iconSize: null }) }).addTo(antMapLayer);
      }

      // 4) base station on top.
      L.circleMarker(bs, { radius: 7, color: '#fff', weight: 2, fillColor: '#c0392b', fillOpacity: 1 })
        .bindTooltip(sector.name + ' · PCI ' + sector.pci, { direction: 'top' }).addTo(antMapLayer);

      const bounds = L.latLngBounds(lobe.concat([bs]));
      if (bounds.isValid()) antMap.fitBounds(bounds.pad(0.25));
      setTimeout(() => { if (antMap) antMap.invalidateSize(); }, 30);
    }

    function renderAntennaPattern() {
      if (!antPatternPlotEl || !antSectorEl) return;

      const outdoor = isOutdoorWorkspace();
      const metric = outdoor ? 'rsrp' : metricEl.value;
      const unit = units[metric] || '';

      // Point counts for the sector picker (PCI → n), environment-aware.
      const pointCounts = new Map();
      if (outdoor) {
        const W = window.OUTDOOR_WALK;
        if (!W) {
          antPatternStatsEl.innerHTML = `<div>Loading the outdoor walk…</div>`;
          Plotly.purge(antPatternPlotEl);
          ensureOutdoorWalk().then(() => {
            if (histSubTab === 'antenna' && currentTab === 'histogram') renderAntennaPattern();
          }).catch(err => {
            antPatternStatsEl.innerHTML = `<div style="color:#b0483f;">${err.message}</div>`;
          });
          return;
        }
        for (const r of W.pts || []) {
          const pci = r[3];
          pointCounts.set(pci, (pointCounts.get(pci) || 0) + 1);
        }
      } else {
        for (const r of tsRecords) {
          if (!Number.isFinite(r.pci)) continue;
          pointCounts.set(r.pci, (pointCounts.get(r.pci) || 0) + 1);
        }
      }

      const sectors = listAntennaSectors(pointCounts);
      populateAntSectors(sectors);
      const sector = sectors.find(s => s.key === antSectorEl.value) || sectors[0];
      if (!sector) {
        Plotly.purge(antPatternPlotEl);
        antMapClear();
        antPatternStatsEl.innerHTML = `<div>No surveyed base-station sectors overlap this ${outdoor ? 'outdoor' : 'indoor'} walk.</div>`;
        return;
      }

      const { samples, note } = collectAntennaSamples(sector, metric);
      if (note && !samples.length) {
        Plotly.purge(antPatternPlotEl);
        antPatternStatsEl.innerHTML = `<div>${note}</div>`;
        return;
      }

      const nBins = Math.max(12, Math.min(72, Number(antBinsEl.value) || 36));
      antBinsEl.value = String(nBins);
      const distNorm = !!(antDistNormEl && antDistNormEl.checked);
      const dists = samples.map(s => s.dist).sort((a, b) => a - b);
      const d0 = dists.length ? dists[Math.floor(dists.length / 2)] : 100;
      const valueOf = (s) => distNorm
        ? s.value + 20 * Math.log10(Math.max(s.dist, 1) / Math.max(d0, 1))
        : s.value;
      const values = samples.map(valueOf);
      const azimuths = samples.map(s => s.az);

      const envelope = binAzimuthMeans(samples, nBins, valueOf);
      let peakAz = null, peakVal = -Infinity;
      for (let i = 0; i < envelope.theta.length - (envelope.theta.length >= 2 ? 1 : 0); i++) {
        if (envelope.r[i] > peakVal) { peakVal = envelope.r[i]; peakAz = envelope.theta[i]; }
      }

      const azSpan = (() => {
        if (azimuths.length < 2) return 0;
        const sorted = [...azimuths].sort((a, b) => a - b);
        let maxGap = 0;
        for (let i = 1; i < sorted.length; i++) maxGap = Math.max(maxGap, sorted[i] - sorted[i - 1]);
        maxGap = Math.max(maxGap, (sorted[0] + 360) - sorted[sorted.length - 1]);
        return Math.max(0, 360 - maxGap);
      })();

      const range = computeDefaultRange(
        samples.map((s, i) => ({ [metric]: values[i] })),
        metric
      );
      const metricLabel = distNorm
        ? `${metric.toUpperCase()} norm (${unit}) @ ${d0.toFixed(0)} m`
        : `${metric.toUpperCase()} (${unit})`;

      const traces = [
        {
          type: 'scatterpolar',
          mode: 'markers',
          r: values,
          theta: azimuths,
          marker: {
            size: 5,
            color: values,
            colorscale: 'Viridis',
            cmin: range.vmin,
            cmax: range.vmax,
            opacity: 0.55,
            colorbar: { title: metricLabel, len: 0.7, thickness: 14 }
          },
          name: 'Walk samples',
          hovertemplate: `Az %{theta:.0f}°<br>${metric.toUpperCase()}: %{r:.1f} ${unit}<extra></extra>`
        }
      ];
      if (envelope.theta.length) {
        traces.push({
          type: 'scatterpolar',
          mode: 'lines',
          r: envelope.r,
          theta: envelope.theta,
          line: { color: '#c0392b', width: 3 },
          name: `Mean · ${nBins} bins`,
          hovertemplate: `Az %{theta:.0f}°<br>Mean: %{r:.1f} ${unit}<extra></extra>`
        });
      }

      Plotly.react(antPatternPlotEl, traces, {
        title: {
          text: `${sector.name} · PCI ${sector.pci} · ${outdoor ? 'Outdoor' : 'Indoor'} measured pattern`,
          x: 0.02
        },
        polar: {
          bgcolor: '#ffffff',
          radialaxis: {
            title: { text: metricLabel },
            angle: 90,
            tickfont: { size: 11 }
          },
          angularaxis: {
            rotation: 90,
            direction: 'clockwise',
            tickmode: 'array',
            tickvals: [0, 45, 90, 135, 180, 225, 270, 315],
            ticktext: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
          }
        },
        showlegend: true,
        legend: { orientation: 'h', y: -0.08 },
        margin: { t: 48, l: 40, r: 40, b: 56 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
      }, { responsive: true, displaylogo: false });

      const distMin = dists[0] || 0, distMax = dists[dists.length - 1] || 0;
      antPatternStatsEl.innerHTML = `
        <div>Sector: <b>${sector.name}</b> · PCI <b>${sector.pci}</b>${sector.carrier ? ` · ${sector.carrier}` : ''}</div>
        <div>Samples: <b>${samples.length}</b> · Azimuth span: <b>${azSpan.toFixed(0)}°</b> · Range: <b>${distMin.toFixed(0)}–${distMax.toFixed(0)} m</b></div>
        <div>Peak lobe (binned mean): <b>${peakAz == null ? '—' : `${peakAz.toFixed(0)}° ${compassLabel(peakAz)}`}</b>${peakAz == null ? '' : ` · ${peakVal.toFixed(1)} ${unit}`}${distNorm ? ` · normalized to ${d0.toFixed(0)} m` : ''}</div>
        <div>Mode: <b>${outdoor ? 'Outdoor walk (RSRP)' : 'Indoor walk (floor-plan georef)'}</b></div>
      `;

      // Mirror the same pattern onto OSM: base station, lobe, dashed peak spike + main-lobe wedge.
      renderAntennaMap({ sector, envelope, peakAz, peakVal, samples, values, range, nBins });
    }

    // Builds the RSRP/RSRQ/CINR/RSSI points table for the Time Elapsed
    // Playback tab. Capped to the most recent N rows for render performance
    // during animated playback; newest-captured point first.
    function renderPlaybackTable(data) {
      const MAX_ROWS = 150;
      const rows = [...data].sort((a, b) => b.t_sec - a.t_sec).slice(0, MAX_ROWS);
      const fmt = (v) => Number.isFinite(v) ? v.toFixed(1) : '—';

      playbackTableBodyEl.innerHTML = rows.map(r => `
        <tr>
          <td>${formatClockTime(r.t_sec) || formatDuration(r.t_sec)}</td>
          <td>${r.network || '—'}</td>
          <td>${Number.isFinite(r.band) ? r.band : '—'}</td>
          <td>${Number.isFinite(r.pci) ? r.pci : '—'}</td>
          <td>${fmt(r.rsrp)}</td>
          <td>${fmt(r.rsrq)}</td>
          <td>${fmt(r.cinr)}</td>
          <td>${fmt(r.rssi)}</td>
        </tr>
      `).join('');

      if (data.length > MAX_ROWS) {
        playbackTableBodyEl.innerHTML += `
          <tr><td colspan="8" style="text-align:center; color: var(--muted); font-style: italic;">
            showing most recent ${MAX_ROWS} of ${data.length} captured
          </td></tr>
        `;
      }
    }

    // The Time Elapsed Playback tab's own mini Map Coverage and Histogram,
    // both built from timeScoped(cutoff) so they always show exactly the
    // same set of points as each other and as the points table below — this
    // is what keeps the three in sync as the slider/play button moves.
    function renderPlayback() {
      const metric = metricEl.value;
      const cutoff = Number(timeSliderEl.value);
      const captured = timeScoped(cutoff);
      const clockText = formatClockTime(cutoff) || formatDuration(cutoff);

      const mapData = captured.filter(r => Number.isFinite(r.px) && Number.isFinite(r.py) && Number.isFinite(r[metric]));
      const mapRange = computeDefaultRange(mapData, metric);

      Plotly.react(playbackMapEl, [{
        type: 'scattergl',
        mode: 'markers',
        x: mapData.map(r => r.px),
        y: mapData.map(r => r.py),
        customdata: mapData.map(r => [r.latitude, r.longitude, 1, r.pci, r.freq]),
        marker: {
          size: 9,
          color: metricValues(mapData, metric),
          colorscale: 'Viridis',
          cmin: mapRange.vmin,
          cmax: mapRange.vmax,
          colorbar: { title: `${metric.toUpperCase()} (${units[metric] || ''})`, len: 0.8, thickness: 14 },
          line: { color: '#111', width: 0.3 },
          opacity: 0.88
        },
        hovertemplate: hoverTemplate(metric)
      }], {
        margin: { l: 12, r: 16, t: 40, b: 12 },
        title: { text: `Captured up to ${clockText} (${mapData.length} points)`, x: 0.02 },
        xaxis: { visible: false, range: [0, 1150] },
        yaxis: { visible: false, range: [515, 0], scaleanchor: 'x' },
        images: [floorPlanImage],
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
      }, { responsive: true, displaylogo: false });

      const metricVals = metricValues(captured, metric);
      Plotly.react(playbackHistogramEl, [{
        type: 'histogram',
        x: metricVals,
        marker: { color: '#0a7f7a' },
        nbinsx: 40,
        hovertemplate: `${metric.toUpperCase()}: %{x:.1f} ${units[metric] || ''}<br>Count: %{y}<extra></extra>`
      }], {
        title: { text: `${metric.toUpperCase()} Distribution (${metricVals.length} captured)` },
        xaxis: { title: { text: `${metric.toUpperCase()} (${units[metric] || ''})` } },
        yaxis: { title: { text: 'Measurement count' } },
        margin: { t: 40, l: 55, r: 16, b: 45 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        bargap: 0.05
      }, { responsive: true, displaylogo: false });

      // Line chart: the selected metric over elapsed time, using only points
      // captured so far — at either raw-measurement granularity or averaged
      // into fixed-size time buckets, with an optional regression trend line.
      // Both the series and the fit recompute from `captured` on every call,
      // so they visibly extend/re-fit as playback advances.
      const aggregation = lineAggregationEl.value;
      const series = buildLineSeries(captured, metric, aggregation);
      const aggLabel = aggregation === 'point' ? 'Raw measurements' : `${aggregation}-minute average`;

      const lineTraces = [{
        type: 'scatter',
        mode: 'lines+markers',
        x: series.x,
        y: series.y,
        line: { color: '#3d6fae', width: 2 },
        marker: { size: aggregation === 'point' ? 4 : 6, color: '#3d6fae' },
        name: aggLabel,
        hovertemplate: `Minute %{x:.1f}<br>${metric.toUpperCase()}: %{y:.1f} ${units[metric] || ''}<extra></extra>`
      }];

      const regressionKind = lineRegressionEl.value;
      if (regressionKind !== 'none' && series.x.length >= 2) {
        const fit = fitRegression(regressionKind, series.x, series.y);
        if (fit) {
          const xMin = Math.min(...series.x);
          const xMax = Math.max(...series.x);
          const steps = 60;
          const fitX = Array.from({ length: steps + 1 }, (_, i) => xMin + (xMax - xMin) * i / steps);
          lineTraces.push({
            type: 'scatter',
            mode: 'lines',
            x: fitX,
            y: fitX.map(x => fit.predict(x)),
            line: { color: '#c0392b', width: 2, dash: 'dash' },
            name: `${regressionKind[0].toUpperCase()}${regressionKind.slice(1)} trend`,
            hoverinfo: 'skip'
          });
        }
      }

      Plotly.react(playbackLineEl, lineTraces, {
        title: { text: `${metric.toUpperCase()} Over Elapsed Time — ${aggLabel} (${series.n} measurements)` },
        xaxis: logTimeEl.checked
          ? { title: { text: 'Elapsed time (minutes)' }, type: 'log' }
          : { title: { text: 'Elapsed time (minutes)' }, type: 'linear', range: [0, Math.max(0.5, maxTSec / 60)] },
        yaxis: { title: { text: `${metric.toUpperCase()} (${units[metric] || ''})` } },
        margin: { t: 40, l: 55, r: 16, b: 45 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        legend: { orientation: 'h', y: -0.2 }
      }, { responsive: true, displaylogo: false });

      renderPlaybackTable(captured);

      const tsTotal = scopedTimeseries().length;
      timeStatsEl.innerHTML = `
        <div>Clock: <b>${clockText}</b></div>
        <div>Elapsed: <b>${formatDuration(cutoff)} / ${formatDuration(maxTSec)}</b></div>
        <div>Measurements captured: <b>${captured.length} / ${tsTotal}</b></div>
      `;
    }

    // Re-renders whichever tab is currently visible. Map Coverage and
    // Histogram are stationary (whole dataset) and don't need the slider
    // position; only the Time Elapsed Playback tab reads it.
    function refresh() {
      if (currentTab === 'histogram') {
        if (histSubTab === 'antenna') renderAntennaPattern();
        else renderHistograms();
      } else if (currentTab === 'time') {
        renderPlayback();
      } else {
        render();
      }
    }

    function setActiveTab(tab) {
      currentTab = tab;
      mapTabEl.style.display = tab === 'map' ? '' : 'none';
      histTabEl.style.display = tab === 'histogram' ? '' : 'none';
      timeTabEl.style.display = tab === 'time' ? '' : 'none';
      const simTabEl = document.getElementById('simTab');
      if (simTabEl) simTabEl.style.display = tab === 'sim' ? '' : 'none';
      tabMapBtn.classList.toggle('active', tab === 'map');
      tabHistogramBtn.classList.toggle('active', tab === 'histogram');
      tabTimeBtn.classList.toggle('active', tab === 'time');
      const simBtn = document.getElementById('tabSimBtn');
      if (simBtn) simBtn.classList.toggle('active', tab === 'sim');
      if (tab !== 'sim') refresh();
    }

    function stopPlayback() {
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
      }
      playBtnEl.textContent = 'Play';
    }

    tabMapBtn.addEventListener('click', () => setActiveTab('map'));
    tabHistogramBtn.addEventListener('click', () => setActiveTab('histogram'));
    tabTimeBtn.addEventListener('click', () => setActiveTab('time'));
    document.getElementById('tabSimBtn')
      .addEventListener('click', () => setActiveTab('sim'));

    if (histSubDistBtn) histSubDistBtn.addEventListener('click', () => setHistSubTab('dist'));
    if (histSubAntBtn) histSubAntBtn.addEventListener('click', () => setHistSubTab('antenna'));
    if (antSectorEl) antSectorEl.addEventListener('change', () => {
      if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
    });
    if (antBinsEl) antBinsEl.addEventListener('change', () => {
      if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
    });
    if (antDistNormEl) antDistNormEl.addEventListener('change', () => {
      if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
    });
    if (antMapPointsEl) antMapPointsEl.addEventListener('change', () => {
      if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
    });

    // Re-render the antenna pattern when the workspace flips Indoor ⇄ Outdoor
    // (landing.js mutates window.appMode.environment).
    window.addEventListener('appmodechange', () => {
      if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
    });
    // Fallback: landing may not dispatch — watch via a light poll on tab show is enough;
    // also hook the landing continue buttons if present.
    document.addEventListener('click', (e) => {
      if (!e.target) return;
      if (e.target.id === 'enterWorkspaceBtn' || e.target.closest?.('#envIndoor, #envOutdoor')) {
        setTimeout(() => {
          if (currentTab === 'histogram' && histSubTab === 'antenna') renderAntennaPattern();
        }, 50);
      }
    });

    // OSM under-layer controls (PR #25) — Map Coverage only. Delegated on document so
    // they work even though the map-modes row is (re)built after this script runs.
    document.addEventListener('change', (e) => {
      if (!e.target) return;
      if ((e.target.id === 'osmBasemap' || e.target.id === 'crossvalSelect') && currentTab === 'map') refresh();
    });
    document.addEventListener('input', (e) => {
      if (e.target && e.target.id === 'floorOpacity' && currentTab === 'map') refresh();
    });
    // Keep the Leaflet surface in sync with the 2D/3D map-mode toggle (also delegated).
    document.addEventListener('click', (e) => {
      if (!e.target) return;
      if (e.target.id === 'mapMode3dBtn') {
        const lm = document.getElementById('leafletMap'); if (lm) lm.style.display = 'none';
      } else if (e.target.id === 'mapMode2dBtn' && currentTab === 'map') {
        setTimeout(refresh, 0);   // let viewer3d.setMode restore #plot first, then re-apply
      }
    });
    // This dashboard is indoor by default; hide the OSM control only for an explicitly outdoor workspace.
    (function gateOsmCtl() {
      const ctl = document.getElementById('osmBasemapCtl');
      if (ctl && window.appMode && window.appMode.environment === 'outdoor') ctl.style.display = 'none';
    })();

    clockFormatEl.addEventListener('change', () => {
      updateTimeLabel();
      refresh();
    });

    lineAggregationEl.addEventListener('change', refresh);
    lineRegressionEl.addEventListener('change', refresh);
    logTimeEl.addEventListener('change', refresh);

    timeSliderEl.addEventListener('input', () => {
      updateTimeLabel();
      refresh();
    });

    playBtnEl.addEventListener('click', () => {
      if (playTimer) {
        stopPlayback();
        return;
      }
      if (Number(timeSliderEl.value) >= Number(timeSliderEl.max)) {
        timeSliderEl.value = '0';
      }
      playBtnEl.textContent = 'Pause';
      const baseStep = Math.max(1, Math.round(maxTSec / 200));
      playTimer = setInterval(() => {
        const step = Math.max(1, Math.round(baseStep * Number(playSpeedEl.value)));
        const next = Number(timeSliderEl.value) + step;
        if (next >= Number(timeSliderEl.max)) {
          timeSliderEl.value = timeSliderEl.max;
          stopPlayback();
        } else {
          timeSliderEl.value = String(next);
        }
        updateTimeLabel();
        refresh();
      }, 100);
    });

    // Accepts either "mm:ss" elapsed time (e.g. "5:30") or "HH:MM:SS" clock
    // time (e.g. "14:30:00", interpreted in the same clock as
    // timeseriesStartTime) and converts it to elapsed seconds. Returns null
    // if the text doesn't parse as either.
    function parseTimeJumpInput(text) {
      const parts = text.trim().split(':').map(p => p.trim());
      if (parts.length < 2 || parts.length > 3 || parts.some(p => p === '' || Number.isNaN(Number(p)))) {
        return null;
      }
      if (parts.length === 2) {
        const [m, s] = parts.map(Number);
        return m * 60 + s;
      }
      if (!testStartTime) return null;
      const [h, m, s] = parts.map(Number);
      const clockSecOfDay = h * 3600 + m * 60 + s;
      const startSecOfDay = testStartTime.getUTCHours() * 3600 + testStartTime.getUTCMinutes() * 60 + testStartTime.getUTCSeconds();
      return clockSecOfDay - startSecOfDay;
    }

    function goToElapsedSeconds(sec) {
      if (!Number.isFinite(sec)) return false;
      timeSliderEl.value = String(Math.round(Math.max(0, Math.min(maxTSec, sec))));
      updateTimeLabel();
      refresh();
      return true;
    }

    timeJumpBtnEl.addEventListener('click', () => {
      const sec = parseTimeJumpInput(timeJumpInputEl.value);
      timeJumpInputEl.style.borderColor = (sec === null) ? '#c0392b' : '';
      if (sec !== null) goToElapsedSeconds(sec);
    });
    timeJumpInputEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') timeJumpBtnEl.click();
    });

    updateTimeLabel();

    metricEl.addEventListener('change', () => {
      applyDefaults(metricEl.value, filtered(metricEl.value));
      refresh();
    });
    coverageEl.addEventListener('change', render);
    networkEl.addEventListener('change', () => {
      populateBands();
      populatePcis();
      populateChannels();
      updateMetricCounts();
      applyDefaults(metricEl.value, filtered(metricEl.value));
      refresh();
    });
    bandEl.addEventListener('change', () => {
      populatePcis();
      populateChannels();
      updateMetricCounts();
      applyDefaults(metricEl.value, filtered(metricEl.value));
      refresh();
    });
    pciEl.addEventListener('change', () => {
      populateChannels();
      updateMetricCounts();
      applyDefaults(metricEl.value, filtered(metricEl.value));
      refresh();
    });
    channelEl.addEventListener('change', () => {
      updateMetricCounts();
      applyDefaults(metricEl.value, filtered(metricEl.value));
      refresh();
    });
    vminEl.addEventListener('change', render);
    vmaxEl.addEventListener('change', render);
    sizeEl.addEventListener('input', render);

    populateNetworks();
    populateBands();
    populatePcis();
    populateChannels();
    updateMetricCounts();
    applyDefaults(metricEl.value, filtered(metricEl.value));
    refresh();
