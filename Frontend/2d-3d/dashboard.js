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
    const ranges = {"rsrp": {"vmin": -112.332, "vmax": -83.488, "unit": "dBm"}, "rsrq": {"vmin": -15.508000000000001, "vmax": -11.06, "unit": "dB"}, "cinr": {"vmin": -0.924, "vmax": 17.923000000000002, "unit": "dB"}, "rssi": {"vmin": -88.92, "vmax": -64.77333333333335, "unit": "dBm"}};
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
        images: [floorPlanImage],
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
      };

      Plotly.react(plotEl, traces, layout, {responsive: true, displaylogo: false});
      updateStats(data, metric);
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
        renderHistograms();
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
