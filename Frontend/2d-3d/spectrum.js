/* Local Spectrum & Channel Reference sidebar + observed-ARFCN buttons —
 * extracted verbatim from Frontend_Data_Display.html. Classic script (IIFE);
 * reads timeseriesRecords, dispatches change events the dashboard listens to,
 * so it loads AFTER dashboard.js. */
    (function () {
      var btn = document.getElementById('spectrumToggle');
      var panel = document.getElementById('spectrumPanel');
      btn.addEventListener('click', function () {
        var open = panel.style.display === 'block';
        panel.style.display = open ? 'none' : 'block';
        btn.textContent = open
          ? '📡 Local Spectrum & Channel Reference ▸'
          : '📡 Local Spectrum & Channel Reference ▾';
      });

      // ---- Carrier selector ----------------------------------------------
      // The walk data itself is single-carrier (all traffic is T-Mobile,
      // MCC 310 / MNC 260), so the Carrier dropdown drives *labelling*: the
      // provider name shown on the observed-ARFCN buttons, and which block is
      // highlighted in the Spectrum Reference panel below. It does not filter
      // the map, since there is no other-carrier data to filter to.
      var CARRIERS = { verizon: 'Verizon', att: 'AT&T', tmobile: 'T-Mobile' };
      var carrierSel = document.getElementById('carrier');
      var currentCarrier = (carrierSel && CARRIERS[carrierSel.value]) ? carrierSel.value : 'tmobile';

      function highlightCarrier() {
        var blocks = panel.querySelectorAll('.carrier-block');
        Array.prototype.forEach.call(blocks, function (b) {
          b.classList.toggle('selected', b.getAttribute('data-carrier') === currentCarrier);
        });
      }

      // ---- Observed ARFCN buttons: channel number + provider + (count) ----
      // ARFCNs are derived from the measured frequency via the 3GPP rasters:
      // NR global raster (5 kHz below 3 GHz, 15 kHz above), LTE per-band
      // EARFCN offsets. The provider label comes from the Carrier dropdown.
      var LTE_OFFS = { 2:[600,1930], 4:[1950,2110], 5:[2400,869], 12:[5010,729],
        13:[5180,746], 14:[5280,758], 17:[5730,734], 25:[8040,1930],
        30:[9770,2350], 41:[39650,2496], 66:[66436,2110], 71:[68586,617] };
      function arfcnOf(network, band, freq) {
        if (network === 'lte') {
          var o = LTE_OFFS[band];
          if (!o) return { label: 'EARFCN ?', n: null };
          return { label: 'EARFCN', n: Math.round(o[0] + (freq - o[1]) * 10) };
        }
        var n = freq < 3000 ? Math.round(freq * 200)
                            : 600000 + Math.round((freq - 3000) * 1000 / 15);
        return { label: 'NR-ARFCN', n: n };
      }
      function bandName(network, band) {
        return (network === 'lte' ? 'B' : 'n') + band;
      }
      var arfcnHost = document.getElementById('arfcnList');
      function renderArfcnList() {
        if (!arfcnHost) return;
        var carrierName = CARRIERS[currentCarrier] || 'T-Mobile';
        var groups = {};
        timeseriesRecords.forEach(function (r) {
          if (r.freq == null || !isFinite(r.freq)) return;   // skip blank freq
          var k = r.network + '|' + r.band + '|' + r.freq;
          (groups[k] = groups[k] || { network: r.network, band: r.band,
                                      freq: r.freq, count: 0 }).count++;
        });
        var list = Object.values(groups).sort(function (a, b) { return b.count - a.count; });
        arfcnHost.innerHTML = '';
        list.forEach(function (g) {
          var a = arfcnOf(g.network, g.band, g.freq);
          var el = document.createElement('button');
          el.type = 'button';
          el.className = 'arfcn-btn';
          el.innerHTML = '<b>' + a.label + (a.n !== null ? ' ' + a.n : '') +
            '</b> · ' + carrierName + ' ' + bandName(g.network, g.band) + ' · ' +
            g.freq.toFixed(1) + ' MHz <b>(' + g.count + ')</b>';
          el.addEventListener('click', function () {
            var fire = function (id, v) {
              var s = document.getElementById(id);
              s.value = v;
              s.dispatchEvent(new Event('change', { bubbles: true }));
            };
            fire('network', g.network);
            fire('band', String(g.band));
            fire('pci', 'all');
            fire('channel', String(g.freq));
          });
          arfcnHost.appendChild(el);
        });
      }

      if (carrierSel) {
        carrierSel.addEventListener('change', function () {
          if (!CARRIERS[this.value]) return;
          currentCarrier = this.value;
          renderArfcnList();
          highlightCarrier();
        });
      }

      try {
        renderArfcnList();
        highlightCarrier();
      } catch (e) { console.error('[spectrum] ARFCN list failed:', e); }
    })();
