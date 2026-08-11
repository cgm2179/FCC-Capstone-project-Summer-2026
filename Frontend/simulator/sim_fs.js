/**
 * sim_fs.js — Full screen / Exit full screen for SIM V3 studios + dashboard embeds.
 *
 * Studio pages: SimFS.wireStudio({ view:'view', btn:'fsBtn', onChange })
 * Dashboard:    SimFS.wireEmbeds([{ wrap, btn, iframe }]) or auto via data-sim-fs attributes.
 */
(function (global) {
  'use strict';

  function isNativeFs() {
    return !!(document.fullscreenElement || document.webkitFullscreenElement);
  }

  function enterFs(el) {
    if (el.requestFullscreen) return el.requestFullscreen();
    if (el.webkitRequestFullscreen) return Promise.resolve(el.webkitRequestFullscreen());
    return Promise.reject(new Error('Fullscreen API unavailable'));
  }

  function exitFs() {
    if (document.exitFullscreen) return document.exitFullscreen();
    if (document.webkitExitFullscreen) return Promise.resolve(document.webkitExitFullscreen());
    return Promise.resolve();
  }

  function reflectBtn(btn, on) {
    if (!btn) return;
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.textContent = on ? '⛶ Exit full screen' : '⛶ Full screen';
  }

  /** Wire a studio page: fullscreen the map column (#view) via #fsBtn. */
  function wireStudio(opts) {
    opts = opts || {};
    const view = document.getElementById(opts.view || 'view');
    const btn = document.getElementById(opts.btn || 'fsBtn');
    if (!view || !btn) return;

    const fallbackClass = opts.fallbackClass || 'fs-fallback';
    const bodyClass = opts.bodyClass || 'fs-fallback-body';

    function isOn() {
      return isNativeFs() || view.classList.contains(fallbackClass);
    }
    function reflect() {
      reflectBtn(btn, isOn());
      if (typeof opts.onChange === 'function') {
        try { opts.onChange(isOn()); } catch (e) { /* ignore */ }
      }
      // Give layout a tick to settle, then notify (3D canvas resize, 2D fit, …).
      requestAnimationFrame(function () {
        try { window.dispatchEvent(new Event('resize')); } catch (e) { /* ignore */ }
        if (typeof opts.onChange === 'function') {
          try { opts.onChange(isOn()); } catch (e2) { /* ignore */ }
        }
      });
    }

    btn.addEventListener('click', async function () {
      try {
        if (isNativeFs()) {
          await exitFs();
        } else if (view.classList.contains(fallbackClass)) {
          view.classList.remove(fallbackClass);
          document.body.classList.remove(bodyClass);
          reflect();
        } else {
          try {
            await enterFs(view);
          } catch (e) {
            view.classList.add(fallbackClass);
            document.body.classList.add(bodyClass);
            reflect();
          }
        }
      } catch (e) {
        view.classList.toggle(fallbackClass);
        document.body.classList.toggle(bodyClass, view.classList.contains(fallbackClass));
        reflect();
      }
    });

    document.addEventListener('fullscreenchange', reflect);
    document.addEventListener('webkitfullscreenchange', reflect);
    reflect();
  }

  /**
   * Wire dashboard embed wraps. Each entry: { wrapId, btnId, iframeId? }.
   * Fullscreens the wrap (iframe / canvas fills the screen). Put the button
   * *inside* the wrap so it only shows when that simulation window is open.
   */
  function wireEmbeds(list) {
    list = list || [];

    function reflectAll() {
      list.forEach(function (item) {
        const wrap = document.getElementById(item.wrapId);
        const btn = document.getElementById(item.btnId);
        if (!wrap || !btn) return;
        const on = document.fullscreenElement === wrap ||
                   document.webkitFullscreenElement === wrap ||
                   wrap.classList.contains('sim-fs-on');
        reflectBtn(btn, on);
      });
    }

    list.forEach(function (item) {
      const wrap = document.getElementById(item.wrapId);
      const btn = document.getElementById(item.btnId);
      const iframe = item.iframeId ? document.getElementById(item.iframeId)
                                   : (wrap && wrap.querySelector('iframe'));
      if (!wrap || !btn) return;

      if (iframe) {
        iframe.setAttribute('allowfullscreen', '');
        const allow = iframe.getAttribute('allow') || '';
        if (!/\bfullscreen\b/.test(allow)) {
          iframe.setAttribute('allow', (allow ? allow + '; ' : '') + 'fullscreen');
        }
      }

      wrap.classList.add('sim-fs-wrap');

      btn.addEventListener('click', async function () {
        const nativeOn = document.fullscreenElement === wrap ||
                         document.webkitFullscreenElement === wrap;
        try {
          if (nativeOn) {
            await exitFs();
            wrap.classList.remove('sim-fs-on');
          } else if (wrap.classList.contains('sim-fs-on')) {
            wrap.classList.remove('sim-fs-on');
            document.body.classList.remove('sim-fs-body');
            reflectBtn(btn, false);
          } else {
            try {
              await enterFs(wrap);
              wrap.classList.add('sim-fs-on');
            } catch (e) {
              wrap.classList.add('sim-fs-on');
              document.body.classList.add('sim-fs-body');
              reflectBtn(btn, true);
            }
          }
        } catch (e) {
          wrap.classList.toggle('sim-fs-on');
          document.body.classList.toggle('sim-fs-body', wrap.classList.contains('sim-fs-on'));
          reflectBtn(btn, wrap.classList.contains('sim-fs-on'));
        }
        requestAnimationFrame(function () {
          try { window.dispatchEvent(new Event('resize')); } catch (err) { /* ignore */ }
        });
      });
    });

    document.addEventListener('fullscreenchange', function () {
      list.forEach(function (item) {
        const wrap = document.getElementById(item.wrapId);
        if (!wrap) return;
        const on = document.fullscreenElement === wrap || document.webkitFullscreenElement === wrap;
        if (!on && isNativeFs() === false) wrap.classList.remove('sim-fs-on');
        if (!on && !document.fullscreenElement && !document.webkitFullscreenElement) {
          wrap.classList.remove('sim-fs-on');
        }
      });
      reflectAll();
    });
    document.addEventListener('webkitfullscreenchange', reflectAll);
    reflectAll();
  }

  /** Convenience: wire the four SIM V3 dashboard embeds + legacy PL canvas wrap. */
  function wireDashboardDefaults() {
    wireEmbeds([
      { wrapId: 'simFwWrap', btnId: 'simFsFw2d', iframeId: 'simFwFrame' },
      { wrapId: 'simFwWrapOut', btnId: 'simFsFw2dOut', iframeId: 'simFwFrameOut' },
      { wrapId: 'simFw3dWrap', btnId: 'simFsFw3d', iframeId: 'simFw3dFrame' },
      { wrapId: 'simFw3dWrapOut', btnId: 'simFsFw3dOut', iframeId: 'simFw3dFrameOut' },
      { wrapId: 'simLegacyWrap', btnId: 'simFsLegacy' },
    ]);
  }

  global.SimFS = { wireStudio: wireStudio, wireEmbeds: wireEmbeds, wireDashboardDefaults: wireDashboardDefaults };
})(typeof window !== 'undefined' ? window : globalThis);
