#!/usr/bin/env python3
"""Headless browser check for M3 outdoor + M4 surrogate (PR verification)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/opt/cursor/artifacts/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8777/Frontend_Data_Display.html"
RESULTS = {}


def go_to_sim(page):
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_selector("#chooserGo", timeout=30_000)
    page.click('#screenChooser [data-env="indoor"]')
    page.click('#screenChooser [data-dim="3d"]')
    page.click("#chooserGo")
    page.click("#importGo")
    page.click("#preGo")
    page.wait_for_selector("#tabSimBtn", timeout=30_000)
    page.click("#tabSimBtn")
    # ensureInit needs a visible viewport
    page.wait_for_timeout(1500)
    page.evaluate("() => window.__sim3d && window.__sim3d.ensureInit()")
    page.wait_for_timeout(1000)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-gl=swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        console = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        go_to_sim(page)

        # ---- M3 outdoor ----
        page.select_option("#mode3dSelect", "outdoor")
        page.wait_for_timeout(2500)
        outdoor = page.evaluate(
            """() => {
              const s = window.__sim3d;
              return {
                mode: s && s.currentMode(),
                gdims: s && s.gdims,
                cellM: s && s.cellM,
                assetsOutdoor: !!window.SIM3D_ASSETS_OUTDOOR,
                collisionOutdoor: !!window.SIM3D_COLLISION_OUTDOOR,
                note: (document.getElementById('mode3dNote')||{}).textContent || '',
                status: (document.getElementById('sim3dStatus')||{}).textContent || '',
              };
            }"""
        )
        RESULTS["outdoor_before_run"] = outdoor
        page.select_option("#viz3dMode", "coverage")
        page.click("#run3dStatic")
        # wait for volume fetch + re-render
        st = {}
        for _ in range(60):
            page.wait_for_timeout(500)
            st = page.evaluate(
                """() => ({
                  status: (document.getElementById('sim3dStatus')||{}).textContent || '',
                  tier: window.__sim3d && window.__sim3d.tier,
                  txid: window.__sim3d && window.__sim3d.volume && window.__sim3d.volume.entry && window.__sim3d.volume.entry.txid,
                  mode: window.__sim3d && window.__sim3d.volume && window.__sim3d.volume.entry && window.__sim3d.volume.entry.mode,
                  shape: window.__sim3d && window.__sim3d.volume && window.__sim3d.volume.entry && window.__sim3d.volume.entry.grid_shape,
                  gdims: window.__sim3d && window.__sim3d.gdims,
                })"""
            )
            if st.get("mode") == "outdoor" and "cached volume" in (st.get("status") or "").lower():
                break
        RESULTS["outdoor_after_run"] = st
        page.screenshot(path=str(OUT / "m3-outdoor-coverage.png"), full_page=False)

        # ---- M4 surrogate (indoor cache miss) ----
        page.select_option("#mode3dSelect", "indoor")
        page.wait_for_timeout(2000)
        surf = page.evaluate(
            """async () => {
              const s = window.__sim3d;
              await s.ensureInit();
              const sur = await s.tryLoadSurrogate();
              const dims = s.gdims || [262,17,132];
              const sceneOk = s.sceneMatchesSurrogate(sur.spec, dims);
              // cache-miss Tx (far from committed indoor volumes)
              const tx = [10, 5, 10];
              const r = await s.runSurrogate(tx, 2442);
              return {
                surrogateState: sur.state,
                surrogateNote: sur.note,
                ran: !!r,
                plFinite: !!(r && r.pl && Number.isFinite(r.pl[0])),
                dims: r && r.dims,
                sceneOk: sceneOk,
                plSample: r ? Number(r.pl[0].toFixed(2)) : null,
              };
            }"""
        )
        RESULTS["surrogate_direct"] = surf
        RESULTS["indoor_status"] = page.evaluate(
            "() => (document.getElementById('sim3dStatus')||{}).textContent || ''"
        )
        RESULTS["surrogate_state"] = page.evaluate(
            """() => {
              const s = window.__sim3d && window.__sim3d.surrogate;
              if (!s) return null;
              return { state: s.state, note: s.note, hasVol: !!(s.volume && s.volume.pl) };
            }"""
        )
        page.screenshot(path=str(OUT / "m4-surrogate-indoor.png"), full_page=False)

        interesting = [
            c for c in console
            if ("pl_unet3d" in c.lower() or ("onnx" in c.lower() and "pl_unet." not in c.lower()))
            and "webgpu" not in c.lower()
        ]
        RESULTS["console_interesting"] = interesting[-40:]
        RESULTS["ok_outdoor"] = (
            RESULTS.get("outdoor_after_run", {}).get("mode") == "outdoor"
            and "cached volume" in (RESULTS.get("outdoor_after_run", {}).get("status") or "").lower()
        )
        RESULTS["ok_surrogate"] = (
            RESULTS.get("surrogate_direct", {}).get("surrogateState") in ("ready", "running")
            and RESULTS.get("surrogate_direct", {}).get("ran") is True
            and RESULTS.get("surrogate_direct", {}).get("plFinite") is True
        )

        summary_path = Path("/opt/cursor/artifacts/m3_m4_browser_verify.json")
        summary_path.write_text(json.dumps(RESULTS, indent=2))
        print(json.dumps(RESULTS, indent=2))
        browser.close()

    if not RESULTS.get("ok_outdoor"):
        raise SystemExit("OUTDOOR VERIFY FAILED: " + json.dumps(RESULTS.get("outdoor_after_run")))
    if not RESULTS.get("ok_surrogate"):
        raise SystemExit("SURROGATE VERIFY FAILED: " + json.dumps(RESULTS.get("surrogate_direct")))
    print("VERIFY PASS")


if __name__ == "__main__":
    main()
