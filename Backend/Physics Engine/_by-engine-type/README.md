# Engines by type — a navigation view

These folders **separate the engines by dimension × physics** so you can browse them
by *what they are* instead of by version number. Each entry is a **symlink to the real
folder** — nothing here is a copy, and the real code still lives under `../2D/` and
`../3D Map Physics/`.

```
2D · analytic (Motley-Keenan)/   SIM (v1) · SIM V2        multi-wall path loss
2D · full-wave (FDTD)/           SIM V3                   + fw_unet2d / fw_unet3d surrogates
3D · eikonal-GO (analytic)/      SIM V1 3D · SIM V1.5 3D  SceneV3 (+ pl_unet3d) — production
3D · full-wave (FDTD)/           SIM V2 (3D)              + 3D field U-Net + hybrid
shared support/                  Wave Behavior · Object and Tranmission
```

**Why a symlink view instead of physically moving the folders:** the bootstraps and
~30 modules (plus the backend, `.gitignore`, the browser ONNX fetches, and the Unity
repo) address these folders by exact name/depth — e.g. `_PHYS / "2D" / "SIM"`,
`Path(__file__).resolve().parent.parent`. Relocating them would break all of that, and
`.resolve()` defeats compat symlinks for the engine's own path math. This view gives the
clean by-type separation with **zero** risk to the running engine, backend, browser, or
Unity build. See `../README.md` for the full layout + the "engines at a glance" map.
