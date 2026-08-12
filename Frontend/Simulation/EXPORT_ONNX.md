# Exporting the SIM V3 ONNX surrogates + running the dashboard

The browser studios (`fw_studio2d.html` indoor, `fw_studio2d_outdoor.html` outdoor)
run the trained surrogates via **onnxruntime-web**. The deploy-path `.onnx` files under
`SIM V1 3D/web/` are **allow-listed in `.gitignore`** and ship with the repo (~30 MB
each) so a fresh clone can run ONNX offline. The `.json` **contracts** next to them
are also committed. If a weight file is missing locally, the studios fall back to
in-browser FDTD (Auto → FDTD; ONNX → a clear error).

Models live in `Physics Engine/3D Map Physics/SIM V1 3D/web/`:

| Studio | ONNX (shipped) | Contract (committed) | Channels |
|---|---|---|---|
| indoor  `fw_studio2d.html`         | `fw_unet2d.onnx` | `fw_unet2d.json` | 9  |
| outdoor `fw_studio2d_outdoor.html` | `fw_bs.onnx`     | `fw_bs.json`    | 10 (+directivity) |

## 1. Export the ONNX from a trained checkpoint

The exporter is `Physics Engine/2D/SIM V3/fw_export.py`. It infers the channel
count from the checkpoint, writes the `.onnx`, runs an onnxruntime smoke check, and
(re)writes the matching `.json` contract — including, for the 10-channel outdoor
model, the **directivity** spec (panel HPBW/Am + the per-family table) the browser
uses to synthesize the 10th channel.

```bash
cd "Physics Engine/2D/SIM V3"

# indoor 9-channel model  (checkpoint fw_unet2d.pt)  -> web/fw_unet2d.onnx + .json
python fw_export.py --model indoor

# outdoor 10-channel base-station model  (checkpoint fw_bs.pt) -> web/fw_bs.onnx + .json
python fw_export.py --model bs
```

Pass the training band LABELS so the contract isn't written single-band (the
outdoor model trains on the 10 clustered bands):

```bash
python fw_export.py --model bs --bands \
  TMO_B71_617 "ATT/TMO/VZW_B12/B13/B14_746" "VZW_B5/B26_885" \
  "ATT/TMO/VZW_B2_1965" "ATT/TMO/VZW_B4/B65/B66_2160" ATT_B30_2355 \
  WLAN_WiFi_2442 TMO_n41_2508 "VZW_n77/n78_3710" VZW_n77_3809
```

Notes:
- The checkpoint (`fw_bs.pt`) comes from the outdoor training notebook
  (`Outdoor V3 Sim/SIM V3_Outdoor_Phase_BCD.ipynb`, Phase C). Download it from the
  Colab run into `Physics Engine/2D/SIM V3/` before exporting.
- `--data-meta` defaults to `fw_data_bs/dataset_meta.json` for `--model bs`; pass
  `--bands` to override.
- If `onnxruntime` isn't installed, the smoke check is skipped (the export still
  works): `pip install onnxruntime onnx`.

## 2. Rebuild the browser city grid (only if the OSM grid changes)

The outdoor studio's map + stations come from `Data/noma_city2d.js`, baked from the
penetrable OSM grid. Regenerate it (and the per-class EM constants the FDTD uses)
with:

```bash
python scripts/bake_noma_city2d.py     # -> Data/noma_city2d.js  (committed, ~570 KB)
```

## 3. Run the dashboard after cloning (VS Code)

The studios `fetch()` the model + data files, so they must be served over **http** —
`file://` blocks the fetch (the page will say *"needs http"* and stay FDTD-only).
From the repo root, any static server works:

```bash
python3 -m http.server 8777
```

Then open <http://localhost:8777/Frontend_Data_Display.html> and choose **Outdoor
→ 2D** (or open the studio directly:
<http://localhost:8777/Frontend/simulator/fw_studio2d_outdoor.html>).

Alternatives:
- **VS Code "Live Server"** extension → *Go Live* (serves the workspace over http).
  A `.claude/launch.json` `static` config is already set up for the in-app preview.
- Serve from the **repo root** specifically — the studios reference the model with
  `../Physics%20Engine/3D%20Map%20Physics/SIM%20V1%203D/web/…`, which only
  resolves when the root is the server root.

## 4. Verify it loaded

Open the studio and check the **engine** line under *Solver*:
- `engine: SIM V3 · ONNX surrogate (full-wave)` → the `.onnx` loaded (WebGPU or wasm).
- `engine: SIM V3 · penetrable FDTD (ONNX unavailable) · <why>` → no weights yet;
  the FDTD fallback is active. Export step 1 and reload.
