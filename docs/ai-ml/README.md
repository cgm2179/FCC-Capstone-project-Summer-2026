# AI/ML in the RF Simulation Studio — enabling it per project + roadmap

The Simulation tab can solve RF coverage two ways:

- **Non‑AI/ML (physics)** — FDTD full‑wave, the Eikonal far‑field, ray tracing, Motley‑Keenan,
  and FSPL. These work on **any** geometry and are the **only** solvers offered for a
  **newly‑created project**.
- **AI/ML (surrogate)** — the trained ONNX U‑Nets (`fw_unet2d.onnx`, `fw_bs.onnx`,
  `fw_unet3d.onnx`) that approximate the full‑wave field ~orders of magnitude faster.

## Why AI/ML is OFF for created projects

Each surrogate was **trained on the DEFAULT dataset only** — the 7th‑floor FCC plan (indoor) and
the NoMa DC tile (outdoor). A surrogate is valid **only for the geometry/материals it was trained
on**; running it on a different building or city produces meaningless numbers. So every project
the backend creates carries **`ai_ml: false`** in its manifest, and the Simulation tab shows the
ONNX/Auto option **disabled** (with the reason). The built‑in **Default** projects carry
`ai_ml: true`. (The disabled‑ONNX presentation + the extra non‑AI/ML solvers are the follow‑on
Simulation‑UI PR.)

## How to enable AI/ML for a NEW project

To make the surrogate valid for your project's geometry, retrain it on that geometry and point
the manifest at the new model:

1. **Voxelize / rasterize** your project (the backend already does this on create:
   `Backend/server/pipeline.py` → `voxelize.py` / `rasterize_floorplan.py`).
2. **Generate synthetic training data** on that grid with the physics engine — sweep transmitter
   positions × bands and solve the field:
   - 2D: `Physics Engine/2D/SIM V3` (CuPy generation) → FDTD near‑field fields.
   - 3D: `Physics Engine/3D Map Physics/SIM V1 3D` (CuPy/JAX generation) → FDTD volumes.
   Aim for the same featurization the trainers use (Tx one‑hot + material channels + band).
3. **Train the surrogate** (teacher = FDTD, student = U‑Net):
   - 2D U‑Net — PyTorch (see the SIM V3 training notebooks). Targets: complex field or PL.
   - 3D U‑Net — JAX (see the SIM V1 3D Phase‑C notebooks). Physics‑weighted loss.
4. **Export to ONNX** and drop the file next to the others in `SIM*/web/` (or the project's
   `data/` dir).
5. **Wire it into the manifest**: set `ai_ml: true` and add the model path, e.g.
   ```json
   { "ai_ml": true, "surrogate": { "2d": "stored:fw_unet2d.onnx" } }
   ```
   Then the Simulation tab re‑enables the ONNX/Auto solver for that project.

Measured surrogate accuracy (capstone): indoor 2D RMSE ≈ 8.89 dB (Spearman +0.70); outdoor 2D
RMSE ≈ 14.7–15.7 dB; 3D RMSE ≈ 10.4 dB — targets < 5 dB. Treat surrogates as fast approximations
and cross‑check against the physics solvers.

## Planned next iterations (roadmap)

- **ADE‑FDTD** — add Auxiliary Differential Equations (Debye / Drude / Lorentz) so material
  permittivity ε(ω) is modeled dispersively, improving accuracy across the 600 MHz–5 GHz range.
- **Conformal / cut‑cell FDTD** — deform boundary cells to follow curved/angled surfaces,
  removing the staircasing error at material boundaries without globally refining the grid.
- **Monte‑Carlo / VMC training** — sample the probabilistic nature of EM propagation (Gaussian
  processes) to produce a most‑likely coverage prediction and uncertainty bands, enriching the
  deep‑learning training distribution.
- **Plan the engine + dashboard up front** — the biggest lesson from v1/v2/v3: design the data
  contracts and grid conventions before generating training shards.
