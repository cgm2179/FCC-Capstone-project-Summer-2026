# Essentials + HTML/

Project essentials: architecture decisions, source floor-plan ground truth, and
the Makefile that regenerates the SIM pipeline. Before changing anything
structural, read **[docs/DECISIONS.md](docs/DECISIONS.md)**.

| Path | What it is |
|---|---|
| **`docs/DECISIONS.md`** | Load-bearing architecture decisions and known-wrong list. |
| **`floorplan/`** | Source ground truth: georeferenced 7th-floor PNG + QGIS `.TAB` / `.aux.xml` / PseudoMercator CSV. Everything derives from these. |
| **`packages/SIM V2.2.zip`** | Packaged SIM V2.2 snapshot. |
| **`Makefile`** | `make everything` / `make test` / `make dataset` / `make assets` (+ 3-D targets via `SIM3D`). Run from repo root: `make -f "Essentials + HTML/Makefile" …`. |

Repo entry point and active app live at the root (`Frontend_Data_Display.html`,
`Frontend/`, `Data/`, `Physics Engine/`). Archived provenance lives in
[`../ARCHIVE/`](../ARCHIVE/README.md).
