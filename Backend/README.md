# Backend/

Home for **new** backend / engine iterations that aren't yet wired into the
existing pipeline.

The active, load-bearing backend deliberately stays at the repo root and is
**not** moved here: its paths are hardcoded in the `Makefile` and in the Colab
training notebooks (which `git clone` the repo and read e.g. `SIM/phase_a.py`,
`SIM V2/manifest_v2.json` by literal string). Moving them would silently break
`make` and training. See [../DECISIONS.md](../DECISIONS.md).

| Where the current backend lives | What it is |
|---|---|
| `../SIM/` | v1 physics generator → dataset → surrogate → web assets (Phases A–F) |
| `../SIM V2/`, `../SIM V2.2 (1)/` | v2 enhanced-Motley-Keenan iteration (7-class, notebooks) |
| `../STEP_1/`, `../STEP_2/` | upstream rasterization + multiband metrics |

**Convention:** start a new engine experiment as `Backend/<name>/` when it
doesn't need the Makefile/notebook wiring; promote it into a `SIM/`-style layout
at the root once it stabilizes.
