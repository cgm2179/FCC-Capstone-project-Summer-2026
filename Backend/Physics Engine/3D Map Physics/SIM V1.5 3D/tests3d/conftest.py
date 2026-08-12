import sys
from pathlib import Path

# make the SIM V1.5 3D modules importable (pathloss_config / _models, link_budget, engine)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
