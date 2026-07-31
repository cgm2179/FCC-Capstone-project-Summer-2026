# UNet Path-Loss Surrogate pipeline (spec §10: one command regenerates all)
#   make everything   = prepare + test + dataset + audit + assets
# Phase C (training) runs on Colab: SIM/phase_c_train_colab.ipynb

PY = python3

.PHONY: everything prepare test sample dataset audit assets optimizer model clean \
        sim3d voxelize calc-3d assets-3d volumes-3d clean-3d

# fetch the trained surrogate (124 MB, too big for git) from the GitHub release
model:
	gh release download surrogate-v1 -p pl_unet.onnx -O SIM/web/pl_unet.onnx \
		--repo cgm2179/indoor-walk-test --clobber

# fetch the 10k-sample training dataset (1.2 GB) instead of regenerating it
dataset-fetch:
	gh release download dataset-v1 -p 'shard_*.npz' -D SIM/dataset \
		--repo cgm2179/indoor-walk-test --clobber

everything: prepare test dataset audit assets

prepare:
	$(PY) SIM/phase_a.py --prepare

test:
	$(PY) SIM/phase_a.py --test
	$(PY) SIM/phase_d_calibrate.py --sanity

sample:
	$(PY) SIM/phase_a.py --sample

dataset:
	$(PY) SIM/phase_b_dataset.py

audit:
	$(PY) SIM/phase_b_dataset.py --audit

assets:
	$(PY) SIM/export_web_assets.py

# --- 3-D pipeline (SIM V1 3D) -------------------------------------------------
# Mirrors the 2-D targets. Run from repo root (the SIM3D symlink resolves there):
#   make -f "Essentials + HTML/Makefile" sim3d
sim3d: voxelize calc-3d assets-3d

voxelize:
	$(PY) SIM3D/voxelize.py

calc-3d:
	$(PY) SIM3D/run_one_calc.py

assets-3d:
	$(PY) SIM3D/export_web3.py
	$(PY) SIM3D/export_collision_3d.py
	$(PY) SIM3D/export_antenna_catalog_3d.py

# real SceneV3 PL + eikonal T volumes for the browser (coverage field + wavefront
# sweep). Auto Tx by default; pass TX="x y z" for a specific voxel.
volumes-3d:
	$(PY) SIM3D/export_pl_volume.py $(if $(TX),--tx $(TX),)

clean-3d:
	rm -rf SIM3D/preview

optimizer:
	$(PY) SIM/phase_e_optimizer.py --objective coverage --stride 8 \
		--tx-power 23 --gain 3

clean:
	rm -rf SIM/preview SIM/dataset/shard_*.npz
