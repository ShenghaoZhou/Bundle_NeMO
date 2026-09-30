# GrowingNeMO × BundleNeMO Integration — Findings

**Dataset:** HOT3D `clip-003312` (birdhouse toy, BOP object ID 26, 150 frames)  
**Branch:** `main`  
**Date:** 2026-09-29

---

## Overview

This document records the results of integrating the **GrowingNeMO** unified-memory incremental pose estimator
(from the standalone `nemo/` pipeline) into the **BundleSDF BundleNeMO** tracking system, and the subsequent
debugging of reconstruction quality.

---

## 1. Integration Architecture

### What changed

| Component | Change |
|---|---|
| `NeMO/growing_nemo.py` | Copied from `nemo/growing_nemo.py`; added `model.nemo(features_2d, ...)` fallback because BundleSDF's `Model` class does not expose `encode_features` directly |
| `bundle_nemo/memory_bank.py` | Added `GrowingNeMOAdapter` — a thin adapter class bridging `GrowingNeMO` to the `BundleNeMOTracker` interface (`clusters` property, `initialize_reference`, `set_canonical_alignment`, `add_cluster`, `decode_query`, `get_canonical_3d_points`, `check_novelty`, `update`) |
| `bundle_nemo/__init__.py` | Exported `GrowingNeMOAdapter` |
| `bundle_nemo/tracker.py` | Added `use_growing_nemo: bool` + 4 `growing_*` params; conditional memory bank instantiation; separate `if self.use_growing_nemo:` branch in `process_frame()` step 5 (keyframe admission via `check_novelty()` + `update()`) |
| `run_bundle_nemo.py` | Added `--use_growing_nemo`, `--growing_max_keyframes`, `--growing_conf_threshold`, `--growing_inlier_threshold` CLI flags; full ATE side-by-side comparison table |
| `evaluate_tracking.py` | Added `umeyama_alignment()` (SVD closed-form SE3/Sim3); enhanced `evaluate_trajectory()` with raw translation RMSE/mean/median, SE(3) ATE RMSE/mean/median, Sim(3) ATE RMSE + scale, rotation RMSE |
| `visualize_rerun.py` | New script — generates comprehensive `.rrd` Rerun recording with all four trajectories (GrowingNeMO, Baseline, GT, Camera), canonical point cloud, Poisson mesh, and per-frame error plots |
| `serve_rrd.py` | CORS-enabled HTTP server for hosting `.rrd` files to the Rerun cloud viewer |

### Key design: GrowingNeMO memory bank adapter

`GrowingNeMOAdapter` maintains a single `GrowingNeMO` model as its only "cluster".
Unlike `AlignedDynamicNeMOMemoryBank` which maintains an ever-growing list of NeMO cluster instances
(one per keyframe, each encoding a separate viewpoint), GrowingNeMO maintains a **unified** 3D feature
volume pruned and updated incrementally. Keyframe admission is controlled by `check_novelty()` which
checks whether confidence coverage over the object mask drops below a threshold.

---

## 2. Trajectory ATE Results (150 frames, HOT3D clip-003312)

All metrics computed vs. HOT3D ground-truth poses.

| Metric | Baseline (Cluster NeMO) | GrowingNeMO (Unified) |
|---|---|---|
| **Throughput** | 8.5 FPS | **16.8 FPS** (+98%) |
| Raw Translation RMSE | 13.14 cm | 13.42 cm |
| Raw Translation Mean | 12.72 cm | 12.62 cm |
| **SE(3) Aligned ATE RMSE** | **9.23 cm** | 10.17 cm |
| SE(3) Aligned ATE Mean | 8.46 cm | 8.92 cm |
| Sim(3) Aligned ATE RMSE | 5.59 cm (scale=1.34) | 9.59 cm (scale=1.14) |
| Mean Geodesic Rotation Error | 31.32° | 44.21° |
| **Median Geodesic Rotation Error** | 33.41° | **26.94°** (−19%) |
| Rotation RMSE | 35.35° | 65.89° |

### Interpretation

- **Median rotation** is better with GrowingNeMO (26.94° vs. 33.41°) — the unified memory makes the
  typical frame more accurate because all keyframes cooperate in a single consistent canonical space.
- **Mean rotation and RMSE are worse** because of a bilateral symmetry failure at frames 125–133 where
  the birdhouse's left-right symmetry causes GrowingNeMO to briefly lock onto the 180° mirrored pose
  (error ~175°). This outlier event dominates the mean/RMSE.
- **Translation ATE** is comparable — both pipelines are in the 9–10 cm SE(3) RMSE range.
- **Throughput doubles** because GrowingNeMO avoids the `O(K)` multi-cluster decode and uses cached
  DINO features for keyframe updates.

---

## 3. Reconstruction Quality Issue and Fix

### Symptom

The fused canonical 3D point cloud from GrowingNeMO had only **4,078 points** vs. **18,681** for the
baseline and **~50 k** for the standalone NeMO incremental pipeline, making the reconstructed object
appear much less complete.

### Root Cause

Two compounding causes:

#### Cause 1 — Point fusion gated on novelty events (dominant)

In [`bundle_nemo/tracker.py`](bundle_nemo/tracker.py), `CanonicalObjectFusion.integrate_points()` was
called only inside:

```python
if keyframe_added and len(self.memory_bank.clusters) > 0:   # LINE 521 (before fix)
```

For GrowingNeMO, `keyframe_added = True` only when `check_novelty()` fires — about **12 frames** out
of 150. The standalone `run_full_clip_growing.py` accumulates decoded `pts3d` at **every PnP-successful
frame** (`if pnp_success and len(valid_3d) > 0`).

#### Cause 2 — Wrong source of points fused

Before the fix, the fused points were `surface_points` — the 1,500 randomly sampled canonical probe
points used as input to the CrossViewEncoder. After the fix, the fused points are `pts3d_cand` — the
high-confidence decoded correspondences directly from the decoder output, strided ×4, which are the
same quantity used in `run_full_clip_growing.py`.

### Fix

Separated the fusion logic for GrowingNeMO from the baseline:

```python
# For GrowingNeMO: fuse pts3d_cand every PnP-successful frame (matches standalone NeMO pipeline)
if self.use_growing_nemo and success and len(pts3d_cand) > 0:
    pts_fuse = pts3d_cand[::4]
    ...
    self.fusion.integrate_points(pts_fuse, colors_per_frame, weights_per_frame)

# For baseline clusters: unchanged — fuse cluster surface on keyframe admission
elif keyframe_added and not self.use_growing_nemo and len(self.memory_bank.clusters) > 0:
    ...
    self.fusion.integrate_points(pts_canon, colors, weights)
```

### Result after fix

| Metric | Before fix | After fix |
|---|---|---|
| Fused points | 4,078 | **36,950** (+9×) |
| Mesh vertices | 17,476 | **81,611** (+4.7×) |

---

## 4. Known Limitations

1. **Symmetry failure (frames 125–133):** The birdhouse has bilateral visual symmetry. GrowingNeMO
   briefly latches onto the 180°-mirrored pose. This inflates mean rotation error by ~12°. A
   symmetry-aware disambiguation (e.g. depth-guided or temporal consistency check) would resolve this.

2. **Scale drift:** Sim(3) estimated scale for GrowingNeMO is 1.14 (vs. 1.34 for baseline) — closer to
   the ideal 1.0, but the metric scale is still estimated from a single depth reading at frame 0. A
   multi-frame scale estimation would improve absolute translation accuracy.

3. **Single-object setting:** Integration tested only on HOT3D clip-003312 (one object class). Full
   generalization to other HOT3D sequences requires validating that `check_novelty()` thresholds work
   well across different object appearances.

---

## 5. How to Reproduce

### Baseline run

```bash
python run_bundle_nemo.py \
  --data_dir ../data/hot3d/extracted/clip-003312 \
  --checkpoint NeMO/checkpoints/checkpoint.pth \
  --out_dir outputs/hot3d_3312_full_clip \
  --object_id 26
```

### GrowingNeMO run (with per-frame dense fusion)

```bash
python run_bundle_nemo.py \
  --data_dir ../data/hot3d/extracted/clip-003312 \
  --checkpoint NeMO/checkpoints/checkpoint.pth \
  --out_dir outputs/hot3d_3312_growing_dense \
  --object_id 26 \
  --use_growing_nemo
```

### Evaluate separately

```bash
python evaluate_tracking.py \
  --pose_dir outputs/hot3d_3312_growing_dense/ob_in_cam \
  --gt_dir ../data/hot3d/extracted/clip-003312 \
  --object_id 26
```

### Visualize with Rerun

```bash
python visualize_rerun.py \
  --growing_dir outputs/hot3d_3312_growing_dense \
  --baseline_dir outputs/hot3d_3312_full_clip \
  --data_dir ../data/hot3d/extracted/clip-003312 \
  --out_rrd outputs/hot3d_3312_growing_dense/growing_bundlenemo_v2.rrd

# Serve the .rrd file (CORS-enabled)
python3 serve_rrd.py 9090 outputs/hot3d_3312_growing_dense
# Then open: https://app.rerun.io/version/0.38.1/?url=http://<your-ip>:9090/growing_bundlenemo_v2.rrd
```
