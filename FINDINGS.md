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

| Metric | Baseline (Cluster NeMO) | GrowingNeMO (Initial) | GrowingNeMO (Smooth & Dense) |
|---|---|---|---|
| **Throughput** | 8.5 FPS | **16.8 FPS** | **16.0 FPS** (+88%) |
| Raw Translation RMSE | 13.14 cm | 13.42 cm | **12.87 cm** (−2.1%) |
| Raw Translation Mean | 12.72 cm | 12.62 cm | **12.30 cm** (−3.3%) |
| **SE(3) Aligned ATE RMSE** | 9.23 cm | 10.17 cm | **8.89 cm** (−3.7% vs baseline) |
| SE(3) Aligned ATE Mean | 8.46 cm | 8.92 cm | **8.13 cm** (−3.9% vs baseline) |
| SE(3) Aligned ATE Median | 8.20 cm | 8.69 cm | **7.78 cm** (−5.1% vs baseline) |
| Sim(3) Aligned ATE RMSE | 5.59 cm (scale=1.34) | 9.59 cm (scale=1.14) | **7.57 cm** (scale=1.20) |
| Mean Geodesic Rotation Error | 31.32° | 44.21° | **32.65°** |
| **Median Geodesic Rotation Error** | 33.41° | 26.94° | **26.52°** (−20.6% vs baseline) |
| Rotation RMSE | 35.35° | 65.89° | **44.35°** |
| Fused 3D Points | 18,681 | 4,078 | **38,046** (+103% vs baseline) |
| Mesh Vertices | 46,340 | 17,476 | **81,734** (+76% vs baseline) |

### Interpretation

- **SE(3) ATE RMSE now outperforms baseline** (8.89 cm vs 9.23 cm baseline) with higher throughput (16 FPS vs 8.5 FPS).
- **Median rotation error is significantly superior** (26.52° vs 33.41° baseline, a 20.6% reduction).
- **Reconstruction completeness** is doubled compared to baseline (38k vs 18k fused points, 81k vs 46k mesh vertices).
- **End-of-trajectory jitter is eliminated**, with consecutive translation deltas dropping from 14–20 cm down to sub-centimeter (0.3–1.0 cm) smooth tracking.

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
| Fused points | 4,078 | **38,046** (+9.3×) |
| Mesh vertices | 17,476 | **81,734** (+4.7×) |

---

## 4. Trajectory Jitter at End: Root Causes and Fixes

### Symptoms

Inspection of the per-frame translation and rotation deltas revealed two distinct jitter phenomena:
1. **Frames 110–121**: Oscillation back and forth between two distant pose hypotheses (`t ≈ [0.24, -0.01, 0.35]` and `t ≈ [0.13, -0.05, 0.28]`), producing 13–15 cm deltas every single frame.
2. **Frames 139–141**: Frame 140 abruptly jumped 19.8 cm away (`t = [0.031, -0.022, 0.371]`) and Frame 141 jumped 17.8 cm back (`t = [0.199, 0.026, 0.337]`), with > 100° rotational swing.

### Root Causes

1. **Buggy Spike Inversion in `BundleNeMOOptimizer`**:
   In `optimizer.py`:
   ```python
   # Previous buggy logic:
   if is_spike and pnp_valid and T_cam_obj_pnp is not None:
       T_init = T_cam_obj_pnp   # Bypassed the filter clamp!
   ...
   if is_spike:
       self.filter.resync(T_opt)  # Overwrote filter state with the outlier!
   ```
   Whenever a PnP jump exceeded the kinematic clamp threshold (`jump_m > 0.055 m` or `theta > 35°`), the code bypassed `T_filt`, initialized BA directly at the outlier `T_cam_obj_pnp`, and resynced the filter to the outlier. The filter was actively accepting and locking onto spikes rather than smoothing them.

2. **Absence of Motion Prior & Point Subsampling in PnP**:
   `cv2.solvePnPRansac` with SQPnP was called without temporal guidance on up to 10,000 dense points. On ambiguous/symmetric viewpoints with 400 iterations, RANSAC would flip between two local minima on alternating frames.

3. **1-Frame Stale Memory Glitch on Keyframe Admission**:
   At Frame 140, low confidence coverage (`cov = 0.42 < 0.45`) triggered keyframe admission. However, in `tracker.py`, pose estimation occurred *before* the keyframe update. Frame 140 was tracked using the stale memory, yielding a degenerate PnP pose, before absorbing the new viewpoint. At Frame 141, the updated memory activated, snapping the pose back and creating a 1-frame 20 cm spike.

4. **Runaway Angular Extrapolation on Lost Frames**:
   When tracking became lost or invalid, `KinematicStateFilter` performed forward extrapolation by rotating by `self.vel_rot`. If the previous step had been clamped to 35°, the filter would spin the object by 35° every subsequent frame.

### Solutions Implemented

1. **Fixed BA Initialization & Resync in Optimizer (`bundle_nemo/optimizer.py`)**:
   - `T_init` is now strictly initialized from the smoothed/clamped kinematic pose `T_filt`.
   - `self.filter.resync(T_opt)` only executes when `not is_spike`, preventing corrupting the filter state with outliers.
   - Decayed linear velocity (`0.5 * vel_pos`) and zeroed angular extrapolation (`vel_rot = I`) on lost frames to prevent runaway spin.

2. **Temporal Continuity Verification & LM Refinement in PnP (`bundle_nemo/correspondence.py`)**:
   - Subsample candidate points to `max_pnp_points = 2048` with `iterations_pnp = 800` (matching standalone NeMO).
   - Added `prior_T` support to `solve_pnp`: when a PnP hypothesis jumps significantly (> 5.5 cm or > 25°), it evaluates the inliers of the locally continuous pose refined via `cv2.solvePnPRefineLM`. If the continuous hypothesis has strong inlier support, the continuous refined pose is preferred over the RANSAC jump.

3. **Immediate Refresh upon Keyframe Admission (`bundle_nemo/tracker.py`)**:
   - When `check_novelty()` triggers a keyframe update, the frame immediately re-decodes and re-solves PnP against the refreshed unified memory, eliminating 1-frame glitches on novel views.

### Results

- Consecutive frame jumps at the end (frames 135–150) reduced from **19.8 cm to ≤ 1.0 cm** (average delta 0.5 cm).
- Alternating 14 cm oscillations between frames 112–121 reduced to **0.2–1.3 cm**.
- **SE(3) Aligned ATE RMSE dropped from 10.17 cm to 8.89 cm** (outperforming baseline).
- **Rotation RMSE dropped from 65.89° to 44.35°**.

---

## 5. Known Limitations

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
