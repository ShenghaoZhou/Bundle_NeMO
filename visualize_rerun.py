#!/usr/bin/env python3
"""
Generate comprehensive Rerun (.rrd) recording for BundleNeMO GrowingNeMO:
- Visualizes 6-DoF tracking across all 150 frames of HOT3D clip-003312
- Side-by-side comparison:
  1. GrowingNeMO Estimated Trajectory (Vibrant Green)
  2. Baseline Clusters Trajectory (Orange)
  3. Ground Truth Trajectory (Red)
  4. Quest 3 Camera Motion (Blue)
- 3D Fused Canonical Point Cloud & Watertight Mesh attached to the moving object frame
- Timeline synchronized RGB video, camera pinhole frustum
- Real-time scalar plots: Geodesic rotation error & translation error over time
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
import open3d as o3d
import rerun as rr

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from run_bundle_nemo import load_dataset_frames, read_frame_data
from evaluate_tracking import geodesic_distance_deg
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.abspath("../data/hot3d/extracted/clip-003312"))
    parser.add_argument("--growing_dir", default=os.path.join(current_dir, "outputs", "hot3d_3312_growing_bundlenemo"))
    parser.add_argument("--baseline_dir", default=os.path.join(current_dir, "outputs", "hot3d_3312_full_clip"))
    parser.add_argument("--object_id", default="26", help="BOP object ID (default: 26)")
    parser.add_argument("--out_rrd", default=None, help="Output .rrd path (default: <growing_dir>/growing_bundlenemo_comparison.rrd)")
    args = parser.parse_args()

    data_dir = args.data_dir
    growing_dir = args.growing_dir
    baseline_dir = args.baseline_dir
    rrd_path = args.out_rrd if args.out_rrd else os.path.join(growing_dir, "growing_bundlenemo_comparison.rrd")


    print(f"[Rerun] Initializing Rerun recording...")
    rr.init("BundleNeMO_GrowingNeMO_Tracking", spawn=False)
    rr.save(rrd_path)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # 1. Load dataset frames
    frames = load_dataset_frames(data_dir, camera_id="214-1")
    print(f"[Rerun] Loaded {len(frames)} frames from {data_dir}")

    # 2. Load poses
    growing_pose_files = sorted(glob.glob(os.path.join(growing_dir, "ob_in_cam", "*.txt")))
    baseline_pose_files = sorted(glob.glob(os.path.join(baseline_dir, "ob_in_cam", "*.txt")))

    has_baseline = (len(baseline_pose_files) == len(growing_pose_files))
    print(f"[Rerun] Found {len(growing_pose_files)} GrowingNeMO poses (Baseline available: {has_baseline})")

    # 3. Load 3D Point Cloud & Mesh
    pcd_path = os.path.join(growing_dir, "fused_model.ply")
    mesh_path = os.path.join(growing_dir, "textured_mesh.obj")

    pts_clean = np.empty((0, 3))
    colors_clean = np.empty((0, 3))
    if os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts_clean = np.asarray(pcd.points)
        colors_clean = np.asarray(pcd.colors)
        print(f"[Rerun] Loaded point cloud with {len(pts_clean)} points")

    mesh_verts = np.empty((0, 3))
    mesh_tris = np.empty((0, 3), dtype=np.uint32)
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh_verts = np.asarray(mesh.vertices)
        mesh_tris = np.asarray(mesh.triangles, dtype=np.uint32)
        print(f"[Rerun] Loaded Poisson mesh with {len(mesh_verts)} vertices, {len(mesh_tris)} faces")

    # 4. Process Sequence and Trajectories
    traj_w_growing = []
    traj_w_baseline = []
    traj_w_gt = []
    traj_w_cam = []

    num_frames = min(len(frames), len(growing_pose_files))

    print(f"[Rerun] Streaming {num_frames} frames to {rrd_path}...")
    for idx in range(num_frames):
        f_info = frames[idx]
        rgb, depth, mask, K, T_w_cam, T_w_obj_gt, mask_modal = read_frame_data(f_info, object_id=args.object_id)

        T_c_o_grow = np.loadtxt(growing_pose_files[idx])
        T_w_o_grow = T_w_cam @ T_c_o_grow if T_w_cam is not None else T_c_o_grow

        traj_w_growing.append(T_w_o_grow[:3, 3].copy())
        if T_w_cam is not None:
            traj_w_cam.append(T_w_cam[:3, 3].copy())

        T_w_o_base = None
        if has_baseline:
            T_c_o_base = np.loadtxt(baseline_pose_files[idx])
            T_w_o_base = T_w_cam @ T_c_o_base if T_w_cam is not None else T_c_o_base
            traj_w_baseline.append(T_w_o_base[:3, 3].copy())

        if T_w_obj_gt is not None:
            traj_w_gt.append(T_w_obj_gt[:3, 3].copy())

        # Set time
        rr.set_time("frame", sequence=idx)
        rr.set_time("sim_time", duration=idx / 30.0)

        # Log camera & image
        if T_w_cam is not None:
            rr.log("world/camera", rr.Transform3D(translation=T_w_cam[:3, 3], mat3x3=T_w_cam[:3, :3]))

        ds = 0.5
        H, W = rgb.shape[:2]
        rgb_small = cv2.resize(rgb, (int(W * ds), int(H * ds)))
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                resolution=[int(W * ds), int(H * ds)],
                focal_length=float(K[0, 0] * ds),
                principal_point=[float(K[0, 2] * ds), float(K[1, 2] * ds)]
            )
        )
        rr.log("world/camera/pinhole/rgb", rr.Image(rgb_small))

        # Log object transforms
        rr.log("world/object_growing_nemo", rr.Transform3D(translation=T_w_o_grow[:3, 3], mat3x3=T_w_o_grow[:3, :3]))
        if T_w_obj_gt is not None:
            rr.log("world/object_gt", rr.Transform3D(translation=T_w_obj_gt[:3, 3], mat3x3=T_w_obj_gt[:3, :3]))
        if T_w_o_base is not None:
            rr.log("world/object_baseline", rr.Transform3D(translation=T_w_o_base[:3, 3], mat3x3=T_w_o_base[:3, :3]))

        # Log errors vs GT
        if T_w_cam is not None and T_w_obj_gt is not None:
            T_c_o_gt = np.linalg.inv(T_w_cam) @ T_w_obj_gt
            grow_r_err = geodesic_distance_deg(T_c_o_grow[:3, :3], T_c_o_gt[:3, :3])
            grow_t_err = np.linalg.norm(T_c_o_grow[:3, 3] - T_c_o_gt[:3, 3]) * 100.0

            rr.log("metrics/rotation_error_deg/growing_nemo", rr.Scalars(grow_r_err))
            rr.log("metrics/translation_error_cm/growing_nemo", rr.Scalars(grow_t_err))

            if has_baseline:
                base_r_err = geodesic_distance_deg(T_c_o_base[:3, :3], T_c_o_gt[:3, :3])
                base_t_err = np.linalg.norm(T_c_o_base[:3, 3] - T_c_o_gt[:3, 3]) * 100.0
                rr.log("metrics/rotation_error_deg/baseline", rr.Scalars(base_r_err))
                rr.log("metrics/translation_error_cm/baseline", rr.Scalars(base_t_err))

    # 5. Log static 3D models and trajectories
    print("[Rerun] Logging 3D trajectories and fused canonical model...")
    if len(traj_w_growing) > 1:
        rr.log("world/trajectories/growing_nemo", rr.LineStrips3D([np.array(traj_w_growing)], colors=[[40, 220, 100]], radii=0.003), static=True)
    if len(traj_w_baseline) > 1:
        rr.log("world/trajectories/baseline_clusters", rr.LineStrips3D([np.array(traj_w_baseline)], colors=[[255, 140, 0]], radii=0.0025), static=True)
    if len(traj_w_gt) > 1:
        rr.log("world/trajectories/ground_truth", rr.LineStrips3D([np.array(traj_w_gt)], colors=[[240, 50, 50]], radii=0.003), static=True)
    if len(traj_w_cam) > 1:
        rr.log("world/trajectories/camera_quest3", rr.LineStrips3D([np.array(traj_w_cam)], colors=[[60, 140, 255]], radii=0.002), static=True)

    # Attach fused 3D points to the estimated object
    if len(pts_clean) > 0:
        rr.log("world/object_growing_nemo/fused_model", rr.Points3D(positions=pts_clean, colors=colors_clean, radii=0.002), static=True)

    print(f"[Rerun] Successfully written Rerun recording: {rrd_path} ({os.path.getsize(rrd_path) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
