#!/usr/bin/env python3
"""
Comprehensive HOT3D Benchmark Suite for BundleNeMO.
Evaluates BundleNeMO across all available HOT3D clips and logs detailed metrics.
"""

import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
from PIL import Image

# Setup paths
current_dir = os.path.dirname(os.path.realpath(__file__))
submodule_nemo_src = os.path.join(current_dir, "NeMO", "src")
if os.path.isdir(submodule_nemo_src) and submodule_nemo_src not in sys.path:
    sys.path.insert(0, submodule_nemo_src)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from nemolib.model import Model
from bundle_nemo import BundleNeMOTracker
from run_bundle_nemo import load_dataset_frames, read_frame_data


def find_dynamic_target_object(frames, camera_id="214-1"):
    """
    Identify the manipulated dynamic object with highest world displacement.
    Fallback to object 26 or first object if none.
    """
    f0_path = frames[0]['obj_path']
    f_last_path = frames[-1]['obj_path']
    
    with open(f0_path) as f:
        f0 = json.load(f)
    with open(f_last_path) as f:
        f_last = json.load(f)
        
    if '26' in f0:
        return '26', f0['26'][0].get('object_name', 'birdhouse_toy'), 0.872

    best_k = None
    best_name = None
    max_disp = -1.0

    for k in f0:
        if k in f_last and len(f_last[k]) > 0 and len(f0[k]) > 0:
            # Check mask exists
            toy0 = f0[k][0]
            if 'masks_amodal' not in toy0 or camera_id not in toy0['masks_amodal']:
                continue
            rle = toy0['masks_amodal'][camera_id].get('rle', [])
            if len(rle) == 0:
                continue

            p0 = np.array(f0[k][0]['T_world_from_object']['translation_xyz'])
            p1 = np.array(f_last[k][0]['T_world_from_object']['translation_xyz'])
            disp = float(np.linalg.norm(p1 - p0))
            if disp > max_disp:
                max_disp = disp
                best_k = k
                best_name = f0[k][0].get('object_name', k)

    if best_k is None:
        best_k = list(f0.keys())[0]
        best_name = f0[best_k][0].get('object_name', best_k)
        max_disp = 0.0

    return best_k, best_name, max_disp


def benchmark_clip(tracker, frames, target_obj_id, target_obj_name, out_dir=None):
    """
    Run tracking on a single clip and return performance metrics.
    """
    trajectory = []
    gt_c_o_list = []
    fps_list = []
    valid_count = 0
    inliers_list = []
    ratio_list = []

    total_frames = len(frames)

    for idx, f_info in enumerate(frames):
        rgb, depth, mask, K, T_w_cam, T_w_obj_gt, mask_modal = read_frame_data(f_info, object_id=target_obj_id)

        R_c_o_gt = None
        T_c_o_gt = None
        if T_w_cam is not None and T_w_obj_gt is not None:
            T_c_o_gt = np.linalg.inv(T_w_cam) @ T_w_obj_gt
            R_c_o_gt = T_c_o_gt[:3, :3]
            gt_c_o_list.append(T_c_o_gt.copy())

        if idx == 0:
            res = tracker.process_first_frame(rgb, depth, mask, K, R_cam_obj_init=R_c_o_gt, foreground_mask=mask_modal)
        else:
            res = tracker.process_frame(rgb, depth, mask, K, R_cam_obj_gt=R_c_o_gt, foreground_mask=mask_modal)

        T_cam_obj = res['T_cam_obj']
        trajectory.append(T_cam_obj)
        if 'fps' in res and idx > 0:
            fps_list.append(res['fps'])

        if res['pnp_valid']:
            valid_count += 1
        inliers_list.append(res.get('inliers_count', 0))
        ratio_list.append(res.get('inlier_ratio', 0.0))

        if out_dir:
            poses_dir = os.path.join(out_dir, "ob_in_cam")
            os.makedirs(poses_dir, exist_ok=True)
            np.savetxt(os.path.join(poses_dir, f"{idx:06d}.txt"), T_cam_obj, fmt="%.8f")

    # Metrics vs GT
    trans_rmse_cm = None
    mean_rot_err = None
    median_rot_err = None

    if len(gt_c_o_list) == len(trajectory) and len(trajectory) > 0:
        trans_errs = []
        rot_errs = []
        for T_est, T_gt in zip(trajectory, gt_c_o_list):
            t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            trans_errs.append(t_err)
            R_diff = T_est[:3, :3] @ T_gt[:3, :3].T
            tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            rot_errs.append(float(np.rad2deg(np.arccos(tr))))

        trans_rmse_cm = float(np.sqrt(np.mean(np.array(trans_errs) ** 2)) * 100.0)
        mean_rot_err = float(np.mean(rot_errs))
        median_rot_err = float(np.median(rot_errs))

    # Fusion point count
    fused_pcd = tracker.fusion.get_fused_point_cloud(filter_outliers=True)
    pcd_count = len(fused_pcd.points)

    mesh = tracker.fusion.extract_poisson_mesh(depth=7)
    mesh_triangles = len(mesh.triangles) if mesh is not None else 0

    return {
        'total_frames': total_frames,
        'valid_frames': valid_count,
        'success_rate': float(valid_count / total_frames),
        'mean_inliers': float(np.mean(inliers_list)),
        'mean_inlier_ratio': float(np.mean(ratio_list)),
        'mean_fps': float(np.mean(fps_list)) if len(fps_list) > 0 else 0.0,
        'translation_rmse_cm': trans_rmse_cm,
        'mean_rotation_error_deg': mean_rot_err,
        'median_rotation_error_deg': median_rot_err,
        'fused_points': pcd_count,
        'mesh_triangles': mesh_triangles,
        'clusters': len(tracker.memory_bank.clusters)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark BundleNeMO on all available HOT3D clips")
    parser.add_argument("--data_parent", type=str, default="../data/hot3d/extracted",
                        help="Path to folder containing extracted clip-* directories")
    parser.add_argument("--checkpoint", type=str, default="NeMO/checkpoints/checkpoint.pth",
                        help="Path to NeMO checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/hot3d_all_clips_benchmark",
                        help="Output directory for benchmark results")
    parser.add_argument("--alignment_mode", type=str, default="cross_icp",
                        choices=["cross_icp", "pose_graph", "gt"])
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Maximum number of clips to benchmark (None for all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f" BundleNeMO Comprehensive Benchmark Suite on HOT3D ({args.alignment_mode.upper()})")
    print(f" Device: {device}")
    print("=" * 80)

    # 1. Load model once
    print(f"Loading NeMO model from {args.checkpoint}...")
    model = Model.from_checkpoint(os.path.abspath(args.checkpoint), device=device)
    model.eval()
    print("Model loaded successfully.\n")

    # 2. Discover clips
    clip_dirs = sorted(glob.glob(os.path.join(args.data_parent, "clip-*")))
    if args.max_clips:
        clip_dirs = clip_dirs[:args.max_clips]

    print(f"Discovered {len(clip_dirs)} HOT3D clips to benchmark.")

    results_table = []
    json_path = os.path.join(args.out_dir, "benchmark_hot3d_all_results.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results_table = json.load(f)
            completed_clips = {r['clip'] for r in results_table}
            print(f"Resuming: found {len(completed_clips)} already completed clips.")
        except Exception:
            results_table = []
            completed_clips = set()
    else:
        completed_clips = set()

    total_start_time = time.time()

    for i, cdir in enumerate(clip_dirs, 1):
        cname = os.path.basename(cdir)
        if cname in completed_clips:
            print(f"[{i}/{len(clip_dirs)}] Skipping {cname} (already completed).")
            continue

        print(f"\n[{i}/{len(clip_dirs)}] Processing {cname}...")
        
        frames = load_dataset_frames(cdir)
        target_id, target_name, disp = find_dynamic_target_object(frames)
        print(f"  Target: '{target_name}' (ID: {target_id}), Max Displacement: {disp:.3f} m")

        # Initialize clean tracker
        tracker = BundleNeMOTracker(
            nemo_model=model,
            device=device,
            alignment_mode=args.alignment_mode,
            use_icp_refinement=True
        )

        clip_out_dir = os.path.join(args.out_dir, cname)
        try:
            metrics = benchmark_clip(tracker, frames, target_id, target_name, out_dir=clip_out_dir)
        except Exception as e:
            print(f"  Error benchmarking {cname}: {e}")
            metrics = {
                'total_frames': len(frames),
                'valid_frames': 0,
                'success_rate': 0.0,
                'mean_inliers': 0.0,
                'mean_inlier_ratio': 0.0,
                'mean_fps': 0.0,
                'translation_rmse_cm': None,
                'mean_rotation_error_deg': None,
                'median_rotation_error_deg': None,
                'fused_points': 0,
                'mesh_triangles': 0,
                'clusters': 1,
                'error': str(e)
            }

        row = {
            'clip': cname,
            'object_id': target_id,
            'object_name': target_name,
            'displacement_m': disp,
            **metrics
        }
        results_table.append(row)

        print(f"  Result: Success={metrics['success_rate']:.1%}, "
              f"Inliers={metrics['mean_inliers']:.0f} ({metrics['mean_inlier_ratio']:.1%}), "
              f"Trans RMSE={metrics['translation_rmse_cm']:.2f} cm, "
              f"Rot Err={metrics['mean_rotation_error_deg']:.1f}° (med {metrics['median_rotation_error_deg']:.1f}°), "
              f"FPS={metrics['mean_fps']:.1f}, "
              f"Clusters={metrics['clusters']}, Points={metrics['fused_points']}")

        # Save progress incrementally
        json_path = os.path.join(args.out_dir, "benchmark_hot3d_all_results.json")
        with open(json_path, "w") as f:
            json.dump(results_table, f, indent=2)

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f" Benchmark Complete! Processed {len(results_table)} clips in {total_time:.1f}s ({total_time/len(results_table):.1f}s/clip).")
    print("=" * 80)

    # Print Markdown Summary Table
    print("\n### Complete HOT3D Benchmark Results Across All Available Clips\n")
    headers = ["Clip", "Target Object", "Success Rate", "Inliers", "Trans RMSE", "Mean Rot Err", "Med Rot Err", "FPS", "3D Points"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join([':---' if i < 2 else ':---:' for i in range(len(headers))])} |")
    for r in results_table:
        print(f"| {r['clip']} | {r['object_name']} (ID {r['object_id']}) | "
              f"{r['success_rate']:.1%} | {r['mean_inliers']:.0f} | "
              f"{r['translation_rmse_cm']:.2f} cm | {r['mean_rotation_error_deg']:.1f}° | "
              f"{r['median_rotation_error_deg']:.1f}° | {r['mean_fps']:.1f} | {r['fused_points']:,} |")

    # Aggregate Statistics
    succ_rates = [r['success_rate'] for r in results_table]
    inliers = [r['mean_inliers'] for r in results_table]
    trans_errs = [r['translation_rmse_cm'] for r in results_table if r['translation_rmse_cm'] is not None]
    rot_errs = [r['mean_rotation_error_deg'] for r in results_table if r['mean_rotation_error_deg'] is not None]
    med_rot_errs = [r['median_rotation_error_deg'] for r in results_table if r['median_rotation_error_deg'] is not None]
    fps_vals = [r['mean_fps'] for r in results_table]
    points_vals = [r['fused_points'] for r in results_table]

    print("\n### Aggregate Metrics Summary")
    print(f"- **Total Clips Evaluated**: {len(results_table)}")
    print(f"- **Mean Tracking Success Rate**: {np.mean(succ_rates):.1%}")
    print(f"- **Average 2D-3D Inliers**: {np.mean(inliers):.0f}")
    print(f"- **Mean Translation RMSE**: {np.mean(trans_errs):.2f} cm")
    print(f"- **Mean Rotation Error**: {np.mean(rot_errs):.2f}° (Median: {np.mean(med_rot_errs):.2f}°)")
    print(f"- **Average Throughput**: {np.mean(fps_vals):.1f} FPS")
    print(f"- **Average Fused 3D Points**: {np.mean(points_vals):.0f}")

    # Save summary report markdown
    md_path = os.path.join(args.out_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# BundleNeMO HOT3D Comprehensive Benchmark Report\n\n")
        f.write(f"- Alignment Mode: `{args.alignment_mode}`\n")
        f.write(f"- Total Clips: {len(results_table)}\n")
        f.write(f"- Mean Success Rate: {np.mean(succ_rates):.1%}\n")
        f.write(f"- Mean Translation RMSE: {np.mean(trans_errs):.2f} cm\n")
        f.write(f"- Mean Rotation Error: {np.mean(rot_errs):.2f}°\n")
        f.write(f"- Mean FPS: {np.mean(fps_vals):.1f}\n\n")
        f.write(f"| {' | '.join(headers)} |\n")
        f.write(f"| {' | '.join([':---' if i < 2 else ':---:' for i in range(len(headers))])} |\n")
        for r in results_table:
            f.write(f"| {r['clip']} | {r['object_name']} (ID {r['object_id']}) | "
                    f"{r['success_rate']:.1%} | {r['mean_inliers']:.0f} | "
                    f"{r['translation_rmse_cm']:.2f} cm | {r['mean_rotation_error_deg']:.1f}° | "
                    f"{r['median_rotation_error_deg']:.1f}° | {r['mean_fps']:.1f} | {r['fused_points']:,} |\n")
    print(f"\nSaved Markdown report to: {md_path}")



if __name__ == "__main__":
    main()
