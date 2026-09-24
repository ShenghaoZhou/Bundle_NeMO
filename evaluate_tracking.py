#!/usr/bin/env python3
"""
Trajectory Evaluation Utility for BundleNeMO.
Computes translation RMSE (cm) and geodesic rotation error (deg) vs Ground Truth,
including symmetry-minimized rotation error when applicable.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np


def geodesic_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute the geodesic angular distance in degrees between two 3x3 rotation matrices."""
    R_diff = R1 @ R2.T
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(tr)))


def symmetry_minimized_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray, symmetry_axis: str = 'z', discrete_order: int = 1) -> float:
    """
    Compute rotation error minimizing over discrete or continuous rotational symmetry.
    """
    min_err = geodesic_distance_deg(R_pred, R_gt)
    if discrete_order > 1:
        angles = np.linspace(0, 2 * np.pi, discrete_order, endpoint=False)
        for ang in angles[1:]:
            c, s = np.cos(ang), np.sin(ang)
            if symmetry_axis == 'z':
                R_sym = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            elif symmetry_axis == 'y':
                R_sym = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R_sym = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            err = geodesic_distance_deg(R_pred, R_gt @ R_sym)
            if err < min_err:
                min_err = err
    return min_err


def evaluate_trajectory(pred_poses: np.ndarray, gt_poses: np.ndarray, symmetry_axis: str = None, discrete_order: int = 1):
    """
    pred_poses: (N, 4, 4)
    gt_poses: (N, 4, 4)
    """
    N = min(len(pred_poses), len(gt_poses))
    if N == 0:
        raise ValueError("Empty trajectory provided for evaluation.")

    trans_errs = []
    rot_errs = []
    sym_rot_errs = []

    for i in range(N):
        P_pred = pred_poses[i]
        P_gt = gt_poses[i]

        t_err = np.linalg.norm(P_pred[:3, 3] - P_gt[:3, 3])
        trans_errs.append(t_err)

        r_err = geodesic_distance_deg(P_pred[:3, :3], P_gt[:3, :3])
        rot_errs.append(r_err)

        if symmetry_axis is not None and discrete_order > 1:
            sym_r_err = symmetry_minimized_rotation_error_deg(
                P_pred[:3, :3], P_gt[:3, :3], symmetry_axis=symmetry_axis, discrete_order=discrete_order
            )
            sym_rot_errs.append(sym_r_err)

    trans_rmse_cm = float(np.sqrt(np.mean(np.array(trans_errs) ** 2)) * 100.0)
    mean_rot_err = float(np.mean(rot_errs))
    median_rot_err = float(np.median(rot_errs))

    results = {
        'num_frames': N,
        'translation_rmse_cm': trans_rmse_cm,
        'mean_rotation_error_deg': mean_rot_err,
        'median_rotation_error_deg': median_rot_err
    }

    if len(sym_rot_errs) > 0:
        results['symmetry_mean_rotation_error_deg'] = float(np.mean(sym_rot_errs))
        results['symmetry_median_rotation_error_deg'] = float(np.median(sym_rot_errs))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 6-DoF Trajectory against Ground Truth")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Path to directory containing predicted ob_in_cam/*.txt poses")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Path to dataset directory with GT cameras.json & objects.json")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Optional output path for evaluation JSON report")
    parser.add_argument("--symmetry_axis", type=str, default=None, choices=['x', 'y', 'z'],
                        help="Object symmetry axis for symmetry-minimized evaluation")
    parser.add_argument("--symmetry_order", type=int, default=1,
                        help="Order of discrete rotational symmetry (e.g. 2 for 180 deg, 4 for 90 deg)")
    args = parser.parse_args()

    # Load predicted poses
    pose_dir = os.path.join(args.pred_dir, "ob_in_cam") if os.path.exists(os.path.join(args.pred_dir, "ob_in_cam")) else args.pred_dir
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    if len(pose_files) == 0:
        print(f"Error: No pose files (*.txt) found in {pose_dir}")
        sys.exit(1)

    pred_poses = np.array([np.loadtxt(f) for f in pose_files])
    print(f"Loaded {len(pred_poses)} predicted poses from {pose_dir}")

    if args.gt_dir:
        from run_bundle_nemo import load_dataset_frames, read_frame_data
        frames = load_dataset_frames(args.gt_dir)
        gt_poses = []
        for f in frames[:len(pred_poses)]:
            _, _, _, _, T_w_cam, T_w_obj_gt, _ = read_frame_data(f)
            if T_w_cam is not None and T_w_obj_gt is not None:
                T_c_o_gt = np.linalg.inv(T_w_cam) @ T_w_obj_gt
                gt_poses.append(T_c_o_gt)

        if len(gt_poses) == len(pred_poses):
            gt_poses = np.array(gt_poses)
            metrics = evaluate_trajectory(
                pred_poses, gt_poses,
                symmetry_axis=args.symmetry_axis,
                discrete_order=args.symmetry_order
            )

            print("\n" + "=" * 60)
            print(" Trajectory Evaluation Report vs Ground Truth")
            print("=" * 60)
            print(f"  Frames Evaluated:           {metrics['num_frames']}")
            print(f"  Translation RMSE:           {metrics['translation_rmse_cm']:.2f} cm")
            print(f"  Mean Rotation Error:        {metrics['mean_rotation_error_deg']:.2f}°")
            print(f"  Median Rotation Error:      {metrics['median_rotation_error_deg']:.2f}°")
            if 'symmetry_mean_rotation_error_deg' in metrics:
                print(f"  Symmetry-min Mean Rot Err:  {metrics['symmetry_mean_rotation_error_deg']:.2f}°")
                print(f"  Symmetry-min Med Rot Err:   {metrics['symmetry_median_rotation_error_deg']:.2f}°")
            print("=" * 60 + "\n")

            if args.out_json:
                with open(args.out_json, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved report to {args.out_json}")


if __name__ == "__main__":
    main()
