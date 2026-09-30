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
from typing import Tuple, Dict, Any, Optional
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


def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = False) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity or rigid transform (s, R, t) such that s * R @ X + t ~ Y
    using the Umeyama (1991) closed-form SVD method.
    X: (N, 3) predicted positions
    Y: (N, 3) ground truth positions
    Returns:
        s: float scale factor (1.0 if with_scale=False)
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        X_aligned: (N, 3) aligned predicted positions
    """
    assert len(X) == len(Y) and len(X) >= 3, "At least 3 points required for Umeyama alignment."
    N, dim = X.shape

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    X_centered = X - mu_x
    Y_centered = Y - mu_y

    var_x = np.mean(np.sum(X_centered ** 2, axis=1))

    # Covariance matrix
    Sigma = (Y_centered.T @ X_centered) / N

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale and var_x > 1e-12:
        s = float(np.sum(D * np.diag(S)) / var_x)
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    X_aligned = s * (X @ R.T) + t

    return s, R, t, X_aligned


def evaluate_trajectory(pred_poses: np.ndarray, gt_poses: np.ndarray, symmetry_axis: str = None, discrete_order: int = 1):
    """
    pred_poses: (N, 4, 4)
    gt_poses: (N, 4, 4)
    """
    N = min(len(pred_poses), len(gt_poses))
    if N == 0:
        raise ValueError("Empty trajectory provided for evaluation.")

    # 1. Raw Absolute Trajectory Error (no alignment)
    trans_errs = []
    rot_errs = []
    sym_rot_errs = []

    pos_pred = pred_poses[:N, :3, 3]
    pos_gt = gt_poses[:N, :3, 3]

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
    rot_rmse_deg = float(np.sqrt(np.mean(np.array(rot_errs) ** 2)))

    # 2. Umeyama SE(3) Rigid Aligned Trajectory Error (Standard ATE)
    _, _, _, pos_pred_se3 = umeyama_alignment(pos_pred, pos_gt, with_scale=False)
    se3_trans_errs = np.linalg.norm(pos_pred_se3 - pos_gt, axis=-1)
    se3_ate_rmse_cm = float(np.sqrt(np.mean(se3_trans_errs ** 2)) * 100.0)
    se3_ate_mean_cm = float(np.mean(se3_trans_errs) * 100.0)
    se3_ate_median_cm = float(np.median(se3_trans_errs) * 100.0)

    # 3. Umeyama Sim(3) Scale+Rigid Aligned Trajectory Error (Scale-invariant ATE)
    scale_sim3, _, _, pos_pred_sim3 = umeyama_alignment(pos_pred, pos_gt, with_scale=True)
    sim3_trans_errs = np.linalg.norm(pos_pred_sim3 - pos_gt, axis=-1)
    sim3_ate_rmse_cm = float(np.sqrt(np.mean(sim3_trans_errs ** 2)) * 100.0)
    sim3_ate_mean_cm = float(np.mean(sim3_trans_errs) * 100.0)

    results = {
        'num_frames': N,
        'raw_translation_rmse_cm': trans_rmse_cm,
        'raw_translation_mean_cm': float(np.mean(trans_errs) * 100.0),
        'raw_translation_median_cm': float(np.median(trans_errs) * 100.0),
        'se3_ate_rmse_cm': se3_ate_rmse_cm,
        'se3_ate_mean_cm': se3_ate_mean_cm,
        'se3_ate_median_cm': se3_ate_median_cm,
        'sim3_ate_rmse_cm': sim3_ate_rmse_cm,
        'sim3_ate_scale': float(scale_sim3),
        'mean_rotation_error_deg': mean_rot_err,
        'median_rotation_error_deg': median_rot_err,
        'rotation_rmse_deg': rot_rmse_deg
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

            print("\n" + "=" * 65)
            print(" Trajectory Evaluation Report vs Ground Truth")
            print("=" * 65)
            print(f"  Frames Evaluated:              {metrics['num_frames']}")
            print(f"  Raw Translation RMSE:          {metrics['raw_translation_rmse_cm']:.2f} cm")
            print(f"  Raw Translation Mean:          {metrics['raw_translation_mean_cm']:.2f} cm")
            print(f"  Raw Translation Median:        {metrics['raw_translation_median_cm']:.2f} cm")
            print(f"  SE(3) Aligned ATE RMSE:        {metrics['se3_ate_rmse_cm']:.2f} cm")
            print(f"  SE(3) Aligned ATE Mean:        {metrics['se3_ate_mean_cm']:.2f} cm")
            print(f"  SE(3) Aligned ATE Median:      {metrics['se3_ate_median_cm']:.2f} cm")
            print(f"  Sim(3) Aligned ATE RMSE:       {metrics['sim3_ate_rmse_cm']:.2f} cm (scale={metrics['sim3_ate_scale']:.4f})")
            print(f"  Mean Geodesic Rotation Error:  {metrics['mean_rotation_error_deg']:.2f}°")
            print(f"  Median Geodesic Rotation Error:{metrics['median_rotation_error_deg']:.2f}°")
            print(f"  Rotation RMSE:                 {metrics['rotation_rmse_deg']:.2f}°")
            if 'symmetry_mean_rotation_error_deg' in metrics:
                print(f"  Symmetry-min Mean Rot Err:     {metrics['symmetry_mean_rotation_error_deg']:.2f}°")
                print(f"  Symmetry-min Med Rot Err:      {metrics['symmetry_median_rotation_error_deg']:.2f}°")
            print("=" * 65 + "\n")

            if args.out_json:
                with open(args.out_json, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved report to {args.out_json}")


if __name__ == "__main__":
    main()
