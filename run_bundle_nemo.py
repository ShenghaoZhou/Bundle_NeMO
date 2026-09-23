#!/usr/bin/env python3
"""
BundleNeMO: Lightweight Neural 6-DoF Tracking and 3D Reconstruction Pipeline.
Reuses the keyframe-based tracking and pose optimization concepts from BundleSDF,
but replaces LoFTR and the implicit SDF Neural Object Field with NeMO.
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

# Ensure self-contained NeMO submodule and bundle_nemo are accessible
current_dir = os.path.dirname(os.path.realpath(__file__))
submodule_nemo_src = os.path.join(current_dir, "NeMO", "src")
if os.path.isdir(submodule_nemo_src) and submodule_nemo_src not in sys.path:
    sys.path.insert(0, submodule_nemo_src)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from nemolib.model import Model
from bundle_nemo import BundleNeMOTracker


def decode_binary_mask_rle(data):
    """Decode RLE mask format used in HOT3D objects.json."""
    starts = np.asarray(data['rle'][0:][::2]) - 1
    ends = starts + np.asarray(data['rle'][1:][::2])
    mask = np.zeros(data['height'] * data['width'], dtype=bool)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = True
    return mask.reshape((data['height'], data['width'])).astype(np.uint8)


def load_dataset_frames(data_dir):
    """
    Detect dataset format (HOT3D JSON-based or standard RGB-D directory).
    Returns list of dicts: {'rgb_path', 'depth_path', 'mask_path', 'K', 'gt_T_cam_obj'}
    """
    # Check for HOT3D format
    hot3d_imgs = sorted(glob.glob(os.path.join(data_dir, "*.image_214-1.jpg")))
    if len(hot3d_imgs) > 0:
        frames = []
        for img_path in hot3d_imgs:
            prefix = img_path.split(".image_214-1.jpg")[0]
            obj_path = f"{prefix}.objects.json"
            cam_path = f"{prefix}.cameras.json"
            frames.append({
                'type': 'hot3d',
                'rgb_path': img_path,
                'obj_path': obj_path,
                'cam_path': cam_path
            })
        return frames

    # Check for standard RGB-D format (rgb/, depth/, masks/, cam_K.txt)
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    mask_dir = os.path.join(data_dir, "masks")
    k_path = os.path.join(data_dir, "cam_K.txt")

    if os.path.exists(rgb_dir):
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.*")))
        K = np.loadtxt(k_path).reshape(3, 3) if os.path.exists(k_path) else None
        frames = []
        for r_file in rgb_files:
            fname = os.path.basename(r_file)
            base_no_ext = os.path.splitext(fname)[0]
            d_file = os.path.join(depth_dir, f"{base_no_ext}.png")
            m_file = os.path.join(mask_dir, f"{base_no_ext}.png")
            frames.append({
                'type': 'rgbd',
                'rgb_path': r_file,
                'depth_path': d_file if os.path.exists(d_file) else None,
                'mask_path': m_file if os.path.exists(m_file) else None,
                'K': K
            })
        return frames

    raise ValueError(f"Could not detect dataset format in {data_dir}")


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def read_frame_data(frame_info):
    """Read RGB, Depth, Mask, Intrinsics, and optional World Poses."""
    T_w_cam = None
    T_w_obj_gt = None

    if frame_info['type'] == 'hot3d':
        rgb = cv2.imread(frame_info['rgb_path'])[:, :, ::-1]  # BGR to RGB
        with open(frame_info['cam_path']) as f:
            cam_data = json.load(f)['214-1']
        fx, cx, cy = cam_data['calibration']['projection_params'][:3]
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)

        if 'T_world_from_camera' in cam_data:
            T_w_cam = np.eye(4, dtype=np.float64)
            T_w_cam[:3, :3] = quat_to_rot(cam_data['T_world_from_camera']['quaternion_wxyz'])
            T_w_cam[:3, 3] = cam_data['T_world_from_camera']['translation_xyz']

        # Read mask from objects.json
        with open(frame_info['obj_path']) as f:
            obj_data = json.load(f)
            key = list(obj_data.keys())[0] if '26' not in obj_data else '26'
            toy = obj_data[key][0]
            mask_rle = toy['masks_amodal']['214-1']
            mask = decode_binary_mask_rle(mask_rle)
            mask_modal = None
            if 'masks_modal' in toy and '214-1' in toy['masks_modal']:
                mask_modal = decode_binary_mask_rle(toy['masks_modal']['214-1'])
            if 'T_world_from_object' in toy:
                T_w_obj_gt = np.eye(4, dtype=np.float64)
                T_w_obj_gt[:3, :3] = quat_to_rot(toy['T_world_from_object']['quaternion_wxyz'])
                T_w_obj_gt[:3, 3] = toy['T_world_from_object']['translation_xyz']

        depth = None
        return rgb, depth, mask, K, T_w_cam, T_w_obj_gt, mask_modal

    elif frame_info['type'] == 'rgbd':
        rgb = cv2.imread(frame_info['rgb_path'])[:, :, ::-1]
        depth = cv2.imread(frame_info['depth_path'], cv2.IMREAD_UNCHANGED) if frame_info['depth_path'] else None
        mask = cv2.imread(frame_info['mask_path'], cv2.IMREAD_GRAYSCALE) if frame_info['mask_path'] else None
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
        K = frame_info['K']
        return rgb, depth, mask, K, None, None, None


def main():
    parser = argparse.ArgumentParser(description="BundleNeMO 6-DoF Object Tracking and Reconstruction")
    parser.add_argument("--data_dir", type=str, default="../data/hot3d/extracted/clip-003312",
                        help="Path to dataset sequence directory")
    parser.add_argument("--checkpoint", type=str, default="NeMO/checkpoints/checkpoint.pth",
                        help="Path to NeMO model checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/bundle_nemo_run",
                        help="Path to output directory")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--metric_scale", type=float, default=None,
                        help="Fixed metric scale factor (leave None for auto-estimation)")
    parser.add_argument("--save_rrd", type=str, default=None,
                        help="Path to save Rerun .rrd recording")
    parser.add_argument("--decode_stride", type=int, default=1,
                        help="NeMO decode stride (1=every frame, 2=skip-frame kinematic prediction)")
    parser.add_argument("--no_icp", action="store_true",
                        help="Disable ICP refinement for newly added clusters")
    parser.add_argument("--alignment_mode", type=str, default="cross_icp",
                        choices=["cross_icp", "pose_graph", "gt"],
                        help="Cluster alignment strategy: cross_icp (Method A+B, Zero-GT), pose_graph (Method C, Zero-GT), or gt (Oracle)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    poses_dir = os.path.join(args.out_dir, "ob_in_cam")
    os.makedirs(poses_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BundleNeMO] Device: {device}")
    print(f"[BundleNeMO] Alignment Mode: {args.alignment_mode.upper()} {'(ZERO-GT / FULLY AUTONOMOUS)' if args.alignment_mode != 'gt' else '(ORACLE REFERENCE)'}")

    # 1. Load NeMO Model
    ckpt_path = os.path.abspath(args.checkpoint)
    print(f"[BundleNeMO] Loading NeMO checkpoint from {ckpt_path}...")
    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval()

    # 2. Discover sequence frames
    frames = load_dataset_frames(os.path.abspath(args.data_dir))
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    total_frames = len(frames)
    print(f"[BundleNeMO] Found {total_frames} frames in sequence.")

    # 3. Initialize BundleNeMO Tracker
    tracker = BundleNeMOTracker(
        nemo_model=model,
        device=device,
        metric_scale=args.metric_scale,
        decode_stride=args.decode_stride,
        use_icp_refinement=(not args.no_icp),
        alignment_mode=args.alignment_mode
    )

    if args.save_rrd:
        import rerun as rr
        os.makedirs(os.path.dirname(os.path.abspath(args.save_rrd)), exist_ok=True)
        rr.init("BundleNeMO_Full_Clip_Tracking", spawn=False)
        rr.save(args.save_rrd)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        print(f"[BundleNeMO] Streaming Rerun recording to {args.save_rrd}")

    trajectory = []
    fps_list = []
    traj_w_est = []
    traj_w_gt = []
    traj_w_cam = []

    print("-" * 75)
    print(f"{'Frame':<8}{'PnP Status':<12}{'Inliers':<10}{'Ratio':<10}{'FPS':<10}{'Keyframe':<10}")
    print("-" * 75)

    # 4. Sequential Tracking Loop
    for idx, f_info in enumerate(frames):
        rgb, depth, mask, K, T_w_cam, T_w_obj_gt, mask_modal = read_frame_data(f_info)

        R_c_o_gt = None
        if T_w_cam is not None and T_w_obj_gt is not None:
            T_c_o_gt = np.linalg.inv(T_w_cam) @ T_w_obj_gt
            R_c_o_gt = T_c_o_gt[:3, :3]

        if idx == 0:
            res = tracker.process_first_frame(rgb, depth, mask, K, R_cam_obj_init=R_c_o_gt, foreground_mask=mask_modal)
        else:
            res = tracker.process_frame(rgb, depth, mask, K, R_cam_obj_gt=R_c_o_gt, foreground_mask=mask_modal)

        T_cam_obj = res['T_cam_obj']
        trajectory.append(T_cam_obj)
        if 'fps' in res:
            fps_list.append(res['fps'])

        # Save pose in BundleSDF format (4x4 matrix txt)
        np.savetxt(os.path.join(poses_dir, f"{idx:06d}.txt"), T_cam_obj, fmt="%.8f")

        if args.save_rrd:
            rr.set_time("frame", sequence=idx)
            rr.set_time("sim_time", duration=idx / 30.0)

            if T_w_cam is not None:
                T_w_obj_est = T_w_cam @ T_cam_obj
                traj_w_est.append(T_w_obj_est[:3, 3].copy())
                traj_w_cam.append(T_w_cam[:3, 3].copy())
                if T_w_obj_gt is not None:
                    traj_w_gt.append(T_w_obj_gt[:3, 3].copy())
                    rr.log("world/object_gt", rr.Transform3D(translation=T_w_obj_gt[:3, 3], mat3x3=T_w_obj_gt[:3, :3]))

                rr.log("world/camera", rr.Transform3D(translation=T_w_cam[:3, 3], mat3x3=T_w_cam[:3, :3]))
                rr.log("world/object_estimated", rr.Transform3D(translation=T_w_obj_est[:3, 3], mat3x3=T_w_obj_est[:3, :3]))

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

        status_str = "VALID" if res['pnp_valid'] else "FAILED"
        kf_str = "YES" if res.get('keyframe_added', False) else "NO"
        fps_val = res.get('fps', 0.0)
        print(f"{idx:<8}{status_str:<12}{res['inliers_count']:<10}{res['inlier_ratio']:<10.2%}{fps_val:<10.1f}{kf_str:<10}")

    print("-" * 75)
    mean_fps = np.mean(fps_list) if len(fps_list) > 0 else 0.0
    print(f"[BundleNeMO] Tracking finished: {total_frames} frames processed at {mean_fps:.1f} mean FPS.")

    # 5. Extract and Export Clean 3D Reconstructed Model
    print("[BundleNeMO] Extracting fused 3D canonical point cloud and Poisson mesh...")
    pcd = tracker.fusion.get_fused_point_cloud(filter_outliers=True)
    pcd_path = os.path.join(args.out_dir, "fused_model.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"  -> Exported point cloud ({len(pcd.points)} points) to {pcd_path}")

    mesh = tracker.fusion.extract_poisson_mesh(depth=8)
    if mesh is not None:
        mesh_path = os.path.join(args.out_dir, "textured_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"  -> Exported Poisson mesh ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces) to {mesh_path}")

    if args.save_rrd:
        if len(traj_w_est) > 1:
            rr.log("world/trajectories/estimated", rr.LineStrips3D([np.array(traj_w_est)], colors=[[40, 220, 100]], radii=0.003), static=True)
        if len(traj_w_gt) > 1:
            rr.log("world/trajectories/ground_truth", rr.LineStrips3D([np.array(traj_w_gt)], colors=[[240, 50, 50]], radii=0.003), static=True)
        if len(traj_w_cam) > 1:
            rr.log("world/trajectories/camera", rr.LineStrips3D([np.array(traj_w_cam)], colors=[[60, 140, 255]], radii=0.002), static=True)

        pts_clean = np.asarray(pcd.points)
        colors_clean = np.asarray(pcd.colors)
        if len(pts_clean) > 0:
            rr.log("object/fused_3d_points", rr.Points3D(positions=pts_clean, colors=colors_clean, radii=0.002), static=True)
        print(f"[BundleNeMO] Rerun recording complete: {args.save_rrd}")

    # Save summary metrics
    summary = {
        'total_frames': total_frames,
        'mean_fps': float(mean_fps),
        'metric_scale': float(tracker.metric_scale),
        'total_clusters': len(tracker.memory_bank.clusters),
        'fused_points_count': len(pcd.points),
        'mesh_vertices': len(mesh.vertices) if mesh is not None else 0
    }
    with open(os.path.join(args.out_dir, "tracking_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[BundleNeMO] Summary saved to {os.path.join(args.out_dir, 'tracking_summary.json')}")


if __name__ == "__main__":
    main()
