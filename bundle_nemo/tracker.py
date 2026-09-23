import os
import time
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List

from .scale_estimator import AutomaticScaleEstimator
from .memory_bank import AlignedDynamicNeMOMemoryBank
from .correspondence import NeMOCorrespondenceEngine
from .optimizer import BundleNeMOOptimizer
from .fusion import CanonicalObjectFusion
from .zero_gt_alignment import ZeroGTAlignmentEngine
from .pose_graph import KeyframePoseGraph
from nemolib.utils import image_to_tensor


def crop_masked_object(
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    foreground_mask: Optional[np.ndarray] = None,
    padding_ratio: float = 0.15,
    crop_size: int = 224
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extract tightly bounded foreground crop on white canvas for NeMO input.
    Uses binary_mask (amodal) for bounding box and foreground_mask (modal) for canvas isolation.
    
    Returns:
        crop_pil: (crop_size, crop_size) RGB PIL image.
        crop_box: (x1, y1, x2, y2) in original image coordinates.
    """
    H, W = binary_mask.shape[:2]
    ys, xs = np.where(binary_mask > 0)

    if len(ys) == 0 or len(xs) == 0:
        pil_img = Image.fromarray(rgb_img)
        return pil_img.resize((crop_size, crop_size)), (0, 0, W, H)

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    bw = x2 - x1
    bh = y2 - y1
    sz = max(bw, bh)
    pad = int(sz * padding_ratio)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_sz = sz // 2 + pad

    nx1 = max(0, cx - half_sz)
    ny1 = max(0, cy - half_sz)
    nx2 = min(W, cx + half_sz)
    ny2 = min(H, cy + half_sz)

    crop_rgb = rgb_img[ny1:ny2, nx1:nx2]
    crop_mask = binary_mask[ny1:ny2, nx1:nx2]

    # White canvas background
    canvas = np.ones((crop_rgb.shape[0], crop_rgb.shape[1], 3), dtype=np.uint8) * 255
    if foreground_mask is not None:
        crop_fg = foreground_mask[ny1:ny2, nx1:nx2]
        fg_mask = (crop_fg > 0)[..., None]
    else:
        fg_mask = (crop_mask > 0)[..., None]
    canvas = np.where(fg_mask, crop_rgb, canvas)

    crop_pil = Image.fromarray(canvas).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    return crop_pil, (nx1, ny1, nx2, ny2)


class BundleNeMOTracker:
    """
    BundleNeMO: Lightweight 6-DoF Tracking and 3D Reconstruction Pipeline.
    
    Reuses BundleSDF's end-to-end multi-view keyframe management and pose graph tracking,
    but replaces both LoFTR and the implicit SDF Neural Object Field with NeMO.
    """
    def __init__(
        self,
        nemo_model,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        metric_scale: Optional[float] = None,
        min_keyframe_rot_deg: float = 28.0,
        voxel_size: float = 0.003,
        decode_stride: int = 1,
        use_icp_refinement: bool = True,
        alignment_mode: str = "cross_icp"  # "cross_icp" (A+B), "pose_graph" (C), or "gt"
    ):
        self.device = device
        self.model = nemo_model
        self.metric_scale = metric_scale
        self.scale_is_fixed = (metric_scale is not None)
        self.decode_stride = decode_stride
        self.use_icp_refinement = use_icp_refinement
        self.alignment_mode = alignment_mode

        # Core Components
        self.memory_bank = AlignedDynamicNeMOMemoryBank(
            model=self.model,
            device=self.device,
            min_keyframe_rot_deg=min_keyframe_rot_deg
        )
        self.corres_engine = NeMOCorrespondenceEngine()
        self.optimizer = BundleNeMOOptimizer(device=self.device)
        self.fusion = CanonicalObjectFusion(voxel_size=voxel_size)

        # Zero-GT Alignment Engines (Methods A, B, and C)
        self.zero_gt_aligner = ZeroGTAlignmentEngine(model=self.model, device=self.device)
        self.pose_graph = KeyframePoseGraph(device=self.device)

        self.frame_count = 0
        self.keyframe_crops_buffer: List[Image.Image] = []
        self.last_T_cam_obj: Optional[np.ndarray] = None

    def process_first_frame(
        self,
        rgb_img: np.ndarray,
        depth_map: Optional[np.ndarray],
        binary_mask: np.ndarray,
        K: np.ndarray,
        R_cam_obj_init: Optional[np.ndarray] = None,
        foreground_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Initialize the tracker with the initial frame observation.
        """
        crop_pil, crop_box = crop_masked_object(rgb_img, binary_mask, foreground_mask=foreground_mask)
        self.keyframe_crops_buffer = [crop_pil]

        if R_cam_obj_init is None:
            R_cam_obj_init = np.eye(3, dtype=np.float64)

        # 1. Initialize reference orientation
        self.memory_bank.initialize_reference(R_cam_obj_init)

        # 2. Add Cluster 0
        self.memory_bank.add_cluster(
            name="Cluster 0 (Initial Aspect)",
            keyframe_crops=[crop_pil],
            R_cam_obj_kf=R_cam_obj_init
        )

        # 3. Decode frame 0 to extract canonical coordinates
        dec_out, best_k = self.memory_bank.decode_query(crop_pil)
        pts3d_local = dec_out['pts3d'][best_k, 0].cpu().numpy()
        conf = dec_out['conf'][best_k, 0].cpu().numpy()

        pts2d_full, pts3d_cand, conf_valid = self.corres_engine.extract_correspondences(
            pts3d_local, conf, crop_box
        )

        # 4. Automatic Scale Calibration from depth
        if not self.scale_is_fixed:
            if depth_map is not None:
                self.metric_scale = AutomaticScaleEstimator.estimate_scale_from_depth(
                    pts3d_cand, pts2d_full, depth_map, K, conf_weights=conf_valid
                )
                print(f"[BundleNeMO] Automatically estimated metric scale: {self.metric_scale:.5f}")
            else:
                self.metric_scale = 0.35577
                print(f"[BundleNeMO] Depth not provided; using metric scale {self.metric_scale}")

        # 5. Solve Initial Pose and Calibrate Canonical Body Alignment
        pts3d_scaled = pts3d_cand * self.metric_scale
        success, T_pnp, inliers, inlier_ratio = self.corres_engine.solve_pnp(
            pts3d_scaled, pts2d_full, K
        )

        if success and inliers is not None:
            R_cam_nemo0 = T_pnp[:3, :3]
            # Align NeMO canonical frame to object reference body frame
            R_canon_align = R_cam_obj_init.T @ R_cam_nemo0
            self.memory_bank.set_canonical_alignment(R_canon_align)
            # Recompute points in aligned canonical frame
            pts3d_canon = self.memory_bank.get_canonical_3d_points(0, pts3d_local, scale=self.metric_scale)
            pts2d_full, pts3d_cand, conf_valid = self.corres_engine.extract_correspondences(
                pts3d_canon, conf, crop_box
            )
            success, T_pnp, inliers, inlier_ratio = self.corres_engine.solve_pnp(
                pts3d_cand, pts2d_full, K
            )

        opt_res = self.optimizer.optimize_step(
            K=K,
            pts3d_canon=pts3d_cand,
            pts2d_pixels=pts2d_full,
            conf_weights=conf_valid,
            T_cam_obj_pnp=T_pnp,
            pnp_valid=success,
            inliers=inliers
        )
        T_cam_obj = opt_res['T_cam_obj']
        self.last_T_cam_obj = T_cam_obj.copy()

        # Register Anchor Node 0 in Keyframe Pose Graph (filter to inliers)
        sub_inl = inliers if (inliers is not None and len(inliers) >= 30) else np.arange(len(pts3d_cand))
        self.pose_graph.add_keyframe(
            frame_idx=0,
            T_cam_obj_init=T_cam_obj,
            pts3d_canon=pts3d_cand[sub_inl],
            pts2d_pixels=pts2d_full[sub_inl],
            conf_weights=conf_valid[sub_inl],
            K=K
        )

        # 6. Fuse initial 3D points (subsampled with stride 4 to prevent clutter)
        if success and inliers is not None and len(inliers) > 0:
            stride = 4
            sub_inl = inliers[::stride]
            inl_pts3d = pts3d_cand[sub_inl]
            u_inl = np.clip(np.round(pts2d_full[sub_inl, 0]).astype(int), 0, rgb_img.shape[1] - 1)
            v_inl = np.clip(np.round(pts2d_full[sub_inl, 1]).astype(int), 0, rgb_img.shape[0] - 1)
            colors = rgb_img[v_inl, u_inl].astype(np.float64) / 255.0
            weights = conf_valid[sub_inl]
            self.fusion.integrate_points(inl_pts3d, colors, weights)

        self.frame_count = 1
        return {
            'T_cam_obj': T_cam_obj,
            'pnp_valid': success,
            'inliers_count': len(inliers) if inliers is not None else 0,
            'inlier_ratio': inlier_ratio,
            'metric_scale': self.metric_scale,
            'keyframe_added': True
        }

    def process_frame(
        self,
        rgb_img: np.ndarray,
        depth_map: Optional[np.ndarray],
        binary_mask: np.ndarray,
        K: np.ndarray,
        R_cam_obj_gt: Optional[np.ndarray] = None,
        foreground_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process incoming video frame:
        1. Crop foreground with optional modal mask.
        2. Decode across memory bank.
        3. Solve SQPnP initial pose.
        4. Refine pose with inlier-filtered sliding-window optimizer.
        5. Check keyframe condition & add new memory cluster if needed.
        6. Fuse observed points into canonical 3D model.
        """
        t0 = time.time()

        # Skip-frame handling for real-time throughput:
        if self.decode_stride > 1 and (self.frame_count % self.decode_stride != 0) and self.last_T_cam_obj is not None:
            pred_res = self.optimizer.filter.step(None, valid=False)
            T_cam_obj = pred_res['T_cam_obj']
            self.last_T_cam_obj = T_cam_obj.copy()
            self.frame_count += 1
            elapsed = time.time() - t0
            return {
                'T_cam_obj': T_cam_obj,
                'pnp_valid': True,
                'inliers_count': 0,
                'inlier_ratio': 0.0,
                'best_cluster': 0,
                'keyframe_added': False,
                'is_spike': False,
                'elapsed_sec': elapsed,
                'fps': 1.0 / max(1e-4, elapsed)
            }

        crop_pil, crop_box = crop_masked_object(rgb_img, binary_mask, foreground_mask=foreground_mask)

        # 1. NeMO Decode across memory bank
        dec_out, best_cluster_idx = self.memory_bank.decode_query(crop_pil)
        pts3d_local = dec_out['pts3d'][best_cluster_idx, 0].cpu().numpy()
        conf = dec_out['conf'][best_cluster_idx, 0].cpu().numpy()

        # 2. Transform to canonical metric space
        pts3d_canon = self.memory_bank.get_canonical_3d_points(
            cluster_idx=best_cluster_idx,
            pts3d_local=pts3d_local,
            scale=self.metric_scale
        )

        pts2d_full, pts3d_cand, conf_valid = self.corres_engine.extract_correspondences(
            pts3d_canon, conf, crop_box
        )

        # 3. Solve Coarse PnP
        success, T_pnp, inliers, inlier_ratio = self.corres_engine.solve_pnp(
            pts3d_cand, pts2d_full, K
        )

        # 4. Refine with Optimizer (strictly on verified geometric inliers)
        opt_res = self.optimizer.optimize_step(
            K=K,
            pts3d_canon=pts3d_cand,
            pts2d_pixels=pts2d_full,
            conf_weights=conf_valid,
            T_cam_obj_pnp=T_pnp,
            pnp_valid=success,
            inliers=inliers
        )
        T_cam_obj = opt_res['T_cam_obj']
        self.last_T_cam_obj = T_cam_obj.copy()

        # 5. Check Keyframe Admission (strict rotation delta, cooldown, healthy inliers)
        keyframe_added = False
        R_for_kf_check = T_cam_obj[:3, :3] if self.alignment_mode != "gt" or R_cam_obj_gt is None else R_cam_obj_gt
        should_add, reason = self.memory_bank.should_register_keyframe(
            current_R_cam_obj=R_for_kf_check,
            current_inlier_ratio=inlier_ratio,
            frame_idx=self.frame_count
        )
        if should_add and success and len(inliers) >= 1500:
            cluster_name = f"Cluster {len(self.memory_bank.clusters)} (Frame {self.frame_count})"

            if self.alignment_mode == "gt" and R_cam_obj_gt is not None:
                # Oracle mode: GT camera-object rotation
                R_kf = R_cam_obj_gt
                R_k_to_0 = self.memory_bank.R_ref0 @ R_kf.T
            elif self.alignment_mode == "pose_graph":
                # Method C: BundleSDF-Style Keyframe Pose Graph Optimization & Multi-View BA
                sub_inl = inliers if (inliers is not None and len(inliers) >= 30) else np.arange(len(pts3d_cand))
                node_id = self.pose_graph.add_keyframe(
                    frame_idx=self.frame_count,
                    T_cam_obj_init=T_cam_obj,
                    pts3d_canon=pts3d_cand[sub_inl],
                    pts2d_pixels=pts2d_full[sub_inl],
                    conf_weights=conf_valid[sub_inl],
                    K=K
                )
                opt_poses = self.pose_graph.optimize()
                R_kf = opt_poses[node_id][:3, :3]
                R_k_to_0 = self.memory_bank.R_ref0 @ R_kf.T
                print(f"[BundleNeMO PoseGraph BA (Method C)] Keyframe {node_id} optimized across {len(self.pose_graph.keyframes)} nodes")
            else:
                # Method A + B: Cross-Cluster 3D Tie Points + Surface ICP Refinement
                tensors = [image_to_tensor(crop_pil, device=self.device)]
                imgs_tensor = torch.cat(tensors, dim=0).unsqueeze(0)
                sample_points = torch.rand(1, 1500, 3, device=self.device) * 2 - 1
                with torch.no_grad():
                    n = self.model.encode_images(imgs_tensor, sample_points)
                    cand_feat3d = n['features_3d'] + self.model.point_encoder(n['surface_points'])
                    cand_surf = n['surface_points'][0].cpu().numpy()

                R_k_to_0, align_info = self.zero_gt_aligner.estimate_and_refine_cluster_rotation(
                    keyframe_crop=crop_pil,
                    memory_bank=self.memory_bank,
                    fusion=self.fusion,
                    new_features_3d=cand_feat3d,
                    new_cluster_surf=cand_surf,
                    fallback_R=T_cam_obj[:3, :3],
                    scale=self.metric_scale
                )
                R_kf = self.memory_bank.R_ref0.T @ R_k_to_0
                print(f"[BundleNeMO Zero-GT A+B] {cluster_name} aligned: {align_info}")

            self.memory_bank.add_cluster(
                name=cluster_name,
                keyframe_crops=[crop_pil],
                R_cam_obj_kf=R_kf,
                frame_idx=self.frame_count,
                R_k_to_0_override=R_k_to_0
            )
            keyframe_added = True

            # Additional ICP refinement if requested and not already done in cross_icp
            if self.use_icp_refinement and self.alignment_mode != "cross_icp":
                cl_idx = len(self.memory_bank.clusters) - 1
                surf = self.memory_bank.clusters[cl_idx]['surface_points']
                pts_surf_canon = self.memory_bank.get_canonical_3d_points(cl_idx, surf, scale=self.metric_scale)
                Delta_T = self.fusion.refine_with_icp(pts_surf_canon)
                Delta_R = Delta_T[:3, :3]
                if not np.allclose(Delta_R, np.eye(3)):
                    self.memory_bank.clusters[cl_idx]['R_k_to_0'] = Delta_R @ self.memory_bank.clusters[cl_idx]['R_k_to_0']
                    print(f"[BundleNeMO] Applied ICP alignment refinement to {cluster_name}")

        # 6. Integrate 3D surface points into Canonical Fusion (subsampled)
        if success and inliers is not None and len(inliers) > 0:
            stride = 4
            sub_inl = inliers[::stride]
            inl_pts3d = pts3d_cand[sub_inl]
            u_inl = np.clip(np.round(pts2d_full[sub_inl, 0]).astype(int), 0, rgb_img.shape[1] - 1)
            v_inl = np.clip(np.round(pts2d_full[sub_inl, 1]).astype(int), 0, rgb_img.shape[0] - 1)
            colors = rgb_img[v_inl, u_inl].astype(np.float64) / 255.0
            weights = conf_valid[sub_inl]
            self.fusion.integrate_points(inl_pts3d, colors, weights)

        self.frame_count += 1
        elapsed = time.time() - t0

        return {
            'T_cam_obj': T_cam_obj,
            'pnp_valid': success,
            'inliers_count': len(inliers) if inliers is not None else 0,
            'inlier_ratio': inlier_ratio,
            'best_cluster': best_cluster_idx,
            'keyframe_added': keyframe_added,
            'is_spike': opt_res['is_spike'],
            'elapsed_sec': elapsed,
            'fps': 1.0 / max(1e-4, elapsed)
        }
