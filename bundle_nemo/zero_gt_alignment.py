import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
from einops import rearrange
from nemolib.utils import image_to_tensor


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form least-squares rigid alignment (Kabsch/Umeyama) without scale.
    Finds R, t minimizing sum ||dst_i - (R @ src_i + t)||^2.
    
    src: (N, 3), dst: (N, 3)
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert len(src) >= 3 and len(src) == len(dst)
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)

    src_c = src - c_src
    dst_c = dst - c_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_dst - R @ c_src
    return R, t


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    dist_thresh: float = 0.015,
    max_iters: int = 300,
    min_inliers_count: int = 15
) -> Tuple[bool, np.ndarray, np.ndarray, int]:
    """
    RANSAC wrapper over Umeyama rigid alignment to eliminate outlier 3D-3D correspondences.
    
    Returns:
        success: bool
        R: (3, 3)
        t: (3,)
        inliers_count: int
    """
    N = len(src)
    if N < min_inliers_count:
        return False, np.eye(3), np.zeros(3), 0

    best_inliers_mask = None
    best_count = 0

    for _ in range(max_iters):
        # Sample 4 points
        idx = np.random.choice(N, size=4, replace=False)
        try:
            R_cand, t_cand = umeyama_alignment(src[idx], dst[idx])
        except Exception:
            continue

        pred = (R_cand @ src.T).T + t_cand
        residuals = np.linalg.norm(pred - dst, axis=-1)
        inliers_mask = (residuals < dist_thresh)
        count = int(np.sum(inliers_mask))

        if count > best_count:
            best_count = count
            best_inliers_mask = inliers_mask

        # Early exit if majority inliers found
        if best_count > 0.8 * N:
            break

    if best_count < min_inliers_count:
        return False, np.eye(3), np.zeros(3), best_count

    # Refit with all inliers
    src_inl = src[best_inliers_mask]
    dst_inl = dst[best_inliers_mask]
    R_final, t_final = umeyama_alignment(src_inl, dst_inl)

    return True, R_final, t_final, best_count


class ZeroGTAlignmentEngine:
    """
    Autonomous Zero-GT Multi-Aspect Alignment Engine (Method A + B).
    
    Method A: Cross-Cluster 3D Tie-Point Alignment:
      Decodes the transition keyframe simultaneously against the existing canonical model
      and the newly created surface cluster, discovering mutual 3D tie points on the object surface,
      then solves for the inter-cluster rotation R_{k->0} via RANSAC-Umeyama.
      
    Method B: Surface ICP Refinement & Verification:
      Verifies and refines the estimated rotation by registering the new cluster's surface points
      against the persistent canonical voxel model using Point-to-Point ICP.
    """
    def __init__(
        self,
        model,
        device: torch.device,
        conf_threshold: float = 0.85,
        ransac_dist_thresh: float = 0.015,
        max_icp_dist: float = 0.025
    ):
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.ransac_dist_thresh = ransac_dist_thresh
        self.max_icp_dist = max_icp_dist

    def find_cross_cluster_transform(
        self,
        keyframe_crop: Image.Image,
        memory_bank,
        new_features_3d: torch.Tensor,
        new_cluster_surf: np.ndarray,
        reference_cluster_idx: int,
        scale: float = 1.0
    ) -> Tuple[bool, np.ndarray, int, str]:
        """
        Estimate R_{k -> ref} mapping the new cluster's centered coordinates
        to the reference cluster's canonical coordinates without ground truth.
        
        Returns:
            success: bool
            R_k_to_ref: (3, 3) rotation matrix
            inl_count: int count of RANSAC inliers
            info: str explanation
        """
        # 1. Decode keyframe crop against reference cluster
        ref_cluster = memory_bank.clusters[reference_cluster_idx]
        ref_feat = ref_cluster['features_3d_updated']  # [1, 1500, C]
        stacked_pair = torch.cat([ref_feat, new_features_3d], dim=0)  # [2, 1500, C]

        t_single = image_to_tensor(
            keyframe_crop, size=224, device=self.device, normalize=False
        ).unsqueeze(1)

        with torch.no_grad():
            imgs = self.model.img_transformations(rearrange(t_single, "b t c h w -> (b t) c h w"))
            _, _, h, w = imgs.shape
            feat2d = self.model.extract_feature(imgs)
            feat2d = self.model.backbone_out(feat2d)
            feat2d_pair = feat2d.expand(2, -1, -1, -1)
            feat2d_pair = rearrange(feat2d_pair, "b c h w -> b 1 c h w")
            dec_out = self.model.dense_xyz_mapping.forward(
                feat2d_pair, stacked_pair, img_shape=(h, w)
            )

        # 0: Reference cluster decode, 1: New cluster decode
        pts_ref = dec_out['pts3d'][0, 0].cpu().numpy()
        conf_ref = dec_out['conf'][0, 0].cpu().numpy()

        pts_k = dec_out['pts3d'][1, 0].cpu().numpy()
        conf_k = dec_out['conf'][1, 0].cpu().numpy()

        # Find mutual confident pixels
        mutual_mask = (conf_ref > self.conf_threshold) & (conf_k > self.conf_threshold)
        mutual_v, mutual_u = np.where(mutual_mask)

        if len(mutual_v) < 20:
            # Lower threshold slightly if needed
            mutual_mask = (conf_ref > 0.65) & (conf_k > 0.65)
            mutual_v, mutual_u = np.where(mutual_mask)

        if len(mutual_v) < 15:
            return False, np.eye(3), 0, f"Insufficient mutual inliers ({len(mutual_v)} < 15)"

        # Points in new cluster local frame (centered)
        c_k = new_cluster_surf.mean(axis=0)
        pts_k_centered = pts_k[mutual_v, mutual_u] - c_k

        # Points in reference cluster local frame (centered)
        ref_cluster = memory_bank.clusters[reference_cluster_idx]
        c_ref = ref_cluster['center']
        pts_ref_centered = pts_ref[mutual_v, mutual_u] - c_ref

        # Solve RANSAC Umeyama: pts_ref_centered = R_k_to_ref @ pts_k_centered
        success, R_k_to_ref, _, inliers = ransac_umeyama(
            src=pts_k_centered,
            dst=pts_ref_centered,
            dist_thresh=self.ransac_dist_thresh,
            min_inliers_count=12
        )

        if not success:
            return False, np.eye(3), 0, f"RANSAC Umeyama rejected (inliers: {inliers})"

        return True, R_k_to_ref, inliers, f"Umeyama solved with {inliers} mutual inliers ({inliers}/{len(mutual_v)})"

    def estimate_and_refine_cluster_rotation(
        self,
        keyframe_crop: Image.Image,
        memory_bank,
        fusion,
        new_features_3d: torch.Tensor,
        new_cluster_surf: np.ndarray,
        fallback_R: np.ndarray,
        scale: float = 1.0
    ) -> Tuple[np.ndarray, str]:
        """
        Complete Method A + B pipeline:
        1. Find 3D tie points against candidate clusters (preceding clusters & cluster 0) (Method A).
        2. Select the candidate transformation with the highest mutual inlier consensus.
        3. Refine resulting canonical surface points against fused voxel cloud via ICP (Method B).
        4. Fall back to tracker pose if overlap is too small.
        """
        best_R = None
        best_inliers = 0
        best_source = "fallback"

        # Tracker coarse pose prior
        R_coarse = memory_bank.R_ref0 @ fallback_R.T

        # Test candidate clusters: immediate previous cluster, and cluster 0
        cand_indices = []
        if len(memory_bank.clusters) > 1:
            cand_indices.append(len(memory_bank.clusters) - 1)
        if 0 not in cand_indices:
            cand_indices.append(0)

        for ref_idx in cand_indices:
            succ, R_k_to_ref, inl_count, info = self.find_cross_cluster_transform(
                keyframe_crop=keyframe_crop,
                memory_bank=memory_bank,
                new_features_3d=new_features_3d,
                new_cluster_surf=new_cluster_surf,
                reference_cluster_idx=ref_idx,
                scale=scale
            )
            if succ and inl_count > best_inliers:
                R_ref_to_0 = memory_bank.clusters[ref_idx]['R_k_to_0']
                R_cand = R_ref_to_0 @ R_k_to_ref

                # Consistency check with tracker motion prior: reject flip solutions (> 75 deg divergence)
                R_diff = R_cand @ R_coarse.T
                tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
                div_deg = float(np.rad2deg(np.arccos(tr)))

                if div_deg < 75.0:
                    best_R = R_cand
                    best_inliers = inl_count
                    best_source = f"Umeyama from Cluster {ref_idx} ({info})"

        # Fallback if cross-cluster decodes had insufficient mutual inliers or were rejected
        if best_R is None:
            best_R = R_coarse
            best_source = "Tracker Coarse Pose Fallback"

        # Method B: Fine-tune using Surface ICP against accumulated Canonical Fusion
        c_k = new_cluster_surf.mean(axis=0)
        c0 = memory_bank.c0
        surf_centered = new_cluster_surf - c_k
        surf_canon = (surf_centered @ best_R.T) + c0
        if memory_bank.R_canon_align is not None:
            surf_canon = surf_canon @ memory_bank.R_canon_align.T
        surf_metric = surf_canon * scale

        Delta_T = fusion.refine_with_icp(surf_metric, max_correspondence_dist=self.max_icp_dist)
        Delta_R = Delta_T[:3, :3]
        if not np.allclose(Delta_R, np.eye(3)):
            best_R = Delta_R @ best_R
            best_source += " + ICP Refined"

        return best_R, best_source
