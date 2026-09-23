import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from einops import rearrange

from nemolib.utils import image_to_tensor


def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute geodesic rotation distance between two SO(3) matrices in degrees."""
    if not (np.isfinite(R1).all() and np.isfinite(R2).all()):
        return 0.0
    R_diff = R1 @ R2.T
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(tr)))


class AlignedDynamicNeMOMemoryBank:
    """
    Maintains an active memory bank of 3D Neural Memory Objects covering multiple surfaces
    with explicit SE(3) orientation alignment to a single canonical reference frame.
    
    Replaces both:
    1. Keyframe pairwise LoFTR matching: Replaced by multi-aspect canonical 2D-3D decoding.
    2. Neural Object Field (SDF): Replaced by instant feed-forward feature lifting.
    """
    def __init__(
        self,
        model,
        device: torch.device,
        min_keyframe_rot_deg: float = 28.0,
        min_inlier_ratio_trigger: float = 0.20
    ):
        self.model = model
        self.device = device
        self.min_keyframe_rot_deg = min_keyframe_rot_deg
        self.min_inlier_ratio_trigger = min_inlier_ratio_trigger

        self.clusters: List[Dict[str, Any]] = []
        self.features_stacked: Optional[torch.Tensor] = None
        self.R_ref0: Optional[np.ndarray] = None
        self.c0: Optional[np.ndarray] = None
        self.R_canon_align: Optional[np.ndarray] = None

    def initialize_reference(self, R_cam_obj_0: np.ndarray):
        """Anchor canonical reference orientation to initial frame object pose."""
        self.R_ref0 = R_cam_obj_0.copy()

    def set_canonical_alignment(self, R_align: np.ndarray):
        """Set alignment matrix from NeMO canonical space to physical model body frame."""
        self.R_canon_align = R_align.copy()

    def should_register_keyframe(
        self,
        current_R_cam_obj: np.ndarray,
        current_inlier_ratio: float,
        frame_idx: int = 0
    ) -> Tuple[bool, str]:
        """
        Evaluate if a new keyframe memory cluster should be admitted into the bank.
        Triggers strictly when:
        1. Viewing angle difference >= min_keyframe_rot_deg from all existing clusters.
        2. Cooldown of at least 20 frames has elapsed.
        3. Current inliers are healthy (>= 35% ratio, not in severe occlusion).
        """
        if len(self.clusters) == 0:
            return True, "Initial cluster"

        # Enforce cooldown of 20 frames
        last_frame = self.clusters[-1].get('frame_idx', 0)
        if frame_idx - last_frame < 20:
            return False, f"Cooldown active ({frame_idx - last_frame} < 20 frames)"

        # Prevent adding keyframes during heavy hand occlusion
        if current_inlier_ratio < 0.35:
            return False, f"Inlier ratio too low ({current_inlier_ratio:.1%} < 35%)"

        # Check geodesic rotation against all existing clusters
        min_angle = float('inf')
        for cluster in self.clusters:
            R_kf = cluster['R_cam_obj']
            ang = geodesic_angle_deg(current_R_cam_obj, R_kf)
            if ang < min_angle:
                min_angle = ang

        if min_angle >= self.min_keyframe_rot_deg:
            return True, f"Rotation delta {min_angle:.1f}° >= {self.min_keyframe_rot_deg}°"

        return False, "Sufficient overlap"

    def add_cluster(
        self,
        name: str,
        keyframe_crops: List[Image.Image],
        R_cam_obj_kf: np.ndarray,
        frame_idx: int = 0,
        sample_points_count: int = 1500,
        R_k_to_0_override: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Feed-forward encode views into a new 3D Neural Memory Object cluster and align to canonical frame.
        Takes <30 ms without backpropagation.
        """
        t0 = time.time()
        tensors = [image_to_tensor(c, device=self.device) for c in keyframe_crops]
        imgs_tensor = torch.cat(tensors, dim=0).unsqueeze(0)
        sample_points = torch.rand(1, sample_points_count, 3, device=self.device) * 2 - 1

        with torch.no_grad():
            n = self.model.encode_images(imgs_tensor, sample_points)
            # Add positional point encoder features
            n['features_3d_updated'] = n['features_3d'] + self.model.point_encoder(n['surface_points'])
            surf = n['surface_points'][0].cpu().numpy()

        if self.R_ref0 is None:
            self.R_ref0 = R_cam_obj_kf.copy()

        # Relative rotation mapping points from cluster k's camera-centric frame to canonical Reference (Cluster 0)
        if R_k_to_0_override is not None:
            R_k_to_0 = R_k_to_0_override.copy()
        else:
            R_k_to_0 = self.R_ref0 @ R_cam_obj_kf.T
        c_k = surf.mean(axis=0)

        if len(self.clusters) == 0:
            self.c0 = c_k.copy()

        cluster_entry = {
            'name': name,
            'frame_idx': frame_idx,
            'R_cam_obj': R_cam_obj_kf.copy(),
            'features_3d_updated': n['features_3d_updated'],
            'surface_points': surf,
            'R_k_to_0': R_k_to_0,
            'center': c_k
        }
        self.clusters.append(cluster_entry)
        self.features_stacked = torch.cat([c['features_3d_updated'] for c in self.clusters], dim=0)

        elapsed = time.time() - t0
        print(f"[NeMO MemoryBank] Added '{name}' (Frame {frame_idx}) in {elapsed:.3f}s. Total clusters: {len(self.clusters)}")
        return cluster_entry

    def decode_query(
        self,
        query_img_crop: Image.Image
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Decode query image crop across all active memory clusters simultaneously.
        Uses cached single-pass DINOv2 feature extraction to avoid redundant backbone passes.
        Selects the best cluster with highest inlier confidence.
        """
        K = len(self.clusters)

        if hasattr(self.model, 'extract_feature') and hasattr(self.model, 'dense_xyz_mapping'):
            t_single = image_to_tensor(
                query_img_crop, size=224, device=self.device, normalize=False
            ).unsqueeze(1)
            with torch.no_grad():
                imgs = self.model.img_transformations(rearrange(t_single, "b t c h w -> (b t) c h w"))
                _, _, h, w = imgs.shape
                feat2d = self.model.extract_feature(imgs)
                feat2d = self.model.backbone_out(feat2d)
                feat2d_K = feat2d.expand(K, -1, -1, -1)
                feat2d_K = rearrange(feat2d_K, "b c h w -> b 1 c h w")
                dec_out = self.model.dense_xyz_mapping.forward(
                    feat2d_K, self.features_stacked, img_shape=(h, w)
                )
        else:
            t_input = image_to_tensor(
                query_img_crop, size=224, device=self.device, normalize=False
            ).repeat(K, 1, 1, 1).unsqueeze(1)
            with torch.no_grad():
                dec_out = self.model.decode_images(t_input, self.features_stacked)

        conf_maxs = dec_out['conf'].amax(dim=[-1, -2]).cpu().numpy().ravel()
        conf_means = dec_out['conf'].mean(dim=[-1, -2]).cpu().numpy().ravel()
        score = conf_maxs + 2.0 * conf_means
        best_cluster_idx = int(np.argmax(score))

        return dec_out, best_cluster_idx

    def get_canonical_3d_points(
        self,
        cluster_idx: int,
        pts3d_local: np.ndarray,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Transform local cluster 3D predictions to unified canonical metric coordinates:
        1. Center by cluster centroid c_k: pts_centered = pts3d_local - c_k
        2. Rotate to canonical frame: pts_canon = (pts_centered @ R_k_to_0.T) + c_0
        3. Align to true body/BOP frame if calibrated: pts_canon = pts_canon @ R_canon_align.T
        4. Scale to physical meters: pts_metric = pts_canon * scale
        """
        c_k = self.clusters[cluster_idx]['center']
        c_0 = self.c0
        R_k_to_0 = self.clusters[cluster_idx]['R_k_to_0']

        pts_centered = pts3d_local - c_k
        pts_canon = np.matmul(pts_centered, R_k_to_0.T) + c_0

        if self.R_canon_align is not None:
            pts_canon = np.matmul(pts_canon, self.R_canon_align.T)

        return pts_canon * scale
