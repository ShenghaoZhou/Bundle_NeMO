import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class NeMOCorrespondenceEngine:
    """
    Direct 2D-3D Canonical Correspondence & PnP Engine.
    
    Replaces LoFTR (detector-free 2D-2D transformer matching):
    Directly extracts dense (u, v) <-> X_canon correspondences from NeMO decoder outputs
    and computes robust initial 6-DoF pose hypotheses via SQPnP-RANSAC.
    """
    def __init__(
        self,
        conf_threshold: float = 1.0,
        min_inliers_count: int = 30,
        reprojection_error: float = 4.0,
        confidence_pnp: float = 0.9999,
        iterations_pnp: int = 500
    ):
        self.conf_threshold = conf_threshold
        self.min_inliers_count = min_inliers_count
        self.reprojection_error = reprojection_error
        self.confidence_pnp = confidence_pnp
        self.iterations_pnp = iterations_pnp

    def extract_correspondences(
        self,
        pts3d_canon: np.ndarray,      # (224, 224, 3) in canonical metric space
        conf: np.ndarray,             # (224, 224)
        crop_box: Tuple[int, int, int, int],  # (x1, y1, x2, y2) in original image
        crop_res: int = 224
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map 224x224 crop predictions back to full-resolution camera pixel coordinates (u, v).
        
        Returns:
            pts2d_full: (N, 2) 2D pixel coordinates in camera frame.
            pts3d_valid: (N, 3) Corresponding 3D canonical points.
            conf_valid: (N,) Confidence scores.
        """
        x1, y1, x2, y2 = crop_box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        v_indices, u_indices = np.where(conf > self.conf_threshold)
        if len(v_indices) == 0:
            return np.empty((0, 2)), np.empty((0, 3)), np.empty((0,))

        pts3d_valid = pts3d_canon[v_indices, u_indices].astype(np.float32)
        conf_valid = conf[v_indices, u_indices].astype(np.float32)

        # Scale crop coordinates (0..223) back to original camera image space
        u_full = x1 + (u_indices / float(crop_res)) * box_w
        v_full = y1 + (v_indices / float(crop_res)) * box_h
        pts2d_full = np.stack([u_full, v_full], axis=-1).astype(np.float32)

        return pts2d_full, pts3d_valid, conf_valid

    def solve_pnp(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        K: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Solve 6-DoF camera pose using SQPnP-RANSAC.
        
        Returns:
            success (bool): Whether PnP succeeded with sufficient inliers.
            T_cam_obj (4x4 or None): Homogeneous transformation matrix.
            inliers (Nx1 array of indices or None).
            inlier_ratio (float): Ratio of inliers to valid candidate points.
        """
        if len(pts3d) < self.min_inliers_count:
            return False, None, None, 0.0

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                K,
                None,
                flags=cv2.SOLVEPNP_SQPNP,
                reprojectionError=self.reprojection_error,
                confidence=self.confidence_pnp,
                iterationsCount=self.iterations_pnp
            )
        except Exception as e:
            return False, None, None, 0.0

        if not success or inliers is None:
            return False, None, None, 0.0

        inlier_ratio = float(len(inliers) / len(pts2d))
        if len(inliers) < self.min_inliers_count:
            return False, None, None, inlier_ratio

        T_cam_obj = np.eye(4, dtype=np.float64)
        R_cam_obj, _ = cv2.Rodrigues(rvec)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = tvec.flatten()

        return True, T_cam_obj, inliers.flatten(), inlier_ratio
