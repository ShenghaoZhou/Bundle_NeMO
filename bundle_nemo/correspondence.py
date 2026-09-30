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
        iterations_pnp: int = 800,
        max_pnp_points: int = 2048
    ):
        self.conf_threshold = conf_threshold
        self.min_inliers_count = min_inliers_count
        self.reprojection_error = reprojection_error
        self.confidence_pnp = confidence_pnp
        self.iterations_pnp = iterations_pnp
        self.max_pnp_points = max_pnp_points

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
        K: np.ndarray,
        prior_T: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Solve 6-DoF camera pose using SQPnP-RANSAC with temporal continuity verification.
        
        Returns:
            success (bool): Whether PnP succeeded with sufficient inliers.
            T_cam_obj (4x4 or None): Homogeneous transformation matrix.
            inliers (Nx1 array of indices or None).
            inlier_ratio (float): Ratio of inliers to valid candidate points.
        """
        if len(pts3d) < self.min_inliers_count:
            return False, None, None, 0.0

        # Subsample points if exceeding max_pnp_points to ensure fast and consistent RANSAC
        if len(pts3d) > self.max_pnp_points:
            step = len(pts3d) // self.max_pnp_points
            sub_indices = np.arange(0, len(pts3d), step)[:self.max_pnp_points]
            sub_3d = pts3d[sub_indices]
            sub_2d = pts2d[sub_indices]
        else:
            sub_indices = np.arange(len(pts3d))
            sub_3d = pts3d
            sub_2d = pts2d

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                sub_3d,
                sub_2d,
                K,
                None,
                flags=cv2.SOLVEPNP_SQPNP,
                reprojectionError=self.reprojection_error,
                confidence=self.confidence_pnp,
                iterationsCount=self.iterations_pnp
            )
        except Exception:
            return False, None, None, 0.0

        if not success or inliers is None or len(inliers) < self.min_inliers_count:
            return False, None, None, 0.0

        inliers_sub = inliers.flatten()

        # Motion continuity check against temporal prior pose if available
        if prior_T is not None:
            R_prior = prior_T[:3, :3].astype(np.float64)
            t_prior = prior_T[:3, 3].astype(np.float64)
            rvec_prior, _ = cv2.Rodrigues(R_prior)

            # Test reprojection error of prior pose
            try:
                proj_prior, _ = cv2.projectPoints(sub_3d, rvec_prior, t_prior, K, None)
                err_prior = np.linalg.norm(proj_prior.reshape(-1, 2) - sub_2d, axis=-1)
                inliers_prior = np.where(err_prior < self.reprojection_error * 1.5)[0]

                if len(inliers_prior) >= self.min_inliers_count:
                    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                        sub_3d[inliers_prior], sub_2d[inliers_prior], K, None,
                        rvec_prior.copy(), t_prior.copy().reshape(3, 1)
                    )
                    proj_ref, _ = cv2.projectPoints(sub_3d, rvec_ref, tvec_ref, K, None)
                    err_ref = np.linalg.norm(proj_ref.reshape(-1, 2) - sub_2d, axis=-1)
                    inliers_ref = np.where(err_ref < self.reprojection_error)[0]
                    num_ref = len(inliers_ref)

                    # Compute jump distance between PnP hypothesis and prior pose
                    dt_pnp = np.linalg.norm(tvec.flatten() - t_prior)
                    R_pnp, _ = cv2.Rodrigues(rvec)
                    tr_diff = np.clip((np.trace(R_pnp @ R_prior.T) - 1.0) / 2.0, -1.0, 1.0)
                    dR_pnp_deg = np.rad2deg(np.arccos(tr_diff))

                    # If PnP made a large jump (> 5.5 cm or > 25 deg), but the continuous pose has strong inliers:
                    # The PnP jump is a symmetry/outlier flip. Retain the continuous refined pose!
                    if (dt_pnp > 0.055 or dR_pnp_deg > 25.0) and (num_ref >= 0.4 * len(inliers_sub) or num_ref >= 80):
                        rvec = rvec_ref
                        tvec = tvec_ref
                        inliers_sub = inliers_ref
                else:
                    # Refine PnP hypothesis with LM on inliers
                    rvec, tvec = cv2.solvePnPRefineLM(
                        sub_3d[inliers_sub], sub_2d[inliers_sub], K, None,
                        rvec.copy(), tvec.copy().reshape(3, 1)
                    )
            except Exception:
                pass
        else:
            # Refine PnP hypothesis with LM
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    sub_3d[inliers_sub], sub_2d[inliers_sub], K, None,
                    rvec.copy(), tvec.copy().reshape(3, 1)
                )
            except Exception:
                pass

        # Map inliers back to full pts3d array indices
        full_inliers = sub_indices[inliers_sub]
        inlier_ratio = float(len(full_inliers) / len(pts2d))

        T_cam_obj = np.eye(4, dtype=np.float64)
        R_cam_obj, _ = cv2.Rodrigues(rvec)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = tvec.flatten()

        return True, T_cam_obj, full_inliers, inlier_ratio
