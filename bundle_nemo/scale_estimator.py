import numpy as np
import cv2
from typing import Optional, Tuple


class AutomaticScaleEstimator:
    """
    Automatically estimates the metric scale factor s between NeMO's normalized canonical
    coordinate space (roughly [-1, 1]^3) and physical metric camera space (meters).
    
    Uses robust unprojected depth points from initial keyframe observations and solves for:
      P_cam ~= R * (s * X_canon) + t
    or via robust median spatial dispersion ratio:
      s = median(||P_cam - center_cam||) / median(||X_canon - center_canon||)
    """

    @staticmethod
    def estimate_scale_from_depth(
        pts3d_canon: np.ndarray,
        pts2d_pixels: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
        conf_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate metric scale using depth observations at corresponding 2D pixel locations.
        
        Args:
            pts3d_canon: (N, 3) NeMO canonical 3D coordinates.
            pts2d_pixels: (N, 2) 2D pixel coordinates (u, v).
            depth_map: (H, W) Metric depth in meters (or uint16 in mm, converted).
            K: (3, 3) Camera intrinsic matrix.
            min_depth: Minimum valid depth in meters.
            max_depth: Maximum valid depth in meters.
            conf_weights: Optional (N,) confidence scores.
            
        Returns:
            scale_factor (float): Multiplier such that X_metric = scale_factor * X_canon.
        """
        H, W = depth_map.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = np.clip(np.round(pts2d_pixels[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(pts2d_pixels[:, 1]).astype(int), 0, H - 1)

        d = depth_map[v, u].astype(np.float64)
        # Handle millimeter depth maps if values > 20
        if np.nanmedian(d[d > 0]) > 20.0:
            d = d / 1000.0

        valid_mask = (d >= min_depth) & (d <= max_depth) & np.isfinite(d)
        if conf_weights is not None:
            valid_mask = valid_mask & (conf_weights > 0.5)

        if np.sum(valid_mask) < 20:
            # Fallback to default scale if depth is insufficient
            return 0.35

        d_valid = d[valid_mask]
        u_valid = pts2d_pixels[valid_mask, 0]
        v_valid = pts2d_pixels[valid_mask, 1]
        canon_valid = pts3d_canon[valid_mask]

        # Unproject to camera 3D space
        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid
        P_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Robust dispersion-based scale:
        center_cam = np.median(P_cam, axis=0)
        center_canon = np.median(canon_valid, axis=0)

        rad_cam = np.linalg.norm(P_cam - center_cam, axis=-1)
        rad_canon = np.linalg.norm(canon_valid - center_canon, axis=-1)

        valid_rad = (rad_canon > 1e-4) & (rad_cam > 1e-4)
        if np.sum(valid_rad) < 10:
            return 0.35

        # Ratio of 75th percentiles / median spread (robust to boundary outliers)
        scale_median = float(np.median(rad_cam[valid_rad] / rad_canon[valid_rad]))
        scale_spread = float(np.percentile(rad_cam[valid_rad], 75) / (np.percentile(rad_canon[valid_rad], 75) + 1e-6))
        
        scale_est = 0.5 * (scale_median + scale_spread)
        # Ensure scale is within plausible physical range (e.g. 2 cm to 1.5 m)
        scale_clamped = float(np.clip(scale_est, 0.02, 1.5))
        return scale_clamped
