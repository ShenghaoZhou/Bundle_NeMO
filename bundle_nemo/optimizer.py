import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple


def exp_so3(omega: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Matrix exponential for so(3) -> SO(3) using Rodrigues formula."""
    theta = torch.norm(omega, p=2, dim=-1, keepdim=True)
    theta_clamped = torch.clamp(theta, min=eps)

    axis = omega / theta_clamped
    wx = axis[..., 0]
    wy = axis[..., 1]
    wz = axis[..., 2]

    zero = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([zero, -wz, wy], dim=-1),
        torch.stack([wz, zero, -wx], dim=-1),
        torch.stack([-wy, wx, zero], dim=-1)
    ], dim=-2)

    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)

    R = I + sin_theta * K + (1.0 - cos_theta) * torch.matmul(K, K)
    # Taylor expansion for small angles
    small_angle = (theta < 1e-4).unsqueeze(-1)
    R_small = I + sin_theta * K
    return torch.where(small_angle, R_small, R)


def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """Matrix exponential for se(3) -> SE(3) homogeneous 4x4 matrix."""
    upsilon = xi[:3]
    omega = xi[3:]

    R = exp_so3(omega)
    theta = torch.norm(omega, p=2)

    if theta < 1e-5:
        V = torch.eye(3, dtype=xi.dtype, device=xi.device)
    else:
        wx = omega[0]
        wy = omega[1]
        wz = omega[2]
        zero = torch.tensor(0.0, dtype=xi.dtype, device=xi.device)
        K = torch.stack([
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero])
        ])
        I = torch.eye(3, dtype=xi.dtype, device=xi.device)
        V = I + ((1.0 - torch.cos(theta)) / (theta ** 2)) * K + ((theta - torch.sin(theta)) / (theta ** 3)) * torch.matmul(K, K)

    t = torch.matmul(V, upsilon.unsqueeze(-1)).squeeze(-1)

    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transform T (4x4) to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return torch.matmul(pts, R.t()) + t


def project_points(K: torch.Tensor, pts_cam: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """Project 3D camera points to 2D pixel coordinates."""
    z = torch.clamp(pts_cam[..., 2:3], min=eps)
    xy = pts_cam[..., :2] / z
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u = fx * xy[..., 0] + cx
    v = fy * xy[..., 1] + cy
    return torch.stack([u, v], dim=-1)


class KinematicStateFilter:
    """
    Causal Kinematic Filter for 6-DoF Trajectory.
    Eliminates high-frequency PnP noise and angular jitter while preventing runaway drift.
    """
    def __init__(
        self,
        max_jump_m: float = 0.045,
        max_step_rot_deg: float = 25.0,
        alpha_pos: float = 0.70,
        alpha_rot: float = 0.70
    ):
        self.max_jump_m = max_jump_m
        self.max_step_rot_deg = max_step_rot_deg
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.pos: Optional[np.ndarray] = None
        self.rot: Optional[np.ndarray] = None
        self.initialized = False
        self.spike_count = 0

    def step(self, T_cam_obj_meas: Optional[np.ndarray], valid: bool) -> Dict[str, Any]:
        is_spike = False

        if not self.initialized:
            if valid and T_cam_obj_meas is not None:
                self.pos = T_cam_obj_meas[:3, 3].astype(np.float64).copy()
                self.rot = T_cam_obj_meas[:3, :3].astype(np.float64).copy()
                self.initialized = True
            else:
                self.pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
                self.rot = np.eye(3, dtype=np.float64)
                self.initialized = True

            T_out = np.eye(4, dtype=np.float64)
            T_out[:3, :3] = self.rot
            T_out[:3, 3] = self.pos
            return {'T_cam_obj': T_out, 'is_spike': False, 'total_spikes': 0}

        if valid and T_cam_obj_meas is not None:
            p_meas = T_cam_obj_meas[:3, 3].astype(np.float64)
            R_meas = T_cam_obj_meas[:3, :3].astype(np.float64)

            # Translation step clamp
            pos_diff = p_meas - self.pos
            jump_m = np.linalg.norm(pos_diff)
            if jump_m > self.max_jump_m:
                pos_diff = pos_diff * (self.max_jump_m / jump_m)
                is_spike = True
                self.spike_count += 1
            self.pos = self.pos + self.alpha_pos * pos_diff

            # Rotation SLERP
            R_diff = self.rot.T @ R_meas
            tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            theta = np.arccos(tr)

            if np.rad2deg(theta) > self.max_step_rot_deg:
                is_spike = True
                self.spike_count += 1

            if theta > 1e-5:
                axis = np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ], dtype=np.float64)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    step_theta = min(theta * self.alpha_rot, np.deg2rad(self.max_step_rot_deg))
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    R_step = np.eye(3) + np.sin(step_theta) * K + (1.0 - np.cos(step_theta)) * (K @ K)
                    self.rot = self.rot @ R_step
        else:
            is_spike = True

        T_out = np.eye(4, dtype=np.float64)
        T_out[:3, :3] = self.rot
        T_out[:3, 3] = self.pos
        return {'T_cam_obj': T_out, 'is_spike': is_spike, 'total_spikes': self.spike_count}


class BundleNeMOOptimizer:
    """
    Sliding-Window SE(3) Bundle Adjustment Optimizer.
    Combines:
    1. Kinematic Lie-group smoothing.
    2. Robust Huber 2D-3D canonical reprojection optimization.
    3. Regularized step clamping on se(3) tangent space.
    """
    def __init__(
        self,
        huber_delta: float = 2.0,
        num_refine_iters: int = 12,
        lr: float = 0.010,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.huber_delta = huber_delta
        self.num_refine_iters = num_refine_iters
        self.lr = lr
        self.device = device
        self.filter = KinematicStateFilter()

    def optimize_step(
        self,
        K: np.ndarray,
        pts3d_canon: np.ndarray,
        pts2d_pixels: np.ndarray,
        conf_weights: np.ndarray,
        T_cam_obj_pnp: Optional[np.ndarray],
        pnp_valid: bool
    ) -> Dict[str, Any]:
        """
        Refine pose using Huber reprojection error over 2D-3D inliers.
        """
        filt_res = self.filter.step(T_cam_obj_pnp, pnp_valid)
        T_init = filt_res['T_cam_obj']
        is_spike = filt_res['is_spike']

        if len(pts3d_canon) >= 20 and not is_spike:
            # Subsample points for fast optimization
            num_pts = min(250, len(pts3d_canon))
            step_sub = max(1, len(pts3d_canon) // num_pts)

            pts3d_t = torch.as_tensor(pts3d_canon[::step_sub], dtype=torch.float32, device=self.device)
            pts2d_t = torch.as_tensor(pts2d_pixels[::step_sub], dtype=torch.float32, device=self.device)
            w_t = torch.as_tensor(conf_weights[::step_sub], dtype=torch.float32, device=self.device)
            K_t = torch.as_tensor(K, dtype=torch.float32, device=self.device)
            T_init_t = torch.as_tensor(T_init, dtype=torch.float32, device=self.device)

            delta_xi = torch.zeros(6, dtype=torch.float32, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([delta_xi], lr=self.lr)

            best_T = T_init_t.clone()
            min_loss = float('inf')

            for _ in range(self.num_refine_iters):
                optimizer.zero_grad()
                T_cand = torch.matmul(exp_se3(delta_xi), T_init_t)
                pts_c = transform_points(T_cand, pts3d_t)
                proj = project_points(K_t, pts_c, eps=0.05)

                diff = proj - pts2d_t
                dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)
                huber = torch.where(
                    dist <= self.huber_delta,
                    0.5 * dist ** 2,
                    self.huber_delta * (dist - 0.5 * self.huber_delta)
                )
                loss = torch.mean(w_t * huber) + 40.0 * torch.sum(delta_xi ** 2)

                if torch.isnan(loss) or torch.isinf(loss):
                    break

                loss.backward()
                optimizer.step()

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_T = T_cand.detach().clone()

            if torch.isfinite(best_T).all():
                T_opt = best_T.cpu().numpy()
            else:
                T_opt = T_init
        else:
            T_opt = T_init

        return {
            'T_cam_obj': T_opt,
            'is_spike': is_spike,
            'total_spikes': filt_res['total_spikes']
        }
