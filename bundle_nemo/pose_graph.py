import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .optimizer import exp_se3, project_points


class KeyframePoseGraph:
    """
    Method C: BundleSDF-Style Keyframe Pose Graph Optimization & Multi-View BA.
    
    Maintains a sparse pose graph of keyframes:
    1. Unary Factors: Multi-view 2D-3D canonical reprojection errors.
    2. Binary Factors: Relative SE(3) tracking odometry constraints between keyframes.
    3. Anchor Factor: Fixes Keyframe 0 to eliminate gauge freedom.
    
    Jointly optimizes all keyframe poses on SE(3) Lie algebra without any ground-truth poses.
    """
    def __init__(self, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.keyframes: List[Dict[str, Any]] = []
        self.relative_edges: List[Dict[str, Any]] = []

    def add_keyframe(
        self,
        frame_idx: int,
        T_cam_obj_init: np.ndarray,
        pts3d_canon: np.ndarray,
        pts2d_pixels: np.ndarray,
        conf_weights: np.ndarray,
        K: np.ndarray
    ) -> int:
        """Add a new keyframe node to the pose graph."""
        node_id = len(self.keyframes)
        # Subsample points for efficient BA
        stride = max(1, len(pts3d_canon) // 250)
        sub_idx = np.arange(0, len(pts3d_canon), stride)

        kf_data = {
            'node_id': node_id,
            'frame_idx': frame_idx,
            'T_init': T_cam_obj_init.copy(),
            'T_opt': T_cam_obj_init.copy(),
            'pts3d': torch.tensor(pts3d_canon[sub_idx], dtype=torch.float32, device=self.device),
            'pts2d': torch.tensor(pts2d_pixels[sub_idx], dtype=torch.float32, device=self.device),
            'weights': torch.tensor(conf_weights[sub_idx], dtype=torch.float32, device=self.device),
            'K': torch.tensor(K, dtype=torch.float32, device=self.device)
        }
        self.keyframes.append(kf_data)

        # Add sequential odometry edge to preceding keyframe
        if node_id > 0:
            prev_kf = self.keyframes[node_id - 1]
            T_prev_inv = np.linalg.inv(prev_kf['T_init'])
            T_rel = T_cam_obj_init @ T_prev_inv  # T_rel @ T_prev = T_curr
            self.relative_edges.append({
                'from': node_id - 1,
                'to': node_id,
                'T_rel': torch.tensor(T_rel, dtype=torch.float32, device=self.device),
                'weight': 5.0
            })

        return node_id

    def optimize(
        self,
        num_iters: int = 35,
        lr: float = 0.015,
        huber_delta: float = 3.0
    ) -> List[np.ndarray]:
        """
        Jointly bundle-adjust all keyframe poses on Lie algebra se(3).
        Keyframe 0 is held fixed to ground the world reference frame.
        
        Returns:
            List of optimized 4x4 SE(3) poses for all keyframes.
        """
        K_nodes = len(self.keyframes)
        if K_nodes <= 1:
            return [kf['T_opt'] for kf in self.keyframes]

        # Parameterize nodes 1..K-1 via Lie algebra se(3) increments xi_i
        xi_params = []
        base_Ts = []

        for i in range(1, K_nodes):
            xi = torch.zeros(6, dtype=torch.float32, device=self.device, requires_grad=True)
            xi_params.append(xi)
            base_Ts.append(torch.tensor(self.keyframes[i]['T_opt'], dtype=torch.float32, device=self.device))

        T0 = torch.tensor(self.keyframes[0]['T_opt'], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(xi_params, lr=lr)

        for _ in range(num_iters):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            # Assemble current poses: T_0 is fixed, T_i = exp(xi_i) @ base_T_i
            current_Ts = [T0]
            for i in range(len(xi_params)):
                delta_T = exp_se3(xi_params[i])
                T_curr = torch.matmul(delta_T, base_Ts[i])
                current_Ts.append(T_curr)

            # 1. Multi-View Reprojection Residuals
            for i in range(1, K_nodes):
                kf = self.keyframes[i]
                T_i = current_Ts[i]
                R_i = T_i[:3, :3]
                t_i = T_i[:3, 3]

                # Project 3D canonical points into keyframe i
                pts_cam = torch.matmul(kf['pts3d'], R_i.t()) + t_i
                pts2d_pred = project_points(kf['K'], pts_cam)
                diff = pts2d_pred - kf['pts2d']
                err_sq = torch.sum(diff ** 2, dim=-1)

                # Huber loss
                huber = torch.where(
                    err_sq < huber_delta ** 2,
                    0.5 * err_sq,
                    huber_delta * (torch.sqrt(err_sq + 1e-6) - 0.5 * huber_delta)
                )
                w_huber = (huber * kf['weights']).mean()
                total_loss = total_loss + w_huber

            # 2. Keyframe-to-Keyframe Odometry Relative Constraints
            for edge in self.relative_edges:
                u, v = edge['from'], edge['to']
                T_u, T_v = current_Ts[u], current_Ts[v]
                T_rel_meas = edge['T_rel']

                # Predicted relative: T_rel_pred = T_v @ inv(T_u)
                T_u_inv = torch.eye(4, device=self.device)
                T_u_inv[:3, :3] = T_u[:3, :3].t()
                T_u_inv[:3, 3] = -torch.matmul(T_u[:3, :3].t(), T_u[:3, 3])
                T_rel_pred = torch.matmul(T_v, T_u_inv)

                # SE(3) difference: Frobenius norm on R and L2 on t
                R_diff = T_rel_pred[:3, :3] - T_rel_meas[:3, :3]
                t_diff = T_rel_pred[:3, 3] - T_rel_meas[:3, 3]
                rel_err = torch.norm(R_diff, p='fro') + 5.0 * torch.norm(t_diff)
                total_loss = total_loss + edge['weight'] * rel_err

            total_loss.backward()
            optimizer.step()

        # Update and extract optimized poses
        opt_poses = [self.keyframes[0]['T_opt']]
        with torch.no_grad():
            for i in range(len(xi_params)):
                delta_T = exp_se3(xi_params[i]).cpu().numpy()
                T_opt = delta_T @ self.keyframes[i + 1]['T_opt']
                self.keyframes[i + 1]['T_opt'] = T_opt
                opt_poses.append(T_opt)

        return opt_poses

    def get_cluster_rotations(self, R_ref0: np.ndarray) -> List[np.ndarray]:
        """
        Derive inter-cluster rotation R_{k -> 0} from bundle-adjusted keyframe poses.
        R_{k -> 0} = R_ref0 @ R_cam_obj(k)^T
        """
        R_list = []
        for kf in self.keyframes:
            R_k = kf['T_opt'][:3, :3]
            R_k_to_0 = R_ref0 @ R_k.T
            R_list.append(R_k_to_0)
        return R_list
