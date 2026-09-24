import unittest
import torch
import numpy as np

from bundle_nemo.optimizer import (
    exp_so3, exp_se3, transform_points, project_points,
    KinematicStateFilter, BundleNeMOOptimizer
)


class TestBundleNeMOOptimizer(unittest.TestCase):
    def test_exp_so3_identity(self):
        omega = torch.zeros(3)
        R = exp_so3(omega)
        np.testing.assert_allclose(R.numpy(), np.eye(3), atol=1e-5)

    def test_exp_so3_rotation(self):
        # 90 degrees around Z axis
        omega = torch.tensor([0.0, 0.0, np.pi / 2.0])
        R = exp_so3(omega)
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        np.testing.assert_allclose(R.numpy(), expected, atol=1e-5)

    def test_exp_se3_transform(self):
        xi = torch.tensor([0.1, -0.2, 0.5, 0.0, 0.0, 0.0])
        T = exp_se3(xi)
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_allclose(T[:3, 3].numpy(), [0.1, -0.2, 0.5], atol=1e-5)
        np.testing.assert_allclose(T[:3, :3].numpy(), np.eye(3), atol=1e-5)

    def test_kinematic_filter_spike_clamp(self):
        filt = KinematicStateFilter(max_jump_m=0.045)
        T0 = np.eye(4)
        T0[:3, 3] = [0.0, 0.0, 1.0]
        r0 = filt.step(T0, True)
        self.assertFalse(r0['is_spike'])

        # Sudden jump of 0.5m (should be flagged as spike and clamped)
        T_spike = np.eye(4)
        T_spike[:3, 3] = [0.5, 0.0, 1.0]
        r1 = filt.step(T_spike, True)
        self.assertTrue(r1['is_spike'])
        jump = np.linalg.norm(r1['T_cam_obj'][:3, 3] - T0[:3, 3])
        self.assertLessEqual(jump, 0.046)

    def test_optimizer_convergence(self):
        optimizer = BundleNeMOOptimizer(num_refine_iters=15, lr=0.02)
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Ground truth pose
        T_gt = np.eye(4)
        T_gt[:3, 3] = [0.05, -0.02, 0.60]

        # Canonical points
        np.random.seed(42)
        pts3d_canon = np.random.uniform(-0.1, 0.1, (50, 3)).astype(np.float32)
        pts3d_cam = (T_gt[:3, :3] @ pts3d_canon.T).T + T_gt[:3, 3]
        pts2d = (K @ pts3d_cam.T).T
        pts2d = pts2d[:, :2] / pts2d[:, 2:3]
        conf = np.ones(50, dtype=np.float32)

        # Perturbed pose
        T_init = T_gt.copy()
        T_init[:3, 3] += [0.02, -0.01, 0.01]

        # Filter step
        optimizer.filter.step(T_gt, True)

        res = optimizer.optimize_step(K, pts3d_canon, pts2d, conf, T_init, True)
        T_opt = res['T_cam_obj']
        pos_err_init = np.linalg.norm(T_init[:3, 3] - T_gt[:3, 3])
        pos_err_opt = np.linalg.norm(T_opt[:3, 3] - T_gt[:3, 3])

        self.assertLess(pos_err_opt, pos_err_init)

    def test_project_points_behind_camera(self):
        K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        # 1 point in front (z=1.0), 1 point behind camera (z=-0.5)
        pts_cam = torch.tensor([[0.1, 0.1, 1.0], [0.1, 0.1, -0.5]])
        proj, in_front = project_points(K, pts_cam, eps=0.05, return_valid_mask=True)
        self.assertTrue(in_front[0])
        self.assertFalse(in_front[1])

    def test_optimizer_with_inliers(self):
        optimizer = BundleNeMOOptimizer(num_refine_iters=15, lr=0.02)
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        T_gt = np.eye(4)
        T_gt[:3, 3] = [0.0, 0.0, 0.70]

        np.random.seed(42)
        pts3d_canon = np.random.uniform(-0.1, 0.1, (60, 3)).astype(np.float32)
        pts3d_cam = (T_gt[:3, :3] @ pts3d_canon.T).T + T_gt[:3, 3]
        pts2d = (K @ pts3d_cam.T).T
        pts2d = pts2d[:, :2] / pts2d[:, 2:3]

        # Add corrupt outliers to last 20 points
        pts2d[-20:] += np.random.uniform(50, 100, (20, 2))
        conf = np.ones(60, dtype=np.float32)

        # Inliers only cover first 40 clean points
        inliers = np.arange(40)
        T_init = T_gt.copy()
        T_init[:3, 3] += [0.015, -0.01, 0.02]

        optimizer.filter.step(T_gt, True)
        res = optimizer.optimize_step(K, pts3d_canon, pts2d, conf, T_init, True, inliers=inliers)
        T_opt = res['T_cam_obj']

        pos_err_init = np.linalg.norm(T_init[:3, 3] - T_gt[:3, 3])
        pos_err_opt = np.linalg.norm(T_opt[:3, 3] - T_gt[:3, 3])
        self.assertLess(pos_err_opt, pos_err_init)

    def test_filter_resync(self):
        filt = KinematicStateFilter(max_jump_m=0.045)
        T0 = np.eye(4)
        T0[:3, 3] = [0.0, 0.0, 1.0]
        filt.step(T0, True)

        # Resync to new pose
        T_new = np.eye(4)
        T_new[:3, 3] = [0.2, -0.1, 1.5]
        filt.resync(T_new)
        self.assertTrue(np.allclose(filt.pos, [0.2, -0.1, 1.5]))


if __name__ == '__main__':
    unittest.main()
