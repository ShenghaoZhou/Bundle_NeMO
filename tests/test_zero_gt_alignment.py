import unittest
import numpy as np
import torch
from bundle_nemo.zero_gt_alignment import umeyama_alignment, ransac_umeyama
from bundle_nemo.pose_graph import KeyframePoseGraph


class TestZeroGTAlignment(unittest.TestCase):
    def test_umeyama_exact(self):
        np.random.seed(42)
        # 3D points
        pts_src = np.random.randn(40, 3) * 0.1
        # 45 deg rotation around Y
        theta = np.deg2rad(45.0)
        R_true = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        t_true = np.array([0.02, -0.01, 0.05])
        pts_dst = (R_true @ pts_src.T).T + t_true

        R_est, t_est = umeyama_alignment(pts_src, pts_dst)
        self.assertTrue(np.allclose(R_est, R_true, atol=1e-5))
        self.assertTrue(np.allclose(t_est, t_true, atol=1e-5))

    def test_ransac_umeyama_with_outliers(self):
        np.random.seed(42)
        pts_src = np.random.randn(50, 3) * 0.1
        theta = np.deg2rad(30.0)
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        t_true = np.array([0.01, 0.02, -0.01])
        pts_dst = (R_true @ pts_src.T).T + t_true

        # Add 15 outliers
        pts_dst[-15:] += np.random.randn(15, 3) * 0.2

        success, R_est, t_est, inliers = ransac_umeyama(
            pts_src, pts_dst, dist_thresh=0.015, max_iters=200
        )
        self.assertTrue(success)
        self.assertGreaterEqual(inliers, 30)
        # Check rotation accuracy
        rot_err = np.linalg.norm(R_est - R_true)
        self.assertLess(rot_err, 0.05)

    def test_pose_graph_optimization_with_loop_closure(self):
        np.random.seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = KeyframePoseGraph(device=device)

        K = np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]], dtype=np.float32)
        pts3d = np.random.randn(50, 3) * 0.05
        pts3d[:, 2] += 0.5  # In front of camera
        pts2d = np.random.randn(50, 2) * 50 + 112
        weights = np.ones(50, dtype=np.float32)

        # Add node 0 (anchor)
        T0 = np.eye(4)
        graph.add_keyframe(0, T0, pts3d, pts2d, weights, K)

        # Add node 1 (20 deg rotation)
        theta1 = np.deg2rad(20.0)
        T1 = np.eye(4)
        T1[:3, :3] = np.array([[np.cos(theta1), -np.sin(theta1), 0], [np.sin(theta1), np.cos(theta1), 0], [0, 0, 1]])
        T1[:3, 3] = np.array([0.02, 0.0, 0.0])
        graph.add_keyframe(10, T1, pts3d, pts2d, weights, K)

        # Add node 2 (< 55 deg from anchor -> triggers loop closure)
        theta2 = np.deg2rad(35.0)
        T2 = np.eye(4)
        T2[:3, :3] = np.array([[np.cos(theta2), -np.sin(theta2), 0], [np.sin(theta2), np.cos(theta2), 0], [0, 0, 1]])
        T2[:3, 3] = np.array([0.04, 0.0, 0.0])
        graph.add_keyframe(20, T2, pts3d, pts2d, weights, K)

        # Verify loop closure edge was created (edges: 0->1, 1->2, and loop closure 0->2)
        self.assertGreaterEqual(len(graph.relative_edges), 3)

        opt_poses = graph.optimize(num_iters=15, lr=0.01)
        self.assertEqual(len(opt_poses), 3)
        self.assertTrue(np.allclose(opt_poses[0], T0))  # Node 0 fixed


if __name__ == '__main__':
    unittest.main()
