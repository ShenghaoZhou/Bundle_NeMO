import unittest
import numpy as np

from bundle_nemo.scale_estimator import AutomaticScaleEstimator


class TestAutomaticScaleEstimator(unittest.TestCase):
    def test_scale_recovery(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        H, W = 480, 640

        true_scale = 0.25  # 25 cm toy in normalized [-1, 1] cube
        np.random.seed(42)
        # Canonical points in [-0.5, 0.5]^3
        pts3d_canon = np.random.uniform(-0.5, 0.5, (100, 3)).astype(np.float32)

        # Scale to metric meters
        pts3d_metric = pts3d_canon * true_scale

        # Camera position
        pts3d_cam = pts3d_metric.copy()
        pts3d_cam[:, 2] += 0.8  # 80 cm away

        # Project to 2D
        proj = (K @ pts3d_cam.T).T
        pts2d = proj[:, :2] / proj[:, 2:3]

        # Construct synthetic depth map
        depth_map = np.zeros((H, W), dtype=np.float32)
        for i in range(len(pts2d)):
            u, v = int(round(pts2d[i, 0])), int(round(pts2d[i, 1]))
            if 0 <= u < W and 0 <= v < H:
                depth_map[v, u] = pts3d_cam[i, 2]

        estimated_scale = AutomaticScaleEstimator.estimate_scale_from_depth(
            pts3d_canon, pts2d, depth_map, K
        )

        self.assertAlmostEqual(estimated_scale, true_scale, delta=0.03)

    def test_scale_recovery_millimeter_depth(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        H, W = 480, 640
        true_scale = 0.25
        np.random.seed(42)
        pts3d_canon = np.random.uniform(-0.5, 0.5, (100, 3)).astype(np.float32)
        pts3d_metric = pts3d_canon * true_scale
        pts3d_cam = pts3d_metric.copy()
        pts3d_cam[:, 2] += 0.8
        proj = (K @ pts3d_cam.T).T
        pts2d = proj[:, :2] / proj[:, 2:3]

        # Depth in mm (e.g. 800 mm)
        depth_map_mm = np.zeros((H, W), dtype=np.float32)
        for i in range(len(pts2d)):
            u, v = int(round(pts2d[i, 0])), int(round(pts2d[i, 1]))
            if 0 <= u < W and 0 <= v < H:
                depth_map_mm[v, u] = pts3d_cam[i, 2] * 1000.0

        estimated_scale = AutomaticScaleEstimator.estimate_scale_from_depth(
            pts3d_canon, pts2d, depth_map_mm, K
        )
        self.assertAlmostEqual(estimated_scale, true_scale, delta=0.03)

    def test_scale_fallback_insufficient_depth(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pts3d_canon = np.zeros((10, 3), dtype=np.float32)
        pts2d = np.zeros((10, 2), dtype=np.float32)
        empty_depth = np.zeros((480, 640), dtype=np.float32)
        scale = AutomaticScaleEstimator.estimate_scale_from_depth(
            pts3d_canon, pts2d, empty_depth, K
        )
        self.assertEqual(scale, AutomaticScaleEstimator.DEFAULT_SCALE)


if __name__ == '__main__':
    unittest.main()
