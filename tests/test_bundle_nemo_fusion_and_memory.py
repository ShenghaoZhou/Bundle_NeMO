import unittest
import numpy as np
import open3d as o3d
from bundle_nemo.fusion import CanonicalObjectFusion
from bundle_nemo.memory_bank import geodesic_angle_deg


class TestFusionAndMemory(unittest.TestCase):
    def test_vectorized_integrate_points(self):
        np.random.seed(42)
        fusion = CanonicalObjectFusion(voxel_size=0.005)
        # 100 points around origin
        pts = np.random.randn(100, 3) * 0.01
        colors = np.ones((100, 3)) * 0.5
        weights = np.ones(100)

        fusion.integrate_points(pts, colors, weights)
        self.assertGreater(len(fusion.voxels), 0)

        # Check that fused point cloud can be extracted
        pcd = fusion.get_fused_point_cloud(filter_outliers=False)
        self.assertGreater(len(pcd.points), 0)

    def test_icp_refinement(self):
        np.random.seed(42)
        fusion = CanonicalObjectFusion(voxel_size=0.005)
        # Create base shape in fusion
        base_pts = np.random.randn(200, 3) * 0.02
        colors = np.ones((200, 3))
        weights = np.ones(200) * 5.0
        fusion.integrate_points(base_pts, colors, weights)

        # Create slightly shifted points (2mm shift)
        shift = np.array([0.002, -0.001, 0.001])
        shifted_pts = base_pts + shift
        T_delta = fusion.refine_with_icp(shifted_pts, max_correspondence_dist=0.02)
        self.assertEqual(T_delta.shape, (4, 4))
        # Refined points should be closer to base points
        pts_refined = (shifted_pts @ T_delta[:3, :3].T) + T_delta[:3, 3]
        init_dist = np.mean(np.linalg.norm(shifted_pts - base_pts, axis=-1))
        refined_dist = np.mean(np.linalg.norm(pts_refined - base_pts, axis=-1))
        self.assertLess(refined_dist, init_dist)

    def test_geodesic_angle(self):
        R1 = np.eye(3)
        # 90 degree rotation around Z
        R2 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        ang = geodesic_angle_deg(R1, R2)
        self.assertAlmostEqual(ang, 90.0, places=3)


if __name__ == '__main__':
    unittest.main()
