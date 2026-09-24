import unittest
import numpy as np
import cv2
from PIL import Image

from bundle_nemo.tracker import crop_masked_object
from bundle_nemo.correspondence import NeMOCorrespondenceEngine
from bundle_nemo.memory_bank import AlignedDynamicNeMOMemoryBank, geodesic_angle_deg


class TestTrackerAndCorrespondence(unittest.TestCase):
    def test_crop_masked_object_with_mask(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        # Put red square at center
        rgb[200:280, 280:360] = [255, 0, 0]
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:280, 280:360] = 1

        # Test without return_mask
        crop_pil, crop_box = crop_masked_object(rgb, mask, return_mask=False)
        self.assertIsInstance(crop_pil, Image.Image)
        self.assertEqual(crop_pil.size, (224, 224))
        self.assertEqual(len(crop_box), 4)

        # Test with return_mask=True
        crop_pil, crop_box, crop_fg_mask = crop_masked_object(rgb, mask, return_mask=True)
        self.assertIsInstance(crop_pil, Image.Image)
        self.assertEqual(crop_fg_mask.shape, (224, 224))
        self.assertGreater(np.sum(crop_fg_mask > 0), 100)

    def test_correspondence_engine_extraction_and_pnp(self):
        np.random.seed(42)
        engine = NeMOCorrespondenceEngine(conf_threshold=1.0)

        # Synthetic 3D points in 224x224 crop
        pts3d_local = np.random.randn(224, 224, 3).astype(np.float32)
        conf = np.ones((224, 224), dtype=np.float32) * 0.5
        # Set high confidence on a 20x20 patch
        conf[100:120, 100:120] = 1.5

        crop_box = (100, 100, 324, 324)
        pts2d_full, pts3d_cand, conf_valid = engine.extract_correspondences(
            pts3d_local, conf, crop_box
        )

        self.assertEqual(len(pts2d_full), 400)
        self.assertEqual(len(pts3d_cand), 400)
        self.assertTrue(np.all(conf_valid > 1.0))

        # Test solve_pnp
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T_true = np.eye(4)
        T_true[:3, 3] = [0.0, 0.0, 0.8]
        pts3d_clean = np.random.uniform(-0.1, 0.1, (100, 3)).astype(np.float32)
        pts3d_cam = (T_true[:3, :3] @ pts3d_clean.T).T + T_true[:3, 3]
        pts2d_clean = (K @ pts3d_cam.T).T
        pts2d_clean = (pts2d_clean[:, :2] / pts2d_clean[:, 2:3]).astype(np.float32)

        succ, T_pnp, inliers, inlier_ratio = engine.solve_pnp(pts3d_clean, pts2d_clean, K)
        self.assertTrue(succ)
        self.assertIsNotNone(inliers)
        self.assertGreaterEqual(len(inliers), 90)
        self.assertAlmostEqual(T_pnp[2, 3], 0.8, delta=0.01)

    def test_keyframe_translation_admission(self):
        class MockModel:
            pass

        bank = AlignedDynamicNeMOMemoryBank(
            model=MockModel(),
            device='cpu',
            min_keyframe_rot_deg=28.0,
            min_keyframe_trans_m=0.08,
            min_inlier_ratio_trigger=0.20
        )
        # Cluster 0 at origin
        bank.clusters.append({
            'name': 'Cluster 0',
            'frame_idx': 0,
            'R_cam_obj': np.eye(3),
            't_cam_obj': np.array([0.0, 0.0, 0.8])
        })

        # Test frame within cooldown (< 20 frames)
        T_curr = np.eye(4)
        T_curr[:3, 3] = [0.0, 0.0, 0.8]
        should_add, _ = bank.should_register_keyframe(T_curr, current_inlier_ratio=0.5, frame_idx=5)
        self.assertFalse(should_add)

        # Test frame after cooldown with pure translation (12 cm translation > 8 cm)
        T_trans = np.eye(4)
        T_trans[:3, 3] = [0.12, 0.0, 0.8]
        should_add, reason = bank.should_register_keyframe(T_trans, current_inlier_ratio=0.5, frame_idx=25)
        self.assertTrue(should_add)
        self.assertIn("Translation delta", reason)


if __name__ == '__main__':
    unittest.main()
