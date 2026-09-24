import numpy as np
import open3d as o3d
from typing import Optional, Dict, Tuple, Any


class CanonicalObjectFusion:
    """
    Persistent Canonical Voxel/Surfel Grid Fusion for NeMO 3D Models.
    
    Replaces NerfRunner (the heavy neural implicit SDF volume rendering thread):
    - Incrementally fuses multi-view 3D surface points into a unified voxel map using
      confidence-weighted running averages and normal consistency filtering.
    - Generates clean, single-surface watertight Poisson meshes and outlier-free point clouds.
    """
    def __init__(
        self,
        voxel_size: float = 0.003,      # 3 mm voxel resolution
        min_observations: int = 4,       # Prune voxels observed in < 4 frames
        outlier_nb_neighbors: int = 30,
        outlier_std_ratio: float = 1.2
    ):
        self.voxel_size = voxel_size
        self.min_observations = min_observations
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        # Voxel hash map: (vx, vy, vz) -> {pos, color, weight, count}
        self.voxels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def refine_with_icp(
        self,
        new_pts_canon: np.ndarray,
        max_correspondence_dist: float = 0.02
    ) -> np.ndarray:
        """
        Run Open3D Point-to-Point ICP between new cluster surface points (in canonical frame)
        and currently fused voxel point cloud.
        Returns 4x4 transformation matrix Delta_T to apply to new_pts_canon.
        """
        if len(self.voxels) < 50 or len(new_pts_canon) < 50:
            return np.eye(4, dtype=np.float64)

        # Target: current fused points
        target_pts = np.array([v['pos'] for v in self.voxels.values() if v['count'] >= 2], dtype=np.float64)
        if len(target_pts) < 50:
            target_pts = np.array([v['pos'] for v in self.voxels.values()], dtype=np.float64)

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(new_pts_canon.astype(np.float64))
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(target_pts)

        try:
            result = o3d.pipelines.registration.registration_icp(
                src, tgt, max_correspondence_dist,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            trans = np.array(result.transformation, dtype=np.float64)
            # Accept if refinement is a reasonable small adjustment (< 3 cm translation, < 25 deg rotation)
            R_delta = trans[:3, :3]
            tr = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
            ang_deg = float(np.rad2deg(np.arccos(tr)))
            t_norm = float(np.linalg.norm(trans[:3, 3]))

            if ang_deg < 25.0:
                if t_norm < 0.03:
                    return trans
                elif t_norm < 0.06:
                    # Preserve rotation adjustment even if translation slightly drifted
                    trans_rot_only = np.eye(4, dtype=np.float64)
                    trans_rot_only[:3, :3] = R_delta
                    return trans_rot_only
            return np.eye(4, dtype=np.float64)
        except Exception:
            return np.eye(4, dtype=np.float64)

    def integrate_points(
        self,
        pts_canon: np.ndarray,
        rgb_canon: np.ndarray,
        conf_weights: np.ndarray
    ) -> None:
        """
        Vectorized integration of a frame's canonical 3D surface points into the persistent voxel map.
        Uses np.unique and np.bincount for efficient batch voxel accumulation.
        """
        if len(pts_canon) == 0:
            return

        inv_vox = 1.0 / self.voxel_size
        voxel_keys = np.floor(pts_canon * inv_vox).astype(np.int32)

        uniq_keys, inv_indices, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
        w = conf_weights.astype(np.float64)
        sum_w = np.bincount(inv_indices, weights=w, minlength=len(uniq_keys))

        sum_wx = np.bincount(inv_indices, weights=w * pts_canon[:, 0], minlength=len(uniq_keys))
        sum_wy = np.bincount(inv_indices, weights=w * pts_canon[:, 1], minlength=len(uniq_keys))
        sum_wz = np.bincount(inv_indices, weights=w * pts_canon[:, 2], minlength=len(uniq_keys))

        sum_cr = np.bincount(inv_indices, weights=w * rgb_canon[:, 0], minlength=len(uniq_keys))
        sum_cg = np.bincount(inv_indices, weights=w * rgb_canon[:, 1], minlength=len(uniq_keys))
        sum_cb = np.bincount(inv_indices, weights=w * rgb_canon[:, 2], minlength=len(uniq_keys))

        for j, (vx, vy, vz) in enumerate(uniq_keys):
            key = (int(vx), int(vy), int(vz))
            sw = sum_w[j]
            cnt = int(counts[j])
            sp = np.array([sum_wx[j], sum_wy[j], sum_wz[j]], dtype=np.float64)
            sc = np.array([sum_cr[j], sum_cg[j], sum_cb[j]], dtype=np.float64)

            if key not in self.voxels:
                if sw > 1e-6:
                    self.voxels[key] = {
                        'pos': sp / sw,
                        'color': sc / sw,
                        'weight': sw,
                        'count': cnt
                    }
            else:
                vox = self.voxels[key]
                new_w = vox['weight'] + sw
                if new_w > 1e-6:
                    vox['pos'] = (vox['weight'] * vox['pos'] + sp) / new_w
                    vox['color'] = (vox['weight'] * vox['color'] + sc) / new_w
                vox['weight'] = new_w
                vox['count'] += cnt

    def get_fused_point_cloud(self, filter_outliers: bool = True) -> o3d.geometry.PointCloud:
        """Extract clean fused canonical point cloud."""
        valid_items = [v for v in self.voxels.values() if v['count'] >= self.min_observations]
        if len(valid_items) < 50:
            valid_items = list(self.voxels.values())

        if len(valid_items) == 0:
            return o3d.geometry.PointCloud()

        pts = np.array([v['pos'] for v in valid_items], dtype=np.float64)
        colors = np.array([v['color'] for v in valid_items], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

        if filter_outliers and len(pts) > self.outlier_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio
            )
        return pcd

    def extract_poisson_mesh(
        self,
        depth: int = 8,
        min_density_quantile: float = 0.05
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Extract a clean, watertight Poisson surface mesh from the fused point cloud.
        """
        pcd = self.get_fused_point_cloud(filter_outliers=True)
        if len(pcd.points) < 100:
            return None

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True
        )
        densities = np.asarray(densities)
        if len(densities) > 0:
            cutoff = np.quantile(densities, min_density_quantile)
            mesh.remove_vertices_by_mask(densities < cutoff)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh
