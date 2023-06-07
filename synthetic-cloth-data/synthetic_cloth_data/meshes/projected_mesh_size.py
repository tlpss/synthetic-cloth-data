"""mesh-based projected area calculation
often referred to as 'coverage' for cloth folding.

Usually this is determined from an orthogonal projection, by segmenting the cloth and counting the pixels.
This is a mesh-based approach, which does not depend on camera extrinsics/intrinsics.

However, it suffers from a similar discretization error due to the voxelization.
Error is expected to be  0.5 * (voxel pitch)^2 * (# voxes on the boundary of the projected area) (verify)

"""

import numpy as np
import trimesh


def project_points_to_plane(points: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    """
    Project points to a plane defined by a normal and a point on the plane.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = -np.dot(plane_normal, plane_point)
    t = -(np.dot(plane_normal, points.T) + d) / np.dot(plane_normal, plane_normal)
    projected_points = points.T + t * plane_normal[:, np.newaxis]
    return projected_points.T


def get_voxel_ground_projected_area(voxel_object: trimesh.voxel.VoxelGrid):
    points = voxel_object.points
    xy_projected_points = points[:, :2]  # project to XY plane, i.e set z=0
    grid_size = np.max(np.abs(xy_projected_points)) * 2
    resolution = voxel_object.pitch[0]
    n_grid_points = int(round(grid_size / resolution)) + 1
    grid = np.zeros((n_grid_points, n_grid_points))
    for point in xy_projected_points:
        grid_x = int(round(point[0] / resolution)) + int(n_grid_points / 2)
        grid_y = int(round(point[1] / resolution)) + int(n_grid_points / 2)
        # print(f"{point} -> {grid_x}, {grid_y}")
        grid[grid_x, grid_y] = 1

    return np.sum(grid) * resolution**2


def get_mesh_projected_xy_area_size(path: str):
    mesh = trimesh.load(path)
    # blender stores meshes with Y-axis up, we use Z-axis up
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    voxels = mesh.voxelized(pitch=0.001, max_iter=10000)
    return get_voxel_ground_projected_area(voxels)


if __name__ == "__main__":
    import pathlib

    filepath = pathlib.Path(__file__).parent
    filepath = filepath / "plane.obj"
    print(get_mesh_projected_xy_area_size(filepath))
