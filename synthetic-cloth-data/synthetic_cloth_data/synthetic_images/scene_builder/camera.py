import dataclasses

import bpy
import numpy as np
from mathutils import Vector
from synthetic_cloth_data.synthetic_images.scene_builder.utils.visible_vertices import is_point_in_camera_frustum


@dataclasses.dataclass
class CameraConfig:
    # intrinsics are for a ZED2i camera
    # based on https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
    horizontal_fov: int = 100  # ZED2i horizontal FOV approx
    horizontal_resolution: int = 512  # rounded to multiple of 256 for MaxViT?
    vertical_resolution: int = 288  # 16:9 aspect ratio
    horizontal_sensor_size: int = 8.67  # 1/3 inch sensor
    # extrinsics
    minimal_camera_height: float = 0.7
    max_sphere_radius: float = 1.8


def add_camera(config: CameraConfig, cloth_object: bpy.types.Object, keypoint_vertices_dict: dict) -> bpy.types.Object:
    camera = bpy.data.objects["Camera"]

    # Set the camera intrinsics
    # cf https://docs.blender.org/manual/en/latest/render/cameras.html for more info.

    # does not really matter as long as FOV is used instead of focal length.
    camera.data.sensor_width = config.horizontal_sensor_size

    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.type = "PERSP"
    camera.data.angle = np.pi / 180 * config.horizontal_fov
    camera.data.lens_unit = "FOV"
    image_width, image_height = config.horizontal_resolution, config.vertical_resolution
    scene = bpy.context.scene
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height

    # TODO: randomize camera parameters?

    def _sample_point_on_unit_sphere(z_min: float) -> np.ndarray:
        """sample a point on the unit sphere, with z coordinate >= z_min, and uniform distribution of the height z in that range"""
        z = np.random.uniform(z_min, 1)
        phi = np.random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        point_on_unit_sphere = np.array([x, y, z])
        return point_on_unit_sphere

    camera_placed = False
    while not camera_placed:
        camera.location = _sample_point_on_unit_sphere(
            z_min=config.minimal_camera_height / config.max_sphere_radius
        ) * np.random.uniform(1, config.max_sphere_radius)
        # Make the camera look at tthe origin, around which the cloth and table are assumed to be centered.
        camera_direction = -camera.location
        camera_direction = Vector(camera_direction)
        camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

        bpy.context.view_layer.update()  # update the scene to propagate the new camera location & orientation
        camera_placed = are_keypoints_in_camera_frustum(cloth_object, keypoint_vertices_dict, camera)

    return camera


## Utils


def are_keypoints_in_camera_frustum(
    cloth_object: bpy.types.Object, keypoint_vertex_dict: dict, camera: bpy.types.Object
) -> bool:
    """Check if all keypoints are in the camera frustum."""
    for _, vertex_id in keypoint_vertex_dict.items():
        point = cloth_object.data.vertices[vertex_id].co
        point = cloth_object.matrix_world @ point
        if not is_point_in_camera_frustum(point, camera):
            return False
    return True
