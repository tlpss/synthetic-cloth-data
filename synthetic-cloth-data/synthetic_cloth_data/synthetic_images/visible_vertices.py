# taken from old airo-blender-toolkit
# https://github.com/airo-ugent/airo-blender-toolkit/blob/main/airo_blender_toolkit/visible_vertices.py

import bpy
from mathutils import Vector
from mathutils.geometry import intersect_ray_tri


def intersect_ray_quad_3d(quad, origin, destination):
    ray = Vector(destination) - Vector(origin)
    p = intersect_ray_tri(quad[0], quad[1], quad[2], ray, origin)
    if p is None:
        p = intersect_ray_tri(quad[2], quad[3], quad[0], ray, origin)
    return p


def intersect_ray_scene(scene, origin, destination):
    direction = destination - origin
    result, _, _, _, object, _ = scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=origin + direction * 0.0001,
        direction=destination,
    )
    return result


def is_vertex_occluded_for_scene_camera(co: Vector) -> bool:
    """Checks if a vertex is occluded by objects in the scene w.r.t. the camera.

    Args:
        co (Vector): the world space x, y and z coordinates of the vertex.

    Returns:
        boolean: visibility
    """
    co = Vector(co)

    bpy.context.view_layer.update()  # ensures camera matrix is up to date
    scene = bpy.context.scene
    camera_obj = scene.camera  # bpy.types.Object
    camera = bpy.data.cameras[camera_obj.name]  # bpy.types.Camera

    view_frame = [camera_obj.matrix_world @ v for v in camera.view_frame(scene=scene)]
    view_center = sum(view_frame, Vector((0, 0, 0))) / len(view_frame)
    view_normal = (view_center - camera_obj.location).normalized()

    d = None
    intersection = intersect_ray_quad_3d(
        view_frame, co, camera_obj.location
    )  # check intersection with the camera frame

    if intersection is not None:
        d = intersection - co
        # only take into account vertices in front of the camera, not behind it.
        if d.dot(view_normal) < 0:
            d = d.length
            # check intersection with all other objects in scene. We revert the direction,
            # ie. look from the camera to avoid self intersection
            if intersect_ray_scene(scene, co, camera_obj.location):
                d = None
        else:
            d = None

    visible = d is not None and d > 0.0
    occluded = not visible
    return occluded
