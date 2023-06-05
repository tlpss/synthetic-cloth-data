""" code to check if blender objects, vertices are occluded in a camera view"""
import bpy
from mathutils import Vector


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

    # add small cube around coord to make sure the ray will intersect
    # as the ray_cast is not always accurate
    # cf https://blender.stackexchange.com/a/87755
    scale = 0.00001
    bpy.ops.mesh.primitive_cube_add(location=co, scale=(scale, scale, scale))
    cube = bpy.context.object
    direction = co - camera_obj.location
    hit, location, _, _, _, _ = scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=camera_obj.location + direction * 0.0001,  # avoid self intersection
        direction=direction,
    )

    # remove the auxiliary cube
    bpy.data.objects.remove(cube, do_unlink=True)

    if not hit:
        raise ValueError("No hit found, this should not happen as the ray should always hit the vertex itself.")
    # if the hit is the vertex itself, it is not occluded
    if (location - co).length < scale * 2:
        return False
    return True


def is_object_occluded_for_scene_camera(obj: bpy.types.Object) -> bool:
    """Checks if all vertices of an object are occluded by objects in the scene w.r.t. the camera.

    Args:
        obj (bpy.types.Object): the object.

    Returns:
        boolean: visibility
    """
    for vertex in obj.data.vertices:
        coords = obj.matrix_world @ vertex.co
        print(f"checking coords {coords}")
        if not is_vertex_occluded_for_scene_camera(coords):
            print("not occluded")
            return False
    return True


if __name__ == "__main__":
    """quick test for the object visibility code."""
    import airo_blender as ab
    import numpy as np

    camera = bpy.data.objects["Camera"]
    camera.location = (-10, 0, 0)
    camera.rotation_euler = (np.pi / 2, 0, -np.pi / 2)

    # add 100 random spheres to scene
    objects = []
    for i in range(100):
        scale = 0.1

        z, y = np.random.random(2)
        z, y = z * 10 - 5, y * 10 - 5
        bpy.ops.mesh.primitive_cube_add(location=(10, y, z), scale=(scale, scale, scale))
        cube_obj = bpy.context.active_object
        objects.append(cube_obj)

    bpy.ops.mesh.primitive_cube_add(location=(10, 0, 1.1), scale=(0.1, 0.1, 0.1))
    cube_obj = bpy.context.active_object
    objects.append(cube_obj)

    for obj in objects:
        occluded = is_object_occluded_for_scene_camera(obj)
        if occluded:
            ab.add_material(obj, (1, 0, 0))
        else:
            ab.add_material(obj, (0, 1, 0))
