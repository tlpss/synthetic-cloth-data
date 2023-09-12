"""utils for checking collisions between objects
adapted from BlenderProc: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/utility/CollisionUtility.py"""

from typing import Callable, List

import bpy
import numpy as np
from mathutils import Vector


def are_object_bboxes_in_collision(object: bpy.types.Object, other_object: bpy.types.Object):
    """takes the axis-aligned bboxs of both blender objects and checks if they overlap"""
    object_bbox = object.bound_box
    object_bbox = [object.matrix_world @ Vector(corner) for corner in object_bbox]

    other_object_bbox = other_object.bound_box
    other_object_bbox = [other_object.matrix_world @ Vector(corner) for corner in other_object_bbox]

    def min_and_max_point(bb):
        """
        Find the minimum and maximum point of the bounding box
        :param bb: bounding box
        :return: min, max
        """
        values = np.array(bb)
        return np.min(values, axis=0), np.max(values, axis=0)

    object_bbox_min, object_bbox_max = min_and_max_point(object_bbox)
    other_object_bbox_min, other_object_bbox_max = min_and_max_point(other_object_bbox)

    return _check_bb_intersection_on_values(
        object_bbox_min, object_bbox_max, other_object_bbox_min, other_object_bbox_max
    )


def _check_bb_intersection_on_values(
    min_b1: List[float],
    max_b1: List[float],
    min_b2: List[float],
    max_b2: List[float],
    used_check: Callable[[float, float], bool] = lambda a, b: a >= b,
):
    """
        Checks if there is an intersection of the given bounding box values. Here we use two different bounding boxes,
        namely b1 and b2. Each of them has a corresponding set of min and max values, this works for 2 and 3 dimensional
        problems.

        :param min_b1: List of minimum bounding box points for b1.
        :param max_b1: List of maximum bounding box points for b1.
        :param min_b2: List of minimum bounding box points for b2.
        :param max_b2: List of maximum bounding box points for b2.
        :param used_check: The operation used inside the is_overlapping1D. With that it possible to change the \
                           collision check from volume and surface check to pure surface or volume checks.
        :return: True if the two bounding boxes intersect with each other
        """
    collide = True
    for min_b1_val, max_b1_val, min_b2_val, max_b2_val in zip(min_b1, max_b1, min_b2, max_b2):
        # inspired by this:
        # https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
        # Checks in each dimension, if there is an overlap if this happens it must be an overlap in 3D, too.
        def is_overlapping_1D(x_min_1, x_max_1, x_min_2, x_max_2):
            # returns true if the min and max values are overlapping
            return used_check(x_max_1, x_min_2) and used_check(x_max_2, x_min_1)

        collide = collide and is_overlapping_1D(min_b1_val, max_b1_val, min_b2_val, max_b2_val)
    return collide


if __name__ == "__main__":
    cube = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(location=(0, 2.1, 0), scale=(1, 1, 1))
    cube2 = bpy.context.active_object
    print(are_object_bboxes_in_collision(cube, cube2))
