import enum
from typing import List

import numpy as np
from synthetic_cloth_data.geometric_templates import (
    ShortsMeshConfig,
    TowelTemplateConfig,
    create_short_vertices,
    create_towel_vertices,
)
from synthetic_cloth_data.mesh_operations import (
    BevelConfig,
    BezierConfig,
    apply_bezier_curves_to_mesh,
    bevel_vertices,
    find_nearest_vertex_ids,
)


def sample_towel_config() -> TowelTemplateConfig:
    length = np.random.uniform(0.5, 1)
    width = np.random.exponential(1.0)
    width = np.clip(width, 0, 1)
    width = length - width / 3 * length

    return TowelTemplateConfig(width, length)


def sample_shorts_config() -> ShortsMeshConfig:
    waist = np.random.uniform(0.4, 0.6)  # loosely based on https://www.sportdirect.com/nl/service/maattabellen

    relative_pipe_width = np.random.uniform(0.4, 0.6)
    pipe_width = waist * relative_pipe_width

    scrotch_height = np.random.uniform(0.1, 0.25)

    length = np.random.uniform(0.2, 0.5)

    waist_pipe_angle = np.random.uniform(0.1, 0.3)
    pipe_outer_angle = np.random.uniform(0.0, 0.2)
    return ShortsMeshConfig(waist, scrotch_height, pipe_width, length, waist_pipe_angle, pipe_outer_angle)


def sample_towel_bezier_config() -> BezierConfig:
    bezier_configs = []
    for i in range(4):  # 4 edges for towels
        end = (i + 1) % 4
        bezier_configs.append(BezierConfig(i, end, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)))
    return bezier_configs


def sample_shorts_bezier_config() -> BezierConfig:
    bezier_configs = [
        BezierConfig(0, 1, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # waist
        BezierConfig(1, 2, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # right pipe outer
        BezierConfig(2, 3, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # right pipe bottom
        BezierConfig(3, 4, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # right pipe inner
        BezierConfig(4, 5, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # left pipe inner
        BezierConfig(5, 6, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # left pipe bottom
        BezierConfig(6, 0, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03)),  # left pipe outer
    ]

    return bezier_configs


def sample_towel_bevel_configs(keypoint_ids: List[int]):
    bevel_configs = []
    for id in keypoint_ids:
        bevel_configs.append(BevelConfig(id, 4, np.random.uniform(0.0, 0.03)))
    return bevel_configs


def sample_shorts_bevel_configs(keypoint_ids: List[int]):
    bevel_configs = []
    for id in keypoint_ids:
        bevel_configs.append(BevelConfig(id, 4, np.random.uniform(0.0, 0.03)))
    return bevel_configs


def create_blender_object_from_vertices(name: str, vertices: List[np.ndarray]):
    # apply bezier curves
    edges = []
    faces = [[i for i in range(len(vertices))]]

    # create blender mesh
    blender_mesh = bpy.data.meshes.new(name)
    blender_mesh.from_pydata(vertices, edges, faces)
    blender_mesh.update()

    # create blender object
    blender_object = bpy.data.objects.new(name, blender_mesh)
    bpy.context.collection.objects.link(blender_object)

    return blender_object


def visualize_keypoints(blender_object, vertex_ids):
    radius = 0.01
    for kid in vertex_ids:
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=blender_object.data.vertices[kid].co, scale=(radius, radius, radius)
        )


CLOTH_TYPES = enum.Enum("CLOTH_TYPES", "TOWEL SHORTS TSHIRT")


def generate_cloth_object(type: CLOTH_TYPES):
    if type == CLOTH_TYPES.TOWEL:
        geometric_vertices, keypoints = create_towel_vertices(sample_towel_config())
        bezier_configs = sample_towel_bezier_config()

    elif type == CLOTH_TYPES.SHORTS:
        geometric_vertices, keypoints = create_short_vertices(sample_shorts_config())
        bezier_configs = sample_shorts_bezier_config()

    towel_vertices = apply_bezier_curves_to_mesh(geometric_vertices, bezier_configs)
    new_keypoint_ids = find_nearest_vertex_ids(towel_vertices, list(keypoints.values()))
    keypoints = {k: towel_vertices[v] for k, v in zip(keypoints.keys(), new_keypoint_ids)}
    blender_object = create_blender_object_from_vertices("towel", towel_vertices)

    if type == CLOTH_TYPES.TOWEL:
        bevel_configs = sample_towel_bevel_configs(new_keypoint_ids)
    elif type == CLOTH_TYPES.SHORTS:
        bevel_configs = sample_shorts_bevel_configs(new_keypoint_ids)
    blender_object = bevel_vertices(blender_object, bevel_configs)
    return blender_object


if __name__ == "__main__":
    import bpy

    bpy.ops.object.delete()  # Delete default cube
    for idx in range(100):
        ob = generate_cloth_object(CLOTH_TYPES.SHORTS)
        ob.location = np.array([idx % 10, idx // 10, 0])
