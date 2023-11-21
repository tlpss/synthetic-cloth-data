import sys
from typing import List

import bpy
import tqdm

print(sys.version)
import numpy as np
from synthetic_cloth_data.meshes.flat_meshes.geometric_templates import (
    ShortsMeshConfig,
    TowelTemplateConfig,
    TshirtMeshConfig,
    create_short_vertices,
    create_towel_vertices,
    create_tshirt_vertices,
)
from synthetic_cloth_data.meshes.utils.mesh_operations import (
    BevelConfig,
    BezierConfig,
    apply_bezier_curves_to_mesh,
    bevel_vertices,
    find_nearest_vertex_ids,
    subdivide_mesh,
    triangulate,
)
from synthetic_cloth_data.utils import CLOTH_TYPES


def sample_towel_config() -> TowelTemplateConfig:
    length = np.random.uniform(0.4, 0.9)
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


def sample_tshirt_config() -> TshirtMeshConfig:
    # based on https://fashion2apparel.com/t-shirt-measurement-guide-with-size-chart/
    chest_width = np.random.uniform(0.38, 0.65)
    shoulder_width = np.random.uniform(0.9, 1.05) * chest_width
    neck_width = np.random.uniform(0.45, 0.55) * chest_width
    sleeve_width = np.random.uniform(0.25, 0.3) * chest_width
    sleeve_length = np.random.uniform(0.19, 0.2)
    total_length = np.random.uniform(0.62, 0.75)
    shoulder_length = total_length - 0.02  # small compensation for shoulder angle, should be based on shoulder angle
    chest_length = shoulder_length * np.random.uniform(
        0.65, 0.7
    )  # small compensation for shoulder angle, should be based on shoulder angle
    waist_width = np.random.uniform(0.95, 1.1) * chest_width
    sleeve_angle = np.random.uniform(0.3, 0.6)
    sleeve_inner_angle = np.random.uniform(-0.1, 0.1)
    shoulder_angle = np.random.uniform(0.0, 0.2)

    return TshirtMeshConfig(
        waist_width,
        chest_width,
        shoulder_width,
        neck_width,
        shoulder_length,
        chest_length,
        sleeve_width,
        sleeve_length,
        sleeve_angle,
        sleeve_inner_angle,
        shoulder_angle,
    )


def sample_towel_bezier_config() -> BezierConfig:
    bezier_configs = []
    for i in range(4):  # 4 edges for towels
        end = (i + 1) % 4
        bezier_configs.append(BezierConfig(i, end, np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)))
    return bezier_configs


def sample_shorts_bezier_config() -> BezierConfig:
    bezier_configs = [
        BezierConfig(0, 1, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # waist
        BezierConfig(1, 2, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # right pipe outer
        BezierConfig(2, 3, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # right pipe bottom
        BezierConfig(3, 4, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # right pipe inner
        BezierConfig(4, 5, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # left pipe inner
        BezierConfig(5, 6, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # left pipe bottom
        BezierConfig(6, 0, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)),  # left pipe outer
    ]

    return bezier_configs


def sample_tshirt_bezier_config() -> BezierConfig:
    x_range = 0.01
    bezier_configs = [
        BezierConfig(0, 1, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # left shoulder
        BezierConfig(1, 2, np.random.uniform(-x_range, x_range), np.random.uniform(-0.15, -0.05)),  # neck
        BezierConfig(2, 3, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # right shoulder
        BezierConfig(3, 4, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # right sleeve
        BezierConfig(4, 5, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # right sleeve side
        BezierConfig(
            5, 6, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)
        ),  # right sleeve bottom
        BezierConfig(6, 7, np.random.uniform(-x_range, x_range), np.random.uniform(-0.03, 0.03)),  # right side
        BezierConfig(7, 8, np.random.uniform(-x_range, x_range), np.random.uniform(-0.03, 0.03)),  # bottom
        BezierConfig(8, 9, np.random.uniform(-x_range, x_range), np.random.uniform(-0.03, 0.03)),  # left side
        BezierConfig(
            9, 10, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)
        ),  # left sleeve bottom
        BezierConfig(10, 11, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # left sleeve side
        BezierConfig(11, 0, np.random.uniform(-x_range, x_range), np.random.uniform(-0.01, 0.01)),  # left sleeve
    ]
    return bezier_configs


def sample_towel_bevel_configs(keypoint_ids: List[int]):
    bevel_configs = []
    for id in keypoint_ids:
        bevel_configs.append(BevelConfig(id, 4, np.random.uniform(0.0, 0.002)))
    return bevel_configs


def sample_shorts_bevel_configs(keypoint_ids: List[int]):
    bevel_configs = []
    for id in keypoint_ids:
        bevel_configs.append(BevelConfig(id, 4, np.random.uniform(0.0, 0.005)))
    return bevel_configs


def sample_tshirt_bevel_configs(keypoint_ids: List[int]):
    bevel_configs = []
    for id in keypoint_ids:
        bevel_configs.append(BevelConfig(id, 4, np.random.uniform(0.0, 0.005)))
    return bevel_configs


def create_blender_object_from_vertices(name: str, vertices: List[np.ndarray]):
    # apply bezier curves
    edges = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
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
            location=blender_object.matrix_world @ blender_object.data.vertices[kid].co, scale=(radius, radius, radius)
        )


def generate_cloth_object(type: CLOTH_TYPES):
    if type == CLOTH_TYPES.TOWEL:
        geometric_vertices, keypoints = create_towel_vertices(sample_towel_config())
        bezier_configs = sample_towel_bezier_config()

    elif type == CLOTH_TYPES.SHORTS:
        geometric_vertices, keypoints = create_short_vertices(sample_shorts_config())
        bezier_configs = sample_shorts_bezier_config()

    elif type == CLOTH_TYPES.TSHIRT:
        geometric_vertices, keypoints = create_tshirt_vertices(sample_tshirt_config())
        bezier_configs = sample_tshirt_bezier_config()

    cloth_vertices = apply_bezier_curves_to_mesh(geometric_vertices, bezier_configs)
    new_keypoint_ids = find_nearest_vertex_ids(cloth_vertices, list(keypoints.values()))
    keypoints = {k: cloth_vertices[v] for k, v in zip(keypoints.keys(), new_keypoint_ids)}

    blender_object = create_blender_object_from_vertices(str.lower(type.name), cloth_vertices)

    if type == CLOTH_TYPES.TOWEL:
        bevel_configs = sample_towel_bevel_configs(new_keypoint_ids)
    elif type == CLOTH_TYPES.SHORTS:
        bevel_configs = sample_shorts_bevel_configs(new_keypoint_ids)
    elif type == CLOTH_TYPES.TSHIRT:
        bevel_configs = sample_tshirt_bevel_configs(new_keypoint_ids)

    blender_object = bevel_vertices(blender_object, bevel_configs)

    blender_object = triangulate(blender_object)
    cloth_vertices = [v.co for v in blender_object.data.vertices]
    new_keypoint_ids = find_nearest_vertex_ids(cloth_vertices, list(keypoints.values()))
    keypoint_ids = {k: v for k, v in zip(keypoints.keys(), new_keypoint_ids)}

    return blender_object, keypoint_ids


if __name__ == "__main__":
    from airo_blender.materials import add_material

    np.random.seed(2023)
    bpy.ops.object.delete()  # Delete default cube
    bpy.ops.mesh.primitive_plane_add(size=12, location=(5, 5, 0))
    bpy.ops.object.modifier_add(type="COLLISION")
    bpy.context.object.collision.cloth_friction = np.random.uniform(5.0, 30.0)
    plane = bpy.context.object
    subdivide_mesh(plane, 10)
    add_material(plane, (1, 0.5, 0.5, 1.0))

    for idx in tqdm.trange(10):
        ob, kp = generate_cloth_object(CLOTH_TYPES.TOWEL)
        # attach_cloth_sim(ob)
        ob.location = np.array([idx % 10, idx // 10, 0.001])
        # update the object's world matrix
        # cf. https://blender.stackexchange.com/questions/27667/incorrect-matrix-world-after-transformation
        bpy.context.view_layer.update()

        # for now no very large crumplings such as folded in half
        # these would probably require pinning some vertices and animating them.
        # see https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/subdivision_surface.html
        # and https://www.youtube.com/watch?v=C8C4GntM60o for animation

        # bpy.data.scenes["Scene"].frame_start = 0
        # for i in tqdm.trange(50):
        #     bpy.context.scene.frame_set(i)
        # visualize_keypoints(ob, list(kp.values()))
