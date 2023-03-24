import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class TowelTemplateConfig:
    width: float = 0.3
    length: float = 0.5


@dataclasses.dataclass
class ShortsMeshConfig:
    waist_width: float = 0.5
    crotch_height: float = 0.3
    pipe_width: float = 0.25
    length: float = 0.5
    waiste_pipe_angle: float = 0.2
    pipe_outer_angle: float = 0.2


@dataclasses.dataclass
class TshirtMeshConfig:
    waist_width: float = 0.65
    chest_width: float = 0.7
    shoulder_width: float = 0.62
    neck_width: float = 0.32
    shoulder_length: float = 0.9
    chest_length: float = 0.6
    sleeve_width: float = 0.2
    sleeve_length: float = 0.3
    sleeve_angle: float = 0.4
    sleeve_inner_angle: float = 0.0
    shoulder_angle: float = 0.3


def create_towel_vertices(config: TowelTemplateConfig) -> List[np.ndarray]:
    width = config.width
    length = config.length

    vertices = [
        np.array([-width / 2, -length / 2, 0.0]),
        np.array([width / 2, -length / 2, 0.0]),
        np.array([width / 2, length / 2, 0.0]),
        np.array([-width / 2, length / 2, 0.0]),
    ]
    keypoints = {
        "corner_0": vertices[0],
        "corner_1": vertices[1],
        "corner_2": vertices[2],
        "corner_3": vertices[3],
    }
    return vertices, keypoints


def create_short_vertices(config: ShortsMeshConfig) -> List[np.ndarray]:
    right_waist = np.array([config.waist_width / 2, 0, 0])
    left_waist = np.array([-config.waist_width / 2, 0, 0])
    crotch = np.array([0, -config.crotch_height, 0])
    right_pipe_outer = right_waist + np.array(
        [np.sin(config.waiste_pipe_angle) * config.length, -np.cos(config.waiste_pipe_angle) * config.length, 0]
    )
    right_pipe_inner = right_pipe_outer + np.array(
        [-np.cos(config.pipe_outer_angle) * config.pipe_width, -np.sin(config.pipe_outer_angle) * config.pipe_width, 0]
    )

    left_pipe_outer = left_waist + np.array(
        [-np.sin(config.waiste_pipe_angle) * config.length, -np.cos(config.waiste_pipe_angle) * config.length, 0]
    )
    left_pipe_inner = left_pipe_outer + np.array(
        [np.cos(config.pipe_outer_angle) * config.pipe_width, -np.sin(config.pipe_outer_angle) * config.pipe_width, 0]
    )

    vertices = [left_waist, right_waist, right_pipe_outer, right_pipe_inner, crotch, left_pipe_inner, left_pipe_outer]

    vertices = np.array(vertices)

    # move origin from center of waist to crotch
    vertices[:, 1] += config.crotch_height
    vertices = list(vertices)

    names = [
        "left_waist",
        "right_waist",
        "right_pipe_outer",
        "right_pipe_inner",
        "crotch",
        "left_pipe_inner",
        "left_pipe_outer",
    ]

    keypoints = {name: vertex for name, vertex in zip(names, vertices)}
    return vertices, keypoints


def create_tshirt_vertices(config: TshirtMeshConfig) -> List[np.ndarray]:
    np.array([-config.waist_width / 2, 0, 0])
    np.array([config.waist_width / 2, 0, 0])
    np.array([-config.chest_width / 2, config.chest_length, 0])
    np.array([config.chest_width / 2, config.chest_length, 0])
    np.array([-config.shoulder_width / 2, config.shoulder_length, 0])
    np.array([config.shoulder_width / 2, config.shoulder_length, 0])

    # TODO: add neck
    # TODO: add sleeves
    raise NotImplementedError


if __name__ == "__main__":

    import bpy

    bpy.ops.object.delete()  # Delete default cube

    vertices, keypoints = create_tshirt_vertices(TshirtMeshConfig())
    print(vertices)
    edges = []
    faces = [[i for i in range(len(vertices))]]

    # create blender mesh
    blender_mesh = bpy.data.meshes.new("test")
    blender_mesh.from_pydata(vertices, edges, faces)
    blender_mesh.update()

    # create blender object
    blender_object = bpy.data.objects.new("test", blender_mesh)
    bpy.context.collection.objects.link(blender_object)
