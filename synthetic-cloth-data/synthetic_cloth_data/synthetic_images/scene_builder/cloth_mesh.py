from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import List

import bpy
import numpy as np
from synthetic_cloth_data import DATA_DIR


@dataclasses.dataclass
class ClothMeshConfig:
    mesh_path: str
    xy_randomization_range: float = 0.1
    mesh_dir: List[str] = dataclasses.field(init=False)

    def __post_init__(self):
        mesh_path = DATA_DIR / pathlib.Path(self.mesh_path)
        cloth_meshes = os.listdir(mesh_path)
        cloth_meshes = [mesh_path / mesh for mesh in cloth_meshes]
        cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]
        self.mesh_dir = cloth_meshes


def load_cloth_mesh(config: ClothMeshConfig):
    # load the obj
    mesh_file = str(np.random.choice(config.mesh_dir))
    bpy.ops.import_scene.obj(filepath=mesh_file, split_mode="OFF")  # keep vertex order with split_mode="OFF"
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    xy_position = np.random.uniform(-config.xy_randomization_range, config.xy_randomization_range, size=2)
    cloth_object.location[0] = xy_position[0]
    cloth_object.location[1] = xy_position[1]

    # make sure the mesh touches the table by having lowest vertex at z=0
    # this is an artifact of the way the meshes were created using blender's cloth simulation
    # which has imperfect collisions with the table
    # but the check does not hurt in general
    cloth_object.location[2] -= np.min([v.co[2] for v in cloth_object.data.vertices])
    # make sure the cloth is a little above the surface for rendering purposes
    cloth_object.location[2] += 0.0001

    cloth_object.rotation_euler[2] = np.random.rand() * 2 * np.pi

    # convention is to have the keypoint vertex ids in a json file with the same name as the obj file
    keypoint_vertex_dict = json.load(open(str(mesh_file).replace(".obj", ".json")))["keypoint_vertices"]
    return cloth_object, keypoint_vertex_dict
