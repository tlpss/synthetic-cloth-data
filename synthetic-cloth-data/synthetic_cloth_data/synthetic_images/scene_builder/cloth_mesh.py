from __future__ import annotations

import dataclasses
import json
from typing import List

import bpy
import numpy as np


@dataclasses.dataclass
class ClothMeshConfig:
    mesh_dir: List[str]
    xy_randomization_range: float = 0.1


def load_cloth_mesh(config: ClothMeshConfig):
    # load the obj
    mesh_file = str(np.random.choice(config.mesh_dir))
    bpy.ops.import_scene.obj(filepath=mesh_file, split_mode="OFF")  # keep vertex order with split_mode="OFF"
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    xy_position = np.random.uniform(-config.xy_randomization_range, config.xy_randomization_range, size=2)
    cloth_object.location[0] = xy_position[0]
    cloth_object.location[1] = xy_position[1]

    cloth_object.location[2] = 0.001  # make sure the cloth is above the surface

    cloth_object.rotation_euler[2] = np.random.rand() * 2 * np.pi

    # convention is to have the keypoint vertex ids in a json file with the same name as the obj file
    keypoint_vertex_dict = json.load(open(str(mesh_file).replace(".obj", ".json")))
    return cloth_object, keypoint_vertex_dict
