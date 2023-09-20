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

    solidify: bool = True
    subdivide: bool = True
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
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = cloth_object
    cloth_object.select_set(True)
    # randomize position & orientation
    xy_position = np.random.uniform(-config.xy_randomization_range, config.xy_randomization_range, size=2)
    cloth_object.location[0] = xy_position[0]
    cloth_object.location[1] = xy_position[1]

    # make sure the mesh touches the table by having lowest vertex at z=0
    # this is an artifact of the way the meshes were created using blender's cloth simulation
    # which has imperfect collisions with the table
    # but the check does not hurt in general
    z_min = np.min([(cloth_object.matrix_world @ v.co)[2] for v in cloth_object.data.vertices])
    cloth_object.location[2] -= z_min
    # make sure the cloth is a little above the surface for rendering purposes
    cloth_object.location[2] += 0.0005

    # randomize orientation
    cloth_object.rotation_euler[2] = np.random.rand() * 2 * np.pi

    # first solidify, then subdivide. The modifier will then also smooth the edges of the solidified mesh.

    if config.solidify:
        thickness = np.random.uniform(0.001, 0.003)  # 1-3 mm cloth thickness.
        # note that this has impacts on the visibility of the keypoints
        # as these are now inside the mesh. Need to either test for 1-ring neighbours or make sure that the auxiliary cubes around a vertex
        # in the visibility check are larger than the solidify modifier thickness. The latter is what we do by default, since the rest distance of the cloth meshes
        # is assumed to be > 1cm.
        cloth_object.location[2] += thickness / 2  # offset for solidify modifier
        # solidify the mesh to give the cloth some thickness.
        bpy.ops.object.modifier_add(type="SOLIDIFY")
        # 2 mm, make sure the particle radius of the cloth simulator is larger than this!
        bpy.context.object.modifiers["Solidify"].thickness = thickness
        bpy.context.object.modifiers["Solidify"].offset = 0.0  # center the thickness around the original mesh

        # disable auto-smooth to enable gpu-accelerated subsurface division modifier

    bpy.ops.object.shade_flat()
    bpy.context.object.data.use_auto_smooth = False
    if config.subdivide:
        #  modifier is more powerful than operator
        # but it is also rather expensive. Make sure it is done on GPU!
        # higher subdivision -> more expensive rendering, so have to find lowest amount that is still good enough
        # also influenced by rendering resolution ofc.

        bpy.ops.object.modifier_add(type="SUBSURF")
        bpy.context.object.modifiers["Subdivision"].render_levels = 2
        bpy.context.object.modifiers["Subdivision"].use_limit_surface = False

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.subdivide(smoothness=1,number_cuts=1)
        # bpy.ops.object.mode_set(mode="OBJECT")

    keypoint_vertex_dict = json.load(open(str(mesh_file).replace(".obj", ".json")))["keypoint_vertices"]
    return cloth_object, keypoint_vertex_dict
