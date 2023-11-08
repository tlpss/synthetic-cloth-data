"""takes a set of meshes and generates a new set of meshes with randomized X and Z scales for those meshes to increase diversity"""


import os
import shutil

import bpy
import numpy as np

np.random.seed(2023)


def generate_randomized_dataset(
    mesh_dir, output_dir, x_relative_scale_range, z_relative_scale_range, n_additional_meshes_per_mesh
):
    """generates a new set of meshes with randomized X and Z scales for those meshes to increase diversity"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    meshes = os.listdir(mesh_dir)
    meshes = [mesh for mesh in meshes if mesh.endswith(".obj")]

    for mesh in meshes:
        # load mesh into blender
        mesh_path = os.path.join(mesh_dir, mesh)
        bpy.ops.import_scene.obj(filepath=mesh_path, split_mode="OFF")  # keep vertex order with split_mode="OFF"

        cloth_object = bpy.context.selected_objects[0]

        # load keypoints json
        json_path = mesh[:-4] + ".json"

        # save original mesh
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = cloth_object
        cloth_object.select_set(True)
        bpy.ops.export_scene.obj(
            filepath=os.path.join(output_dir, mesh), use_selection=True, use_normals=False, use_materials=False
        )
        print(output_dir)
        print(mesh_dir)
        shutil.copy(os.path.join(mesh_dir, json_path), os.path.join(output_dir, json_path))
        for i in range(n_additional_meshes_per_mesh):
            x_scale = 1 + np.random.uniform(-x_relative_scale_range, x_relative_scale_range)
            z_scale = 1 + np.random.uniform(-z_relative_scale_range, z_relative_scale_range)
            cloth_object.scale[0] = x_scale
            cloth_object.scale[2] = z_scale
            filename = f"{mesh[:-4]}_x_scale_{x_scale:.2f}_z_scale_{z_scale:.2f}.obj"
            bpy.ops.export_scene.obj(
                filepath=os.path.join(output_dir, filename), use_selection=True, use_normals=False, use_materials=False
            )
            shutil.copy(os.path.join(mesh_dir, json_path), os.path.join(output_dir, filename[:-4] + ".json"))

        bpy.ops.object.delete()


if __name__ == "__main__":
    from synthetic_cloth_data import DATA_DIR

    mesh_dir = str(DATA_DIR / "flat_meshes" / "TSHIRT" / "Cloth3D-5-flat")
    output_dir = str(DATA_DIR / "flat_meshes" / "TSHIRT" / "Cloth3D-5-flat-randomized-scale")
    x_relative_scale_range = 0.2
    z_relative_scale_range = 0.2
    n_additional_meshes_per_mesh = 9
    generate_randomized_dataset(
        mesh_dir, output_dir, x_relative_scale_range, z_relative_scale_range, n_additional_meshes_per_mesh
    )
