import json
import os
import pathlib

import bpy
import tqdm
from synthetic_cloth_data.meshes.cloth_meshes import CLOTH_TYPES, generate_cloth_object


def _unwrap_cloth_mesh(towel_object: bpy.types.Object):
    """unwrap the mesh to create UV coordinates for texture mapping."""

    # activate the object and enter edit mode
    bpy.context.view_layer.objects.active = towel_object
    bpy.ops.object.mode_set(mode="EDIT")

    # unwrap UV for rendering
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)

    # exit edit mode
    bpy.ops.object.mode_set(mode="OBJECT")


def generate_dataset(cloth_type: CLOTH_TYPES, num_samples: int, output_dir: str):
    dir = pathlib.Path(output_dir) / cloth_type.name
    dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm.trange(num_samples):
        blender_object, keypoint_ids = generate_cloth_object(cloth_type)
        _unwrap_cloth_mesh(blender_object)

        filename = f"{i:06d}.obj"
        # select new object and  save as obj file
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = blender_object
        blender_object.select_set(True)
        bpy.ops.export_scene.obj(
            filepath=os.path.join(dir, filename),
            use_selection=True,
            use_materials=False,
            keep_vertex_order=True,  # important for keypoints
            check_existing=False,
            use_uvs=True,  # save UV mappings
        )
        # write keypoints to json file
        with open(os.path.join(dir, filename.replace(".obj", ".json")), "w") as f:
            json.dump(keypoint_ids, f)
        bpy.ops.object.delete()


if __name__ == "__main__":
    import bpy
    import numpy as np
    from synthetic_cloth_data import DATA_DIR

    np.random.seed(2023)
    output_dir = DATA_DIR / "flat_meshes"
    generate_dataset(CLOTH_TYPES.TOWEL, 100, output_dir)
