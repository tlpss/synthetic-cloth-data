import json
import os
import pathlib

from synthetic_cloth_data.meshes.cloth_meshes import CLOTH_TYPES, generate_cloth_object


def generate_dataset(cloth_type: CLOTH_TYPES, num_samples: int, output_dir: str):
    dir = pathlib.Path(output_dir) / cloth_type.name
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        blender_object, keypoint_ids = generate_cloth_object(cloth_type)

        filename = f"{i:06d}.obj"
        # select new object and  save as obj file
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = blender_object
        blender_object.select_set(True)
        bpy.ops.export_scene.obj(
            filepath=os.path.join(dir, filename),
            use_selection=True,
            use_materials=False,
            keep_vertex_order=True,
            check_existing=False,
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
    generate_dataset(CLOTH_TYPES.TSHIRT, 10, output_dir)
