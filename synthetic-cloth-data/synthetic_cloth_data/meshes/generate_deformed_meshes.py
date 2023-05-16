import json
import os
import pathlib

import tqdm
from synthetic_cloth_data.meshes.cloth_meshes import CLOTH_TYPES
from synthetic_cloth_data.meshes.deformed_towel import generate_random_deformed_towel


def generate_dataset(cloth_type: CLOTH_TYPES, num_samples: int, output_dir: str):
    dir = pathlib.Path(output_dir) / cloth_type.name
    dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm.trange(num_samples):
        blender_object, keypoint_ids = generate_random_deformed_towel(i)

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
    from synthetic_cloth_data import DATA_DIR

    output_dir = DATA_DIR / "deformed_meshes"
    generate_dataset(CLOTH_TYPES.TOWEL, 25, output_dir)
