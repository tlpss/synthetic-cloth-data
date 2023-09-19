import hashlib
import json
import os
import pathlib

import bpy
import tqdm
from synthetic_cloth_data.meshes.flat_meshes.cloth_meshes import CLOTH_TYPES, generate_cloth_object
from synthetic_cloth_data.meshes.utils.projected_mesh_area import get_mesh_projected_xy_area
from synthetic_cloth_data.utils import get_metadata_dict_for_dataset


def _unwrap_cloth_mesh(towel_object: bpy.types.Object):
    """unwrap the mesh to create UV coordinates for texture mapping."""

    # activate the object and enter edit mode
    bpy.context.view_layer.objects.active = towel_object
    bpy.ops.object.mode_set(mode="EDIT")

    # unwrap UV for rendering
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)

    # exit edit mode
    bpy.ops.object.mode_set(mode="OBJECT")


def generate_dataset(cloth_type: CLOTH_TYPES, num_samples: int, output_dir: str, dataset_tag: str = "dev"):
    dir = pathlib.Path(output_dir) / cloth_type.name / dataset_tag
    dir.mkdir(parents=True, exist_ok=True)

    # write metadata
    metadata = {
        "num_samples": num_samples,
        "cloth_type": cloth_type.name,
        "info": "<info about mesh generation",
    }
    metadata.update(get_metadata_dict_for_dataset())
    with open(os.path.join(dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

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
            use_normals=False,  # don't save because these will change during deformations.
            check_existing=False,
            use_uvs=True,  # save UV mappings
        )

        # save data to json file
        data = {
            "keypoint_vertices": keypoint_ids,
            "area": get_mesh_projected_xy_area(os.path.join(dir, filename)),
            "obj_md5_hash": hashlib.md5(open(os.path.join(dir, filename), "rb").read()).hexdigest(),
        }
        with open(os.path.join(dir, filename.replace(".obj", ".json")), "w") as f:
            json.dump(data, f)
        bpy.ops.object.delete()


if __name__ == "__main__":
    import argparse
    import sys

    import bpy
    import numpy as np
    from synthetic_cloth_data import DATA_DIR

    np.random.seed(2023)
    output_dir = DATA_DIR / "flat_meshes"

    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth_type", type=CLOTH_TYPES, default="TOWEL")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--dataset_tag", type=str, default="dev")
    args = parser.parse_args(argv)
    generate_dataset(CLOTH_TYPES(args.cloth_type), args.num_samples, output_dir, args.dataset_tag)
