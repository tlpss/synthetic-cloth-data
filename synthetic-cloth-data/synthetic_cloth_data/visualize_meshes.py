import json
import pathlib

import bpy
import numpy as np
import tqdm
from airo_blender.materials import add_material
from synthetic_cloth_data.meshes.cloth_meshes import visualize_keypoints


def visualize_meshes(mesh_dir: str, show_keypoints: bool = False, max_amount_to_visualize: int = 100):
    bpy.ops.object.delete()  # Delete default cube
    bpy.ops.mesh.primitive_plane_add(size=12, location=(5, 5, 0))
    bpy.ops.object.modifier_add(type="COLLISION")
    bpy.context.object.collision.cloth_friction = np.random.uniform(5.0, 30.0)
    plane = bpy.context.object
    add_material(plane, (1, 0.5, 0.5, 1.0))

    meshes = list(pathlib.Path(mesh_dir).glob("*.obj"))
    meshes = sorted(meshes, key=lambda x: x.stem)
    print(f"Found {len(meshes)} meshes in {mesh_dir}")
    n_meshes = min(len(meshes), max_amount_to_visualize)
    for idx in tqdm.trange(n_meshes):
        bpy.ops.import_scene.obj(filepath=str(meshes[idx]), split_mode="OFF")
        blender_obj = bpy.context.selected_objects[0]

        blender_obj.location = np.array([idx % 10, idx // 10, 0.01])
        # update the object's world matrix
        # cf. https://blender.stackexchange.com/questions/27667/incorrect-matrix-world-after-transformation
        bpy.context.view_layer.update()
        if show_keypoints:
            keypoint_ids = json.load(open(str(meshes[idx]).replace(".obj", ".json")))["keypoint_vertices"]
            visualize_keypoints(blender_obj, vertex_ids=[int(k) for k in keypoint_ids.values()])


if __name__ == "__main__":
    import sys

    from synthetic_cloth_data import DATA_DIR

    relative_dir = sys.argv[sys.argv.index("--dir") + 1]
    show_keypoints = "--show-keypoints" in sys.argv
    mesh_dir = DATA_DIR / relative_dir
    visualize_meshes(mesh_dir, show_keypoints, max_amount_to_visualize=100)
