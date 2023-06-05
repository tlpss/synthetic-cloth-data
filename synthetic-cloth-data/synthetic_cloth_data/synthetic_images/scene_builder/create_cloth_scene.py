from __future__ import annotations

import dataclasses
import json

import bpy
import numpy as np
from synthetic_cloth_data.synthetic_images.scene_builder.annotator import create_coco_annotations
from synthetic_cloth_data.synthetic_images.scene_builder.background import (
    HDRIConfig,
    add_polyhaven_hdri_background_to_scene,
)
from synthetic_cloth_data.synthetic_images.scene_builder.camera import CameraConfig, add_camera
from synthetic_cloth_data.synthetic_images.scene_builder.cloth_material import (
    ClothMaterialConfig,
    TowelMaterialConfig,
    add_material_to_cloth_mesh,
)
from synthetic_cloth_data.synthetic_images.scene_builder.cloth_mesh import ClothMeshConfig, load_cloth_mesh
from synthetic_cloth_data.synthetic_images.scene_builder.distractors import DistractorConfig, add_distractors_to_scene
from synthetic_cloth_data.synthetic_images.scene_builder.renderer import (
    CyclesRendererConfig,
    RendererConfig,
    render_scene,
)
from synthetic_cloth_data.synthetic_images.scene_builder.surface import SurfaceConfig, add_cloth_surface_to_scene
from synthetic_cloth_data.utils import CLOTH_TYPES


@dataclasses.dataclass
class ClothSceneConfig:
    cloth_type: CLOTH_TYPES
    cloth_mesh_config: ClothMeshConfig
    cloth_material_config: ClothMaterialConfig
    camera_config: CameraConfig
    hdri_config: HDRIConfig
    surface_config: SurfaceConfig
    distractor_config: DistractorConfig


def create_cloth_scene(config: ClothSceneConfig):
    bpy.ops.object.delete()
    add_polyhaven_hdri_background_to_scene(config.hdri_config)
    surface = add_cloth_surface_to_scene(config.surface_config)
    cloth_object, keypoint_vertex_ids = load_cloth_mesh(config.cloth_mesh_config)
    cloth_object.pass_index = 1  # TODO: make more generic  # mark for segmentation mask rendering
    add_camera(config.camera_config, cloth_object, keypoint_vertex_ids)
    # TODO: check if all keypoints are visible in the camera view, resample (?) if not.

    add_material_to_cloth_mesh(
        cloth_object=cloth_object, cloth_type=config.cloth_type, config=config.cloth_material_config
    )
    add_distractors_to_scene(config.distractor_config, cloth_object, surface)
    return cloth_object, keypoint_vertex_ids


def create_sample(scene_config: ClothSceneConfig, render_config: RendererConfig, output_dir: str, coco_id: int):
    cloth_object, keypoint_vertex_dict = create_cloth_scene(scene_config)
    render_scene(render_config, output_dir)
    create_coco_annotations(scene_config.cloth_type, output_dir, coco_id, cloth_object, keypoint_vertex_dict)


if __name__ == "__main__":
    import os
    import sys

    from synthetic_cloth_data import DATA_DIR
    from synthetic_cloth_data.synthetic_images.assets.make_assets_snapshots import (
        GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_PATH,
        POLYHAVEN_ASSETS_SNAPSHOT_PATH,
    )

    id = 7
    # check if id was passed as argument
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        id = int(argv[argv.index("--id") + 1])

    # FIX THIS PART WITH HYDRA CONFIG.
    hdri_path = POLYHAVEN_ASSETS_SNAPSHOT_PATH
    distractor_path = GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_PATH
    cloth_mesh_path = DATA_DIR / "deformed_meshes" / "TOWEL"
    dataset_dir = DATA_DIR / "synthetic_images" / "deformed_test"
    cloth_type = CLOTH_TYPES.TOWEL

    output_dir = os.path.join(dataset_dir, f"{id:06d}")
    np.random.seed(2023 + id)

    # load HDRIS
    with open(hdri_path, "r") as file:
        assets = json.load(file)["assets"]
    worlds = [asset for asset in assets if asset["type"] == "worlds" and "indoor" in asset["tags"]]
    materials = [asset for asset in assets if asset["type"] == "materials"]

    # load cloth meshes

    cloth_meshes = os.listdir(cloth_mesh_path)
    cloth_meshes = [cloth_mesh_path / mesh for mesh in cloth_meshes]
    cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]

    # distractor assets
    with open(distractor_path, "r") as file:
        distractor_assets = json.load(file)["assets"]

    config = ClothSceneConfig(
        cloth_type=cloth_type,
        cloth_mesh_config=ClothMeshConfig(
            mesh_dir=cloth_meshes,
        ),
        hdri_config=HDRIConfig(polyhaven_hdri_asset_list=worlds),
        cloth_material_config=TowelMaterialConfig(),  # TODO: must be adapted to cloth type. -> Config.
        camera_config=CameraConfig(),
        surface_config=SurfaceConfig(materials_list=materials),
        distractor_config=DistractorConfig(distractor_list=distractor_assets),
    )
    render_config = CyclesRendererConfig()

    create_sample(config, render_config, output_dir, id)
