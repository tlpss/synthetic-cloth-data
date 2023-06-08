from __future__ import annotations

import dataclasses

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
    renderer_config: RendererConfig
    coco_id: int
    relative_dataset_dir: str


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


def create_sample(scene_config: ClothSceneConfig):

    output_dir = os.path.join(scene_config.relative_dataset_dir, f"{scene_config.coco_id:06d}")
    np.random.seed(2023 + scene_config.coco_id)

    cloth_object, keypoint_vertex_dict = create_cloth_scene(scene_config)
    render_scene(scene_config.renderer_config, output_dir)
    create_coco_annotations(
        scene_config.cloth_type, output_dir, scene_config.coco_id, cloth_object, keypoint_vertex_dict
    )


if __name__ == "__main__":
    import argparse
    import os
    import sys

    import hydra
    from omegaconf import OmegaConf
    from synthetic_cloth_data import DATA_DIR

    # parse arguments in blender compatible way by using -- as separator
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_config", type=str, default="dev")
    parser.add_argument("--hydra", nargs="*", default=[])
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        args = parser.parse_args([])
    config_name = args.hydra_config
    argv = args.hydra

    # initialize hydra
    hydra.initialize(config_path="../configs", job_name="create_cloth_scene")
    cfg = hydra.compose(config_name=config_name, overrides=argv)
    print(f"hydra config: \n{OmegaConf.to_yaml(cfg)}")

    # FIX THIS PART WITH HYDRA CONFIG.
    config = ClothSceneConfig(
        cloth_type=CLOTH_TYPES[cfg["cloth_type"]],
        cloth_mesh_config=hydra.utils.instantiate(cfg["cloth_mesh"]),
        hdri_config=HDRIConfig(),
        cloth_material_config=TowelMaterialConfig(),
        camera_config=hydra.utils.instantiate(cfg["camera"]),
        surface_config=SurfaceConfig(),
        distractor_config=DistractorConfig(),
        renderer_config=CyclesRendererConfig(),
        coco_id=cfg["id"],
        relative_dataset_dir=DATA_DIR / cfg["relative_dataset_path"],
    )
    # print(json.dumps(dataclasses.asdict(config), indent=4))
    # print(config)
    create_sample(config)
