import dataclasses
from typing import List

import airo_blender as ab
import bpy
import numpy as np
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import POLYHAVEN_ASSETS_SNAPSHOT_PATH
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig


@dataclasses.dataclass
class HDRIConfig(AssetConfig):
    tags: List[str] = dataclasses.field(default_factory=lambda: ["indoor"])
    types: List[str] = dataclasses.field(default_factory=lambda: ["worlds"])
    asset_json_path: str = POLYHAVEN_ASSETS_SNAPSHOT_PATH


def add_polyhaven_hdri_background_to_scene(config: HDRIConfig):
    """adds a polyhaven HDRI background to the scene."""
    hdri_dict = np.random.choice(config.asset_list)
    world = ab.load_asset(**hdri_dict)
    bpy.context.scene.world = world

    # set Polyhaven HDRI resolution to 4k
    # requires creating manual context override, although this is not documented ofc.
    override = bpy.context.copy()
    override["world"] = bpy.context.scene.world
    with bpy.context.temp_override(**override):
        bpy.ops.pha.resolution_switch(res="4k", asset_id=bpy.context.world.name)
    return world
