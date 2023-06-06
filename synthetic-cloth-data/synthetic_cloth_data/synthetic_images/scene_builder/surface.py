import dataclasses
from typing import List, Tuple

import airo_blender as ab
import bpy
import numpy as np
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import POLYHAVEN_ASSETS_SNAPSHOT_PATH
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig
from synthetic_cloth_data.synthetic_images.scene_builder.utils.colors import hsv_to_rgb, sample_hsv_color


@dataclasses.dataclass
class PolyhavenMaterials(AssetConfig):
    asset_json_path: str = POLYHAVEN_ASSETS_SNAPSHOT_PATH
    types: List[str] = dataclasses.field(default_factory=lambda: ["materials"])


@dataclasses.dataclass
class SurfaceConfig:
    size_range: Tuple[float, float] = (1, 3)
    polyhaven_material_probability: float = 0.5
    polyhaven_materials: PolyhavenMaterials = PolyhavenMaterials()


def add_cloth_surface_to_scene(config: SurfaceConfig) -> bpy.types.Object:
    size = np.random.uniform(*config.size_range, size=2)
    bpy.ops.mesh.primitive_plane_add(size=1)
    # scale the plane to the desired size (cannot do this on creation bc of weir thing in blender API
    # :https://devtalk.blender.org/t/setting-scale-on-primitive-creation/28348 )
    bpy.ops.transform.resize(value=(size[0], size[1], 1))
    plane = bpy.context.object

    if np.random.rand() < config.polyhaven_material_probability and len(config.polyhaven_materials.asset_list) > 0:
        material_dict = np.random.choice(config.polyhaven_materials.asset_list)
        material = ab.load_asset(**material_dict)

        # disable actual mesh displacements as they change the geometry of the surface
        # and are not used in collision checking, which can cause the cloth to become 'invisible' in renders
        material.cycles.displacement_method = "BUMP"
        plane.data.materials.append(material)
    else:
        hsv = sample_hsv_color()
        rgb = hsv_to_rgb(hsv)
        ab.add_material(plane, color=rgb)

    return plane
