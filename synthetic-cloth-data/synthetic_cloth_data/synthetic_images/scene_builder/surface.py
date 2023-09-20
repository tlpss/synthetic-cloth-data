import dataclasses
from typing import List, Tuple

import airo_blender as ab
import bpy
import numpy as np
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import POLYHAVEN_ASSETS_SNAPSHOT_RELATIVE_PATH
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig
from synthetic_cloth_data.synthetic_images.scene_builder.utils.colors import hsv_to_rgb, sample_hsv_color


@dataclasses.dataclass
class PolyhavenMaterials(AssetConfig):
    asset_json_relative_path: str = POLYHAVEN_ASSETS_SNAPSHOT_RELATIVE_PATH
    types: List[str] = dataclasses.field(default_factory=lambda: ["materials"])


@dataclasses.dataclass
class SurfaceConfig:
    size_range: Tuple[float, float] = (1, 3)
    polyhaven_material_probability: float = 0.95
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
        assert isinstance(material, bpy.types.Material)

        # add a color mix node before the principled BSDF color
        # to randomize the base color hue

        # use multiply to limit the change in brightness (which is always an issue with addition)
        # colors should be close to (1,1,1) to avoid darkening the material too much (this is the issue with multiplying..)
        # so set value to 1 and keep saturation low.
        hue = np.random.uniform(0, 1)
        saturation = np.random.uniform(0.0, 0.5)
        value = 1.0
        base_hsv = np.array([hue, saturation, value])
        base_rgb = hsv_to_rgb(base_hsv)

        multiply_node = material.node_tree.nodes.new("ShaderNodeMixRGB")
        multiply_node.blend_type = "MULTIPLY"
        multiply_node.inputs["Fac"].default_value = 1.0
        multiply_node.inputs["Color2"].default_value = (*base_rgb, 1.0)

        # map original input of the BSDF base color to the multiply node
        color_input_node = material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].links[0].from_node
        color_input_node_socket = (
            material.node_tree.nodes["Principled BSDF"]
            .inputs["Base Color"]
            .links[0]
            .from_socket.identifier  # use identifier, names are not unique!
        )
        material.node_tree.links.new(color_input_node.outputs[color_input_node_socket], multiply_node.inputs["Color1"])

        # map the output of the multiply node to the BSDF base color
        material.node_tree.links.new(
            material.node_tree.nodes["Principled BSDF"].inputs["Base Color"], multiply_node.outputs["Color"]
        )

        # disable actual mesh displacements as they change the geometry of the surface
        # and are not used in collision checking, which can cause the cloth to become 'invisible' in renders
        material.cycles.displacement_method = "BUMP"
        plane.data.materials.append(material)
    else:
        base_hsv = sample_hsv_color()
        base_rgb = hsv_to_rgb(base_hsv)
        ab.add_material(plane, color=base_rgb)

    return plane
