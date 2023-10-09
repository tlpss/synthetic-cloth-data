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
    pass


@dataclasses.dataclass
class PolyHavenTexturedSurfaceConfig(SurfaceConfig):
    """adds a 2d plane and applies a random material from the polyhaven snapshots' material snapshot. Also randomizes (mildly) the base color of the material.
    if probability is higher than poly_probability: a random RGB color is used instead."""

    size_range: Tuple[float, float] = (1, 3)
    polyhaven_material_probability: float = 0.95
    polyhaven_materials: PolyhavenMaterials = PolyhavenMaterials()


@dataclasses.dataclass
class HSVTableSurfaceConfig(SurfaceConfig):
    """create a plane and randomizes the color around the specified HSV color range."""

    size_range: Tuple[float, float] = (1, 3)

    h_range: List[float] = dataclasses.field(default_factory=lambda: [0.0, 1.0])
    s_range: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.7])
    v_range: List[float] = dataclasses.field(default_factory=lambda: [1.0, 1.0])


# TODO: should also provide a class for generic 3D assets that can be imported as blender assets.


def add_cloth_surface_to_scene(config: SurfaceConfig) -> bpy.types.Object:
    if isinstance(config, PolyHavenTexturedSurfaceConfig):
        return add_textured_surface_to_scene(config)
    elif isinstance(config, HSVTableSurfaceConfig):
        return add_rgb_surface_to_scene(config)


def _add_plane_to_scene(size: Tuple[float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1)
    # scale the plane to the desired size (cannot do this on creation bc of weir thing in blender API
    # :https://devtalk.blender.org/t/setting-scale-on-primitive-creation/28348 )
    bpy.ops.transform.resize(value=(size[0], size[1], 1))
    plane = bpy.context.object
    return plane


def add_textured_surface_to_scene(config: PolyHavenTexturedSurfaceConfig) -> bpy.types.Object:
    size = np.random.uniform(*config.size_range, size=2)
    plane = _add_plane_to_scene(size)
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
        saturation = np.random.uniform(0.0, 0.7)
        value = 1.0
        base_hsv = np.array([hue, saturation, value])
        base_rgb = hsv_to_rgb(base_hsv)

        multiply_node = material.node_tree.nodes.new("ShaderNodeMixRGB")
        multiply_node.blend_type = "MULTIPLY"
        multiply_node.inputs["Fac"].default_value = 1.0
        multiply_node.inputs["Color2"].default_value = (*base_rgb, 1.0)

        # map original input of the BSDF base color to the multiply node
        # cannot search on "Name" because they can have suffixes like ".001"
        for node in material.node_tree.nodes:
            if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
                break

        bsdf_node = node
        color_input_node = bsdf_node.inputs["Base Color"].links[0].from_node
        color_input_node_socket = (
            bsdf_node.inputs["Base Color"].links[0].from_socket.identifier
        )  # use identifier, names are not unique!
        material.node_tree.links.new(color_input_node.outputs[color_input_node_socket], multiply_node.inputs["Color1"])

        # map the output of the multiply node to the BSDF base color
        material.node_tree.links.new(bsdf_node.inputs["Base Color"], multiply_node.outputs["Color"])

        # disable actual mesh displacements as they change the geometry of the surface
        # and are not used in collision checking, which can cause the cloth to become 'invisible' in renders
        material.cycles.displacement_method = "BUMP"
        plane.data.materials.append(material)
    else:
        base_hsv = sample_hsv_color()
        base_rgb = hsv_to_rgb(base_hsv)
        ab.add_material(plane, color=base_rgb)

    return plane


def add_rgb_surface_to_scene(config: HSVTableSurfaceConfig) -> bpy.types.Object:
    size = np.random.uniform(*config.size_range, size=2)
    plane = _add_plane_to_scene(size)

    hue = np.random.uniform(*config.h_range)
    saturation = np.random.uniform(*config.s_range)
    value = np.random.uniform(*config.v_range)
    hsv = np.array([hue, saturation, value])
    rgb = hsv_to_rgb(hsv)
    print(rgb)
    ab.add_material(plane, color=rgb)
    return plane
