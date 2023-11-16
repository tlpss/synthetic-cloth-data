import dataclasses
from pathlib import Path
from typing import List

import airo_blender as ab
import bpy
import numpy as np
from synthetic_cloth_data.materials.common import (
    FabricMaterialConfig,
    ImageOnTextureConfig,
    add_fabric_material_to_bsdf,
    add_image_to_material_base_color,
    create_evenly_colored_material,
)
from synthetic_cloth_data.materials.towels import (
    create_gridded_dish_towel_material,
    create_striped_material,
    modify_bsdf_to_cloth,
)
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import (
    ASSETS_CODE_PATH,
    POLYHAVEN_ASSETS_SNAPSHOT_RELATIVE_PATH,
)
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig
from synthetic_cloth_data.synthetic_images.scene_builder.utils.colors import hsv_to_rgb, sample_hsv_color
from synthetic_cloth_data.utils import CLOTH_TYPES


@dataclasses.dataclass
class ClothMaterialConfig:
    pass


@dataclasses.dataclass
class TowelMaterialConfig(ClothMaterialConfig):
    uniform_color_probability: float = 0.4  # probability of a uniform color material
    striped_probability: float = 0.3  # probability of a striped material


@dataclasses.dataclass
class HSVMaterialConfig(ClothMaterialConfig):
    h_range: List = dataclasses.field(default_factory=lambda: [0, 1])
    s_range: List = dataclasses.field(default_factory=lambda: [0, 1])
    v_range: List = dataclasses.field(default_factory=lambda: [0.5, 1])
    add_procedural_fabric_texture: bool = True


@dataclasses.dataclass
class TowelStriped001(ClothMaterialConfig):
    pass


@dataclasses.dataclass
class LegoBatteryMaterialConfig(ClothMaterialConfig):
    pass


@dataclasses.dataclass
class TshirtMaterialConfig(ClothMaterialConfig):
    uniform_color_probability: float = 0.8  # probability of a uniform color material
    image_probability: float = 0.1  # probability of a logo image added on top of the material
    image_path: float = ASSETS_CODE_PATH / "coco_images"


@dataclasses.dataclass
class ShortsMaterialConfig(ClothMaterialConfig):
    uniform_color_probability: float = 0.8  # probability of a uniform color material


@dataclasses.dataclass
class PolyhavenMaterials(AssetConfig):
    asset_json_relative_path: str = POLYHAVEN_ASSETS_SNAPSHOT_RELATIVE_PATH
    types: List[str] = dataclasses.field(default_factory=lambda: ["materials"])


@dataclasses.dataclass
class PolyhavenMaterialConfig(ClothMaterialConfig):
    polyhaven_materials: PolyhavenMaterials = PolyhavenMaterials()


def add_material_to_cloth_mesh(config: ClothMaterialConfig, cloth_object: bpy.types.Object, cloth_type: CLOTH_TYPES):
    if isinstance(config, TowelMaterialConfig):
        _add_towel_material_to_mesh(config, cloth_object)

    elif isinstance(config, TshirtMaterialConfig):
        _add_tshirt_material_to_mesh(config, cloth_object)

    elif isinstance(config, ShortsMaterialConfig):
        _add_shorts_material_to_mesh(config, cloth_object)
    elif isinstance(config, HSVMaterialConfig):
        _add_rgb_material_to_mesh(config, cloth_object)

    # TODO: this needs to become more generic.
    elif isinstance(config, TowelStriped001):
        _001_striped_towel(cloth_object)

    elif isinstance(config, PolyhavenMaterialConfig):
        _add_polyhaven_material_to_object(config, cloth_object)

    elif isinstance(config, LegoBatteryMaterialConfig):
        pass


def _add_towel_material_to_mesh(config: TowelMaterialConfig, cloth_object: bpy.types.Object):
    material_sample = np.random.rand()

    if material_sample < config.uniform_color_probability:
        hsv = sample_hsv_color()
        rgb = hsv_to_rgb(hsv)
        rgba = np.concatenate([rgb, [1]])
        material = create_evenly_colored_material(rgba)

    elif material_sample < config.uniform_color_probability + config.striped_probability:
        amount_of_stripes = np.random.randint(2, 20)
        relative_stripe_width = np.random.uniform(0.1, 0.5)
        stripe_color = hsv_to_rgb(sample_hsv_color())
        background_color = hsv_to_rgb(sample_hsv_color())
        vertical_orientation = np.random.rand() < 0.5

        # rgb to rgba
        stripe_color = np.array([*stripe_color, 1])
        background_color = np.array([*background_color, 1])
        material = create_striped_material(
            amount_of_stripes, relative_stripe_width, stripe_color, background_color, vertical_orientation
        )
    else:

        background_color = hsv_to_rgb(sample_hsv_color())
        vertical_color = hsv_to_rgb(sample_hsv_color())
        horizontal_color = hsv_to_rgb(sample_hsv_color())
        intersection_color = hsv_to_rgb(sample_hsv_color())

        # rgb to rgba
        background_color = np.array([*background_color, 1])
        vertical_color = np.array([*vertical_color, 1])
        horizontal_color = np.array([*horizontal_color, 1])
        intersection_color = np.array([*intersection_color, 1])

        n_vertical_stripes = np.random.randint(2, 20)
        n_horizontal_stripes = np.random.randint(2, 20)
        vertical_stripe_relative_width = np.random.uniform(0.05, 0.5)
        horizontal_stripe_relative_width = np.random.uniform(0.05, 0.5)

        material = create_gridded_dish_towel_material(
            n_vertical_stripes,
            n_horizontal_stripes,
            vertical_stripe_relative_width,
            horizontal_stripe_relative_width,
            vertical_color,
            horizontal_color,
            intersection_color,
        )

    material = modify_bsdf_to_cloth(material)
    if config.add_procedural_fabric_texture:
        material = _add_procedural_fabric_texture_to_bsdf(material)
    cloth_object.data.materials[0] = material


def _add_tshirt_material_to_mesh(config: TshirtMaterialConfig, cloth_object: bpy.types.Object):
    if np.random.rand() < config.uniform_color_probability:
        hsv = sample_hsv_color()
        rgb = hsv_to_rgb(hsv)
        rgba = np.concatenate([rgb, [1]])
        material = create_evenly_colored_material(rgba)
    else:
        # create striped material
        amount_of_stripes = np.random.randint(2, 20)
        relative_stripe_width = np.random.uniform(0.1, 0.5)
        stripe_color = hsv_to_rgb(sample_hsv_color())
        background_color = hsv_to_rgb(sample_hsv_color())
        vertical_orientation = np.random.rand() < 0.5
        stripe_color = np.array([*stripe_color, 1])
        background_color = np.array([*background_color, 1])
        material = create_striped_material(
            amount_of_stripes, relative_stripe_width, stripe_color, background_color, vertical_orientation
        )
    if np.random.rand() < config.image_probability:
        # add image on top of material
        x_width = np.random.uniform(0.02, 0.2)
        y_width = np.random.uniform(0.01, 0.1)
        x_center = np.random.uniform(0.0, 1.0)
        y_center = np.random.uniform(0.0, 0.5)  # uv maps only on bottom half of the (0,0) to (1,1) square
        image_scale = np.random.uniform(2.0, 20.0)
        image_config = ImageOnTextureConfig(
            uv_x_position=x_center,
            uv_y_position=y_center,
            uv_x_width=x_width,
            uv_y_width=y_width,
            image_x_scale=image_scale,
            image_y_scale=image_scale,
        )
        image_dir = config.image_path
        image_dir = Path(image_dir)
        images = list(image_dir.glob("*.jpg"))
        images.extend(list(image_dir.glob("*.png")))
        print(images)
        image_path = np.random.choice(images)
        add_image_to_material_base_color(material, str(image_path), image_config)

    material = modify_bsdf_to_cloth(material)
    material = _add_procedural_fabric_texture_to_bsdf(material)
    cloth_object.data.materials[0] = material


def _add_shorts_material_to_mesh(config: ShortsMaterialConfig, cloth_object: bpy.types.Object):
    if np.random.rand() < config.uniform_color_probability:
        hsv = sample_hsv_color()
        rgb = hsv_to_rgb(hsv)
        rgba = np.concatenate([rgb, [1]])
        material = create_evenly_colored_material(rgba)
    else:
        # create striped material
        amount_of_stripes = np.random.randint(2, 20)
        relative_stripe_width = np.random.uniform(0.1, 0.5)
        stripe_color = hsv_to_rgb(sample_hsv_color())
        background_color = hsv_to_rgb(sample_hsv_color())
        vertical_orientation = np.random.rand() < 0.5
        stripe_color = np.array([*stripe_color, 1])
        background_color = np.array([*background_color, 1])
        material = create_striped_material(
            amount_of_stripes, relative_stripe_width, stripe_color, background_color, vertical_orientation
        )

    material = modify_bsdf_to_cloth(material)
    material = _add_procedural_fabric_texture_to_bsdf(material)
    cloth_object.data.materials[0] = material


def _add_rgb_material_to_mesh(config: HSVMaterialConfig, cloth_object: bpy.types.Object):
    h = np.random.uniform(config.h_range[0], config.h_range[1])
    s = np.random.uniform(config.s_range[0], config.s_range[1])
    v = np.random.uniform(config.v_range[0], config.v_range[1])
    hsv = np.array([h, s, v])
    rgb = hsv_to_rgb(hsv)
    rgba = np.concatenate([rgb, [1]])
    material = create_evenly_colored_material(rgba)
    material = modify_bsdf_to_cloth(material)
    material = _add_procedural_fabric_texture_to_bsdf(material)

    cloth_object.data.materials[0] = material


def _log_uniform(low, high):
    # check for zero values
    assert low >= 0
    assert high > 0
    if low < 1e-6:
        low = 1e-6
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def _001_striped_towel(cloth_object):
    amount_of_stripes = 18
    relative_stripe_width = 0.1
    stripe_color = np.array([0.54, 0.19, 0.02])
    stripe_color = hsv_to_rgb(stripe_color)
    background_color = np.array([0.05, 0.05, 0.9])
    background_color = hsv_to_rgb(background_color)
    vertical_orientation = 0

    # rgb to rgba
    stripe_color = np.array([*stripe_color, 1])
    background_color = np.array([*background_color, 1])
    material = create_striped_material(
        amount_of_stripes, relative_stripe_width, stripe_color, background_color, vertical_orientation
    )
    material = modify_bsdf_to_cloth(material)
    material = _add_procedural_fabric_texture_to_bsdf(material)

    cloth_object.data.materials[0] = material


def _add_procedural_fabric_texture_to_bsdf(material):
    # these are manually tuned to provide appropriate noise levels
    fabric_material_config = FabricMaterialConfig()

    fabric_material_config.high_frequency_distance = _log_uniform(0.001, 0.008)
    fabric_material_config.low_frequency_noise_distance = _log_uniform(0.001, 0.05)
    fabric_material_config.high_frequency_noise_scale = np.random.uniform(30, 150)
    fabric_material_config.low_frequency_noise_scale = np.random.uniform(2, 10)
    fabric_material_config.wave_distance = _log_uniform(0.0, 0.05)
    fabric_material_config.wave_scale = np.random.uniform(50, 150)
    material = add_fabric_material_to_bsdf(material, fabric_material_config)
    return material


def _add_polyhaven_material_to_object(config: PolyhavenMaterialConfig, object: bpy.types.Object):

    num_object_materials = len(object.material_slots)
    for material_idx in range(num_object_materials):

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

        material.cycles.displacement_method = "BUMP"
        object.data.materials[material_idx] = material
