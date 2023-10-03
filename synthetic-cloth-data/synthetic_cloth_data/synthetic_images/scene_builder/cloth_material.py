import dataclasses
from typing import List

import bpy
import numpy as np
from synthetic_cloth_data.materials.common import (
    FabricMaterialConfig,
    add_fabric_material_to_bsdf,
    create_evenly_colored_material,
)
from synthetic_cloth_data.materials.towels import (
    create_gridded_dish_towel_material,
    create_striped_material,
    modify_bsdf_to_cloth,
)
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


def add_material_to_cloth_mesh(config: ClothMaterialConfig, cloth_object: bpy.types.Object, cloth_type: CLOTH_TYPES):
    if isinstance(config, TowelMaterialConfig):
        _add_towel_material_to_mesh(config, cloth_object)
    elif isinstance(config, HSVMaterialConfig):
        _add_rgb_material_to_mesh(config, cloth_object)


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

    # these are manually tuned to provide appropriate noise levels
    fabric_material_config = FabricMaterialConfig()
    fabric_material_config.high_frequency_distance = np.random.uniform(0.01, 0.4)
    fabric_material_config.low_frequency_noise_distance = np.random.uniform(0.0005, 0.05)
    fabric_material_config.high_frequency_noise_scale = np.random.uniform(30, 150)
    fabric_material_config.low_frequency_noise_scale = np.random.uniform(2, 10)
    fabric_material_config.wave_distance = np.random.uniform(0.0, 0.02)
    fabric_material_config.wave_scale = np.random.uniform(50, 150)
    material = add_fabric_material_to_bsdf(material, fabric_material_config)
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

    # these are manually tuned to provide appropriate noise levels
    fabric_material_config = FabricMaterialConfig()
    fabric_material_config.high_frequency_distance = np.random.uniform(0.01, 0.4)
    fabric_material_config.low_frequency_noise_distance = np.random.uniform(0.0005, 0.05)
    fabric_material_config.high_frequency_noise_scale = np.random.uniform(30, 150)
    fabric_material_config.low_frequency_noise_scale = np.random.uniform(2, 10)
    fabric_material_config.wave_distance = np.random.uniform(0.0, 0.02)
    fabric_material_config.wave_scale = np.random.uniform(50, 150)
    material = add_fabric_material_to_bsdf(material, fabric_material_config)

    cloth_object.data.materials[0] = material
