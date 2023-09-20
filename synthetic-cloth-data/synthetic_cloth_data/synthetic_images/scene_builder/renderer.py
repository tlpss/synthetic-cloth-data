import dataclasses
import os

import bpy
import numpy as np


@dataclasses.dataclass
class RendererConfig:
    exposure_min: float = -1.0
    exposure_max: float = 1.0
    device: str = "GPU"


class CyclesRendererConfig(RendererConfig):
    num_samples: int = 32


def render_scene(render_config: RendererConfig, output_dir: str):
    scene = bpy.context.scene

    if isinstance(render_config, CyclesRendererConfig):
        scene.render.engine = "CYCLES"
        scene.cycles.samples = render_config.num_samples
        scene.cycles.device = render_config.device
    else:
        raise NotImplementedError(f"Renderer config {render_config} not implemented")

    scene.view_settings.exposure = np.random.uniform(render_config.exposure_min, render_config.exposure_max)
    scene.view_settings.gamma = 1.0

    # Make a directory to organize all the outputs
    os.makedirs(output_dir, exist_ok=True)

    scene.view_layers["ViewLayer"].use_pass_object_index = True
    scene.use_nodes = True

    image_name = "rgb"
    # Add a file output node to the scene
    tree = scene.node_tree
    links = tree.links
    nodes = tree.nodes
    node = nodes.new("CompositorNodeOutputFile")
    node.location = (500, 200)
    node.base_path = output_dir
    slot_image = node.file_slots["Image"]
    slot_image.path = "rgb"
    slot_image.format.color_mode = "RGB"

    # Prevent the 0001 suffix from being added to the file name

    segmentation_name = "segmentation"
    node.file_slots.new(segmentation_name)
    slot_segmentation = node.file_slots[segmentation_name]

    # slot_segmentation.path = f"{random_seed:08d}_segmentation"
    slot_segmentation.format.color_mode = "BW"
    slot_segmentation.use_node_format = False
    slot_segmentation.save_as_render = False

    render_layers_node = nodes["Render Layers"]
    links.new(render_layers_node.outputs["Image"], node.inputs[0])

    # Other method, use the mask ID node
    mask_id_node = nodes.new("CompositorNodeIDMask")
    mask_id_node.index = 1  # TODO: make this configurable, instead of hardcoding cloth ID
    mask_id_node.location = (300, 200)
    links.new(render_layers_node.outputs["IndexOB"], mask_id_node.inputs[0])
    links.new(mask_id_node.outputs[0], node.inputs[slot_segmentation.path])

    # Rendering the scene into an image
    bpy.ops.render.render(animation=False)

    # Annoying fix, because Blender adds a 0001 suffix to the file name which can't be disabled
    image_path = os.path.join(output_dir, f"{image_name}0001.png")
    image_path_new = os.path.join(output_dir, f"{image_name}.png")
    os.rename(image_path, image_path_new)

    segmentation_path = os.path.join(output_dir, f"{segmentation_name}0001.png")
    segmentation_path_new = os.path.join(output_dir, f"{segmentation_name}.png")
    os.rename(segmentation_path, segmentation_path_new)
