import dataclasses

import bpy
from synthetic_cloth_data.materials import RGBAColor


def create_evenly_colored_material(color: RGBAColor) -> bpy.types.Material:
    material = bpy.data.materials.new(name="Simple Colored Material")
    material.use_nodes = True
    # create color node
    color_node = material.node_tree.nodes.new(type="ShaderNodeRGB")
    color_node.outputs["Color"].default_value = color
    bsdf_node = material.node_tree.nodes["Principled BSDF"]
    material.node_tree.links.new(color_node.outputs[0], bsdf_node.inputs["Base Color"])

    return material


@dataclasses.dataclass
class ImageOnTextureConfig:
    uv_x_position: float = 0.5
    uv_y_position: float = 0.5
    uv_x_width: float = 0.2
    uv_y_width: float = 0.2
    image_rotation_on_uv: float = 0.0
    image_x_scale: float = 1.0
    image_y_scale: float = 0.5


def add_image_to_material_base_color(
    material: bpy.types.Material, image_path: str, config: ImageOnTextureConfig
) -> bpy.types.Material:
    # find the node and socket that are connected to the base color input of the principled bsdf

    color_input_node = material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].links[0].from_node
    color_input_node_socket = (
        material.node_tree.nodes["Principled BSDF"]
        .inputs["Base Color"]
        .links[0]
        .from_socket.identifier  # use identifier, names are not unique!
    )

    # create nodes
    image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    # load the image and attach to the blender node
    image = bpy.data.images.load(image_path)
    image_node.image = image

    tex_coord_node = material.node_tree.nodes.new("ShaderNodeTexCoord")
    image_uv_mapping_node = material.node_tree.nodes.new("ShaderNodeMapping")
    # set size of the image
    image_uv_mapping_node.inputs["Scale"].default_value = [config.image_x_scale, config.image_y_scale, 1.0]
    separation_uv_mapping_node1 = material.node_tree.nodes.new("ShaderNodeMapping")
    # create frame  to map center of the image on
    separation_uv_mapping_node1.inputs["Location"].default_value = [-config.uv_x_position, -config.uv_y_position, 0.0]
    separation_uv_mapping_node2 = material.node_tree.nodes.new("ShaderNodeMapping")
    # rotate locally
    separation_uv_mapping_node2.inputs["Rotation"].default_value = [0.0, 0.0, config.image_rotation_on_uv]
    separate_node = material.node_tree.nodes.new("ShaderNodeSeparateXYZ")
    multiply_node = material.node_tree.nodes.new("ShaderNodeMath")
    multiply_node.operation = "MULTIPLY"
    compare_x_node = material.node_tree.nodes.new("ShaderNodeMath")
    compare_x_node.operation = "COMPARE"
    compare_x_node.inputs[2].default_value = config.uv_x_width
    compare_x_node.inputs[1].default_value = 0.0  # already in local frame
    compare_y_node = material.node_tree.nodes.new("ShaderNodeMath")
    compare_y_node.inputs[2].default_value = config.uv_y_width
    compare_y_node.inputs[1].default_value = 0.0  # already in local frame
    compare_y_node.operation = "COMPARE"

    mix_node = material.node_tree.nodes.new("ShaderNodeMixRGB")

    # links
    material.node_tree.links.new(tex_coord_node.outputs["UV"], image_uv_mapping_node.inputs["Vector"])
    material.node_tree.links.new(tex_coord_node.outputs["UV"], separation_uv_mapping_node1.inputs["Vector"])
    material.node_tree.links.new(
        separation_uv_mapping_node1.outputs["Vector"], separation_uv_mapping_node2.inputs["Vector"]
    )
    material.node_tree.links.new(separation_uv_mapping_node2.outputs["Vector"], separate_node.inputs["Vector"])
    material.node_tree.links.new(image_uv_mapping_node.outputs["Vector"], image_node.inputs["Vector"])
    material.node_tree.links.new(separate_node.outputs["X"], compare_x_node.inputs[0])
    material.node_tree.links.new(separate_node.outputs["Y"], compare_y_node.inputs[0])
    material.node_tree.links.new(compare_x_node.outputs["Value"], multiply_node.inputs[0])
    material.node_tree.links.new(compare_y_node.outputs["Value"], multiply_node.inputs[1])
    material.node_tree.links.new(multiply_node.outputs["Value"], mix_node.inputs["Fac"])
    material.node_tree.links.new(image_node.outputs["Color"], mix_node.inputs["Color2"])
    material.node_tree.links.new(color_input_node.outputs[color_input_node_socket], mix_node.inputs["Color1"])

    material.node_tree.links.new(
        mix_node.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
    )

    return material


def modify_bsdf_to_cloth(material: bpy.types.Material) -> bpy.types.Material:
    bsdf_node = material.node_tree.nodes["Principled BSDF"]

    # Sheen was made for a cloth looks, and dish towel fabric is generally not shiny at all.
    bsdf_node.inputs["Sheen"].default_value = 1.0
    bsdf_node.inputs["Roughness"].default_value = 1.0

    return material


def _add_white_stripes_on_black_nodes(
    node_tree: bpy.types.ShaderNodeTree,
    input_socket: bpy.types.NodeSocket,
    amount_of_stripes: int,
    stripe_width: float,
) -> bpy.types.NodeSocket:
    """
    Add nodes to a node tree to create a pattern with white stripes on a black background.
    Args:
        node_tree: The matieral node tree to add the nodes to.
        input_socket: This input should be a linear 0->1 range, e.g. the X component of a UV coordinate.
        amount_of_stripes: The amount of white stripes on the black background.
        stripe_width: The relative width of the white stripes, in range 0.0 to 1.0. 0.0 is solid black, 1.0 is solid white.
    Returns:
        The Color output socket that contains the striped pattern.
    """
    nodes = node_tree.nodes
    links = node_tree.links

    # Add nodes
    # Math node set to multiply by 10
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(input_socket, multiply.inputs[0])
    multiply.inputs[1].default_value = amount_of_stripes

    # Math node set to fraction
    fraction = nodes.new(type="ShaderNodeMath")
    fraction.operation = "FRACT"
    links.new(multiply.outputs["Value"], fraction.inputs[0])

    # Math node set to compare to 0.5, with epsilon in range 0.0 to 0.5 to control the stripe width
    compare = nodes.new(type="ShaderNodeMath")
    compare.operation = "COMPARE"
    links.new(fraction.outputs["Value"], compare.inputs[0])
    compare.inputs[1].default_value = 0.5
    compare.inputs[2].default_value = 0.5 * stripe_width
    output_socket = compare.outputs["Value"]

    return output_socket


def create_striped_material(
    amount_of_stripes: int,
    stripe_width: float,
    stripe_color: tuple[float, float, float, float],
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    vertical: bool = True,
) -> bpy.types.Material:
    """
    Create a striped dish towel material.
    Args:
        amount_of_stripes: The amount of stripes.
        stripe_width: The relative width of the stripes, 0.0 means only background color, 1.0 means only stripe color.
        stripe_color: The color of the stripes.
        background_color: The color of the background. Defaults to white.
        vertical: If True, the stripes run vertically, else horizontally.
    Returns:
        The created material.
    """
    material = bpy.data.materials.new(name="Striped Dish Towel")
    material.use_nodes = True

    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # First set up the texture coordinate node for access to the UVs
    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    separate_xyz = nodes.new(type="ShaderNodeSeparateXYZ")

    # Connect the texture coordinate node to the separate XYZ node
    links.new(texture_coordinates.outputs["UV"], separate_xyz.inputs["Vector"])

    output_name = "X" if vertical else "Y"
    stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs[output_name], amount_of_stripes, stripe_width
    )

    # There are several ways to turn a black and white pattern into a colored pattern.
    # Here we use a Mix node to mix the stripe color with the background color.
    # TODO: consider a better way of specifying inputs and outputs than using indices
    # We can't use the string names because they are not unique.
    mix = nodes.new(type="ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(stripes, mix.inputs[0])
    mix.inputs[6].default_value = background_color
    mix.inputs[7].default_value = stripe_color
    colored_stripes = mix.outputs[2]

    links.new(colored_stripes, nodes["Principled BSDF"].inputs["Base Color"])
    return material
