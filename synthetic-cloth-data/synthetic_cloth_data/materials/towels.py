import bpy
from synthetic_cloth_data.materials.common import (
    _add_white_stripes_on_black_nodes,
    create_striped_material,
    modify_bsdf_to_cloth,
)


def create_gridded_dish_towel_material(
    amount_of_vertical_stripes: int,
    amount_of_horizontal_stripes: int,
    vertical_stripe_width: float,
    horizontal_stripe_width: float,
    vertical_stripe_color: tuple[float, float, float, float],
    horizontal_stripe_color: tuple[float, float, float, float],
    intersection_color: tuple[float, float, float, float],
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> bpy.types.Material:
    """Creates a dish towel material with horizontal and vertical stripes.
    Args:
        amount_of_vertical_stripes: The amount of vertical stripes.
        amount_of_horizontal_stripes: The amount of horizontal stripes.
        vertical_stripe_width: The relative width of the vertical stripes.
        horizontal_stripe_width: The relative width of the horizontal stripes.
        vertical_stripe_color: The color of the vertical stripes.
        horizontal_stripe_color: The color of the horizontal stripes.
        intersection_color: The color at the intersections of the vertical and horizontal stripes.
        background_color: The color of the background. Defaults to white.
    Returns:
        bpy.types.Material: The created material.
    """
    material = bpy.data.materials.new(name="Gridded Dish Towel")
    material.use_nodes = True

    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    separate_xyz = nodes.new(type="ShaderNodeSeparateXYZ")

    links.new(texture_coordinates.outputs["UV"], separate_xyz.inputs["Vector"])

    vertical_stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs["X"], amount_of_vertical_stripes, vertical_stripe_width
    )
    horizontal_stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs["Y"], amount_of_horizontal_stripes, horizontal_stripe_width
    )

    # Find where the stripes intersect by multiplying the vertical and horizontal stripes
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(vertical_stripes, multiply.inputs[0])
    links.new(horizontal_stripes, multiply.inputs[1])
    stripes = multiply.outputs["Value"]

    # Now create the colors by using a Mix node
    mix = nodes.new(type="ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(vertical_stripes, mix.inputs[0])
    mix.inputs[6].default_value = background_color
    mix.inputs[7].default_value = vertical_stripe_color

    # Add another Mix node for the horizontal stripes
    mix2 = nodes.new(type="ShaderNodeMix")
    mix2.data_type = "RGBA"
    links.new(horizontal_stripes, mix2.inputs[0])
    links.new(mix.outputs[2], mix2.inputs[6])
    mix2.inputs[7].default_value = horizontal_stripe_color

    # Add a third mix node for the intersection of the stripes
    mix3 = nodes.new(type="ShaderNodeMix")
    mix3.data_type = "RGBA"
    links.new(stripes, mix3.inputs[0])
    links.new(mix2.outputs[2], mix3.inputs[6])
    mix3.inputs[7].default_value = intersection_color
    colored_stripes = mix3.outputs[2]

    links.new(colored_stripes, nodes["Principled BSDF"].inputs["Base Color"])
    return material


if __name__ == "__main__":
    # Delete the default cube and add a plane with the dish towel material
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object
    red = (1.0, 0.0, 0.0, 1.0)
    material = create_striped_material(3, 0.5, red)
    material = modify_bsdf_to_cloth(material)
    plane.data.materials.append(material)

    # Add a second plane with the gridded dish towel material
    bpy.ops.mesh.primitive_plane_add()
    plane_gridded = bpy.context.object
    plane_gridded.location = (2.1, 0.0, 0.0)
    pale_blue = (0.5, 0.5, 1.0, 1.0)
    pale_yellow = (1.0, 1.0, 0.5, 1.0)
    green = (0.0, 1.0, 0.0, 1.0)
    material = create_gridded_dish_towel_material(5, 5, 0.2, 0.2, pale_blue, pale_yellow, green)
    # material = create_evenly_colored_material(green)
    material = modify_bsdf_to_cloth(material)
    plane_gridded.data.materials.append(material)
