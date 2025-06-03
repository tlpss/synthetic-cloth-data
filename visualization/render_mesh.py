import bpy


def srgb2lin(s):
    if s <= 0.0404482362771082:
        lin = s / 12.92
    else:
        lin = pow(((s + 0.055) / 1.055), 2.4)
    return lin


colors = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (70, 240, 240),
    (240, 50, 230),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

# blender user the linear color space by default
# but our colors were defined in sRGB space
# so we need to convert them to linear space before using them in blender!
# cf https://blender.stackexchange.com/questions/153048/blender-2-8-python-input-rgb-doesnt-match-hex-color-nor-actual-color
linear_colors = [(srgb2lin(r), srgb2lin(g), srgb2lin(b)) for r, g, b in colors]


# Set Cycles as the renderer
bpy.context.scene.render.engine = "CYCLES"

# Clear existing objects
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()

# Load the mesh
mesh_path = "/home/tlips/Documents/synthetic-cloth-data/visualization/poster/meshes/001502.obj"
# mesh_path = "/home/tlips/Documents/synthetic-cloth-data/visualization/poster/meshes/deformed.obj"
bpy.ops.import_scene.obj(filepath=mesh_path, split_mode="OFF")


mesh = bpy.context.selected_objects[0]
# set as active object
bpy.context.view_layer.objects.active = mesh
bpy.context.view_layer.update()


# add  material
material = bpy.data.materials.new(name="Material")
material.use_nodes = True
principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
principled_bsdf.inputs[0].default_value = (0.2, 0.24, 0.38, 1.0)  # Set the color with full alpha
bpy.context.object.data.materials.append(material)
mesh.active_material = material

# add smooth shading
bpy.ops.object.shade_smooth()

# add solidify modifier
bpy.ops.object.modifier_add(type="SOLIDIFY")
solidify = bpy.context.object.modifiers["Solidify"]
solidify.thickness = 0.005


# Create a top-down camera
bpy.ops.object.camera_add(location=(0, 0, 2))
camera = bpy.context.object
camera.rotation_euler = (0, 0, 0)

# Set camera settings
bpy.context.scene.camera = camera
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.image_settings.file_format = "PNG"

bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = "RGBA"  # Enable transparency
bpy.context.scene.render.filepath = "mesh.png"

# set max rays per sample
bpy.context.scene.cycles.samples = 64
# render on GPU
bpy.context.scene.cycles.device = "GPU"

# remove default light source

bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="LIGHT")
bpy.ops.object.delete()

# Add a new light source, area with size =5, distance 2m and angled towards the origin
bpy.ops.object.light_add(type="AREA", radius=5, location=(0, 1, 2))
light = bpy.context.object
light.data.energy = 50

# add second light source, point light with energy 10
bpy.ops.object.light_add(type="POINT", location=(0, -1, 0.5))
light2 = bpy.context.object
light2.data.energy = 10


# load all keypoints


def visualize_keypoints(blender_object, vertex_id, color):
    radius = 0.02
    bpy.ops.mesh.primitive_ico_sphere_add(
        location=blender_object.matrix_world @ blender_object.data.vertices[vertex_id].co,
        scale=(radius, radius, radius),
        subdivisions=4,
    )

    # add material for the keypoint
    material = bpy.data.materials.new(name="KeypointMaterial")
    material.use_nodes = True
    principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
    principled_bsdf.inputs[0].default_value = color + (1.0,)  # Set the color with full alpha
    bpy.context.object.data.materials.append(material)


keypoints_path = mesh_path.replace(".obj", ".json")

# read the json file
import json

with open(keypoints_path, "r") as f:
    keypoints = json.load(f)
keypoints = keypoints["keypoint_vertices"]

for i, (id, vertex_id) in enumerate(keypoints.items()):
    visualize_keypoints(mesh, int(vertex_id), linear_colors[int(i) % len(linear_colors)])
# Render the image
bpy.ops.render.render(write_still=True)
