from __future__ import annotations

import dataclasses
import json
from typing import List, Tuple

import airo_blender as ab
import bpy
import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointAnnotation
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from synthetic_cloth_data.materials.common import create_evenly_colored_material, modify_bsdf_to_cloth
from synthetic_cloth_data.materials.towels import create_gridded_dish_towel_material, create_striped_material
from synthetic_cloth_data.utils import CLOTH_TYPE_TO_COCO_CATEGORY_ID, CLOTH_TYPES


@dataclasses.dataclass
class ClothSceneConfig:
    cloth_type: CLOTH_TYPES
    cloth_mesh_config: ClothMeshConfig
    cloth_material_config: ClothMaterialConfig
    camera_config: CameraConfig
    hdri_config: HDRIConfig
    surface_config: SurfaceConfig


def create_cloth_scene(config: ClothSceneConfig):
    bpy.ops.object.delete()
    add_hdri_background(config.hdri_config)
    create_surface(config.surface_config)
    cloth_object, keypoint_vertex_ids = load_cloth_mesh(config.cloth_mesh_config)
    cloth_object.pass_index = 1  # mark for segmentation mask rendering
    add_camera(config.camera_config, cloth_object, keypoint_vertex_ids)
    # TODO: check if all keypoints are visible in the camera view, resample (?) if not.
    add_material_to_cloth_mesh(
        cloth_object=cloth_object, cloth_type=config.cloth_type, config=config.cloth_material_config
    )
    return cloth_object, keypoint_vertex_ids


@dataclasses.dataclass
class HDRIConfig:
    hdri_asset_list: List[dict]  # blender HDRI assets as exported


def add_hdri_background(config: HDRIConfig):
    hdri_dict = np.random.choice(config.hdri_asset_list)
    world = ab.load_asset(**hdri_dict)
    bpy.context.scene.world = world

    # set Polyhaven HDRI resolution to 4k
    # requires creating manual context override, although this is not documented ofc.
    override = bpy.context.copy()
    override["world"] = bpy.context.scene.world
    with bpy.context.temp_override(**override):
        bpy.ops.pha.resolution_switch(res="4k", asset_id=bpy.context.world.name)


@dataclasses.dataclass
class SurfaceConfig:
    size_range: Tuple[float, float] = (1, 3)
    materials_list: List[dict] = dataclasses.field(default_factory=list)
    polyhaven_material_probability: float = 0.5


def _sample_hsv_color():
    hue = np.random.uniform(0, 180)
    saturation = np.random.uniform(0.0, 1)
    value = np.random.uniform(0.0, 1)
    return np.array([hue, saturation, value])


def _hsv_to_rgb(hsv: np.ndarray):
    assert hsv.shape == (3,)
    hsv = hsv.astype(np.float32)
    rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, ...], cv2.COLOR_HSV2RGB)
    return rgb[0][0]


def create_surface(config: SurfaceConfig) -> bpy.types.Object:
    size = np.random.uniform(*config.size_range, size=2)
    bpy.ops.mesh.primitive_plane_add(size=1)
    # scale the plane to the desired size (cannot do this on creation bc of weir thing in blender API
    # :https://devtalk.blender.org/t/setting-scale-on-primitive-creation/28348 )
    bpy.ops.transform.resize(value=(size[0], size[1], 1))
    plane = bpy.context.object

    if np.random.rand() < config.polyhaven_material_probability and len(config.materials_list) > 0:
        material_dict = np.random.choice(config.materials_list)
        material = ab.load_asset(**material_dict)

        # disable actual mesh displacements as they change the geometry of the surface
        # and are not used in collision checking, which can cause the cloth to become 'invisible' in renders
        material.cycles.displacement_method = "BUMP"
        plane.data.materials.append(material)
    else:
        hsv = _sample_hsv_color()
        rgb = _hsv_to_rgb(hsv)
        ab.add_material(plane, color=rgb)

    return plane


@dataclasses.dataclass
class CameraConfig:
    # intrinsics
    focal_length: int = 20  # ZED2i focal length
    horizontal_resolution: int = 512
    vertical_resolution: int = 288  # 16:9 aspect ratio
    horizontal_sensor_size = 38  # ZED2i horizontal sensor size

    # extrinsics
    minimal_camera_height: float = 0.7
    max_sphere_radius: float = 1.8


def add_camera(config: CameraConfig, cloth_object: bpy.types.Object, keypoint_vertices_dict: dict) -> bpy.types.Object:
    camera = bpy.data.objects["Camera"]

    # Set the camera intrinsics
    # cf https://docs.blender.org/manual/en/latest/render/cameras.html for more info.
    camera.data.lens = config.focal_length
    camera.data.sensor_width = config.horizontal_sensor_size
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.type = "PERSP"

    image_width, image_height = config.horizontal_resolution, config.vertical_resolution
    scene = bpy.context.scene
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height

    # TODO: randomize camera parameters?

    def _sample_point_on_unit_sphere(z_min: float) -> np.ndarray:
        """sample a point on the unit sphere, with z coordinate >= z_min, and uniform distribution of the height z in that range"""
        z = np.random.uniform(z_min, 1)
        phi = np.random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        point_on_unit_sphere = np.array([x, y, z])
        return point_on_unit_sphere

    camera_placed = False
    while not camera_placed:
        camera.location = _sample_point_on_unit_sphere(z_min=config.minimal_camera_height) * np.random.uniform(
            1, config.max_sphere_radius
        )
        # Make the camera look at tthe origin, around which the cloth and table are assumed to be centered.
        camera_direction = -camera.location
        camera_direction = Vector(camera_direction)
        camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

        bpy.context.view_layer.update()  # update the scene to propagate the new camera location & orientation
        camera_placed = _are_keypoints_in_camera_frustum(cloth_object, keypoint_vertices_dict, camera)
        camera_placed = True

    return camera


@dataclasses.dataclass
class ClothMaterialConfig:
    pass


class TowelMaterialConfig(ClothMaterialConfig):
    uniform_color_probability: float = 0.4  # probability of a uniform color material
    striped_probability: float = 0.3  # probability of a striped material


def add_material_to_cloth_mesh(config: ClothMaterialConfig, cloth_object: bpy.types.Object, cloth_type: CLOTH_TYPES):
    if cloth_type == CLOTH_TYPES.TSHIRT:
        pass
    elif cloth_type == CLOTH_TYPES.TOWEL:
        _add_material_to_towel_mesh(config, cloth_object)




def _add_material_to_towel_mesh(config: TowelMaterialConfig, cloth_object: bpy.types.Object):

    material_sample = np.random.rand()

    if material_sample < config.uniform_color_probability:
        hsv = _sample_hsv_color()
        rgb = _hsv_to_rgb(hsv)
        rgba = np.concatenate([rgb, [1]])
        material = create_evenly_colored_material(rgba)

    elif material_sample < config.uniform_color_probability + config.striped_probability:
        amount_of_stripes = np.random.randint(1, 8)
        relative_stripe_width = np.random.uniform(0.1, 0.5)
        stripe_color = _hsv_to_rgb(_sample_hsv_color())
        background_color = _hsv_to_rgb(_sample_hsv_color())
        vertical_orientation = np.random.rand() < 0.5

        # rgb to rgba
        stripe_color = np.array([*stripe_color, 1])
        background_color = np.array([*background_color, 1])
        material = create_striped_material(
            amount_of_stripes, relative_stripe_width, stripe_color, background_color, vertical_orientation
        )
    else:

        background_color = _hsv_to_rgb(_sample_hsv_color())
        vertical_color = _hsv_to_rgb(_sample_hsv_color())
        horizontal_color = _hsv_to_rgb(_sample_hsv_color())
        intersection_color = _hsv_to_rgb(_sample_hsv_color())

        # rgb to rgba
        background_color = np.array([*background_color, 1])
        vertical_color = np.array([*vertical_color, 1])
        horizontal_color = np.array([*horizontal_color, 1])
        intersection_color = np.array([*intersection_color, 1])

        n_vertical_stripes = np.random.randint(1, 8)
        n_horizontal_stripes = np.random.randint(1, 8)
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
    cloth_object.data.materials[0] = material


@dataclasses.dataclass
class ClothMeshConfig:
    mesh_dir: List[str]
    xy_randomization_range: float = 0.1


def load_cloth_mesh(config: ClothMeshConfig):
    # load the obj
    mesh_file = str(np.random.choice(config.mesh_dir))
    bpy.ops.import_scene.obj(filepath=mesh_file, split_mode="OFF")  # keep vertex order with split_mode="OFF"
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    xy_position = np.random.uniform(-config.xy_randomization_range, config.xy_randomization_range, size=2)
    cloth_object.location[0] = xy_position[0]
    cloth_object.location[1] = xy_position[1]

    cloth_object.location[2] = 0.001  # make sure the cloth is above the surface

    cloth_object.rotation_euler[2] = np.random.rand() * 2 * np.pi

    # convention is to have the keypoint vertex ids in a json file with the same name as the obj file
    keypoint_vertex_dict = json.load(open(str(mesh_file).replace(".obj", ".json")))
    return cloth_object, keypoint_vertex_dict


@dataclasses.dataclass
class RendererConfig:
    exposure: float = 0.0
    gamma: float = 1.0
    device: str = "GPU"


class CyclesRendererConfig(RendererConfig):
    num_samples: int = 64


def render_scene(render_config: RendererConfig, output_dir: str):
    scene = bpy.context.scene

    if isinstance(render_config, CyclesRendererConfig):
        scene.render.engine = "CYCLES"
        scene.cycles.samples = render_config.num_samples
        scene.cycles.device = render_config.device
    else:
        raise NotImplementedError(f"Renderer config {render_config} not implemented")

    # TODO: randomize exposure and gamma
    scene.view_settings.exposure = render_config.exposure
    scene.view_settings.gamma = render_config.gamma

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
    mask_id_node.index = 1
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


def _are_keypoints_in_camera_frustum(
    cloth_object: bpy.types.Object, keypoint_vertex_dict: dict, camera: bpy.types.Object
) -> bool:
    """Check if all keypoints are in the camera frustum."""
    for _, vertex_id in keypoint_vertex_dict.items():
        point = cloth_object.data.vertices[vertex_id].co
        point = cloth_object.matrix_world @ point
        if not _is_point_in_camera_frustum(point, camera):
            return False
    return True


def _is_point_in_camera_frustum(point: Vector, camera: bpy.types.Object) -> bool:
    """Check if a point is in the camera frustum."""
    # Project the point
    scene = bpy.context.scene
    projected_point = world_to_camera_view(scene, camera, point)
    # Check if the point is in the frustum
    return (
        0 <= projected_point[0] <= 1
        and 0 <= projected_point[1] <= 1
        and camera.data.clip_start <= projected_point[2] <= camera.data.clip_end
    )


def create_annotations(
    cloth_type, output_dir: str, coco_id: int, cloth_object: bpy.types.Object, keypoint_vertex_dict: dict
):
    image_name = "rgb"
    segmentation_name = "segmentation"
    image_path = os.path.join(output_dir, f"{image_name}.png")
    segmentation_path = os.path.join(output_dir, f"{segmentation_name}.png")

    segmentation_mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    segmentation_mask = segmentation_mask > 0

    segmentation = BinarySegmentationMask(segmentation_mask)
    rle_mask = segmentation.as_compressed_rle

    bbox = segmentation.bbox
    x_min, y_min, width, height = bbox
    x_min + width
    y_min + height

    image_height, image_width = segmentation_mask.shape[0], segmentation_mask.shape[1]
    coco_image = CocoImage(file_name=image_path, height=image_height, width=image_width, id=coco_id)

    keypoints_3D = [
        cloth_object.matrix_world @ cloth_object.data.vertices[vid].co for vid in keypoint_vertex_dict.values()
    ]
    scene = bpy.context.scene
    camera = bpy.context.scene.camera
    keypoints_2D = [world_to_camera_view(scene, camera, corner) for corner in keypoints_3D]

    coco_keypoints = []
    num_labeled_keypoints = 0
    for keypoint_2D in keypoints_2D:
        u, v, _ = keypoint_2D
        px = image_width * u
        py = image_height * (1.0 - v)
        visible_flag = 2

        # Currently we set keypoints outside the image to be "not labeled"
        if px < 0 or py < 0 or px > image_width or py > image_height:
            visible_flag = 0
            px = 0.0
            py = 0.0

        if visible_flag > 0:
            num_labeled_keypoints += 1

        coco_keypoints += (px, py, visible_flag)

    category_id = CLOTH_TYPE_TO_COCO_CATEGORY_ID[cloth_type.name]
    annotation = CocoKeypointAnnotation(
        category_id=category_id,
        id=coco_id,  # only one annotation per image, so we can use the image id for now. #TODO: make this more generic.
        image_id=coco_id,
        keypoints=coco_keypoints,
        num_keypoints=num_labeled_keypoints,
        segmentation=rle_mask,
        area=segmentation.area,
        bbox=bbox,
        iscrowd=0,
    )

    # Save CocoImage to disk as json
    coco_image_json = "coco_image.json"
    coco_image_json_path = os.path.join(output_dir, coco_image_json)
    with open(coco_image_json_path, "w") as file:
        json.dump(coco_image.dict(exclude_none=True), file, indent=4)

    # Save CocoKeypointAnnotation to disk as json
    coco_annotation_json = "coco_annotation.json"
    coco_annotation_json_path = os.path.join(output_dir, coco_annotation_json)
    with open(coco_annotation_json_path, "w") as file:
        json.dump(annotation.dict(exclude_none=True), file, indent=4)


def create_sample(scene_config: ClothSceneConfig, render_config: RendererConfig, output_dir: str, coco_id: int):
    cloth_object, keypoint_vertex_dict = create_cloth_scene(scene_config)
    render_scene(render_config, output_dir)
    create_annotations(scene_config.cloth_type, output_dir, coco_id, cloth_object, keypoint_vertex_dict)


if __name__ == "__main__":
    import os
    import sys

    from synthetic_cloth_data import DATA_DIR
    from synthetic_cloth_data.synthetic_images.make_polyhaven_assets_snapshot import POLYHAVEN_ASSETS_SNAPSHOT_PATH

    hdri_path = POLYHAVEN_ASSETS_SNAPSHOT_PATH
    cloth_mesh_path = DATA_DIR / "deformed_meshes" / "towel"
    dataset_dir = DATA_DIR / "synthetic_images" / "deformed_test"
    cloth_type = CLOTH_TYPES.TOWEL

    id = 7
    # check if id was passed as argument
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        id = int(argv[argv.index("--id") + 1])

    output_dir = os.path.join(dataset_dir, f"{id:06d}")
    np.random.seed(2023 + id)

    # load HDRIS
    with open(hdri_path, "r") as file:
        assets = json.load(file)["assets"]
    worlds = [asset for asset in assets if asset["type"] == "worlds" and "indoor" in asset["tags"]]
    materials = [asset for asset in assets if asset["type"] == "materials"]

    # load cloth meshes

    cloth_meshes = os.listdir(cloth_mesh_path)
    cloth_meshes = [cloth_mesh_path / mesh for mesh in cloth_meshes]
    cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]

    config = ClothSceneConfig(
        cloth_type=cloth_type,
        cloth_mesh_config=ClothMeshConfig(
            mesh_dir=cloth_meshes,
        ),
        hdri_config=HDRIConfig(hdri_asset_list=worlds),
        cloth_material_config=TowelMaterialConfig(),  # TODO: must be adapted to cloth type. -> Config.
        camera_config=CameraConfig(),
        surface_config=SurfaceConfig(materials_list=materials),
    )
    render_config = CyclesRendererConfig()

    create_sample(config, render_config, output_dir, id)
