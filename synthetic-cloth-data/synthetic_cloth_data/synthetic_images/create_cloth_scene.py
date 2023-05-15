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
from synthetic_cloth_data.materials.towels import (
    create_evenly_colored_material,
    create_gridded_dish_towel_material,
    create_striped_material,
)
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


@dataclasses.dataclass
class SurfaceConfig:
    size_range: Tuple[float, float] = (1, 3)
    textures_list: List[dict] = dataclasses.field(default_factory=list)
    texture_probability: float = 0.5


def _sample_hsv_color():
    hue = np.random.uniform(0, 180)
    saturation = np.random.uniform(0.5, 1)
    value = np.random.uniform(0.5, 1)
    return np.array([hue, saturation, value])


def _hsv_to_rgb(hsv: np.ndarray):
    assert hsv.shape == (3,)
    hsv = hsv.astype(np.float32)
    print(hsv)
    rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, ...], cv2.COLOR_HSV2RGB)
    print(rgb)
    return rgb[0][0]


def create_surface(config: SurfaceConfig) -> bpy.types.Object:
    size = np.random.uniform(*config.size_range, size=2)
    bpy.ops.mesh.primitive_plane_add(size=1)
    # scale the plane to the desired size (cannot do this on creation bc of weir thing in blender API
    # :https://devtalk.blender.org/t/setting-scale-on-primitive-creation/28348 )
    bpy.ops.transform.resize(value=(size[0], size[1], 1))
    plane = bpy.context.object

    print(f"len config.textures_list: {len(config.textures_list)}")
    if np.random.rand() < config.texture_probability and len(config.textures_list) > 0:
        texture_dict = np.random.choice(config.textures_list)
        texture = ab.load_asset(**texture_dict)
        plane.data.materials.append(texture)
    else:
        hsv = _sample_hsv_color()
        rgb = _hsv_to_rgb(hsv)
        ab.add_material(plane, color=rgb)

    return plane


@dataclasses.dataclass
class CameraConfig:
    focal_length: int = 32
    minimal_camera_height: float = 0.5


def add_camera(config: CameraConfig, cloth_object: bpy.types.Object, keypoint_vertices_dict: dict) -> bpy.types.Object:
    camera = bpy.data.objects["Camera"]

    def _sample_point_on_unit_sphere() -> np.ndarray:
        point_gaussian_3D = np.random.randn(3)
        point_on_unit_sphere = point_gaussian_3D / np.linalg.norm(point_gaussian_3D)
        return point_on_unit_sphere
        # TODO: better camera randomization?
        # in upper half of unit sphere?

    camera_placed = False
    while not camera_placed:
        camera.location = _sample_point_on_unit_sphere()
        camera_placed = camera.location[2] > config.minimal_camera_height  # reasonable view heights
        camera_placed = camera_placed and _are_keypoints_in_camera_frustum(cloth_object, keypoint_vertices_dict)

    # Make the camera look at tthe origin, around which the cloth and table are assumed to be centered.
    camera_direction = -camera.location
    camera_direction = Vector(camera_direction)
    camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

    # Set the camera focal length
    camera.data.lens = config.focal_length
    # TODO: randomize other camera parameters?
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


def _unwrap_towel(towel_object: bpy.types.Object):
    # UV unwrap the towel template

    # activate the object and enter edit mode
    bpy.context.view_layer.objects.active = towel_object
    bpy.ops.object.mode_set(mode="EDIT")

    # unwrap UV for rendering
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)

    # exit edit mode
    bpy.ops.object.mode_set(mode="OBJECT")


def _add_material_to_towel_mesh(config: TowelMaterialConfig, cloth_object: bpy.types.Object):

    _unwrap_towel(cloth_object)
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
    cloth_object.data.materials[0] = material


@dataclasses.dataclass
class ClothMeshConfig:
    mesh_path: str
    xy_randomization_range: float = 0.1


def load_cloth_mesh(config: ClothMeshConfig):
    # load the obj

    bpy.ops.import_scene.obj(
        filepath=str(config.mesh_path), split_mode="OFF"
    )  # keep vertex order with split_mode="OFF"
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    xy_position = np.random.uniform(-config.xy_randomization_range, config.xy_randomization_range, size=2)
    cloth_object.location[0] = xy_position[0]
    cloth_object.location[1] = xy_position[1]

    cloth_object.location[2] = 0.001  # make sure the cloth is above the surface

    cloth_object.rotation_euler[2] = np.random.rand() * 2 * np.pi

    # convention is to have the keypoint vertex ids in a json file with the same name as the obj file
    keypoint_vertex_dict = json.load(open(str(config.mesh_path).replace(".obj", ".json")))
    return cloth_object, keypoint_vertex_dict


@dataclasses.dataclass
class RendererConfig:
    width: int = 512
    height: int = 512
    exposure: float = 0.0
    gamma: float = 1.0


class CyclesRendererConfig(RendererConfig):
    num_samples: int = 16


def render_scene(render_config: RendererConfig, output_dir: str):
    scene = bpy.context.scene

    if isinstance(render_config, CyclesRendererConfig):
        scene.render.engine = "CYCLES"
        scene.cycles.samples = render_config.num_samples
    else:
        raise NotImplementedError(f"Renderer config {render_config} not implemented")

    image_width, image_height = render_config.width, render_config.height
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height

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


def _are_keypoints_in_camera_frustum(keypoint_vertex_dict: dict, camera: bpy.types.Object) -> bool:
    """Check if all keypoints are in the camera frustum."""
    for _, vertex_id in keypoint_vertex_dict.items():
        point = bpy.data.objects["cloth"].data.vertices[vertex_id].co
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
        -1 <= projected_point[0] <= 1
        and -1 <= projected_point[1] <= 1
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

    from synthetic_cloth_data import DATA_DIR
    from synthetic_cloth_data.synthetic_images.make_polyhaven_assets_snapshot import POLYHAVEN_ASSETS_SNAPSHOT_PATH

    hdri_path = POLYHAVEN_ASSETS_SNAPSHOT_PATH
    cloth_mesh_path = DATA_DIR / "flat_meshes" / "TOWEL"
    dataset_dir = DATA_DIR / "synthetic_images" / "test"

    id = 0
    cloth_type = CLOTH_TYPES.TOWEL

    output_dir = os.path.join(dataset_dir, str(id))
    # np.random.seed(2023)

    # load HDRIS
    with open(hdri_path, "r") as file:
        assets = json.load(file)["assets"]
    worlds = [asset for asset in assets if asset["type"] == "worlds"]
    materials = [asset for asset in assets if asset["type"] == "materials"]

    # load cloth meshes

    cloth_meshes = os.listdir(cloth_mesh_path)
    cloth_meshes = [cloth_mesh_path / mesh for mesh in cloth_meshes]
    cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]

    config = ClothSceneConfig(
        cloth_type=cloth_type,
        cloth_mesh_config=ClothMeshConfig(
            mesh_path=cloth_meshes[0],
        ),
        hdri_config=HDRIConfig(hdri_asset_list=worlds),
        cloth_material_config=TowelMaterialConfig(),
        camera_config=CameraConfig(),
        surface_config=SurfaceConfig(textures_list=materials),
    )
    render_config = CyclesRendererConfig()
    create_sample(config, render_config, output_dir, id)
