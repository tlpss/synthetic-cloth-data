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
    add_camera(config.camera_config, cloth_object)
    # TODO: check if all keypoints are visible in the camera view, resample (?) if not.
    add_material_to_cloth_mesh(
        cloth_object=cloth_object, cloth_type=config.cloth_type, cloth_material_config=config.cloth_material_config
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
    size: int = 5
    rgb: Tuple[float, float, float] = (0.5, 0.5, 1)


def create_surface(config: SurfaceConfig) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=config.size)
    plane = bpy.context.object
    ab.add_material(plane, color=config.rgb)

    # TODO: more complex surfaces:
    # - random textures
    # - random images
    return plane


@dataclasses.dataclass
class CameraConfig:
    focal_length: int = 32


def add_camera(config: CameraConfig, cloth_object) -> bpy.types.Object:
    camera = bpy.data.objects["Camera"]

    def _sample_point_on_unit_sphere() -> np.ndarray:
        point_gaussian_3D = np.random.randn(3)
        point_on_unit_sphere = point_gaussian_3D / np.linalg.norm(point_gaussian_3D)
        return point_on_unit_sphere

    camera.location = _sample_point_on_unit_sphere()

    # Make the camera look at the towel center
    camera_direction = cloth_object.location - camera.location  # Note: these are mathutils Vectors
    camera_direction = Vector(camera_direction)
    camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

    # Set the camera focal length to 32 mm
    camera.data.lens = config.focal_length
    return camera


@dataclasses.dataclass
class ClothMaterialConfig:
    pass


def add_material_to_cloth_mesh(
    cloth_material_config: ClothMaterialConfig, cloth_object: bpy.types.Object, cloth_type: CLOTH_TYPES
):
    pass
    # determine materials.


@dataclasses.dataclass
class ClothMeshConfig:
    mesh_path: str
    position_in_world_frame: np.ndarray = np.array([0, 0, 0.01])
    z_rotation: float = 0.0


def load_cloth_mesh(config: ClothMeshConfig):
    # load the obj

    bpy.ops.import_scene.obj(
        filepath=str(config.mesh_path), split_mode="OFF"
    )  # keep vertex order with split_mode="OFF"
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    cloth_object.location = config.position_in_world_frame
    cloth_object.rotation_euler[2] = config.z_rotation

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
    num_samples: int = 32


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


def _is_point_in_camera_frustum(point: np.ndarray, camera: bpy.types.Object) -> bool:
    """Check if a point is in the camera frustum."""
    # Get the camera matrix
    raise NotImplementedError


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
    cloth_mesh_path = DATA_DIR / "flat_meshes" / "TSHIRT"
    dataset_dir = DATA_DIR / "synthetic_images" / "test"
    id = 0
    cloth_type = CLOTH_TYPES.TSHIRT
    output_dir = os.path.join(dataset_dir, str(id))

    # load HDRIS
    with open(hdri_path, "r") as file:
        assets = json.load(file)["assets"]
    worlds = [asset for asset in assets if asset["type"] == "worlds"]

    # load cloth meshes

    cloth_meshes = os.listdir(cloth_mesh_path)
    cloth_meshes = [DATA_DIR / "flat_meshes" / "TSHIRT" / mesh for mesh in cloth_meshes]
    cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]

    config = ClothSceneConfig(
        cloth_type=cloth_type,
        cloth_mesh_config=ClothMeshConfig(
            mesh_path=cloth_meshes[0],
        ),
        hdri_config=HDRIConfig(hdri_asset_list=worlds),
        cloth_material_config=ClothMaterialConfig(),
        camera_config=CameraConfig(),
        surface_config=SurfaceConfig(),
    )
    render_config = CyclesRendererConfig()
    create_sample(config, render_config, output_dir, id)
