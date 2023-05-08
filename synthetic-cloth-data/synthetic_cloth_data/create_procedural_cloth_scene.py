import json

import airo_blender as ab
import bpy
import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointAnnotation
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from bpy_extras.object_utils import world_to_camera_view
from synthetic_cloth_data.utils import CLOTH_TYPE_TO_COCO_CATEGORY_ID, CLOTH_TYPES


def build_procedural_cloth_scene(cloth_type: CLOTH_TYPES, worlds: list, cloth_meshes: list[str]):
    add_hdri_background(worlds)
    create_surface()
    cloth_object, keypoint_vertex_ids = load_cloth_mesh(cloth_meshes)
    cloth_object.pass_index = 1  # mark for segmentation mask rendering
    add_camera(target_location=cloth_object.location)
    # TODO: check if all keypoints are visible in the camera view, resample if not.
    add_material_to_cloth_mesh(cloth_object, cloth_type)
    return cloth_object, keypoint_vertex_ids


def add_hdri_background(worlds):
    random_world_info = np.random.choice(worlds)
    world = ab.load_asset(**random_world_info)
    bpy.context.scene.world = world


def create_surface():
    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object
    ab.add_material(plane, (0.5, 0.5, 1))
    # simple RGB colors
    # random textures
    # random images
    return plane


def add_camera(target_location: np.ndarray) -> bpy.types.Object:
    camera = bpy.data.objects["Camera"]

    def _sample_point_on_unit_sphere() -> np.ndarray:
        point_gaussian_3D = np.random.randn(3)
        point_on_unit_sphere = point_gaussian_3D / np.linalg.norm(point_gaussian_3D)
        return point_on_unit_sphere

    # Sample a point on the top part of the unit sphere
    high_point = _sample_point_on_unit_sphere()
    while high_point[2] < 0.75:
        high_point = _sample_point_on_unit_sphere()

    # Place the camera above the table
    high_point[2] += target_location[2]
    camera.location = high_point

    # Make the camera look at the towel center
    camera_direction = target_location - camera.location  # Note: these are mathutils Vectors
    camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

    # Set the camera focal length to 32 mm
    camera.data.lens = 32
    return camera


def add_material_to_cloth_mesh(cloth_object, cloth_type):
    pass
    # determine materials.


def load_cloth_mesh(cloth_meshes):
    cloth_mesh = np.random.choice(cloth_meshes)
    # load the obj
    bpy.ops.import_scene.obj(filepath=str(cloth_mesh), split_mode="OFF")
    cloth_object = bpy.context.selected_objects[0]
    # randomize position & orientation
    cloth_object.location = (0.0, 0.0, 0.001)
    cloth_object.rotation_euler[2] = 0.2

    keypoint_vertex_dict = json.load(open(str(cloth_mesh).replace(".obj", ".json")))
    return cloth_object, keypoint_vertex_dict


def render_scene(output_dir: str):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16

    image_width, image_height = 512, 512
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height

    # scene.view_settings.exposure = np.random.uniform(-2, 2)
    # scene.view_settings.gamma = np.random.uniform(0.9, 1.1)

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
    polygon = segmentation.as_polygon

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
        segmentation=polygon,
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


def create_dataset_sample(cloth_type: CLOTH_TYPES, dataset_dir: str, id: int, cloth_mesh_path: str, hdri_path: str):
    """should be called from a new blender instance."""
    # remove default object
    bpy.ops.object.delete()

    with open(hdri_path, "r") as file:
        assets = json.load(file)["assets"]
    worlds = [asset for asset in assets if asset["type"] == "worlds"]

    # load cloth meshes

    cloth_meshes = os.listdir(cloth_mesh_path)
    cloth_meshes = [DATA_DIR / "flat_meshes" / "TSHIRT" / mesh for mesh in cloth_meshes]
    cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]

    cloth_object, keypoint_vertex_ids = build_procedural_cloth_scene(cloth_type, worlds, cloth_meshes)

    dirname = os.path.join(dataset_dir, str(id))
    render_scene(dirname)
    create_annotations(cloth_type, dirname, id, cloth_object, keypoint_vertex_ids)


if __name__ == "__main__":
    import os

    from synthetic_cloth_data import DATA_DIR
    from synthetic_cloth_data.make_polyhaven_assets_snapshot import POLYHAVEN_ASSETS_SNAPSHOT_PATH

    # TODO: how to pass configuration to blender script?
    # cannot use CLI because blender has its own CLI
    # have to run under subprocess instead of multiprocessing for the same reason
    # and want to use a different process due to blender getting slower over time..
    # so how to pass configs? dump config dict to json and read from blender script?
    # @click.command()
    # @click.option("--cloth_type", type=str, default="TSHIRT")
    # @click.option("--dataset_dir", type=str, default=DATA_DIR / "coco_test")
    # @click.option("--id", type=int, default=0)
    # @click.option("--cloth_meshes_path", type=str, default=DATA_DIR / "flat_meshes" / "TSHIRT")
    # @click.option("--hdri_path", type=str, default=POLYHAVEN_ASSETS_SNAPSHOT_PATH)
    # def create_dataset_sample_cli(cloth_type, dataset_dir, id, cloth_meshes_path, hdri_path):
    #     create_dataset_sample(cloth_type, dataset_dir, id, cloth_meshes_path,hdri_path)

    create_dataset_sample(
        CLOTH_TYPES.TSHIRT,
        DATA_DIR / "coco_test",
        1,
        DATA_DIR / "flat_meshes" / "TSHIRT",
        POLYHAVEN_ASSETS_SNAPSHOT_PATH,
    )
