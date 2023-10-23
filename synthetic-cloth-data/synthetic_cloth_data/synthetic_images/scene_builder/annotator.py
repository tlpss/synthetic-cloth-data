from __future__ import annotations

import json
import os

import bpy
import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointAnnotation
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from bpy_extras.object_utils import world_to_camera_view
from synthetic_cloth_data.synthetic_images.scene_builder.utils.visible_vertices import (
    is_vertex_occluded_for_scene_camera,
)
from synthetic_cloth_data.utils import (
    CATEGORY_NAME_TO_KEYPOINTS_DICT,
    CLOTH_TYPE_TO_COCO_CATEGORY_ID,
    TSHIRT_KEYPOINTS,
)


def create_coco_annotations(
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

    image_height, image_width = segmentation_mask.shape[0], segmentation_mask.shape[1]
    coco_image = CocoImage(file_name=image_path, height=image_height, width=image_width, id=coco_id)

    # Save CocoImage to disk as json
    coco_image_json = "coco_image.json"
    coco_image_json_path = os.path.join(output_dir, coco_image_json)
    with open(coco_image_json_path, "w") as file:
        json.dump(coco_image.dict(exclude_none=True), file, indent=4)

    if segmentation.area < 1.0:
        # this object is not visible in the image
        # so no annotation needs to be added.
        return

    # get the 3D coordinates of the keypoints, in the desired order for the coco category
    category_keypoints = CATEGORY_NAME_TO_KEYPOINTS_DICT[cloth_type.name]
    keypoints_3D = [
        cloth_object.matrix_world @ cloth_object.data.vertices[keypoint_vertex_dict[keypoint_name]].co
        for keypoint_name in category_keypoints
    ]
    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    keypoints_2D = np.array([np.array(world_to_camera_view(scene, camera, corner))[:2] for corner in keypoints_3D])

    # flip y-axis because blender uses y-up and we use y-down (as does coco)
    keypoints_2D[:, 1] = 1.0 - keypoints_2D[:, 1]

    # scale keypoints to pixel coordinates
    keypoints_2D[:, 0] = keypoints_2D[:, 0] * image_width
    keypoints_2D[:, 1] = keypoints_2D[:, 1] * image_height

    # order keypoints to deal with symmetries.
    if cloth_type == "TOWEL":
        keypoints_2D, keypoints_3D = _order_towel_keypoints(keypoints_2D, keypoints_3D, bbox)
    if cloth_type == "TSHIRT":
        keypoints_2D, keypoints_3D = _order_tshirt_keypoints(keypoints_2D, keypoints_3D, bbox)

    # check if cloth object has a solidify modifier and remove it temporarily because it affects the ray cast and hence the visibility check.
    solidify_modifier = None
    is_solidify_modifier = [modifier.type == "SOLIDIFY" for modifier in cloth_object.modifiers]
    if any(is_solidify_modifier):
        solidify_modifier = cloth_object.modifiers[is_solidify_modifier.index(True)]
        solidifier_thickness = solidify_modifier.thickness
        solidify_modifier.thickness = 0.0

    # gather the keypoints
    coco_keypoints = []
    num_labeled_keypoints = 0
    for keypoint_idx, (keypoint_3D, keypoint_2D) in enumerate(zip(keypoints_3D, keypoints_2D)):
        u, v = keypoint_2D
        px, py = u, v

        visible_flag = 1 if is_vertex_occluded_for_scene_camera(keypoint_3D) else 2
        print(f"{keypoint_3D} -> visible_flag: {visible_flag}")

        # we set keypoints outside the image to be "not labeled"
        if px < 0 or py < 0 or px > image_width or py > image_height:
            visible_flag = 0
            px = 0.0
            py = 0.0

        if visible_flag > 0:
            num_labeled_keypoints += 1

        coco_keypoints += (px, py, visible_flag)

        # for debugging:
        # add 3D sphere around each keypoint
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=keypoint_3D)
        bpy.context.object.name = f"keypoint_{TSHIRT_KEYPOINTS[keypoint_idx]}"

    # add the solidifier back if required
    if solidify_modifier is not None:
        solidify_modifier.thickness = solidifier_thickness

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

    # Save CocoKeypointAnnotation to disk as json
    coco_annotation_json = "coco_annotation.json"
    coco_annotation_json_path = os.path.join(output_dir, coco_annotation_json)
    with open(coco_annotation_json_path, "w") as file:
        json.dump(annotation.dict(exclude_none=True), file, indent=4)


def _order_towel_keypoints(keypoints_2D, keypoints_3D, bbox):
    x_min, y_min, width, height = bbox

    # keypoints are in cyclical order but we need to break symmetries by having a starting point in the image viewpoint
    bbox_top_left = (x_min, y_min)

    # find the keypoint that is closest to the top left corner of the bounding box
    distances = [np.linalg.norm(np.array(keypoint_2D) - np.array(bbox_top_left)) for keypoint_2D in keypoints_2D]
    starting_keypoint_index = np.argmin(distances)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    # closest to the topright corner of the bbox

    bbox_top_right = (x_min + width, y_min)
    distances = [
        np.linalg.norm(
            np.array(keypoints_2D[(starting_keypoint_index + i) % len(keypoints_2D)]) - np.array(bbox_top_right)
        )
        for i in [-1, +1]
    ]
    direction = -1 if np.argmin(distances) == 0 else +1
    second_keypoint_index = (starting_keypoint_index + direction) % len(keypoints_2D)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    direction = second_keypoint_index - starting_keypoint_index

    order = [starting_keypoint_index]
    for i in range(1, len(keypoints_2D)):
        order.append((starting_keypoint_index + i * direction) % len(keypoints_2D))

    new_keypoints_2D = [keypoints_2D[i] for i in order]
    new_keypoints_3D = [keypoints_3D[i] for i in order]

    keypoints_2D = new_keypoints_2D
    keypoints_3D = new_keypoints_3D
    return keypoints_2D, keypoints_3D


def _order_tshirt_keypoints(keypoints_2D: np.ndarray, keypoints_3D: np.ndarray, bbox: tuple):

    # left == side of which the waist kp is closest to the bottom left corner of the bbox in 2D.
    # simply serves to break symmetries and find adjacent keypoints, does not correspond with human notion of left and right,
    # which is determind in 3D. This can be determiend later in the pipeline if desired, once the 2D keypoints are lifted to 3D somehow.
    # we use waist kp as this one has been estimated to be least deformed by real pipelines.

    x_min, y_min, width, height = bbox

    bottom_left_bbox_corner = (x_min, y_min + height)

    waist_left_idx = TSHIRT_KEYPOINTS.index("waist_left")
    waist_right_idx = TSHIRT_KEYPOINTS.index("waist_right")
    waist_left_2D = keypoints_2D[waist_left_idx]
    waist_right_2D = keypoints_2D[waist_right_idx]

    distance_waist_left = np.linalg.norm(np.array(waist_left_2D) - np.array(bottom_left_bbox_corner))
    distance_waist_right = np.linalg.norm(np.array(waist_right_2D) - np.array(bottom_left_bbox_corner))

    if distance_waist_left > distance_waist_right:
        should_tshirt_be_flipped = True
    else:
        should_tshirt_be_flipped = False
    print(f"should_tshirt_be_flipped: {should_tshirt_be_flipped}")
    if should_tshirt_be_flipped:
        for idx, keypoint in enumerate(TSHIRT_KEYPOINTS):
            if "left" in keypoint:
                right_idx = TSHIRT_KEYPOINTS.index(keypoint.replace("left", "right"))
                print(f"swapping {keypoint} with {TSHIRT_KEYPOINTS[right_idx]}")
                # swap the rows in the numpy array, cannot do this as with lists
                # https://stackoverflow.com/questions/21288044/row-exchange-in-numpy
                keypoints_2D[[idx, right_idx]] = keypoints_2D[[right_idx, idx]]
                keypoints_3D[idx], keypoints_3D[right_idx] = keypoints_3D[right_idx], keypoints_3D[idx]

    return keypoints_2D, keypoints_3D
