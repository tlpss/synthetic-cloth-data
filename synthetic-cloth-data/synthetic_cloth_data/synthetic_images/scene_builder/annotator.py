from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Tuple

import bpy
import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointAnnotation
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from bpy_extras.object_utils import world_to_camera_view
from synthetic_cloth_data.meshes.utils.n_ring_neighbours import build_neighbour_dict, get_strict_n_ring_neighbours
from synthetic_cloth_data.synthetic_images.scene_builder.utils.visible_vertices import (
    is_vertex_occluded_for_scene_camera,
)
from synthetic_cloth_data.utils import (
    CATEGORY_NAME_TO_KEYPOINTS_DICT,
    CLOTH_TYPE_TO_COCO_CATEGORY_ID,
    SHORTS_KEYPOINTS,
    TOWEL_KEYPOINTS,
    TSHIRT_KEYPOINTS,
)


def get_3D_and_2D_keypoints_from_vertices(
    cloth_object: bpy.types.Object,
    keypoint_vertex_dict: dict,
    category_keypoints: List[str],
    image_width: int,
    image_height: int,
):
    # get the 3D coordinates of the keypoints, in the desired order for the coco category
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

    return keypoints_2D, keypoints_3D


@dataclasses.dataclass
class COCOAnnotatorConfig:
    # N-ring of vertices to check for visibility of keypoints.
    # If any of the vertices in the n-ring is visible, the keypoint is considered visible.
    # this value should be tuned for a particular mesh triangulation/resolution.
    annotations_n_ring_visibility: int = 4


def create_coco_annotations(  # noqa: C901
    cloth_type,
    output_dir: str,
    coco_id: int,
    cloth_object: bpy.types.Object,
    keypoint_vertex_dict: dict,
    annotator_config: COCOAnnotatorConfig,
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

    category_keypoints = CATEGORY_NAME_TO_KEYPOINTS_DICT[cloth_type.name]
    keypoints_2D, keypoints_3D = get_3D_and_2D_keypoints_from_vertices(
        cloth_object, keypoint_vertex_dict, category_keypoints, image_width, image_height
    )

    # order keypoints to deal with symmetries.
    # uses the 2D pixel locations, so that it can also be applied on the 2D keypoints from the real data.
    # but we need to make sure that the 2D keypoints are in the same order as the 3D keypoints and that the keypoint_vertex_dict is also updated.
    if cloth_type == "TOWEL":
        keypoints_2D, keypoints_3D, keypoint_vertex_dict = _order_towel_keypoints(
            keypoints_2D, keypoints_3D, keypoint_vertex_dict, bbox
        )
    if cloth_type == "TSHIRT":
        keypoints_2D, keypoints_3D, keypoint_vertex_dict = _order_tshirt_keypoints(
            keypoints_2D, keypoints_3D, keypoint_vertex_dict, bbox
        )
    if cloth_type == "SHORTS":
        keypoints_2D, keypoints_3D, keypoint_vertex_dict = order_shorts_keypoints(
            keypoints_2D, keypoints_3D, keypoint_vertex_dict, bbox
        )

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
    neighbours_dict = build_neighbour_dict(cloth_object)
    for keypoint_idx, (keypoint_3D, keypoint_2D) in enumerate(zip(keypoints_3D, keypoints_2D)):
        u, v = keypoint_2D
        px, py = u, v

        # take N-ring neighbourhood of the keypoint and check if any of the vertices in the neighbourhood are not occluded
        # if so, we set the keypoint to be visible
        # location remains at the original keypoint location
        is_kp_visible = False
        for i in range(annotator_config.annotations_n_ring_visibility + 1):
            n_ring_neighbours = get_strict_n_ring_neighbours(
                neighbours_dict, keypoint_vertex_dict[category_keypoints[keypoint_idx]], i
            )
            for neighbour in n_ring_neighbours:
                neighbour_coords = cloth_object.matrix_world @ cloth_object.data.vertices[neighbour].co
                # add sphere on neighbour for debugging
                # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.001, location=neighbour_coords)
                if not is_vertex_occluded_for_scene_camera(neighbour_coords):
                    is_kp_visible = True
                    print(f"found visible neighbour at n={i}")
                    break
            if is_kp_visible:
                break
        visible_flag = 2 if is_kp_visible else 1
        print(f"{keypoint_3D} -> visible_flag: {visible_flag}")

        # we set keypoints outside the image to be "not labeled"
        if px < 0 or py < 0 or px > image_width or py > image_height:
            print("keypoint outside image")
            visible_flag = 0
            px = 0.0
            py = 0.0

        if visible_flag > 0:
            num_labeled_keypoints += 1

        coco_keypoints += (px, py, visible_flag)

        # for debugging:
        # add 3D sphere around each keypoint
        if visible_flag == 2:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=keypoint_3D)
            bpy.context.object.name = f"keypoint_{category_keypoints[keypoint_idx]}"

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


def _order_towel_keypoints(keypoints_2D, keypoints_3D, vertex_dict, bbox):
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
    vertices = list(vertex_dict.values())
    new_vertices = [vertices[i] for i in order]
    new_vertex_dict = dict(zip(TOWEL_KEYPOINTS, new_vertices))

    keypoints_2D = new_keypoints_2D
    keypoints_3D = new_keypoints_3D
    return keypoints_2D, keypoints_3D, new_vertex_dict


def _order_tshirt_keypoints(keypoints_2D: np.ndarray, keypoints_3D: List, keypoints_vertex_dict, bbox: tuple):

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
    # print(f"should_tshirt_be_flipped: {should_tshirt_be_flipped}")
    if should_tshirt_be_flipped:
        for idx, keypoint in enumerate(TSHIRT_KEYPOINTS):
            if "left" in keypoint:
                right_idx = TSHIRT_KEYPOINTS.index(keypoint.replace("left", "right"))
                # print(f"swapping {keypoint} with {TSHIRT_KEYPOINTS[right_idx]}")
                # swap the rows in the numpy array, cannot do this as with lists
                # https://stackoverflow.com/questions/21288044/row-exchange-in-numpy
                keypoints_2D[[idx, right_idx]] = keypoints_2D[[right_idx, idx]]
                keypoints_3D[idx], keypoints_3D[right_idx] = keypoints_3D[right_idx], keypoints_3D[idx]
                keypoints_vertex_dict[keypoint], keypoints_vertex_dict[TSHIRT_KEYPOINTS[right_idx]] = (
                    keypoints_vertex_dict[TSHIRT_KEYPOINTS[right_idx]],
                    keypoints_vertex_dict[keypoint],
                )
    return keypoints_2D, keypoints_3D, keypoints_vertex_dict


def order_shorts_keypoints(
    keypoints_2D: np.ndarray, keypoints_3D: List, keypoint_vertices_dict: dict, bbox: Tuple[int]
) -> np.ndarray:
    x_min, y_min, width, height = bbox

    top_left_bbox_corner = (x_min, y_min)

    waist_left_idx = SHORTS_KEYPOINTS.index("waist_left")
    waist_right_idx = SHORTS_KEYPOINTS.index("waist_right")
    waist_left_2D = keypoints_2D[waist_left_idx][:2]
    waist_right_2D = keypoints_2D[waist_right_idx][:2]

    distance_waist_left = np.linalg.norm(np.array(waist_left_2D) - np.array(top_left_bbox_corner))
    distance_waist_right = np.linalg.norm(np.array(waist_right_2D) - np.array(top_left_bbox_corner))

    if distance_waist_left > distance_waist_right:
        should_shorts_be_flipped = True
    else:
        should_shorts_be_flipped = False

    if should_shorts_be_flipped:
        for idx, keypoint in enumerate(SHORTS_KEYPOINTS):
            if "left" in keypoint:
                right_idx = SHORTS_KEYPOINTS.index(keypoint.replace("left", "right"))
                # swap the rows in the numpy array, cannot do this as with lists
                # https://stackoverflow.com/questions/21288044/row-exchange-in-numpy
                keypoints_2D[[idx, right_idx]] = keypoints_2D[[right_idx, idx]]
                keypoints_3D[idx], keypoints_3D[right_idx] = keypoints_3D[right_idx], keypoints_3D[idx]
                keypoint_vertices_dict[keypoint], keypoint_vertices_dict[SHORTS_KEYPOINTS[right_idx]] = (
                    keypoint_vertices_dict[SHORTS_KEYPOINTS[right_idx]],
                    keypoint_vertices_dict[keypoint],
                )
    return keypoints_2D, keypoints_3D, keypoint_vertices_dict
