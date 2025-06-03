import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from PIL import Image, ImageDraw

TOWEL_KEYPOINTS = [
    "corner0",
    "corner1",
    "corner2",
    "corner3",
]

SHORTS_KEYPOINTS = [
    "waist_left",
    "waist_right",
    "pipe_right_outer",
    "pipe_right_inner",
    "crotch",
    "pipe_left_inner",
    "pipe_left_outer",
]

TSHIRT_KEYPOINTS = [
    "shoulder_left",
    "neck_left",
    "neck_right",
    "shoulder_right",
    "sleeve_right_top",
    "sleeve_right_bottom",
    "armpit_right",
    "waist_right",
    "waist_left",
    "armpit_left",
    "sleeve_left_bottom",
    "sleeve_left_top",
]

# https://sashamaps.net/docs/resources/20-colors/
DISTINCT_COLORS = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#ffffff",
    "#000000",
]


def reorder_inference_keypoints_to_gt_keypoints(
    keypoints: List[List[Tuple[int, int]]], channel_config: str, cloth_type: str
) -> List[List[Tuple[int, int]]]:
    channel_keypoints = channel_config.split(":")
    if cloth_type == "TOWEL":
        keypoints_names = TOWEL_KEYPOINTS

    elif cloth_type == "SHORTS":
        keypoints_names = SHORTS_KEYPOINTS
    elif cloth_type == "TSHIRT":
        keypoints_names = TSHIRT_KEYPOINTS
    else:
        raise ValueError(f"Unknown cloth type {cloth_type}")

    reordered_keypoints = []
    for i, keypoint_name in enumerate(keypoints_names):
        if keypoint_name not in channel_config:
            raise ValueError(f"Keypoint {keypoint_name} not in channel config {channel_config}")
        idx = channel_keypoints.index(keypoint_name)
        reordered_keypoints.append(keypoints[idx])
    return reordered_keypoints


def draw_keypoints_on_image(image: Image, image_keypoints: List[List[Tuple[int, int]]], scale=8) -> Image:
    """adds all keypoints to the PIL image, with different colors for each channel."""
    color_pool = DISTINCT_COLORS
    draw = ImageDraw.Draw(image)
    for channel_idx, channel_keypoints in enumerate(image_keypoints):
        for _, keypoint in enumerate(channel_keypoints):
            u, v = keypoint
            draw.ellipse((u - scale, v - scale, u + scale, v + scale), fill=color_pool[channel_idx])
    return image


def local_inference(model, image: Image, device="cuda", abs_confidence_threshold=0.1, max_keypoints_per_channel=20):
    """inference on a single image as if you would load the image from disk or get it from a camera.
    Returns a list of the extracted keypoints for each channel.


    """
    # assert model is in eval mode! (important for batch norm layers)
    model.eval().to(device)

    # convert image to tensor with correct shape (channels, height, width) and convert to floats in range [0,1]
    # add batch dimension
    # and move to device
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # pass through model
    with torch.no_grad():
        heatmaps = model(image).squeeze(0)

    # extract keypoints from heatmaps
    predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(
        heatmaps.unsqueeze(0), abs_max_threshold=abs_confidence_threshold, max_keypoints=max_keypoints_per_channel
    )[0]

    return predicted_keypoints


def get_ground_truth_keypoints(image_base_dir, image_path, annotations_json_path, channel_config):
    # load the coco dataset annotations
    coco_dataset = CocoKeypointsDataset(**json.load(open(annotations_json_path, "r")))

    relative_image_path = os.path.relpath(image_path, image_base_dir)
    image_id = None
    for image in coco_dataset.images:
        if image.file_name == relative_image_path:
            image_id = image.id
            break
    if image_id is None:
        raise ValueError(f"Could not find image {image_path} in annotations {annotations_json_path}")
    for annotation in coco_dataset.annotations:
        if annotation.image_id == image_id:
            category = [category for category in coco_dataset.categories if category.id == annotation.category_id][0]
            [keypoint for keypoint in category.keypoints]

            # convert the keypoints to the order of the channel config
            # keypoints_order = [keypoints_names.index(keypoint_name) for keypoint_name in channel_config.split(":")]
            keypoints = annotation.keypoints
            keypoints = np.array(keypoints)
            keypoints = keypoints.reshape(-1, 3)[:, :2]
            # keypoints = np.array([keypoints[keypoint_order] for keypoint_order in keypoints_order])

            keypoints = keypoints[np.newaxis, :]
            keypoints = keypoints.reshape(-1, 1, 2)
            return keypoints

    raise ValueError(f"Could not find image {image_path} in annotations {annotations_json_path}")


def horizontally_combine_images(images: List[Image.Image]) -> Image.Image:
    """horizontally combines a list of images"""
    return combine_images(images, n_rows=1)


def combine_images(images: List[Image.Image], n_rows) -> Image.Image:
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    n_images = len(images)
    n_cols = math.ceil(n_images / n_rows)
    combined_image = Image.new("RGB", (max_width * n_cols, max_height * n_rows))
    for i, image in enumerate(images):
        x = i % n_cols
        y = i // n_cols
        combined_image.paste(image, (x * max_width, y * max_height))
    return combined_image


if __name__ == "__main__":
    from state_estimation.keypoint_detection.common import data_dir
    from state_estimation.keypoint_detection.final_checkpoints import SYNTHETIC_TSHIRTS_CHECKPOINT
    from state_estimation.keypoint_detection.real_baselines import ARTF_TSHIRT_TEST_PATH, TSHIRT_CHANNEL_CONFIG

    image_path = (
        data_dir
        / "artf_data"
        / "towels-test_resized_512x256"
        / "test"
        / "location_2"
        / "towels"
        / "2023-04-24_12-20-18_rgb_zed.png"
    )
    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_2"
        / "tshirts"
        / "2023-04-24_12-03-02_rgb_zed.png"
    )
    image = Image.open(image_path)

    json_annotations_path = ARTF_TSHIRT_TEST_PATH
    base_dir = json_annotations_path.parent

    gt_keypoints = get_ground_truth_keypoints(base_dir, image_path, json_annotations_path, TSHIRT_CHANNEL_CONFIG)

    keypoints = local_inference(get_model_from_wandb_checkpoint(SYNTHETIC_TSHIRTS_CHECKPOINT), image)
    keypoints = reorder_inference_keypoints_to_gt_keypoints(keypoints, TSHIRT_CHANNEL_CONFIG, "TSHIRT")
    image = draw_keypoints_on_image(image, keypoints)
    image.save("predictions.png")

    image = Image.open(image_path)
    image = draw_keypoints_on_image(image, gt_keypoints)
    image.save("gt.png")
