import json
import os
import pathlib

import cv2
from airo_dataset_tools.data_parsers.coco import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)
from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.utils import SHORT_KEYPOINTS, TOWEL_KEYPOINTS, TSHIRT_KEYPOINTS
from tqdm import tqdm

towel_category = CocoKeypointCategory(
    supercategory="cloth",
    id=0,
    name="towel",
    keypoints=TOWEL_KEYPOINTS,
    skeleton=[(0, 1), (1, 2), (2, 3)],
)
tshirt_category = CocoKeypointCategory(
    supercategory="cloth",
    id=2,
    name="tshirt",
    keypoints=TSHIRT_KEYPOINTS,
)
short_category = CocoKeypointCategory(
    supercategory="cloth",
    id=1,
    name="short",
    keypoints=SHORT_KEYPOINTS,
)


def create_coco_dataset_from_intermediates(relative_target_directory, relative_source_directory: str):

    categories = [towel_category, tshirt_category, short_category]
    annotations = []
    images = []

    source_directory = pathlib.Path(DATA_DIR / relative_source_directory)
    target_directory = pathlib.Path(DATA_DIR / relative_target_directory)
    image_directory = target_directory / "images"

    target_directory.mkdir(parents=True, exist_ok=True)
    image_directory.mkdir(parents=True, exist_ok=True)

    data_samples = os.listdir(source_directory)

    # This could be a seperate script that just looks for directories with the right name
    for data_sample_dir in data_samples:
        image_json_path = f"{source_directory}/{data_sample_dir}/coco_image.json"
        annotation_path = f"{source_directory}/{data_sample_dir}/coco_annotation.json"

        # sometimes, for some reason, one of the outputs is missing. We skip those samples for now.
        if not os.path.exists(image_json_path) or not os.path.exists(annotation_path):
            print(f"skipping {data_sample_dir} because {image_json_path} or {annotation_path} does not exist")
            continue

        with open(image_json_path, "r") as file:
            coco_image = CocoImage(**json.load(file))
        images.append(coco_image)

        with open(annotation_path, "r") as file:
            coco_annotation = CocoKeypointAnnotation(**json.load(file))
        annotations.append(coco_annotation)

    # copy images and make jpg
    print("copying images")
    for coco_image in tqdm(images):
        assert isinstance(coco_image, CocoImage)
        src_path = coco_image.file_name
        dst_path = image_directory / f"{coco_image.id}.jpg"
        # copy image to new location and encode as jpg
        img = cv2.imread(src_path)
        cv2.imwrite(str(dst_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # update image file name to relative path wrt image directory
        coco_image.file_name = str(dst_path.relative_to(target_directory))

    labels = CocoKeypointsDataset(categories=categories, images=images, annotations=annotations)
    annotations_json = target_directory / "annotations.json"
    with open(annotations_json, "w") as file:
        json.dump(labels.dict(exclude_none=True), file, indent=4)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--target_dir", default="datasets/TOWEL/dev", help="Target directory")
    @click.option("--src_dir", default="synthetic_images/TOWEL/dev", help="Source directory")
    def cli_create_coco_dataset_from_intermediates(target_dir, src_dir):
        create_coco_dataset_from_intermediates(target_dir, src_dir)

    cli_create_coco_dataset_from_intermediates()
