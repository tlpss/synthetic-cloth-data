import json
import os

from airo_dataset_tools.data_parsers.coco import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)
from synthetic_cloth_data.utils import SHORT_KEYPOINTS, TOWEL_KEYPOINTS, TSHIRT_KEYPOINTS


def create_coco_dataset_from_intermediates(directory: str):
    towel_category = CocoKeypointCategory(
        supercategory="cloth",
        id=0,
        name="towel",
        keypoints=TOWEL_KEYPOINTS,
        skeleton=[],
    )
    tshirt_category = CocoKeypointCategory(
        supercategory="cloth",
        id=1,
        name="tshirt",
        keypoints=TSHIRT_KEYPOINTS,
    )
    short_category = CocoKeypointCategory(
        supercategory="cloth",
        id=2,
        name="short",
        keypoints=SHORT_KEYPOINTS,
    )

    categories = [towel_category, tshirt_category, short_category]
    annotations = []
    images = []

    data_samples = os.listdir(directory)

    # This could be a seperate script that just looks for directories with the right name
    for data_sample_dir in data_samples:
        image_json_path = f"{directory}/{data_sample_dir}/coco_image.json"

        with open(image_json_path, "r") as file:
            coco_image = CocoImage(**json.load(file))
        images.append(coco_image)

        # use glob to find all annotation files
        annotation_path = f"{directory}/{data_sample_dir}/coco_annotation.json"
        with open(annotation_path, "r") as file:
            coco_annotation = CocoKeypointAnnotation(**json.load(file))
        annotations.append(coco_annotation)

    labels = CocoKeypointsDataset(categories=categories, images=images, annotations=annotations)
    annotations_json = "coco.json"
    # TODO: copy the images to a new directory, update the image paths to be relative to the new directory and then save the json there
    # dir
    # /images
    # coco.json (with image paths relative to /images)

    with open(annotations_json, "w") as file:
        json.dump(labels.dict(exclude_none=True), file, indent=4)


if __name__ == "__main__":
    from synthetic_cloth_data import DATA_DIR

    test_dir = DATA_DIR / "synthetic_images" / "deformed_test"
    create_coco_dataset_from_intermediates(test_dir)
