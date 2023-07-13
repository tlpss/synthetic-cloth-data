"""script to replace background of images in a coco dataset with random backgrounds"""
import json
import os
import pathlib

import cv2
import numpy as np
import tqdm
from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointAnnotation, CocoKeypointsDataset
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask


def create_random_background_dataset(
    coco_json_path: str, target_dir: str, backgrounds_dir: str, num_backgrounds_per_sample: int = 1
):
    # load coco json into the pydantic model
    with open(coco_json_path, "r") as f:
        coco_json = json.load(f)
        coco_dataset = CocoKeypointsDataset(**coco_json)

    # prepare target directory
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)
    target_image_dir = target_dir / "images"
    target_image_dir.mkdir(exist_ok=True, parents=True)

    # gather all background images
    backgrounds = [
        os.path.join(backgrounds_dir, f)
        for f in os.listdir(backgrounds_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    assert len(backgrounds) > 0, f"no backgrounds found in {backgrounds_dir}"

    image_id_to_image = {image.id: image for image in coco_dataset.images}
    new_coco_images = []
    new_coco_annotations = []

    for annotation in tqdm.tqdm(coco_dataset.annotations):
        for _ in range(num_backgrounds_per_sample):

            coco_image = image_id_to_image[annotation.image_id]
            random_background = np.random.choice(backgrounds)
            # load the background image & resize to the size of the coco image
            background_image = cv2.imread(random_background)
            background_image = cv2.resize(background_image, (coco_image.width, coco_image.height))

            # take the segmentation mask of the dataset image and use it as a mask for the background image to paste
            # the coco object on top of the background
            image_path = pathlib.Path(coco_json_path).parent / coco_image.file_name
            image = cv2.imread(str(image_path))
            new_image = np.zeros_like(image)
            segmentation_mask = BinarySegmentationMask.from_coco_segmentation_mask(
                annotation.segmentation, coco_image.width, coco_image.height
            )
            segmentation_mask = np.repeat(segmentation_mask.bitmap[:, :, np.newaxis], 3, axis=2)
            new_image = background_image * (1 - segmentation_mask) + image * segmentation_mask

            # save the new image and add it to the new coco dataset with the original annotations
            new_image_id = len(new_coco_images) + 1
            new_image_path = pathlib.Path(target_dir) / f"images/{new_image_id}.jpg"
            cv2.imwrite(str(new_image_path), new_image)
            new_coco_images.append(
                CocoImage(
                    id=new_image_id,
                    file_name=f"images/{new_image_id}.jpg",
                    width=coco_image.width,
                    height=coco_image.height,
                )
            )

            new_annotation = CocoKeypointAnnotation(**annotation.dict())
            new_annotation.image_id = new_coco_images[-1].id
            new_coco_annotations.append(new_annotation)

    # save the new coco dataset
    dataset = CocoKeypointsDataset(
        images=new_coco_images, annotations=new_coco_annotations, categories=coco_dataset.categories
    )
    with open(target_dir / "annotations.json", "w") as f:
        f.write(dataset.json(indent=2))


if __name__ == "__main__":

    coco_json = (
        "/home/tlips/Code/synthetic-cloth-data/synthetic-cloth-data/data/datasets/TOWEL/pyflex/annotations.json"
    )
    target_dir_json = "/home/tlips/Code/synthetic-cloth-data/synthetic-cloth-data/data/datasets/TOWEL/coco_bg"
    background_dir = "/media/roblaundry/tlips/coco/test2017/"
    create_random_background_dataset(coco_json, target_dir_json, background_dir, num_backgrounds_per_sample=2)
