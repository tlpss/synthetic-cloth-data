import glob
import json
import os
import subprocess

from airo_dataset_tools.coco.coco_parser import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)
from tqdm import tqdm

dataset_size = 2

script = "tutorial_3.py"

file_directory = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(file_directory, script)

# We generate each sample is a separate process for robustness
print("Generating samples")
for seed in tqdm(range(dataset_size)):
    command = f"blender -b -P {script_path} -- --seed {seed}"
    subprocess.run([command], shell=True, stdout=subprocess.DEVNULL)


# This code should move to a airo-datasets package in the future
towel_keypoints = [
    "corner1",
    "corner2",
    "corner3",
    "corner4",
]

towel_category = CocoKeypointCategory(
    supercategory="cloth",
    id=0,
    name="towel",
    keypoints=towel_keypoints,
    skeleton=[],
)

categories = [towel_category]
annotations = []
images = []

# This could be a seperate script that just looks for directories with the right name
for seed in range(dataset_size):
    seed_padded = f"{seed:08d}"
    seed_directory = seed_padded
    image_path = f"{seed_directory}/{seed_padded}.png"
    image_json_path = f"{seed_directory}/{seed_padded}_coco_image.json"

    with open(image_json_path, "r") as file:
        coco_image = CocoImage(**json.load(file))
    images.append(coco_image)

    # use glob to find all annotation files
    annotation_paths = glob.glob(f"{seed_directory}/*_coco_annotation_*.json")
    image_annotations = []
    for annotation_path in annotation_paths:
        with open(annotation_path, "r") as file:
            coco_annotation = CocoKeypointAnnotation(**json.load(file))
        image_annotations.append(coco_annotation)

    annotations.extend(image_annotations)


labels = CocoKeypointsDataset(categories=categories, images=images, annotations=annotations)
annotations_json = "annotations.json"
with open(annotations_json, "w") as file:
    json.dump(labels.dict(exclude_none=True), file, indent=4)
