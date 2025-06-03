from pathlib import Path

from PIL import Image
from state_estimation.keypoint_detection.real_baselines import (
    SHORTS_CHANNEL_CONFIG,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
)
from state_estimation.keypoint_detection.visualizations.visualization import (
    combine_images,
    draw_keypoints_on_image,
    get_ground_truth_keypoints,
)

data_dir = Path(__file__).parent / "images"


def examples():
    image_path1 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/images/259.jpg"
    base_dir_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/"
    annotations_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/annotations.json"

    image1 = Image.open(image_path1)
    keypoints1 = get_ground_truth_keypoints(base_dir_path, image_path1, annotations_path, TSHIRT_CHANNEL_CONFIG)
    image1 = draw_keypoints_on_image(image1, keypoints1, scale=4)

    image_path2 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/SHORTS/single-layer-random-material-10K/images/840.jpg"
    base_dir_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/SHORTS/single-layer-random-material-10K/"
    annotations_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/SHORTS/single-layer-random-material-10K/annotations.json"
    image2 = Image.open(image_path2)
    keypoints2 = get_ground_truth_keypoints(base_dir_path, image_path2, annotations_path, SHORTS_CHANNEL_CONFIG)
    image2 = draw_keypoints_on_image(image2, keypoints2, scale=4)

    image_path3 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TOWEL/single-layer-random-material-10K/images/116.jpg"
    base_dir_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TOWEL/single-layer-random-material-10K/"
    annotations_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TOWEL/single-layer-random-material-10K/annotations.json"
    image3 = Image.open(image_path3)
    keypoints3 = get_ground_truth_keypoints(base_dir_path, image_path3, annotations_path, TOWEL_CHANNEL_CONFIG)
    image3 = draw_keypoints_on_image(image3, keypoints3, scale=4)

    image_path4 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/images/61.jpg"
    base_dir_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/"
    annotations_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/annotations.json"
    image4 = Image.open(image_path4)
    keypoints4 = get_ground_truth_keypoints(base_dir_path, image_path4, annotations_path, TSHIRT_CHANNEL_CONFIG)
    image4 = draw_keypoints_on_image(image4, keypoints4, scale=4)

    images = [image1, image2, image3, image4]
    combined_image = combine_images(images, n_rows=2)
    combined_image.save(data_dir / "icra25-sim-examples.png")


if __name__ == "__main__":
    examples()
