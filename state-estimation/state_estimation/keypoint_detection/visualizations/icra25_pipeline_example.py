from pathlib import Path

from PIL import Image
from state_estimation.keypoint_detection.real_baselines import TSHIRT_CHANNEL_CONFIG
from state_estimation.keypoint_detection.visualizations.visualization import (
    draw_keypoints_on_image,
    get_ground_truth_keypoints,
)

image_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/VIS/images/11.jpg"
image_base_dir = (
    "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/VIS/"
)
annotations_path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/VIS/annotations_train.json"


keypoints = get_ground_truth_keypoints(image_base_dir, image_path, annotations_path, TSHIRT_CHANNEL_CONFIG)
image = Image.open(image_path)
image = draw_keypoints_on_image(image, keypoints)

data_dir = Path(__file__).parent / "images"

image.save(data_dir / "icra25_pipeline_example.png")
