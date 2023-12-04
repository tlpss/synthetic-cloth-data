from pathlib import Path

from PIL import Image
from state_estimation.keypoint_detection.visualizations.visualization import combine_images

data_dir = Path(__file__).parent / "images"


def examples():
    image_path1 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/images/259.jpg"
    image_path2 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/SHORTS/single-layer-random-material-10K/images/840.jpg"
    image_path3 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TOWEL/single-layer-random-material-10K/images/116.jpg"
    image_path4 = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/single-layer-random-material-10K/images/61.jpg"
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image3 = Image.open(image_path3)
    image4 = Image.open(image_path4)
    images = [image1, image2, image3, image4]
    combined_image = combine_images(images, n_rows=2)
    combined_image.save(data_dir / "sim-examples.png")


if __name__ == "__main__":
    examples()
