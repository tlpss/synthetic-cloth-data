from pathlib import Path

from PIL import Image
from state_estimation.keypoint_detection.visualizations.visualization import combine_images

data_dir = Path(__file__).parent / "images"


def examples():
    image_id = 4967
    image_path2 = f"/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/06-single-layer-flat-random-material/images/{image_id}.jpg"
    image_path1 = f"/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/05-single-layer-random-material/images/{image_id}.jpg"
    image_path3 = f"/home/tlips/Documents/synthetic-cloth-data/state-estimation/state_estimation/data/synthetic-data/TSHIRT/02-cloth3d-random-material/images/{image_id}.jpg"
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image3 = Image.open(image_path3)
    images = [image1, image2, image3]
    # cut left and right sides of image
    # images = [image.crop((100, 0, image.width - 100, image.height)) for image in images]
    combined_image = combine_images(images, n_rows=1)
    combined_image.save(data_dir / "sim-meshes.png")


if __name__ == "__main__":
    examples()
