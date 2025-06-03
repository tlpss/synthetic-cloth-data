from pathlib import Path

from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from PIL import Image
from state_estimation.keypoint_detection.common import data_dir
from state_estimation.keypoint_detection.final_checkpoints import (
    FINETUNED_SYNTHETIC_SHORTS_CHECKPOINT,
    FINETUNED_SYNTHETIC_TOWELS_CHECKPOINT,
    FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT,
)
from state_estimation.keypoint_detection.real_baselines import (
    SHORTS_CHANNEL_CONFIG,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
)
from state_estimation.keypoint_detection.visualizations.visualization import (
    combine_images,
    draw_keypoints_on_image,
    local_inference,
    reorder_inference_keypoints_to_gt_keypoints,
)

example_dir = Path(__file__).parent / "images" / "icra25-examples"
example_dir.mkdir(exist_ok=True, parents=True)


def example(example_name, image_path, checkpoint, channel_config, cloth_type):
    image = Image.open(image_path)
    keypoints = local_inference(get_model_from_wandb_checkpoint(checkpoint), image, max_keypoints_per_channel=1)
    keypoints = reorder_inference_keypoints_to_gt_keypoints(keypoints, channel_config, cloth_type)
    image = draw_keypoints_on_image(image, keypoints, scale=6)
    image.save(example_dir / f"{example_name}.png")
    return image


if __name__ == "__main__":
    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_4"
        / "tshirts"
        / "2023-04-24_14-51-30_rgb_zed.png"
    )

    image1 = example("example1", image_path, FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT, TSHIRT_CHANNEL_CONFIG, "TSHIRT")

    # image_path = data_dir  /"artf_data" / "towels-test_resized_512x256" / "test" /"location_7" / "towels" / "2023-04-25_14-39-57_rgb_zed.png"
    image_path = (
        data_dir
        / "artf_data"
        / "towels-test_resized_512x256"
        / "test"
        / "location_7"
        / "towels"
        / "2023-04-25_14-49-43_rgb_zed.png"
    )
    image2 = example("example2", image_path, FINETUNED_SYNTHETIC_TOWELS_CHECKPOINT, TOWEL_CHANNEL_CONFIG, "TOWEL")

    image_path = (
        data_dir
        / "artf_data"
        / "shorts-test_resized_512x256"
        / "test"
        / "location_8"
        / "shorts"
        / "2023-04-25_15-44-31_rgb_smartphone.png"
    )
    image3 = example("example3", image_path, FINETUNED_SYNTHETIC_SHORTS_CHECKPOINT, SHORTS_CHANNEL_CONFIG, "SHORTS")

    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_6"
        / "tshirts"
        / "2023-04-24_13-02-22_rgb_zed.png"
    )
    image4 = example("example4", image_path, FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT, TSHIRT_CHANNEL_CONFIG, "TSHIRT")

    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_9"
        / "tshirts"
        / "2023-04-26_09-58-20_rgb_zed.png"
    )
    image5 = example("example5", image_path, FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT, TSHIRT_CHANNEL_CONFIG, "TSHIRT")

    image_path = (
        data_dir
        / "artf_data"
        / "shorts-test_resized_512x256"
        / "test"
        / "location_8"
        / "shorts"
        / "2023-04-25_15-43-10_rgb_zed.png"
    )
    image6 = example("example6", image_path, FINETUNED_SYNTHETIC_SHORTS_CHECKPOINT, SHORTS_CHANNEL_CONFIG, "SHORTS")

    images = [image4, image2, image3, image1, image6, image5]
    combined_image = combine_images(images, n_rows=2)
    combined_image.save(example_dir / "examples-combined.png")
