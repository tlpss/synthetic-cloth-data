import pathlib

from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from PIL import Image
from state_estimation.keypoint_detection.common import data_dir
from state_estimation.keypoint_detection.final_checkpoints import (
    FINETUNED_SYNTHETIC_SHORTS_CHECKPOINT,
    FINETUNED_SYNTHETIC_TOWELS_CHECKPOINT,
    FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT,
    REAL_SHORTS_CHECKPOINT,
    REAL_TOWELS_CHECKPOINT,
    REAL_TSHIRTS_CHECKPOINT,
    SYNTHETIC_SHORTS_CHECKPOINT,
    SYNTHETIC_TOWELS_CHECKPOINT,
    SYNTHETIC_TSHIRTS_CHECKPOINT,
)
from state_estimation.keypoint_detection.real_baselines import (
    ARTF_SHORTS_TEST_PATH,
    ARTF_TOWEL_TEST_PATH,
    ARTF_TSHIRT_TEST_PATH,
    SHORTS_CHANNEL_CONFIG,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
)
from state_estimation.keypoint_detection.visualizations.visualization import (
    draw_keypoints_on_image,
    get_ground_truth_keypoints,
    horizontally_combine_images,
    local_inference,
)

output_dir = pathlib.Path(__file__).parent / "images"


def example_1():
    """shows confusion / FP on the detector trained on real data"""
    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_9"
        / "tshirts"
        / "2023-04-26_09-58-20_rgb_zed.png"
    )
    json_annotations_path = ARTF_TSHIRT_TEST_PATH
    json_annotations_path.parent
    example(
        "imperfect_example1",
        image_path,
        json_annotations_path,
        TSHIRT_CHANNEL_CONFIG,
        REAL_TSHIRTS_CHECKPOINT,
        SYNTHETIC_TSHIRTS_CHECKPOINT,
        FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT,
    )


def example2():
    """shows how all struggle with folds"""
    image_path = (
        data_dir
        / "artf_data"
        / "towels-test_resized_512x256"
        / "test"
        / "location_5"
        / "towels"
        / "2023-04-24_16-20-05_rgb_zed.png"
    )
    json_annotations_path = ARTF_TOWEL_TEST_PATH
    json_annotations_path.parent
    example(
        "imperfect_example2",
        image_path,
        json_annotations_path,
        TOWEL_CHANNEL_CONFIG,
        REAL_TOWELS_CHECKPOINT,
        SYNTHETIC_TOWELS_CHECKPOINT,
        FINETUNED_SYNTHETIC_TOWELS_CHECKPOINT,
    )


def example3():
    """label ambiguity"""
    image_path = (
        data_dir
        / "artf_data"
        / "tshirts-test_resized_512x256"
        / "test"
        / "location_3"
        / "tshirts"
        / "2023-04-24_17-13-02_rgb_zed.png"
    )
    json_annotations_path = ARTF_TSHIRT_TEST_PATH
    json_annotations_path.parent
    example(
        "imperfect_example3",
        image_path,
        json_annotations_path,
        TSHIRT_CHANNEL_CONFIG,
        REAL_TSHIRTS_CHECKPOINT,
        SYNTHETIC_TSHIRTS_CHECKPOINT,
        FINETUNED_SYNTHETIC_TSHIRTS_CHECKPOINT,
    )


def example4():
    """mesh features not seen -> confused"""
    image_path = (
        data_dir
        / "artf_data"
        / "shorts-test_resized_512x256"
        / "test"
        / "location_2"
        / "shorts"
        / "2023-04-24_12-17-00_rgb_zed.png"
    )
    json_annotations_path = ARTF_SHORTS_TEST_PATH
    json_annotations_path.parent
    example(
        "imperfect_example4",
        image_path,
        json_annotations_path,
        SHORTS_CHANNEL_CONFIG,
        REAL_SHORTS_CHECKPOINT,
        SYNTHETIC_SHORTS_CHECKPOINT,
        FINETUNED_SYNTHETIC_SHORTS_CHECKPOINT,
    )


def example(
    example_name,
    image_path,
    json_annotations_path,
    channel_config,
    real_checkpoint,
    sim_checkpoint,
    finetuned_checkpoint,
):

    example_path = output_dir / example_name
    example_path.mkdir(exist_ok=True, parents=True)

    base_dir = json_annotations_path.parent

    real_image = Image.open(image_path)
    keypoints = local_inference(get_model_from_wandb_checkpoint(real_checkpoint), real_image)
    real_image = draw_keypoints_on_image(real_image, keypoints)
    real_image.save(example_path / f"{example_name}_real_predictions.png")

    sim_image = Image.open(image_path)
    keypoints = local_inference(get_model_from_wandb_checkpoint(sim_checkpoint), sim_image)
    sim_image = draw_keypoints_on_image(sim_image, keypoints)
    sim_image.save(example_path / f"{example_name}_sim_predictions.png")

    sim_real_image = Image.open(image_path)
    keypoints = local_inference(get_model_from_wandb_checkpoint(finetuned_checkpoint), sim_real_image)
    sim_real_image = draw_keypoints_on_image(sim_real_image, keypoints)
    sim_real_image.save(example_path / f"{example_name}_sim+real_predictions.png")

    gt_image = Image.open(image_path)
    gt_keypoints = get_ground_truth_keypoints(base_dir, image_path, json_annotations_path, channel_config)
    gt_image = draw_keypoints_on_image(gt_image, gt_keypoints)
    gt_image.save(example_path / f"{example_name}_ground_truth.png")

    combined_image = horizontally_combine_images([gt_image, real_image, sim_image, sim_real_image])
    combined_image.save(example_path / f"{example_name}_combined.png")


if __name__ == "__main__":
    example_1()
    example2()
    example3()
    example4()
