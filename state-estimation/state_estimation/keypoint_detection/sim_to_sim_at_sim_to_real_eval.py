"""evaluate the sim2sim performance on a synthetic validation dataset of best sim2real checkpoints.
Note that this is not the common sim2sim vs sim2real measure, which would be to compare sim2real best checkpoint against sim2sim best checkpoint."""


import pathlib
import subprocess

data_dir = pathlib.Path(__file__).parents[1] / "data"
COMMAND = "keypoint-detection train --detect_only_visible_keypoints --augment_train"


from state_estimation.keypoint_detection.final_checkpoints import (
    SYNTHETIC_SHORTS_CHECKPOINT,
    SYNTHETIC_TOWELS_CHECKPOINT,
    SYNTHETIC_TSHIRTS_CHECKPOINT,
)


def evaluate(wandb_checkpoint, dataset_path):
    COMMAND = f"keypoint-detection eval --wandb_checkpoint {wandb_checkpoint} --test_json_path {dataset_path}"
    subprocess.run(COMMAND, shell=True, check=True)


if __name__ == "__main__":
    tshirts = data_dir / "synthetic-data" / "TSHIRT" / "single-layer-random-material-10K" / "annotations_val.json"
    towels = data_dir / "synthetic-data" / "TOWEL" / "single-layer-random-material-10K" / "annotations_train.json"
    shorts = data_dir / "synthetic-data" / "SHORTS" / "single-layer-random-material-10K" / "annotations_val.json"

    evaluate(SYNTHETIC_TSHIRTS_CHECKPOINT, tshirts)
    evaluate(SYNTHETIC_TOWELS_CHECKPOINT, towels)
    evaluate(SYNTHETIC_SHORTS_CHECKPOINT, shorts)
