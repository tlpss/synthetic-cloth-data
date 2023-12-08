"""similar to synthetic-main.py, but with a synthetic validation dataset. Used to quantify sim2sim performance, for illustration of the sim2real gap."""

import subprocess

from state_estimation.keypoint_detection.common import (
    SHORTS_CHANNEL_CONFIG,
    SYNTH_DEFAULT_DICT,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
    data_dir,
)

if __name__ == "__main__":
    print(" synthetic tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["max_epochs"] = 20

    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_validation_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "single-layer-random-material-10K" / "annotations_val.json"
    )
    arg_dict["json_test_dataset_path"] = arg_dict["json_validation_dataset_path"]
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-sim2sim"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic towels")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TOWEL_CHANNEL_CONFIG

    arg_dict["json_validation_dataset_path"] = (
        data_dir / "synthetic-data" / "TOWEL" / "single-layer-random-material-10K" / "annotations_val.json"
    )
    arg_dict["json_test_dataset_path"] = arg_dict["json_validation_dataset_path"]
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TOWEL" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-towels-sim2sim"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic shorts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = SHORTS_CHANNEL_CONFIG
    arg_dict["json_validation_dataset_path"] = (
        data_dir / "synthetic-data" / "SHORTS" / "single-layer-random-material-10K" / "annotations_val.json"
    )
    arg_dict["json_test_dataset_path"] = arg_dict["json_validation_dataset_path"]

    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "SHORTS" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-shorts-sim2sim"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
