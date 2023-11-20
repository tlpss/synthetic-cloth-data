import subprocess

from state_estimation.keypoint_detection.common import (
    SYNTH_DEFAULT_DICT,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
    data_dir,
)
from state_estimation.keypoint_detection.real_baselines import ARTF_TSHIRT_TEST_PATH, ARTF_TSHIRT_TRAIN_PATH

if __name__ == "__main__":
    print("default-material tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "00-cloth3d-default-material" / "annotations.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-material-full"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("hsv material tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "01-cloth3d-hsv-material" / "annotations.json"
    )

    arg_dict["wandb_name"] = "synthetic-tshirts-material-hsv"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("random material tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "02-cloth3d-random-material" / "annotations.json"
    )

    arg_dict["wandb_name"] = "synthetic-tshirts-material-random"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
