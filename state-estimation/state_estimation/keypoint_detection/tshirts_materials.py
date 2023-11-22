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
        data_dir / "synthetic-data" / "TSHIRT" / "03-single-layer-full-material" / "annotations.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-single-full-material"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("hsv material tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "07-single-layer-hsv-material" / "annotations.json"
    )

    arg_dict["wandb_name"] = "synthetic-tshirts-single-hsv-material"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("random material tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "05-single-layer-random-material" / "annotations.json"
    )

    arg_dict["wandb_name"] = "synthetic-tshirts-single-random-material"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
