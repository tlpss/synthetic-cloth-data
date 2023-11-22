import subprocess

from state_estimation.keypoint_detection.common import (
    SYNTH_DEFAULT_DICT,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
    data_dir,
)
from state_estimation.keypoint_detection.real_baselines import ARTF_TSHIRT_TEST_PATH, ARTF_TSHIRT_TRAIN_PATH

if __name__ == "__main__":
    print("cloth3d meshes tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "02-cloth3d-random-material" / "annotations.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-cloth3d-random-material"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("single layer meshes tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "05-single-random-material" / "annotations.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-single-layer-random-material"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("single layer flat meshes tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "06-single-layer-flat-random-material" / "annotations.json"
    )

    arg_dict["wandb_name"] = "synthetic-tshirts-meshes-single-flat-random-material"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
