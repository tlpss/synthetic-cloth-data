import subprocess

from state_estimation.keypoint_detection.common import (
    SHORTS_CHANNEL_CONFIG,
    SYNTH_DEFAULT_DICT,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
    data_dir,
)
from state_estimation.keypoint_detection.real_baselines import (
    ARTF_SHORTS_TEST_PATH,
    ARTF_SHORTS_TRAIN_PATH,
    ARTF_TOWEL_TEST_PATH,
    ARTF_TOWEL_TRAIN_PATH,
    ARTF_TSHIRT_TEST_PATH,
    ARTF_TSHIRT_TRAIN_PATH,
)

if __name__ == "__main__":
    print(" synthetic towels")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TOWEL_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TOWEL_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TOWEL_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TOWEL" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-towels-main"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic shorts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = SHORTS_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_SHORTS_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_SHORTS_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "SHORTS" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-shorts-main"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic tshirts")
    arg_dict = SYNTH_DEFAULT_DICT.copy()
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_dataset_path"] = (
        data_dir / "synthetic-data" / "TSHIRT" / "single-layer-random-material-10K" / "annotations_train.json"
    )
    arg_dict["wandb_name"] = "synthetic-tshirts-main"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
