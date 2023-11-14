import subprocess

from state_estimation.keypoint_detection.common import (
    DEFAULT_DICT,
    SHORTS_CHANNEL_CONFIG,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
    data_dir,
)

ARTF_TOWEL_TRAIN_PATH = data_dir / "artf_data" / "towels-train_resized_512x256/towels-train.json"
ARTF_TOWEL_TEST_PATH = data_dir / "artf_data" / "towels-test_resized_512x256/towels-test.json"
ARTF_TOWEL_VAL_PATH = data_dir / "artf_data" / "towels-val_resized_512x256/towels-val.json"

ARTF_TSHIRT_TRAIN_PATH = data_dir / "artf_data" / "tshirts-train_resized_512x256/tshirts-train.json"
ARTF_TSHIRT_TEST_PATH = data_dir / "artf_data" / "tshirts-test_resized_512x256/tshirts-test.json"
ARTF_TSHIRT_VAL_PATH = data_dir / "artf_data" / "tshirts-val_resized_512x256/tshirts-val.json"

ARTF_SHORTS_TRAIN_PATH = data_dir / "artf_data" / "shorts-train_resized_512x256/shorts-train.json"
ARTF_SHORTS_TEST_PATH = data_dir / "artf_data" / "shorts-test_resized_512x256/shorts-test.json"
ARTF_SHORTS_VAL_PATH = data_dir / "artf_data" / "shorts-val_resized_512x256/shorts-val.json"

if __name__ == "__main__":
    print("Towel baseline")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["json_dataset_path"] = ARTF_TOWEL_TRAIN_PATH
    arg_dict["json_test_dataset_path"] = ARTF_TOWEL_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TOWEL_VAL_PATH
    arg_dict["keypoint_channel_configuration"] = TOWEL_CHANNEL_CONFIG
    arg_dict["wandb_name"] = "towels-real"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("Tshirt baseline")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["json_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_VAL_PATH
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["wandb_name"] = "tshirts-real"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print("Shorts baseline")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["json_dataset_path"] = ARTF_SHORTS_TRAIN_PATH
    arg_dict["json_test_dataset_path"] = ARTF_SHORTS_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_SHORTS_VAL_PATH
    arg_dict["keypoint_channel_configuration"] = SHORTS_CHANNEL_CONFIG
    arg_dict["wandb_name"] = "shorts-real"
    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
