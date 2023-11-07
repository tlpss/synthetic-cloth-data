import pathlib
import subprocess

data_dir = pathlib.Path(__file__).parents[1] / "data"
COMMAND = "keypoint-detection train --detect_only_visible_keypoints --augment_train"

DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 10,
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
    "json_validation_dataset_path": "",
    "max_epochs": 150,
    "maximal_gt_keypoint_pixel_distances": " '2 4 8'",  # quotes are need to avoid splitting in list
    "minimal_keypoint_extraction_pixel_distance": 1,
    "precision": 16,
    "seed": 2022,
    # determined based on hparam sweep
    "heatmap_sigma": 3,
    "learning_rate": 0.0002,
    "batch_size": 8,
    ###
    # "wandb_entity": "tlips",
    "wandb_project": "cloth-keypoints-paper",
    "wandb_name": None,
}


def create_train_command_from_arg_dict(arg_dict):
    command = COMMAND
    for key, value in arg_dict.items():
        command += f" --{key} {value}"
    return command


TOWEL_CHANNEL_CONFIG = "corner0:corner1:corner2:corner3"
TSHIRT_CHANNEL_CONFIG = "neck_left:neck_right:waist_left:waist_right:sleeve_left_top:sleeve_right_top:shoulder_left:shoulder_right:armpit_left:armpit_right:sleeve_left_bottom:sleeve_right_bottom"
SHORTS_CHANNEL_CONFIG = (
    "waist_left:waist_right:pipe_left_inner:pipe_right_inner:pipe_right_outer:pipe_left_outer:crotch"
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
