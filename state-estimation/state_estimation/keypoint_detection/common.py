import pathlib

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

SYNTH_DEFAULT_DICT = DEFAULT_DICT.copy()
SYNTH_DEFAULT_DICT["ap_epoch_freq"] = 3
SYNTH_DEFAULT_DICT["max_epochs"] = 15


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
