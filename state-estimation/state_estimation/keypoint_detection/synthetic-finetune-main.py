"""finetune the sim only models on the aRTF dataset.
Checkpoints are obtained using the sythetic-main.py script"""

import subprocess

from state_estimation.keypoint_detection.common import (
    DEFAULT_DICT,
    SHORTS_CHANNEL_CONFIG,
    TOWEL_CHANNEL_CONFIG,
    TSHIRT_CHANNEL_CONFIG,
    create_train_command_from_arg_dict,
)
from state_estimation.keypoint_detection.real_baselines import (
    ARTF_SHORTS_TEST_PATH,
    ARTF_SHORTS_TRAIN_PATH,
    ARTF_SHORTS_VAL_PATH,
    ARTF_TOWEL_TEST_PATH,
    ARTF_TOWEL_TRAIN_PATH,
    ARTF_TOWEL_VAL_PATH,
    ARTF_TSHIRT_TEST_PATH,
    ARTF_TSHIRT_TRAIN_PATH,
    ARTF_TSHIRT_VAL_PATH,
)

if __name__ == "__main__":
    print(" synthetic tshirts")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["max_epochs"] = 20  # converges after a few epochs..
    arg_dict["keypoint_channel_configuration"] = TSHIRT_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TSHIRT_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TSHIRT_VAL_PATH
    arg_dict["json_dataset_path"] = ARTF_TSHIRT_TRAIN_PATH
    arg_dict["wandb_name"] = "synthetic-tshirts-main-finetune"
    arg_dict["wandb_checkpoint_artifact"] = "tlips/cloth-keypoints-paper/model-j6ny4mrs:v0"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic towels")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["max_epochs"] = 20  # converges after a few epochs..
    arg_dict["keypoint_channel_configuration"] = TOWEL_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_TOWEL_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_TOWEL_VAL_PATH
    arg_dict["json_dataset_path"] = ARTF_TOWEL_TRAIN_PATH
    arg_dict["wandb_name"] = "synthetic-towel-main-finetune"
    arg_dict["wandb_checkpoint_artifact"] = "tlips/cloth-keypoints-paper/model-u9sb4732:v0"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)

    print(" synthetic shorts")
    arg_dict = DEFAULT_DICT.copy()
    arg_dict["max_epochs"] = 20  # converges after a few epochs..
    arg_dict["keypoint_channel_configuration"] = SHORTS_CHANNEL_CONFIG
    arg_dict["json_test_dataset_path"] = ARTF_SHORTS_TEST_PATH
    arg_dict["json_validation_dataset_path"] = ARTF_SHORTS_VAL_PATH
    arg_dict["json_dataset_path"] = ARTF_SHORTS_TRAIN_PATH
    arg_dict["wandb_name"] = "synthetic-shorts-main-finetune"
    arg_dict["wandb_checkpoint_artifact"] = "tlips/cloth-keypoints-paper/model-rw5ta1cd:v0"

    subprocess.run(create_train_command_from_arg_dict(arg_dict), shell=True)
