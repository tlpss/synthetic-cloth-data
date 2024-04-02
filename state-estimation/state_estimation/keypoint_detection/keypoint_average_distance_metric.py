import torch
from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_average_error_for_dataset(
    model: KeypointDetector, dataset_json_path, channel_config: list[list[str]], detect_only_visible_keypoints
):
    dataset = COCOKeypointsDataset(
        dataset_json_path,
        keypoint_channel_configuration=channel_config,
        detect_only_visible_keypoints=detect_only_visible_keypoints,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    model.cuda()

    errors = [[] for _ in range(len(channel_config))]
    for image, keypoints in tqdm(dataloader):
        image = image.cuda()
        heatmaps = model(image)
        keypoints = keypoints
        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, max_keypoints=1)[0]
        for i in range(len(channel_config)):
            if len(predicted_keypoints[i]) == 0:
                print("no keypoints found")
                continue
            if len(keypoints[i]) == 0:
                # print("no GT keypoints found")
                continue
            kp = torch.tensor(predicted_keypoints[i][0], dtype=torch.float32)
            gt_kp = torch.tensor(keypoints[i][0], dtype=torch.float32)
            l2_error = torch.norm(kp - gt_kp)
            errors[i].append(l2_error.item())

    average_errors = [sum(errors[i]) / len(errors[i]) for i in range(len(channel_config))]
    for i in range(len(channel_config)):
        print(f"Average error for channel {channel_config[i]}: {average_errors[i]}")
    print(f"Average error: {sum(average_errors)/len(average_errors)}")

    mae_dict = {}
    full_dict = {}
    for i in range(len(channel_config)):
        channel_name = "" + "-".join(channel_config[i])
        mae_dict[channel_name] = average_errors[i]
        full_dict[channel_name] = errors[i]
    mae_dict["average"] = sum(average_errors) / len(average_errors)

    return mae_dict, full_dict


if __name__ == "__main__":
    from state_estimation.keypoint_detection.common import (
        SHORTS_CHANNEL_CONFIG,
        TOWEL_CHANNEL_CONFIG,
        TSHIRT_CHANNEL_CONFIG,
        data_dir,
    )
    from state_estimation.keypoint_detection.final_checkpoints import ARTIFACT_DICT
    from state_estimation.keypoint_detection.real_baselines import (
        ARTF_SHORTS_TEST_PATH,
        ARTF_TOWEL_TEST_PATH,
        ARTF_TSHIRT_TEST_PATH,
    )

    error_dict = {}
    for key, value in ARTIFACT_DICT.items():
        if "tshirt" in key:
            wandb_checkpoint = value
            if "sim" in key:
                dataset = (
                    data_dir
                    / "synthetic-data"
                    / "TSHIRT"
                    / "single-layer-random-material-10K"
                    / "annotations_val.json"
                )
            else:
                dataset = ARTF_TSHIRT_TEST_PATH
            keypoints = TSHIRT_CHANNEL_CONFIG.split(":")
            keypoints = [channel.split(",") for channel in keypoints]
        elif "towel" in key:
            wandb_checkpoint = value
            if "sim" in key:
                dataset = (
                    data_dir / "synthetic-data" / "TOWEL" / "single-layer-random-material-10K" / "annotations_val.json"
                )
            else:
                dataset = ARTF_TOWEL_TEST_PATH
            keypoints = TOWEL_CHANNEL_CONFIG.split(":")
            keypoints = [channel.split(",") for channel in keypoints]
        elif "shorts" in key:
            wandb_checkpoint = value
            if "sim" in key:
                dataset = (
                    data_dir
                    / "synthetic-data"
                    / "SHORTS"
                    / "single-layer-random-material-10K"
                    / "annotations_val.json"
                )
            else:
                dataset = ARTF_SHORTS_TEST_PATH
            keypoints = SHORTS_CHANNEL_CONFIG.split(":")
            keypoints = [channel.split(",") for channel in keypoints]
        else:
            raise ValueError("Unknown artifact key")

        print(f"Calculating average error for {key}")
        print(f"dataset = {dataset}")
        print(f"keypoints = {keypoints}")

        model = get_model_from_wandb_checkpoint(wandb_checkpoint).cuda()
        avg_errors, d = calculate_average_error_for_dataset(
            model, dataset, keypoints, detect_only_visible_keypoints=True
        )
        error_dict[key] = avg_errors

        # save dict as json
        import json
        import pathlib

        file_path = pathlib.Path(__file__).parent
        # with open(file_path / "average_keypoint_distances.json", "w") as f:
        #     json.dump(error_dict, f, indent=4)

        with open(file_path / "akd" / f"{key}.json", "w") as f:
            json.dump(d, f, indent=4)
