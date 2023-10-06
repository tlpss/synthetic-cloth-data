"""takes in a COCO CLoth dataset and orders the keypoints (as apparent in the image) to deal with symmetries.

For towels this is most complex and orders the keypoints based on their position in the image, under the assumption that the keypoints have already been labeled
cyclical.

 For others it is simply matter of 'flipping' if the cloth is upside down, because we don't take this into account.
"""


from typing import List

import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset


def order_dataset_keypoints(dataset: CocoKeypointsDataset) -> CocoKeypointsDataset:
    category_id_to_category_mapping = {category.id: category for category in dataset.categories}

    for annotation in dataset.annotations:
        if category_id_to_category_mapping[annotation.category_id].name == "towel":
            annotation.keypoints = _order_towel_keypoints(annotation.keypoints, annotation.bbox)

        elif category_id_to_category_mapping[annotation.category_id].name == "tshirt":
            raise NotImplementedError

        else:
            raise ValueError(f"unknown category {category_id_to_category_mapping[annotation.category_id].name}")

    return dataset


def _order_towel_keypoints(keypoints: List[float], bbox: List[float]) -> List[float]:
    x_min, y_min, width, height = bbox
    keypoints = np.array(keypoints).reshape(-1, 3)
    keypoints_2D = keypoints[:, :2]

    # keypoints are in cyclical order but we need to break symmetries by having a starting point in the image viewpoint
    bbox_top_left = (x_min, y_min)

    # find the keypoint that is closest to the top left corner of the bounding box
    distances = [np.linalg.norm(np.array(keypoint_2D) - np.array(bbox_top_left)) for keypoint_2D in keypoints_2D]
    starting_keypoint_index = np.argmin(distances)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    # closest to the topright corner of the bbox

    bbox_top_right = (x_min + width, y_min)
    distances = [
        np.linalg.norm(
            np.array(keypoints_2D[(starting_keypoint_index + i) % len(keypoints_2D)]) - np.array(bbox_top_right)
        )
        for i in [-1, +1]
    ]
    direction = -1 if np.argmin(distances) == 0 else +1
    second_keypoint_index = (starting_keypoint_index + direction) % len(keypoints_2D)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    direction = second_keypoint_index - starting_keypoint_index

    order = [starting_keypoint_index]
    for i in range(1, len(keypoints_2D)):
        order.append((starting_keypoint_index + i * direction) % len(keypoints_2D))

    print(order)
    new_keypoints = np.array([keypoints[i] for i in order])
    return new_keypoints.flatten().tolist()


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("annotations-json-path", type=click.Path(exists=True))
    def order_dataset_keypoints_cli(annotations_json_path: str) -> None:
        """Order the keypoints in a COCO dataset"""
        dataset = CocoKeypointsDataset.parse_file(annotations_json_path)
        dataset = order_dataset_keypoints(dataset)
        target_path = annotations_json_path.replace(".json", "_ordered.json")
        with open(target_path, "w") as f:
            f.write(dataset.json(indent=4))

    order_dataset_keypoints_cli()
