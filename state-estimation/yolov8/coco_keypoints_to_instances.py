import copy
import json
from typing import List, Optional

from airo_dataset_tools.data_parsers.coco import (
    CocoCategory,
    CocoInstanceAnnotation,
    CocoInstancesDataset,
    CocoKeypointsDataset,
)


def convert_coco_keypoints_to_bboxes(  # noqa: C901
    coco_keypoints_dataset: CocoKeypointsDataset,
    bbox_size: int = 20,
    keypoints_to_combine: Optional[List[List[str]]] = None,
    map_occluded_keypoints: bool = False,
) -> CocoInstancesDataset:
    categories = coco_keypoints_dataset.categories
    categories_id_mapping = {category.id: category for category in categories}
    new_categories = [category for category in categories]

    keypoint_type_mapping = {}
    if keypoints_to_combine is not None:
        for mapping in keypoints_to_combine:
            for keypoint_type in mapping:
                keypoint_type_mapping[keypoint_type] = mapping[0]

    for category in categories:
        for idx, keypoint_type in enumerate(category.keypoints):
            identifier = f"{category.name}.{keypoint_type}"
            if identifier in keypoint_type_mapping and keypoint_type_mapping[identifier] != identifier:
                continue
            new_categories.append(
                CocoCategory(
                    id=int("999" + str(category.id) + "999" + str(idx)), name=identifier, supercategory=category.name
                )
            )

    new_categories_name_to_id = {category.name: category.id for category in new_categories}
    new_annotations = copy.deepcopy(coco_keypoints_dataset.annotations)
    annotation_id = max([annotation.id for annotation in coco_keypoints_dataset.annotations]) + 1
    for annotation in coco_keypoints_dataset.annotations:
        category_id = annotation.category_id
        category = categories_id_mapping[category_id]
        keypoints = annotation.keypoints
        # split array in chunks of 3
        keypoints = [keypoints[i : i + 3] for i in range(0, len(keypoints), 3)]
        for idx, keypoint in enumerate(keypoints):
            if keypoint[2] == 0:
                continue
            if keypoint[2] == 1 and not map_occluded_keypoints:
                continue
            keypoint_category_name = f"{category.name}.{category.keypoints[idx]}"
            if keypoint_category_name in keypoint_type_mapping.keys():
                keypoint_category_name = keypoint_type_mapping[keypoint_category_name]
            keypoint_category_id = new_categories_name_to_id[keypoint_category_name]
            new_annotations.append(
                CocoInstanceAnnotation(
                    id=annotation_id,
                    image_id=annotation.image_id,
                    category_id=keypoint_category_id,
                    bbox=[keypoint[0] - bbox_size, keypoint[1] - bbox_size, 2 * bbox_size, 2 * bbox_size],
                    area=4 * bbox_size**2,
                    segmentation=[],
                    iscrowd=0,
                )
            )
            annotation_id += 1
    return CocoInstancesDataset(
        images=coco_keypoints_dataset.images, annotations=new_annotations, categories=new_categories
    )


if __name__ == "__main__":

    def convert_coco_keypoints_dataset_to_instances_dataset(
        json_path: str,
        keypoints_mapping: Optional[List[List[str]]] = None,
        map_occluded_keypoints: bool = False,
        bbox_size: int = 20,
    ) -> CocoInstancesDataset:
        with open(json_path, "r") as f:
            coco_keypoints_dataset = CocoKeypointsDataset(**json.load(f))
        coco_instances_dataset = convert_coco_keypoints_to_bboxes(
            coco_keypoints_dataset,
            keypoints_to_combine=keypoints_mapping,
            bbox_size=bbox_size,
            map_occluded_keypoints=map_occluded_keypoints,
        )
        with open(json_path.replace(".json", f"_keypoints_to_instances_b{bbox_size}.json"), "w") as f:
            json.dump(coco_instances_dataset.dict(), f, indent=2)
        return coco_instances_dataset

    path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/yolov8/data/sim/TOWEL/06-tailored-meshes/annotations_train.json"
    path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/yolov8/data/sim/TOWEL/06-tailored-meshes/annotations_val.json"
    # path = "/home/tlips/Documents/synthetic-cloth-data/state-estimation/yolov8/data/aRTFClothes/towels-train_resized_512x256/towels-train.json"
    # path ="/home/tlips/Documents/synthetic-cloth-data/state-estimation/yolov8/data/aRTFClothes/towels-test_resized_512x256/towels-test.json"
    convert_coco_keypoints_dataset_to_instances_dataset(
        path, [["towel.corner0", "towel.corner1", "towel.corner2", "towel.corner3"]], bbox_size=20
    )
