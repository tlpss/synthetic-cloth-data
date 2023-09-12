import dataclasses

import airo_blender as ab
import bpy
import numpy as np
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import (
    GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_RELATIVE_PATH,
)
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig
from synthetic_cloth_data.synthetic_images.scene_builder.utils.collisions import are_object_bboxes_in_collision


@dataclasses.dataclass
class DistractorConfig(AssetConfig):
    max_distractors: int = 6
    asset_json_relative_path: str = GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_RELATIVE_PATH


def add_distractors_to_scene(
    distractor_config: DistractorConfig, cloth_object: bpy.types.Object, surface_object: bpy.types.Object
):
    """adds a number of distractors on the table surface
    and makes sure they do not intersect with the cloth (though they could occlude keypoints of the cloth).)"""

    plane_x_size, plane_y_size = surface_object.dimensions[0], surface_object.dimensions[1]
    plane_x_min, plane_x_max = (
        surface_object.location[0] - plane_x_size / 2,
        surface_object.location[0] + plane_x_size / 2,
    )
    plane_y_min, plane_y_max = (
        surface_object.location[1] - plane_y_size / 2,
        surface_object.location[1] + plane_y_size / 2,
    )

    border_delta = 0.05

    n_distractors = np.random.randint(0, distractor_config.max_distractors + 1)
    distractor_objects = []
    for _ in range(n_distractors):
        distractor = np.random.choice(distractor_config.asset_list)
        distractor = ab.load_asset(**dict(distractor))
        distractor_objects.append(distractor)
        assert isinstance(distractor, bpy.types.Object)
        # add object to scene
        bpy.context.scene.collection.objects.link(distractor)

        # simple distractor placement

        # we only check with the axis-aligned bounding box of the cloth for simplicity.
        # in theory we should also check if there are no collisions between the distractors
        # assuming this does not matter too much for the representation learning task

        for _ in range(20):  # try 10 times to place the distractor, otherwise give up
            x = np.random.uniform(plane_x_min + border_delta, plane_x_max - border_delta)
            y = np.random.uniform(plane_y_min + border_delta, plane_y_max - border_delta)
            z = 0.001  # make sure the distractor is above the surface
            distractor.location[0] = x
            distractor.location[1] = y
            distractor.location[2] = z
            # apply the location
            bpy.context.view_layer.update()
            # check if the distractor is inside the cloth bbox
            if are_object_bboxes_in_collision(distractor, cloth_object):
                continue
            else:
                break
        if are_object_bboxes_in_collision(distractor, cloth_object):
            print("Warning: could not place distractor without collision with cloth")
            # remove the distractor
            bpy.context.scene.collection.objects.unlink(distractor)

    return distractor_objects
