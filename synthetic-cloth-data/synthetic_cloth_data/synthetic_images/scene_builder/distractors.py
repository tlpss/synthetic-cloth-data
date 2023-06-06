import dataclasses

import airo_blender as ab
import bpy
import numpy as np
from mathutils import Vector
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import (
    GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_RELATIVE_PATH,
)
from synthetic_cloth_data.synthetic_images.scene_builder.utils.assets import AssetConfig


@dataclasses.dataclass
class DistractorConfig(AssetConfig):
    max_distractors: int = 2
    asset_json_relative_path: str = GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_RELATIVE_PATH


def add_distractors_to_scene(
    distractor_config: DistractorConfig, cloth_object: bpy.types.Object, surface_object: bpy.types.Object
):
    """adds a number of distractors on the cloth surface
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

    cloth_bbox = cloth_object.bound_box
    cloth_bbox = [cloth_object.matrix_world @ Vector(corner) for corner in cloth_bbox]
    cloth_x_min = min([corner[0] for corner in cloth_bbox])
    cloth_x_max = max([corner[0] for corner in cloth_bbox])
    cloth_y_min = min([corner[1] for corner in cloth_bbox])
    cloth_y_max = max([corner[1] for corner in cloth_bbox])

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
        # in teory we should also check if there are no collisions between the distractors
        # assuming this does not matter too much for the representation learning task

        for _ in range(10):  # try 10 times to place the distractor, otherwise give up
            x = np.random.uniform(plane_x_min + border_delta, plane_x_max - border_delta)
            y = np.random.uniform(plane_y_min + border_delta, plane_y_max - border_delta)
            z = 0.001  # make sure the distractor is above the surface

            # check if the distractor is inside the cloth bbox
            if x > cloth_x_min and x < cloth_x_max and y > cloth_y_min and y < cloth_y_max:
                # sample new points.
                continue
            else:
                break

        distractor.location[0] = x
        distractor.location[1] = y
        distractor.location[2] = z

    return distractor_objects
