from typing import Tuple

import bpy
import loguru
import numpy as np
from airo_blender.materials import add_material
from airo_typing import Vector3DType, Vectors3DType
from linen.blender.curve import add_line_segment
from linen.blender.path import add_path
from linen.blender.points import add_points
from linen.folding.trajectories.circular_fold import circular_arc_position_trajectory
from linen.path.path import Path
from synthetic_cloth_data.meshes.cloth_meshes import attach_cloth_sim, generate_cloth_object, visualize_keypoints
from synthetic_cloth_data.meshes.generate_flat_meshes import _unwrap_cloth_mesh
from synthetic_cloth_data.meshes.mesh_operations import subdivide_mesh
from synthetic_cloth_data.utils import CLOTH_TYPES

logger = loguru.logger


def project_vector_onto_plane(vector: Vector3DType, plane_normal: Vector3DType) -> Vector3DType:
    return vector - np.dot(vector, plane_normal) * plane_normal


def random_point_in_convex_hull(points: Vectors3DType) -> Vector3DType:
    weights = np.random.uniform(0, 1, len(points))
    weights /= weights.sum()
    return (points * weights[:, None]).sum(axis=0)


def random_towel_fold_line(keypoints: Vectors3DType) -> Tuple[Vector3DType, Vector3DType]:
    fold_line_point = random_point_in_convex_hull(keypoints)

    # TODO consider case where these are parallel
    towel_normal = np.cross(keypoints[1] - keypoints[0], keypoints[2] - keypoints[0])
    towel_normal /= np.linalg.norm(towel_normal)

    random_direction = np.random.uniform(-1, 1, 3)
    random_direction /= np.linalg.norm(random_direction)

    random_direction_projected = project_vector_onto_plane(random_direction, towel_normal)

    return fold_line_point, random_direction_projected


def animate_object_location_along_path(object: bpy.types.Object, position_path: Path, frame_start: int = 1):
    fps = bpy.context.scene.render.fps
    frame_interval = 1 / fps
    num_frames = int(np.ceil(position_path.duration / frame_interval))

    for i in range(num_frames):
        t = i * frame_interval
        position = position_path(t)
        object.location = position
        object.keyframe_insert(data_path="location", frame=frame_start + i)


def animate_grasped_vertex(ob: bpy.types.Object, grasped_vertex_id: int, position_trajectory: Path, frame_start=1):
    bpy.ops.object.empty_add()
    empty = bpy.context.object
    empty.empty_display_size = 0.1
    animate_object_location_along_path(empty, position_trajectory, frame_start=frame_start)
    empty.location = position_trajectory(0)  # reset location for hook modifier
    bpy.context.view_layer.update()

    path_end_frame = 1 + int(np.ceil(position_trajectory.duration * bpy.context.scene.render.fps))

    # Create a vertex group for the single grasped vertex
    grasped_vertex_group = ob.vertex_groups.new(name="grasped")
    grasped_vertex_group.add([grasped_vertex_id], 1.0, "REPLACE")

    # Add a VertexWeightMix modifier to allow animating the vertex weight of the grsaped vertex
    # Setting the vertex weight to 0 will release the vertex from the cloth pin group and hook.
    weight_modifier = ob.modifiers.new(name="VertexWeightMix", type="VERTEX_WEIGHT_MIX")
    weight_modifier.vertex_group_a = grasped_vertex_group.name
    weight_modifier.mix_set = "OR"

    weight_modifier.default_weight_b = 1.0
    weight_modifier.keyframe_insert(data_path="default_weight_b", frame=1)
    weight_modifier.default_weight_b = 0.0
    weight_modifier.keyframe_insert(data_path="default_weight_b", frame=path_end_frame)
    # set the interpolation mode to constant
    action = ob.animation_data.action
    fcurve = action.fcurves.find(data_path='modifiers["VertexWeightMix"].default_weight_b')
    for point in fcurve.keyframe_points:
        point.interpolation = "CONSTANT"

    # Add a hook modifier to the empty and set the object to the empty and the vertex group to the grasped vertex group
    hook_modifier = ob.modifiers.new(name="Hook", type="HOOK")
    hook_modifier.vertex_group = grasped_vertex_group.name
    hook_modifier.object = empty
    return grasped_vertex_group, path_end_frame


def generate_random_deformed_towel(random_seed: int = 2023, debug_visualizations=False) -> bpy.types.Object:
    np.random.seed(random_seed)
    bpy.ops.object.delete()  # Delete default cube
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    bpy.ops.object.modifier_add(type="COLLISION")
    bpy.context.object.collision.cloth_friction = np.random.uniform(5.0, 30.0)
    plane = bpy.context.object
    subdivide_mesh(plane, 10)
    add_material(plane, (1, 0.5, 0.5, 1.0))

    # for idx in tqdm.trange(10):
    ob, kp = generate_cloth_object(CLOTH_TYPES.TOWEL)
    # TODO: reuse existing meshes
    _unwrap_cloth_mesh(ob)
    # attach_cloth_sim(ob)
    ob.location = np.array([0, 0, 1.0])
    # update the object's world matrix
    # cf. https://blender.stackexchange.com/questions/27667/incorrect-matrix-world-after-transformation
    x_rot, y_rot = np.random.uniform(0, np.pi / 2 * 0.2, 2)
    ob.rotation_euler = np.array([x_rot, y_rot, 0])
    bpy.context.view_layer.update()

    # for now no very large crumplings such as folded in half
    # these would probably require pinning some vertices and animating them.
    # see https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/subdivision_surface.html
    # and https://www.youtube.com/watch?v=C8C4GntM60o for animation

    # apply all towel transforms
    # ob.select_set(True)
    # bpy.context.view_layer.objects.active = ob
    # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    keypoints = np.array([ob.matrix_world @ ob.data.vertices[kid].co for kid in kp.values()])

    fold_line = random_towel_fold_line(keypoints)

    fold_line_point, fold_line_direction = fold_line

    if debug_visualizations:
        add_points([fold_line_point], radius=0.02, color=(1, 0, 0, 1))
        add_line_segment(fold_line_point - fold_line_direction, fold_line_point + fold_line_direction)

    # choose a random keypoint from kp
    grasped_vertex_id = int(np.random.choice(list(kp.values())))
    grasp_location = np.array(ob.matrix_world @ ob.data.vertices[grasped_vertex_id].co)

    fold_arc_angle = np.random.uniform(0, np.pi)
    position_trajectory = circular_arc_position_trajectory(
        grasp_location, *fold_line, max_angle=fold_arc_angle, speed=0.4
    )

    if debug_visualizations:
        red = (1, 0, 0, 1)
        add_path(position_trajectory, points_per_second=25, color=red)

    grasped_vertex_group, path_end_frame = animate_grasped_vertex(ob, grasped_vertex_id, position_trajectory)

    # Set the pin group to the vertex group
    attach_cloth_sim(ob, solifify=True)
    cloth_modifier = ob.modifiers["Cloth"]
    cloth_modifier.settings.vertex_group_mass = grasped_vertex_group.name

    scene = bpy.context.scene

    # Make gravity starts only after the fold has been completed
    scene.use_gravity = False
    scene.keyframe_insert(data_path="use_gravity", frame=1)
    scene.use_gravity = True
    scene.keyframe_insert(data_path="use_gravity", frame=path_end_frame)

    scene.frame_start = 1

    if debug_visualizations:
        visualize_keypoints(ob, list(kp.values()))

    for i in range(1, path_end_frame):
        scene.frame_set(i)

    # let gravity work for a while
    MAX_GRAVITY_FRAMES = 100
    current_frame = path_end_frame
    max_frames = MAX_GRAVITY_FRAMES + current_frame
    while current_frame < max_frames:
        scene.frame_set(current_frame)
        current_frame += 1
        # check if all vertices' z coordinates are close enough to zero,
        # indicating that the cloth has fallen to the ground
        if all([(ob.matrix_world @ v.co).z < 0.05 for v in ob.data.vertices]):
            logger.debug(f"cloth has fallen to the ground at frame {current_frame}")
            break

    scene.frame_end = current_frame
    bpy.context.view_layer.update()

    return ob, kp


if __name__ == "__main__":
    import json
    import os
    import sys

    from synthetic_cloth_data import DATA_DIR

    id = 16
    debug = True
    output_dir = DATA_DIR / "deformed_meshes" / "TOWEL"
    output_dir.mkdir(parents=True, exist_ok=True)

    # check if id was passed as argument
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        id = int(argv[argv.index("--id") + 1])
        debug = "--debug" in argv
    logger.info(f"generating deformed towel with id {id}")
    blender_object, keypoint_ids = generate_random_deformed_towel(id, debug_visualizations=debug)
    if not debug:
        logger.info("saving files")
        filename = f"{id:06d}.obj"
        # select new object and  save as obj file
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = blender_object
        blender_object.select_set(True)
        bpy.ops.export_scene.obj(
            filepath=os.path.join(output_dir, filename),
            use_selection=True,
            use_materials=False,
            keep_vertex_order=True,  # important for keypoints
            check_existing=False,
            use_uvs=True,  # save UV coordinates
        )
        # write keypoints to json file
        with open(os.path.join(output_dir, filename.replace(".obj", ".json")), "w") as f:
            json.dump(keypoint_ids, f)
    logger.info("completed")
