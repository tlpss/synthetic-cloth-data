import dataclasses
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
from synthetic_cloth_data.meshes.cloth_meshes import visualize_keypoints
from synthetic_cloth_data.meshes.generate_flat_meshes import _unwrap_cloth_mesh
from synthetic_cloth_data.meshes.mesh_operations import subdivide_mesh
from synthetic_cloth_data.meshes.projected_mesh_area import get_mesh_projected_xy_area

logger = loguru.logger


def attach_cloth_sim(blender_object: bpy.types.Object, solifify=True) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = blender_object
    blender_object.select_set(True)
    # add solidify modifier
    if solifify:
        bpy.ops.object.modifier_add(type="SOLIDIFY")
        bpy.context.object.modifiers["Solidify"].thickness = np.random.uniform(0.001, 0.005)
    bpy.ops.object.modifier_add(type="CLOTH")
    # helps with smoothing of the mesh after deformation
    bpy.ops.object.shade_smooth()

    # physics
    blender_object.modifiers["Cloth"].settings.quality = 5
    blender_object.modifiers["Cloth"].settings.mass = np.random.uniform(0.1, 2.0)  # mass per vertex!
    # air resistance - higher will result in more wrinkles during free fall. if zero, cloth falls 'rigidly'.
    blender_object.modifiers["Cloth"].settings.air_damping = np.random.uniform(0.1, 3.0)
    blender_object.modifiers["Cloth"].settings.tension_stiffness = np.random.uniform(2.0, 50.0)
    blender_object.modifiers["Cloth"].settings.compression_stiffness = np.random.uniform(2.0, 50.0)
    blender_object.modifiers["Cloth"].settings.shear_stiffness = np.random.uniform(2.0, 20.0)
    blender_object.modifiers["Cloth"].settings.bending_stiffness = np.random.uniform(0.01, 50.0)
    # collision
    blender_object.modifiers["Cloth"].collision_settings.collision_quality = 2
    blender_object.modifiers["Cloth"].collision_settings.use_self_collision = True
    blender_object.modifiers["Cloth"].collision_settings.distance_min = 0.003
    blender_object.modifiers["Cloth"].collision_settings.self_distance_min = 0.003
    blender_object.modifiers["Cloth"].collision_settings.self_friction = np.random.uniform(0.1, 2.0)


def project_vector_onto_plane(vector: Vector3DType, plane_normal: Vector3DType) -> Vector3DType:
    return vector - np.dot(vector, plane_normal) * plane_normal


def sample_point_in_convex_hull(points: Vectors3DType) -> Vector3DType:
    weights = np.random.uniform(0, 1, len(points))
    weights /= weights.sum()
    return (points * weights[:, None]).sum(axis=0)


def get_random_fold_line(keypoints: Vectors3DType) -> Tuple[Vector3DType, Vector3DType]:
    fold_line_point = sample_point_in_convex_hull(keypoints)

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


@dataclasses.dataclass
class DeformationConfig:
    max_arc_angle_rad: float = np.pi / 2
    max_num_falling_physics_steps: int = 150
    falling_termination_height: float = 0.05
    max_mesh_xy_rotation_rad: float = np.pi / 6


def generate_deformed_mesh(
    deformation_config: DeformationConfig, flat_mesh_path: str, debug_visualizations=False
) -> bpy.types.Object:
    bpy.ops.object.delete()  # Delete default cube
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    bpy.ops.object.modifier_add(type="COLLISION")
    bpy.context.object.collision.cloth_friction = np.random.uniform(5.0, 30.0)
    plane = bpy.context.object
    subdivide_mesh(plane, 10)
    add_material(plane, (1, 0.5, 0.5, 1.0))

    bpy.ops.import_scene.obj(filepath=flat_mesh_path, split_mode="OFF")  # keep vertex order for keypoints!
    ob = bpy.context.selected_objects[0]
    ob.name = ob.name.split(".")[0] + "_blender_deformed"
    kp = json.load(open(flat_mesh_path.replace(".obj", ".json")))["keypoint_vertices"]

    _unwrap_cloth_mesh(ob)
    ob.location = np.array([0, 0, 1.0])

    # randomize orientation, which has large impact on crumpling due to simulated air friction
    # (and internal collisions of the cloth mesh)

    # apply relative rotation, to not mess with the obj <-> scene import which already applies rotations
    # to go from y-up to z-up coordinate system.
    ob.rotation_euler[0] += np.random.uniform(0, deformation_config.max_mesh_xy_rotation_rad)
    ob.rotation_euler[1] += np.random.uniform(0, deformation_config.max_mesh_xy_rotation_rad)
    ob.rotation_euler[2] = np.random.uniform(0, 2 * np.pi)

    # update the object's world matrix
    # cf. https://blender.stackexchange.com/questions/27667/incorrect-matrix-world-after-transformation
    bpy.context.view_layer.update()

    keypoints = np.array([ob.matrix_world @ ob.data.vertices[kid].co for kid in kp.values()])

    fold_line = get_random_fold_line(keypoints)

    fold_line_point, fold_line_direction = fold_line

    if debug_visualizations:
        add_points([fold_line_point], radius=0.02, color=(1, 0, 0, 1))
        add_line_segment(fold_line_point - fold_line_direction, fold_line_point + fold_line_direction)

    # choose a random keypoint from kp
    grasped_vertex_id = int(np.random.choice(list(kp.values())))
    grasp_location = np.array(ob.matrix_world @ ob.data.vertices[grasped_vertex_id].co)

    fold_arc_angle = np.random.uniform(0, deformation_config.max_arc_angle_rad)
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
    current_frame = path_end_frame
    max_frames = deformation_config.max_num_falling_physics_steps + current_frame
    while current_frame < max_frames:
        scene.frame_set(current_frame)
        current_frame += 1
        # check if all vertices' z coordinates are close enough to zero,
        # indicating that the cloth has fallen to the ground

        # have to get the evaluated vertices, otherwise the coordinates are not updated..
        evaluated_vertices = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
        if all(
            [(ob.matrix_world @ v.co).z < deformation_config.falling_termination_height for v in evaluated_vertices]
        ):
            logger.debug(f"cloth has fallen to the ground at frame {current_frame}")
            break

    logger.debug("max frames reached for cloth falling")

    scene.frame_end = current_frame
    bpy.context.view_layer.update()

    return ob, kp


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    from synthetic_cloth_data import DATA_DIR

    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--id", type=int, default=17)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--mesh_dir_relative_path", type=str, default="flat_meshes/TOWEL/dev")
    argparser.add_argument("--output_dir", type=str, default="deformed_meshes/TOWEL/dev")
    args = argparser.parse_args(argv)

    output_dir = DATA_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    id = args.id

    logger.info(f"generating deformed towel with id {id}")

    np.random.seed(id)

    mesh_dir_path = DATA_DIR / args.mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    mesh_path = np.random.choice(mesh_paths)

    blender_object, keypoint_ids = generate_deformed_mesh(
        DeformationConfig(), mesh_path, debug_visualizations=args.debug
    )
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
    flat_mesh_data = json.load(open(mesh_path.replace(".obj", ".json")))
    # write data to json file
    data = {
        "keypoint_vertices": keypoint_ids,
        "area": get_mesh_projected_xy_area(os.path.join(output_dir, filename)),
        "flat_mesh": {
            "relative_path": mesh_path.replace(f"{DATA_DIR}/", ""),
            "obj_md5_hash": flat_mesh_data["obj_md5_hash"],
            "area": flat_mesh_data["area"],  # duplication, but makes it easier to use later on..
        },
    }
    with open(os.path.join(output_dir, filename.replace(".obj", ".json")), "w") as f:
        json.dump(data, f)
    logger.info("completed")
