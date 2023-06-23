import loguru
import numpy as np
from pyflex_utils import (
    ClothParticleSystem,
    ParticleGrasper,
    PyFlexStepWrapper,
    create_pyflex_cloth_scene_config,
    wait_until_scene_is_stable,
)
from pyflex_utils.utils import create_obj_with_new_vertex_positions, load_cloth_mesh_in_simulator
from synthetic_cloth_data.meshes.projected_mesh_area import get_mesh_projected_xy_area

import pyflex

logger = loguru.logger


class DeformationConfig:
    max_fold_distance: float = 0.4

    max_bending_stiffness: float = 0.005
    max_stretch_stiffness: float = 1.0

    fold_probability: float = 0.5
    flip_probability: float = 0.5


def generate_deformed_mesh(
    deformation_config: DeformationConfig, undeformed_mesh_path: str, target_mesh_path: str, gui: bool = False
):
    # create pyflex scene
    pyflex.init(not gui, gui, 480, 480, 0)
    static_friction = np.random.uniform(0.0, 0.5)
    config = create_pyflex_cloth_scene_config(static_friction=static_friction)
    pyflex.set_scene(0, config)
    pyflex.set_camera_params(config["camera_params"][config["camera_name"]])

    # import the mesh
    stretch_stiffness = np.random.uniform(
        0.5, deformation_config.max_stretch_stiffness
    )  # O.5 seems reasonable lower bound
    bend_stiffness = np.random.uniform(0.0, deformation_config.max_bending_stiffness)
    cloth_vertices, _ = load_cloth_mesh_in_simulator(
        undeformed_mesh_path, cloth_stretch_stiffness=stretch_stiffness, cloth_bend_stiffness=bend_stiffness
    )
    n_particles = len(cloth_vertices)
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)
    cloth_system.center_object()

    # randomize orientation & flip w/ 50% probability before folding
    # TODO: this is not yet implemented

    # drop mesh
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper)

    # fold towards a random point around the grasp point
    if np.random.uniform() < deformation_config.fold_probability:
        grasp_particle_idx = np.random.randint(0, n_particles)
        grasper = ParticleGrasper(pyflex_stepper)

        grasper.grasp_particle(grasp_particle_idx)

        fold_distance = np.random.uniform(0.1, deformation_config.max_fold_distance)
        fold_direction = np.random.uniform(0.0, 2 * np.pi)
        fold_vector = np.array([np.cos(fold_direction), 0, np.sin(fold_direction)]) * fold_distance
        grasper.circular_fold_particle(fold_vector, np.pi)
        grasper.release_particle()

    if np.random.uniform() < deformation_config.flip_probability:
        # lift, flip and drop again to have occluded folds

        # turn upside down, lift and drop..
        pass

    # export mesh
    create_obj_with_new_vertex_positions(cloth_system.get_positions(), undeformed_mesh_path, target_mesh_path)

    if gui:
        # log all the randomization params
        logger.debug(f"static friction: {static_friction}")
        logger.debug(f"stretch stiffness: {stretch_stiffness}")
        logger.debug(f"bend stiffness: {bend_stiffness}")
        logger.debug(f"fold distance: {fold_distance}")


if __name__ == "__main__":
    import argparse
    import json

    from synthetic_cloth_data import DATA_DIR

    # TODO: this part has some code duplication with the blender variant..
    # would be nice to clean this up a little bit

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--id", type=int, default=17)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--mesh_dir_relative_path", type=str, default="flat_meshes/TOWEL/dev")
    argparser.add_argument("--output_dir", type=str, default="deformed_meshes/TOWEL/pyflex/dev")
    argparser.add_argument("--gui", action="store_true")
    args = argparser.parse_args()

    id = args.id
    np.random.seed(id)
    logger.info(f"generating deformation with id {id} using flex")

    mesh_dir_path = DATA_DIR / args.mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    mesh_path = np.random.choice(mesh_paths)

    filename = f"{id:06d}.obj"
    output_dir = DATA_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # generate deformed mesh
    generate_deformed_mesh(DeformationConfig(), mesh_path, output_path, gui=args.gui)

    # create json file
    flat_mesh_data = json.load(open(mesh_path.replace(".obj", ".json")))
    # write data to json file
    data = {
        "keypoint_vertices": flat_mesh_data["keypoint_vertices"],
        "area": get_mesh_projected_xy_area(output_path),
        "flat_mesh": {
            "relative_path": mesh_path.replace(f"{DATA_DIR}/", ""),
            "obj_md5_hash": flat_mesh_data["obj_md5_hash"],
            "area": flat_mesh_data["area"],  # duplication, but makes it easier to use later on..
        },
    }
    with open(str(output_path).replace(".obj", ".json"), "w") as f:
        json.dump(data, f)

    logger.info("completed")
