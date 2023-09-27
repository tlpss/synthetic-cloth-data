import dataclasses
import json
import pathlib
import random

import loguru
import numpy as np
from airo_spatial_algebra.se3 import SE3Container
from pyflex_utils import (
    ClothParticleSystem,
    ParticleGrasper,
    PyFlexStepWrapper,
    create_pyflex_cloth_scene_config,
    wait_until_scene_is_stable,
)
from pyflex_utils.utils import create_obj_with_new_vertex_positions_the_hacky_way, load_cloth_mesh_in_simulator
from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.meshes.utils.projected_mesh_area import get_mesh_projected_xy_area
from synthetic_cloth_data.utils import get_metadata_dict_for_dataset

import pyflex

logger = loguru.logger


@dataclasses.dataclass
class DeformationConfig:
    max_fold_distance: float = 0.6  # should allow to fold the cloth in half

    max_bending_stiffness: float = 0.02  # higher becomes unrealistic
    max_stretch_stiffness: float = 1.0
    max_drag: float = 0.002  # higher -> cloth will start to fall down very sloooow
    max_orientation_angle: float = np.pi / 4  # higher will make the cloth more crumpled

    fold_probability: float = 0.6
    grasp_keypoint_vertex_probability: float = 0.5
    flip_probability: float = 0.5


def deform_mesh(
    deformation_config: DeformationConfig, undeformed_mesh_path: str, target_mesh_path: str, gui: bool = False
):
    # create pyflex scene
    pyflex.init(not gui, gui, 480, 480, 0)

    # https://en.wikipedia.org/wiki/Friction
    static_friction = np.random.uniform(0.3, 1.0)
    dynamic_friction = np.random.uniform(0.3, 1.0)
    particle_friction = np.random.uniform(0.3, 1.0)
    drag = np.random.uniform(0.0, deformation_config.max_drag)
    config = create_pyflex_cloth_scene_config(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        particle_friction=particle_friction,
        drag=drag,
    )
    pyflex.set_scene(0, config["scene_config"])
    pyflex.set_camera_params(config["camera_params"][config["camera_name"]])

    # import the mesh

    # 0.5 is arbitrary but we don't want too much stretching
    stretch_stiffness = np.random.uniform(0.5, deformation_config.max_stretch_stiffness)
    bend_stiffness = np.random.uniform(0.0, deformation_config.max_bending_stiffness)
    cloth_vertices, _ = load_cloth_mesh_in_simulator(
        undeformed_mesh_path, cloth_stretch_stiffness=stretch_stiffness, cloth_bending_stiffness=bend_stiffness
    )
    n_particles = len(cloth_vertices)
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)

    # randomize masses for drag to have effect
    # (artifact of pyflex?)

    inverse_masses = cloth_system.get_masses()
    masses = 1.0 / inverse_masses
    masses += np.random.uniform(-np.max(masses) / 10, np.max(masses) / 10, size=masses.shape)
    inverse_masses = 1.0 / masses
    cloth_system.set_masses(inverse_masses)

    # separate the rotations, otherwise the y-rotation will be applied before the Z-rotation
    # which can increase the angle of the Z-rotation and thus make the cloth more crumpled
    rotation_matrix = SE3Container.from_euler_angles_and_translation(
        np.array(
            [
                np.random.uniform(0, deformation_config.max_orientation_angle),
                0,
                np.random.uniform(0, deformation_config.max_orientation_angle),
            ]
        )
    ).rotation_matrix

    y_rotation_matrix = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.random.uniform(0, 2 * np.pi), 0])
    ).rotation_matrix

    cloth_system.set_positions(cloth_system.get_positions() @ rotation_matrix @ y_rotation_matrix)
    cloth_system.center_object()

    # drop mesh
    # tolerance empirically determined
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=200, tolerance=0.08)

    # fold towards a random point around the grasp point
    if np.random.uniform() < deformation_config.fold_probability:
        logger.debug("folding")

        if np.random.uniform() < deformation_config.grasp_keypoint_vertex_probability:
            # load keypoints from json file
            json_path = undeformed_mesh_path.replace(".obj", ".json")
            keypoints = json.load(open(json_path))["keypoint_vertices"]
            grasp_particle_idx = random.choice(list(keypoints.values()))
        else:
            grasp_particle_idx = np.random.randint(0, n_particles)

        grasper = ParticleGrasper(pyflex_stepper)
        grasper.grasp_particle(grasp_particle_idx)

        fold_distance = np.random.uniform(0.1, deformation_config.max_fold_distance)

        cloth_center = cloth_system.get_center_of_mass()
        vertex_position = cloth_system.get_positions()[grasp_particle_idx]
        center_direction = np.arctan2(cloth_center[2] - vertex_position[2], cloth_center[0] - vertex_position[0])

        # 70% of time wihtin pi/3 of the center direction. folds outside of the mesh are less interesting.
        fold_direction = np.random.normal(center_direction, np.pi / 3)

        fold_vector = np.array([np.cos(fold_direction), 0, np.sin(fold_direction)]) * fold_distance
        logger.debug(f"fold vector: {fold_vector}")

        # don't fold all the way, as that makes the sim 'force it back to flat' due to the inifite weight of the grasped particle
        grasper.circular_fold_particle(fold_vector, np.pi * 0.8)
        grasper.release_particle()

        if np.random.uniform() < deformation_config.flip_probability:
            # lift, flip and drop again to have occluded folds
            logger.debug("flipping after folding")
            cloth_system.center_object()
            # rotate 180 degrees around x axis
            cloth_system.set_positions(
                cloth_system.get_positions()
                @ SE3Container.from_euler_angles_and_translation(np.array([np.pi, 0, 0])).rotation_matrix
            )
            # lift and drop
            cloth_system.set_positions(cloth_system.get_positions() + np.array([0, 0.5, 0]))
            wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper)

    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=200)
    cloth_system.center_object()

    # export mesh
    create_obj_with_new_vertex_positions_the_hacky_way(
        cloth_system.get_positions(), undeformed_mesh_path, target_mesh_path
    )

    logger.debug(f"static friction: {static_friction}")
    logger.debug(f"dynamic friction: {dynamic_friction}")
    logger.debug(f"particle friction: {particle_friction}")
    logger.debug(f"stretch stiffness: {stretch_stiffness}")
    logger.debug(f"bend stiffness: {bend_stiffness}")
    logger.debug(f"drag: {drag}")

    # cannot use this multiple times in the same process (segfault)
    # so start in new process, in which case there is no need to actually call the clean since all memory will be released anyways.
    # pyflex.clean()


def generate_deformed_mesh(
    deformation_config: DeformationConfig,
    mesh_dir_relative_path: str,
    output_dir_relative_path: str,
    id: int,
    debug: bool = False,
):

    np.random.seed(id)
    logger.info(f"generating deformation with id {id} using flex")

    mesh_dir_path = DATA_DIR / mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    mesh_path = np.random.choice(mesh_paths)

    logger.debug(f"mesh path: {mesh_path}")

    filename = f"{id:06d}.obj"
    output_dir_relative_path = DATA_DIR / output_dir_relative_path
    output_dir_relative_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_relative_path / filename

    # generate deformed mesh
    deform_mesh(deformation_config, mesh_path, output_path, gui=debug)

    # # create json file
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


if __name__ == "__main__":
    import hydra
    import tqdm
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="configs", config_name="config")
    def generate_deformed_meshes(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg))

        # write metadata
        data = {
            "num_samples": cfg.num_samples,
            "flat_mesh_dir": cfg.mesh_dir,
        }
        data.update(get_metadata_dict_for_dataset())
        data.update({"hydra_config": OmegaConf.to_container(cfg, resolve=True)})
        output_dir = DATA_DIR / pathlib.Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.json"
        json.dump(data, open(metadata_path, "w"))
        print(f"Metadata written to {metadata_path}")

        deformation_config = hydra.utils.instantiate(cfg["deformation_config"])
        print(deformation_config)
        for id in tqdm.trange(cfg.start_id, cfg.start_id + cfg.num_samples):

            # call python method in new process to avoid memory buildup
            # (cannot use pyflex.clean() multiple times in the same process)
            import multiprocessing

            p = multiprocessing.Process(
                target=generate_deformed_mesh, args=(deformation_config, cfg.mesh_dir, cfg.output_dir, id, cfg.debug)
            )
            p.start()
            p.join()

            # generate_deformed_mesh(deformation_config, cfg.mesh_dir, cfg.output_dir, id, debug=cfg.debug)

    generate_deformed_meshes()
