import dataclasses
import json
import pathlib

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
from pyflex_utils.utils import create_obj_with_new_vertex_positions, load_cloth_mesh_in_simulator
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

    # randomize orientation & flip w/ 50% probability before folding
    rotation_matrix = SE3Container.from_euler_angles_and_translation(
        np.array(
            [
                np.random.uniform(0, deformation_config.max_orientation_angle),
                np.random.uniform(0, 2 * np.pi),
                np.random.uniform(0, deformation_config.max_orientation_angle),
            ]
        )
    ).rotation_matrix
    cloth_system.set_positions(cloth_system.get_positions() @ rotation_matrix)
    cloth_system.center_object()

    # drop mesh

    # tolerance empirically determined
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=200, tolerance=0.08)

    # fold towards a random point around the grasp point
    if np.random.uniform() < deformation_config.fold_probability:
        logger.debug("folding")
        grasp_particle_idx = np.random.randint(0, n_particles)
        grasper = ParticleGrasper(pyflex_stepper)

        grasper.grasp_particle(grasp_particle_idx)

        fold_distance = np.random.uniform(0.1, deformation_config.max_fold_distance)
        fold_direction = np.random.uniform(0.0, 2 * np.pi)
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
    create_obj_with_new_vertex_positions(cloth_system.get_positions(), undeformed_mesh_path, target_mesh_path)

    logger.debug(f"static friction: {static_friction}")
    logger.debug(f"dynamic friction: {dynamic_friction}")
    logger.debug(f"particle friction: {particle_friction}")
    logger.debug(f"stretch stiffness: {stretch_stiffness}")
    logger.debug(f"bend stiffness: {bend_stiffness}")
    logger.debug(f"drag: {drag}")


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
        for id in range(cfg.start_id, cfg.start_id + cfg.num_samples):
            generate_deformed_mesh(deformation_config, cfg.mesh_dir, cfg.output_dir, id, debug=cfg.debug)

    generate_deformed_meshes()
