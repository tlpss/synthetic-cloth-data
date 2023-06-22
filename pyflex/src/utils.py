import numpy as np

import pyflex


class PyFlexStepWrapper:
    def __init__(self):
        pass

    def step(self):
        pyflex.step()


class RenderPyFlexStepWrapper(PyFlexStepWrapper):
    def __init__(self, render):
        self.render = render

    def step(self):
        super().step()
        # get an image of the scene
        # and save it somewhere.
        raise NotImplementedError


class ClothParticleSystem:
    def __init__(self, n_particles, pyflex_stepper: PyFlexStepWrapper) -> None:
        self.n_particles = n_particles
        self.pyflex_stepper = pyflex_stepper

    def get_positions(self):
        return pyflex.get_positions().reshape(-1, 4)[: self.n_particles, :3]

    def set_positions(self, positions):
        pyflex_positions = pyflex.get_positions().reshape(-1, 4)
        pyflex_positions[: self.n_particles, :3] = positions
        pyflex.set_positions(pyflex_positions.flatten())

    def get_velocities(self):
        return pyflex.get_velocities().reshape(-1, 4)[: self.n_particles, :3]

    def set_velocities(self, velocities):
        pyflex_velocities = pyflex.get_velocities().reshape(-1, 3)
        pyflex_velocities[: self.n_particles, :3] = velocities
        pyflex.set_velocities(pyflex_velocities.flatten())

    def get_masses(self):
        return pyflex.get_positions().reshape(-1, 4)[: self.n_particles, 3]

    def set_masses(self, masses: np.ndarray):
        pyflex_masses = pyflex.get_positions().reshape(-1, 4)
        pyflex_masses[: self.n_particles, 3] = masses
        pyflex.set_positions(pyflex_masses.flatten())

    def center_object(self):
        pos = self.get_positions()
        mid_x = (np.max(pos[:, 0]) + np.min(pos[:, 0])) / 2
        mid_z = (np.max(pos[:, 2]) + np.min(pos[:, 2])) / 2
        pos[:, [0, 2]] -= np.array([mid_x, mid_z])
        self.set_positions(pos)
        self.pyflex_stepper.step()


def wait_until_scene_is_stable(max_steps=100, tolerance=1e-2, pyflex_stepper: PyFlexStepWrapper = None):
    pyflex.step()
    for _ in range(max_steps):
        particle_velocity = pyflex.get_velocities()
        if np.abs(particle_velocity).max() < tolerance:
            return True
        pyflex_stepper.step()
    return False


class ParticleGrasper:
    def __init__(self, pyflex_stepper: PyFlexStepWrapper):
        self.particle_idx = None
        self.original_inv_mass = None
        self.pyflex_stepper = pyflex_stepper

    def grasp_particle(self, particle_idx):
        self.particle_idx = particle_idx
        pyflex_positions = pyflex.get_positions().reshape(-1, 4)
        self.original_inv_mass = pyflex_positions[particle_idx, 3]
        pyflex_positions[particle_idx, 3] = 0.0
        pyflex.set_positions(pyflex_positions.flatten())

    def release_particle(self):
        pyflex_positions = pyflex.get_positions().reshape(-1, 4)
        pyflex_positions[self.particle_idx, 3] = self.original_inv_mass
        pyflex.set_positions(pyflex_positions.flatten())
        self.particle_idx = None
        self.original_inv_mass = None

    def is_grasping(self):
        return self.particle_idx is not None

    def get_particle_position(self):
        return pyflex.get_positions().reshape(-1, 4)[self.particle_idx, :3]

    def move_particle(self, target_position, speed=0.01):
        assert self.is_grasping(), "No particle is being grasped."
        curr_pos = pyflex.get_positions()
        pickpoint = self.particle_idx
        init_point = curr_pos[pickpoint * 4 : pickpoint * 4 + 3].copy()
        for j in range(int(1 / speed)):

            curr_pos = pyflex.get_positions()
            curr_vel = pyflex.get_velocities()
            pickpoint_pos = (target_position - init_point) * (j * speed) + init_point
            curr_pos[pickpoint * 4 : pickpoint * 4 + 3] = pickpoint_pos
            curr_pos[pickpoint * 4 + 3] = 0
            curr_vel[pickpoint * 3 : pickpoint * 3 + 3] = [0, 0, 0]

            pyflex.set_positions(curr_pos)
            pyflex.set_velocities(curr_vel)
            self.pyflex_stepper.step()

    def squared_fold_particle(self, lift_height, displacement):
        self.move_particle(self.get_particle_position() + np.array([0, lift_height, 0]))
        wait_until_scene_is_stable(pyflex_stepper=self.pyflex_stepper)
        self.move_particle(self.get_particle_position() + displacement, 0.01)
        wait_until_scene_is_stable(pyflex_stepper=self.pyflex_stepper)


def get_cloth_scene_config(
    particle_radius: float = 0.0175, cloth_stiffness: tuple = (0.75, 0.02, 0.02), scale: float = 1.0
):
    config = {
        "scale": 1.0,
        "cloth_pos": [0.0, 1.0, 0.0],
        "cloth_size": [int(0.6 / particle_radius), int(0.368 / particle_radius)],
        "cloth_stiff": cloth_stiffness,  # Stretch, Bend and Shear
        "cloth_mass": 0.2,
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "render_type": ["cloth"],
                "cam_position": [0, 5, 0],
                "cam_angle": [np.pi / 2, -np.pi / 2, 0],
                "cam_size": [480, 480],
                "cam_fov": 39.5978 / 180 * np.pi,
            }
        },
        "scene_config": {
            "scene_id": 0,
            "radius": particle_radius * scale,
            "buoyancy": 0,
            "numExtraParticles": 20000,
            "collisionDistance": 0.0006,
            "msaaSamples": 0,
        },
        "flip_mesh": 0,
    }

    return config


def load_tri_cloth_mesh_into_config(obj_path: str, config: dict):
    """loads an tri-obj. Only tri-mesh is acceptable!"""
    vertices, faces = [], []
    with open(obj_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith("v "):
            vertices.append([float(n) for n in line.replace("v ", "").split(" ")])
        # Face
        elif line.startswith("f "):
            idx = [n.split("/") for n in line.replace("f ", "").split(" ")]
            face = [int(n[0]) - 1 for n in idx]
            assert len(face) == 3
            faces.append(face)

    stretch_edges = []
    for face in faces:
        x, y, z = face
        stretch_edges.append([x, y])
        stretch_edges.append([y, z])
        stretch_edges.append([z, x])

    vertices = np.array(vertices)
    faces = np.array(faces)
    stretch_constraints = np.array(stretch_edges)
    bend_constraints = np.array([])
    shear_constraints = np.array([])
    uvs = np.array([])

    config["mesh_verts"] = vertices.flatten()
    config["mesh_faces"] = faces.flatten()
    config["mesh_stretch_edges"] = stretch_constraints.flatten()
    config["mesh_bend_edges"] = bend_constraints.flatten()
    config["mesh_shear_edges"] = shear_constraints.flatten()
    config["mesh_nocs_verts"] = uvs.flatten()


if __name__ == "__main__":
    import time

    pyflex.init(False, True, 480, 480, 0)

    config = get_cloth_scene_config()
    pyflex.set_scene(config["scene_config"]["scene_id"], config["scene_config"])
    pyflex.set_camera_params(config["camera_params"][config["camera_name"]])

    mesh_path = "/home/tlips/Documents/cloth-funnels/cloth_funnels/rtf/000000.obj"
    load_tri_cloth_mesh_into_config(mesh_path, config)

    pyflex.add_cloth_mesh(
        position=config["cloth_pos"],
        verts=config["mesh_verts"],
        faces=config["mesh_faces"],
        stretch_edges=config["mesh_stretch_edges"],
        bend_edges=config["mesh_bend_edges"],
        shear_edges=config["mesh_shear_edges"],
        stiffness=config["cloth_stiff"],
        uvs=config["mesh_nocs_verts"],
        mass=config["cloth_mass"],
    )

    n_particles = len(config["mesh_verts"]) // 3
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)

    cloth_system.center_object()

    # drop cloth to the ground
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper)

    grasp_particle_idx = np.random.randint(0, n_particles)
    grasper = ParticleGrasper(pyflex_stepper)

    grasper.grasp_particle(grasp_particle_idx)
    grasper.squared_fold_particle(0.1, np.array([0.2, 0, 0]))
    time.sleep(2)
