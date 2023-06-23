import numpy as np
import trimesh
from airo_spatial_algebra import SE3Container
from trimesh.exchange.obj import export_obj

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

    def circular_fold_particle(self, arc_displacement, angle):
        initial_position = self.get_particle_position()
        final_position = initial_position + arc_displacement
        point = (initial_position + final_position) / 2

        # get rotation vector as the cross produc of the displacement and the upright vector
        line = np.cross((final_position - initial_position), np.array([0, 1, 0]))
        line = line / np.linalg.norm(line)
        line = -line  # the rotation vector is in the opposite direction of the cross product

        n_steps = 5

        for i in range(n_steps + 1):
            rotation_matrix = SE3Container.from_rotation_vector_and_translation(
                line * angle / n_steps * i
            ).rotation_matrix
            new_position = np.matmul(rotation_matrix, initial_position - point) + point
            # make the fold motion elliptical
            # to reduce y-axis strain to avoid displacements during folding.
            # cf. https://www.frontiersin.org/articles/10.3389/fnbot.2022.989702/full
            new_position[1] *= 0.7
            self.move_particle(new_position)
            print(new_position)
        wait_until_scene_is_stable(pyflex_stepper=self.pyflex_stepper)


def get_pyflex_cloth_scene_config(
    dynamic_friction: float = 0.75, particle_friction: float = 1.0, static_friction: float = 0.0
):
    """default values taken from Cloth Funnels codebase for now."""
    pyflex_config = {
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "render_type": ["cloth"],
                "cam_position": [0, 2, 0],
                "cam_angle": [np.pi / 2, -np.pi / 2, 0.0],
                "cam_size": [480, 480],
                "cam_fov": 80 / 180 * np.pi,
            }
        },
        "scene_config": {
            "scene_id": 0,  # the Empty Scene.
            "dynamic_friction": dynamic_friction,  # friction between cloth and rigid objects
            "particle_friction": particle_friction,  # friction between cloth particles
            "static_friction": static_friction,  # friction between rigid objects
        },
    }

    return pyflex_config


def read_obj_mesh(obj_path: str):
    """loads an tri-obj. accepts only trimeshes!"""
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
    return vertices, faces


def create_constraints(vertices, faces):
    stretch_edges = []
    for face in faces:
        x, y, z = face
        stretch_edges.append([x, y])
        stretch_edges.append([y, z])
        stretch_edges.append([z, x])

    stretch_constraints = np.array(stretch_edges)
    bend_constraints = np.array([])
    shear_constraints = np.array([])

    return stretch_constraints, bend_constraints, shear_constraints


def load_cloth_mesh_in_simulator(
    obj_path: str,
    position=None,
    cloth_stretch_stiffness: float = 0.6,
    cloth_bending_stiffness: float = 0.02,
    cloth_shear_stiffness: float = 0.02,
    cloth_mass: float = 20.0,
):
    vertices, faces = read_obj_mesh(obj_path)
    stretch_constraints, bend_constraints, shear_constraints = create_constraints(vertices, faces)

    if position is None:
        position = np.array([[0, 1.0, 0]])  # default
    pyflex.add_cloth_mesh(
        position=position,
        verts=np.array(vertices).flatten(),
        faces=np.array(faces).flatten(),
        stretch_edges=stretch_constraints.flatten(),
        bend_edges=bend_constraints.flatten(),
        shear_edges=shear_constraints.flatten(),
        stiffness=(cloth_stretch_stiffness, cloth_bending_stiffness, cloth_shear_stiffness),
        uvs=np.array([]),
        mass=cloth_mass,
    )

    return vertices, faces


def create_obj_with_new_vertex_positions(positions: np.ndarray, obj_path: str, target_obj_path: str):
    """Creates a new obj mesh by replacing the positions of the vertices in the original obj mesh. The order positions must match the order of the vertices in the original mesh."""
    mesh = trimesh.load(obj_path, process=False)  # keep order!
    assert len(mesh.vertices) == len(
        positions
    ), "Cannot update positions if the number of vertices does not match the number of positions!"
    mesh.vertices = positions
    # remove materials to avoid exporting .mtl files
    mesh.visual = trimesh.visual.ColorVisuals()
    # export to string and then to new obj
    obj_string = export_obj(mesh)
    with open(target_obj_path, "w") as f:
        f.write(obj_string)


if __name__ == "__main__":
    import time

    pyflex.init(False, True, 480, 480, 0)

    config = get_pyflex_cloth_scene_config()
    pyflex.set_scene(config["scene_config"]["scene_id"], config["scene_config"])
    pyflex.set_camera_params(config["camera_params"][config["camera_name"]])

    mesh_path = "/home/tlips/Documents/cloth-funnels/cloth_funnels/rtf/000000.obj"
    cloth_vertices, _ = load_cloth_mesh_in_simulator(mesh_path)

    print(len(cloth_vertices))

    n_particles = len(cloth_vertices)
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)
    cloth_system.center_object()
    pyflex.set_gravity(0, -9.81, 0)

    # drop cloth to the ground
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper)

    grasp_particle_idx = np.random.randint(0, n_particles)
    grasper = ParticleGrasper(pyflex_stepper)

    grasper.grasp_particle(grasp_particle_idx)
    # grasper.squared_fold_particle(0.05, np.array([0.2, 0, 0]))

    idx = np.random.randint(0, n_particles)
    point = cloth_system.get_positions()[idx]
    print(point)
    point = np.array([0.0, 0, 0])
    grasper.circular_fold_particle(np.array([0.3, 0, 0]), np.pi)
    grasper.release_particle()

    for _ in range(200):
        pyflex_stepper.step()

    create_obj_with_new_vertex_positions(cloth_system.get_positions(), mesh_path, "test.obj")

    pyflex.render()
    time.sleep(2)
