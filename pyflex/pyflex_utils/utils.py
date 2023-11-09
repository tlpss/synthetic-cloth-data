"""a bunch of utility functions and classes for deforming cloth meshes with pyflex.
parts of this code are inspired on https://github.com/columbia-ai-robotics/cloth-funnels/blob/main/cloth_funnels/utils/task_utils.py """
from collections import defaultdict

import numpy as np
import trimesh
from airo_spatial_algebra import SE3Container
from trimesh.exchange.obj import export_obj

import pyflex


class PyFlexStepWrapper:
    """base class for wrapping pyflex.step() calls. Can be inherited to add functionality such as logging or rendering."""

    def __init__(self):
        pass

    def step(self):
        pyflex.step()


class ClothParticleSystem:
    """Convenience class for interacting with Flex particles that represent a cloth item"""

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

    def get_center_of_mass(self):
        return np.mean(self.get_positions(), axis=0)

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
    """simulates a gripper that can grasp and release a single particle. Used to (partially) fold clothes"""

    def __init__(self, pyflex_stepper: PyFlexStepWrapper):
        self.particle_idx = None
        self.original_inv_mass = None
        self.pyflex_stepper = pyflex_stepper

    def grasp_particle(self, particle_idx):
        """grasp particle by setting its inverse mass to 0, so that it is rigidly attached to the 'gripper'"""
        self.particle_idx = particle_idx
        pyflex_positions = pyflex.get_positions().reshape(-1, 4)
        self.original_inv_mass = pyflex_positions[particle_idx, 3]

        pyflex_positions[particle_idx, 3] = 0.0
        pyflex.set_positions(pyflex_positions.flatten())

    def release_particle(self):
        """release particle by setting its inverse mass to its original value"""
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
        """move the grasped particle along a straight line to a target position with a given speed"""

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

    def circular_fold_particle(self, displacement, angle):
        initial_position = self.get_particle_position()
        final_position = initial_position + displacement
        point = (initial_position + final_position) / 2

        # get rotation vector as the cross produc of the displacement and the upright vector
        line = np.cross((final_position - initial_position), np.array([0, 1, 0]))
        line = line / np.linalg.norm(line)
        line = -line  # the rotation vector is in the opposite direction of the cross product

        n_steps = int(angle / np.pi * 20)

        for i in range(n_steps + 1):
            rotation_matrix = SE3Container.from_rotation_vector_and_translation(
                line * angle / n_steps * i
            ).rotation_matrix
            new_position = np.matmul(rotation_matrix, initial_position - point) + point
            # make the fold motion elliptical
            # to reduce y-axis strain to avoid displacements during folding.
            # cf. https://www.frontiersin.org/articles/10.3389/fnbot.2022.989702/full
            new_position[1] *= 0.7
            self.move_particle(new_position, 0.05)  # as fast as possible while remaining quasi-static
        wait_until_scene_is_stable(pyflex_stepper=self.pyflex_stepper)

    def lift_particle(self, lift_height):
        self.move_particle(self.get_particle_position() + np.array([0, lift_height, 0]))
        wait_until_scene_is_stable(pyflex_stepper=self.pyflex_stepper)


def create_pyflex_cloth_scene_config(
    dynamic_friction: float = 0.75,
    particle_friction: float = 1.0,
    static_friction: float = 0.0,
    drag: float = 0.002,
    particle_radius: float = 0.02,
    solid_rest_distance: float = 0.006,
):

    # default values taken from Cloth Funnels codebase for now.
    pyflex_config = {
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "render_type": ["cloth"],
                "cam_position": [1, 0.5, 0],
                "cam_angle": [np.pi / 2, -np.pi / 8, 0.0],
                "cam_size": [480, 480],
                "cam_fov": 80 / 180 * np.pi,
            }
        },
        "scene_config": {
            "scene_id": 0,  # the Empty Scene.
            "particle_radius": particle_radius,  # particle radius
            "collision_distance": 0.0005,  # collision distance between particles and shapes (such as ground surface)
            "solid_rest_distance": solid_rest_distance,  # rest distance for solid particles -> a.o. 'cloth thickness' for self collisions, best close to edge lengths
            "dynamic_friction": dynamic_friction,  # friction between cloth and rigid objects
            "particle_friction": particle_friction,  # friction between cloth particles
            "static_friction": static_friction,  # friction between rigid objects
            "drag": drag,  # drag coefficient of the cloth - air interaction (requires unequal particle masses to have effect?)
        },
    }

    return pyflex_config


def read_obj_mesh(obj_path: str):
    """loads a tri-mesh obj file and returns vertices and faces"""
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


def get_1_ring_neighbourhood(faces):
    neighbourhood = defaultdict(set)
    for face in faces:
        x, y, z = face
        neighbourhood[x].add(y)
        neighbourhood[x].add(z)
        neighbourhood[y].add(x)
        neighbourhood[y].add(z)
        neighbourhood[z].add(x)
        neighbourhood[z].add(y)

    return neighbourhood


def get_2_ring_neighborhoods(faces):
    neighbourhood = get_1_ring_neighbourhood(faces)
    neighbourhood_2 = defaultdict(set)
    for vertex in neighbourhood:
        two_ring_candidates = set()
        for neighbour in neighbourhood[vertex]:
            two_ring_candidates = two_ring_candidates.union(neighbourhood[neighbour])
        # subtract the 1-ring neighbours
        two_ring_neighbours = two_ring_candidates.difference(neighbourhood[vertex])

        neighbourhood_2[vertex] = two_ring_neighbours
    return neighbourhood_2


def create_constraints(faces):
    stretch_edges = []
    for face in faces:
        x, y, z = face
        stretch_edges.append([x, y])
        stretch_edges.append([y, z])
        stretch_edges.append([z, x])

    stretch_constraints = np.array(stretch_edges)

    bend_constraints = []
    two_ring_neighbourhood_dict = get_2_ring_neighborhoods(faces)
    for vertex_id, two_ring_neighbours in two_ring_neighbourhood_dict.items():
        for neighbour in two_ring_neighbours:
            if neighbour > vertex_id:
                bend_constraints.append([vertex_id, neighbour])

    bend_constraints = np.array(bend_constraints)

    shear_constraints = np.array([])

    return stretch_constraints, bend_constraints, shear_constraints


def load_cloth_mesh_in_simulator(
    obj_path: str,
    position=None,
    cloth_stretch_stiffness: float = 0.6,
    cloth_bending_stiffness: float = 0.0,
    cloth_shear_stiffness: float = 0.0,
    cloth_mass: float = 1.0,  # lower -> drag makes free fall slower
):
    """loads a cloth mesh in the simulator and creates a particle system with stretch constraints between the 1-ring neighbours and bending constraints between the 2-ring neighbours.
    as in the Nivida Flex docs. The resting state of all constraints is set to the initial mesh configuration.
    """
    vertices, faces = read_obj_mesh(obj_path)
    stretch_constraints, bend_constraints, shear_constraints = create_constraints(faces)

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


def create_obj_with_new_vertex_positions_the_hacky_way(
    new_vertex_positions: np.ndarray, file_path: str, target_file_path: str
):
    """trimesh keeps messing with faces and vertices, so this function does a one-on-one replacement of the vertex positions in the obj file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    vertex_idx = 0
    for line_idx, line in enumerate(lines):
        if line.startswith("v "):
            lines[
                line_idx
            ] = f"v {new_vertex_positions[vertex_idx][0]} {new_vertex_positions[vertex_idx][1]} {new_vertex_positions[vertex_idx][2]}\n"
            vertex_idx += 1
    with open(target_file_path, "w") as f:
        f.writelines(lines)


def create_obj_with_new_vertex_positions(positions: np.ndarray, obj_path: str, target_obj_path: str):
    """Creates a new obj mesh by replacing the positions of the vertices in the original obj mesh. The order positions must match the order of the vertices in the original mesh."""
    # allow order change for faces to avoid artifacts in the texture coordinates
    # this should not influence the vertex positions if the mesh does not contain duplicate uv coordinates for a single vertex.
    mesh = trimesh.load(obj_path, process=False, maintain_order=False)
    assert len(mesh.vertices) == len(
        positions
    ), "Cannot update positions if the number of vertices does not match the number of positions!"
    mesh.vertices = positions
    # remove materials to avoid exporting .mtl files
    # cannot use this trick as it will remove the texture coordinates from the obj..
    # mesh.visual = trimesh.visual.ColorVisuals()
    # export to string and then to new obj
    obj_string = export_obj(mesh, include_texture=True)
    # remove second and third line from obj string
    # which corresponds to the material
    obj_string = "\n".join(obj_string.split("\n")[3:])
    with open(target_obj_path, "w") as f:
        f.write(obj_string)


if __name__ == "__main__":
    import time

    pyflex.init(False, True, 480, 480, 0)

    # pyflex becomes unstable if radius is set to higher values (self-collisions)
    # and rest_distance seems to be most stable if it is close to the highest edge lengths in the mesh.
    config = create_pyflex_cloth_scene_config(drag=0.0, particle_radius=0.02, solid_rest_distance=0.02)
    pyflex.set_scene(config["scene_config"]["scene_id"], config["scene_config"])
    pyflex.set_camera_params(config["camera_params"][config["camera_name"]])

    # mesh_path = "/home/tlips/Documents/synthetic-cloth-data/pyflex/pyflex_utils/00421.obj"
    mesh_path = "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/flat_meshes/SHORTS/Cloth3D-5-flat/01375.obj"
    cloth_vertices, _ = load_cloth_mesh_in_simulator(
        mesh_path, cloth_bending_stiffness=0.01, cloth_stretch_stiffness=0.5
    )

    n_particles = len(cloth_vertices)
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)
    cloth_system.center_object()
    # pyflex.set_gravity(0, 0, 0)

    # drop cloth to the ground
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=400)

    # grasp_particle_idx = np.random.randint(0, n_particles)
    # grasper = ParticleGrasper(pyflex_stepper)

    # grasper.grasp_particle(grasp_particle_idx)

    # idx = np.random.randint(0, n_particles)
    # point = cloth_system.get_positions()[idx]
    # point = np.array([0.0, 0, 0])
    # grasper.circular_fold_particle(np.array([-0.3, 0, 0]), np.pi)
    # grasper.release_particle()
    cloth_system.center_object()
    create_obj_with_new_vertex_positions_the_hacky_way(cloth_system.get_positions(), mesh_path, "test.obj")

    time.sleep(2)
