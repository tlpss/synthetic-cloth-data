import dataclasses
from typing import List

import bpy
import numpy as np
import triangle


def quadratic_bezier(start: np.ndarray, control: np.ndarray, end: np.ndarray, steps: int) -> List[np.ndarray]:
    t = np.linspace(0, 1, steps, endpoint=False)
    t = t[:, np.newaxis]
    points = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end
    return list(points)


@dataclasses.dataclass
class BezierConfig:
    """specifies control point relative to the middle between start and enpoint"""

    start_vertex_id: int
    end_vertex_id: int
    relative_x: float = 0.0
    relative_y: float = 0.0
    n_samples: int = 10


@dataclasses.dataclass
class BevelConfig:
    """specifies control point relative to the middle between start and enpoint"""

    vertex_id: int
    segments: int = 4
    offset: float = 0.01


def apply_bezier_curves_to_mesh(vertices: List[np.ndarray], bezier_configs: List[BezierConfig]):
    """apply bezier curves to the mesh"""
    beziered_vertices = [[bezier.start_vertex_id, bezier.end_vertex_id] for bezier in bezier_configs]
    beziered_vertices = sorted(beziered_vertices, key=lambda x: x[0])  # sort by start vertex id
    for i in range(len(beziered_vertices)):
        start_idx = beziered_vertices[i][0]
        end_idx = beziered_vertices[i][1]
        assert (end_idx == start_idx + 1) or (
            start_idx == len(vertices) - 1 and end_idx == 0
        ), "bezier curve must be between two consecutive vertices"
        start = vertices[start_idx]
        end = vertices[end_idx]
        control_point = (start + end) / 2
        x = (end - start) / np.linalg.norm(end - start)
        y = np.cross(np.array([0, 0, 1]), x)
        control_point += y * bezier_configs[i].relative_y
        control_point += x * bezier_configs[i].relative_x

        new_vertices = vertices[:start_idx]
        new_vertices.extend(quadratic_bezier(start, control_point, end, bezier_configs[i].n_samples))
        if end_idx > start_idx:
            new_vertices.extend(vertices[end_idx:])

        vertices = new_vertices

        for j in range(len(beziered_vertices)):
            if beziered_vertices[j][0] >= end_idx:
                beziered_vertices[j][0] += bezier_configs[i].n_samples - 1
            if beziered_vertices[j][1] >= end_idx:
                beziered_vertices[j][1] += bezier_configs[i].n_samples - 1
    return vertices


def bevel_vertices(blender_object: bpy.types.Object, bevel_configs: List[BevelConfig]):

    vertex_ids = [b.vertex_id for b in bevel_configs]

    # select & activate!
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = blender_object
    blender_object.select_set(True)

    ## BEVEL all corners (with bookkeeping for keypoints)
    i = 0
    while i < len(vertex_ids):
        id = vertex_ids[i]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        blender_object.data.vertices[id].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.bevel(offset=bevel_configs[i].offset, segments=bevel_configs[i].segments, affect="VERTICES")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        # so current vertex is deleted. All future vertices are shifted by one to the left.
        # and segment+1 newly are added, of which the middle is the newest keypoint
        vertex_ids = [kid - 1 for kid in vertex_ids]
        vertex_ids[i] = len(blender_object.data.vertices) - 2
        i += 1

    return blender_object


def subdivide_mesh(blender_object, n_cuts=2):
    ## subdivide mesh to increase resolution
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = blender_object
    blender_object.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=n_cuts)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def find_nearest_vertex_ids(vertices: List[np.ndarray], points: List[np.ndarray]) -> List[int]:
    """find the nearest vertex for each point"""
    distances = np.linalg.norm(np.array(vertices)[:, np.newaxis, :] - np.array(points)[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=0).tolist()


def triangulate(blender_object: bpy.types.Object):
    """triangulate a planar mesh (typically a cloth template)"""
    vertices, edges, faces = blender_object.data.vertices, blender_object.data.edges, blender_object.data.polygons
    vertices = np.array([v.co for v in vertices])
    vertices = vertices[:, :2]

    triangle_input = {
        "vertices": vertices,
        "segments": np.array([[e.vertices[0], e.vertices[1]] for e in edges]),
    }
    area = 0.00005
    triangle_output = triangle.triangulate(triangle_input, f"qpa{area:.32f}")
    vertices = triangle_output["vertices"]
    vertices = np.concatenate([vertices, np.zeros((len(vertices), 1))], axis=1)
    faces = triangle_output["triangles"]
    mesh = bpy.data.meshes.new("mesh")
    mesh.from_pydata(vertices, [], faces)
    blender_object.data = mesh
    return blender_object
