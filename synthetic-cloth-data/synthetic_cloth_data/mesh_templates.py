import dataclasses
from typing import List
import numpy as np 
import bpy, bmesh
@dataclasses.dataclass
class TowelMeshConfig:
    width: float = 0.2
    length: float = 0.6

class TowelMesh:
    def __init__(self, config: TowelMeshConfig) -> None:
        self._config = config
        width = config.width
        length = config.length

        vertices = [
            np.array([-width / 2, -length / 2, 0.0]),
            np.array([-width / 2, length / 2, 0.0]),
            np.array([width / 2, length / 2, 0.0]),
            np.array([width / 2, -length / 2, 0.0]),
        ]
        vertices 
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        faces = [(0, 1, 2, 3)]

        name = "TowelMesh"
        self._mesh = bpy.data.meshes.new(name)
        self._mesh.from_pydata(vertices, edges, faces)
        self._mesh.update()

    @property
    def mesh(self):
        return self._mesh
    
@dataclasses.dataclass
class ShortsMeshConfig:
    waist_width: float = 0.5
    crotch_height: float = 0.3
    pipe_width: float = 0.25
    length: float = 0.5
    waiste_pipe_angle: float = 0.2
    pipe_outer_angle: float = 0.2

class ShortsMesh:
    def __init__(self, config: ShortsMeshConfig) -> None:
        self._config = config
        self._mesh = self._create_mesh()

    def _create_mesh(self):
        right_waist = np.array([self._config.waist_width/2,0,0])
        left_waist = np.array([-self._config.waist_width/2,0,0])
        crotch = np.array([0,-self._config.crotch_height,0])
        right_pipe_outer = right_waist + np.array([np.sin(self._config.waiste_pipe_angle) * self._config.length, -np.cos(self._config.waiste_pipe_angle) * self._config.length,0])
        right_pipe_inner = right_pipe_outer + np.array([-np.cos(self._config.pipe_outer_angle) * self._config.pipe_width, - np.sin(self._config.pipe_outer_angle) * self._config.pipe_width,0])

        left_pipe_outer =  left_waist + np.array([- np.sin(self._config.waiste_pipe_angle) * self._config.length, - np.cos(self._config.waiste_pipe_angle) * self._config.length,0])
        left_pipe_inner = left_pipe_outer + np.array([np.cos(self._config.pipe_outer_angle) * self._config.pipe_width, -np.sin(self._config.pipe_outer_angle) * self._config.pipe_width,0])


        vertices = [
            left_waist,
            right_waist,
            right_pipe_outer,
            right_pipe_inner,
            crotch,
            left_pipe_inner,
            left_pipe_outer
        ]

        vertices = np.array(vertices)


        # move origin from center of waist to crotch
        vertices[:,1] += self._config.crotch_height 
        print(vertices)

        edges = []
        faces = [list(range(len(vertices)))]
        name = "TowelMesh"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, edges, faces)
        mesh.update()
        return mesh

    @property
    def mesh(self):
        return self._mesh

    

@dataclasses.dataclass
class TshirtMeshConfig:
    bottom_width: float = 0.65
    neck_width:float =0.32
    neck_depth:float =0.08
    shoulder_width:float= 0.62
    shoulder_height:float =0.9
    sleeve_width_start:float =0.28
    sleeve_width_end:float =0.18
    sleeve_length:float =0.76
    sleeve_angle:float =-3
    scale:float =0.635
class TshirtMesh:

    def __init__(self,config: TshirtMeshConfig):
        self._config = config
        self._mesh = self._create_mesh()

    def _create_mesh(self):
         # First we make the right half of the shirt
        print(self._config.bottom_width)
        bottom_side = np.array([self._config.bottom_width / 2.0, 0.0, 0.0])

        neck_offset = self._config.neck_width / 2.0
        neck_top = np.array([neck_offset, 1.0, 0.0])
        neck_bottom = np.array([0, 1.0 - self._config.neck_depth, 0.0])

        shoulder = np.array([self._config.shoulder_width / 2.0, self._config.shoulder_height, 0.0])

        A = np.abs(self._config.bottom_width - self._config.shoulder_width) / 2.0
        C = self._config.sleeve_width_start
        B = np.sqrt(C**2 - A**2)
        armpit_height = self._config.shoulder_height - B
        armpit = np.array([self._config.bottom_width / 2.0, armpit_height, 0.0])

        sleeve_middle = (shoulder + armpit) / 2.0
        sleeve_end = sleeve_middle + np.array([self._config.sleeve_length, 0.0, 0.0])
        sleeve_end_top = sleeve_end + np.array([0.0, self._config.sleeve_width_end / 2.0, 0.0])
        sleeve_end_bottom = sleeve_end - np.array([0.0, self._config.sleeve_width_end / 2.0, 0.0])

        # a = np.deg2rad(self.sleeve_angle)
        # up = np.array([0.0, 0.0, 1.0])
        # sleeve_end_top = abt.rotate_point_3D(sleeve_end_top, -a, sleeve_middle, up)
        # sleeve_end_bottom = abt.rotate_point_3D(sleeve_end_bottom, -a, sleeve_middle, up)

        vertices = [
            bottom_side,
            armpit,
            sleeve_end_bottom,
            sleeve_end_top,
            shoulder,
            neck_top,
            neck_bottom,
        ]

        for vertex in reversed(vertices[0:-1]):
            mirrored_vertex = vertex.copy()
            mirrored_vertex[0] *= -1
            vertices.append(mirrored_vertex)

        vertices = np.array(vertices)
        vertices[:, 1] -= 0.5  # move origin to center of shirt

        faces = [list(range(len(vertices)))]
        mesh = bpy.data.meshes.new("TshirtMesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        return mesh
        
    @property
    def mesh(self):
        return self._mesh
    

if __name__ == "__main__":
    towel_mesh = ShortsMesh(ShortsMeshConfig()).mesh


    towel = bpy.data.objects.new("Towel", towel_mesh)
    bpy.context.collection.objects.link(towel)
    print("Towel created!")
    #print([towel_mesh.keypoints["corners"][0].co])

    