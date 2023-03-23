import numpy as np 
from typing import List
import dataclasses
import bpy


def quadratic_bezier(start: np.ndarray, control:np.ndarray, end:np.ndarray, steps:int) -> List[np.ndarray]:
    t = np.linspace(0,1,steps,endpoint=False)
    t = t[:,np.newaxis]
    points =  (1-t)**2 * start + 2*(1-t)*t*control + t**2 * end
    return points.tolist()


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

        ## create the geometric template
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

        keypoint_ids = list(range(len(vertices)))

        ## create the bezier trajectories for all edges

        control_y_offset = -0.05
        control_x_offset = 0.01
        steps = 10
        beziered_vertices = []
        for i in range(len(vertices)):
            start = vertices[i]
            if i == len(vertices) -1:
                end = vertices[0]
            else:
                end = vertices[i+1]
            control_point = (start + end) / 2
            x = (end-start) / np.linalg.norm(end-start)
            y = np.cross(np.array([0,0,1]),x)
            control_point += y * control_y_offset
            control_point += x * control_x_offset

            beziered_vertices.extend(quadratic_bezier(start,control_point,end,steps))
            
            keypoint_ids[i +1:] = [kid + steps -1 for kid in keypoint_ids[i +1:]]
        # move origin from center of waist to crotch
        #beziered_vertices[:,1] += self._config.crotch_height 


        edges = []
        faces = [list(range(len(beziered_vertices)))]
        name = "TowelMesh"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(beziered_vertices, edges, faces)
        mesh.update()
        return mesh, keypoint_ids
    

if __name__ == "__main__":
    bpy.ops.object.delete()  # Delete default cube

    mesh,keypoints = ShortsMesh(ShortsMeshConfig())._create_mesh()


    towel = bpy.data.objects.new("mesh", mesh)
    bpy.context.collection.objects.link(towel)

    # select & activate!
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = towel
    towel.select_set(True)


    ## BEVEL all corners (with bookkeeping for keypoints)
    i = 0
    while i < len(keypoints):
        kid = keypoints[i]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        towel.data.vertices[kid].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.bevel(offset=0.5,segments=4,affect="VERTICES")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        # so current vertex is deleted. All future vertices are shifted by one to the left.
        # and segment+1 newly are added, of which the middle is the newest keypoint
        keypoints = [kid -1 for kid in keypoints]
        keypoints[i] = len(towel.data.vertices) - 2
        print(keypoints)
        i+=1




    radius = 0.01
    for kid in keypoints:
        bpy.ops.mesh.primitive_ico_sphere_add(location=towel.data.vertices[kid].co,scale=(radius,radius,radius))
    print("created!")

    