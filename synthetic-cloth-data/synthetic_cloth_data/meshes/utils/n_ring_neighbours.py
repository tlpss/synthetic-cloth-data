import bpy


def build_neighbour_dict(blender_object):
    edges, vertices = blender_object.data.edges, blender_object.data.vertices
    neighbour_dict = {}
    for vertex in vertices:
        neighbour_dict[vertex.index] = []
    for edge in edges:
        neighbour_dict[edge.vertices[0]].append(edge.vertices[1])
        neighbour_dict[edge.vertices[1]].append(edge.vertices[0])
    return neighbour_dict


def get_1_ring_neighbours(neigbour_dict, vertex_id):
    return [vertex_id] + neigbour_dict[vertex_id]


def get_n_ring_neighbours(neighbour_dict, vertex_id, n):
    assert n >= 0
    if n == 0:
        return [vertex_id]
    else:
        neighbours = get_1_ring_neighbours(neighbour_dict, vertex_id)
        for _ in range(n - 1):
            new_neighbours = []
            for neighbour in neighbours:
                new_neighbours += get_1_ring_neighbours(neighbour_dict, neighbour)
            neighbours = new_neighbours
        return list(set(neighbours))


def get_strict_n_ring_neighbours(neighbour_dict, vertex_id, n):
    if n == 0:
        return [vertex_id]

    n_ring = get_n_ring_neighbours(neighbour_dict, vertex_id, n)
    n_minus_1_ring = get_n_ring_neighbours(neighbour_dict, vertex_id, n - 1)
    return list(set(n_ring) - set(n_minus_1_ring))


if __name__ == "__main__":
    # add monkey
    bpy.ops.mesh.primitive_monkey_add()
    monkey = bpy.context.active_object
    vertex_id = 74
    neighbours = get_strict_n_ring_neighbours(build_neighbour_dict(monkey), vertex_id, 2)

    # add cube to vertex
    bpy.ops.mesh.primitive_cube_add(location=monkey.data.vertices[vertex_id].co, scale=(0.01, 0.01, 0.01))
    # add sphere to each neighbour
    for neighbour in neighbours:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=monkey.data.vertices[neighbour].co)
