import numpy as np
import pymeshlab as pml
import trimesh
import torch


def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=int(target), preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh') # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces

def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices() # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct)) # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces() # faces defined by the same verts
    ms.meshing_remove_null_faces() # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(min_d))
    
    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces

def connected_components(vertices, faces):
    """
    Compute the connected components of the mesh.
    """
    # Using union-find to identify connected components
    parent = list(range(len(vertices)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootX] = rootY
    
    for face in faces:
        union(face[0], face[1])
        union(face[1], face[2])
        union(face[0], face[2])
    
    components = {}
    for i, p in enumerate(parent):
        root = find(p)
        if root in components:
            components[root].append(i)
        else:
            components[root] = [i]
            
    return components

def clean_mesh_connected(vertices, faces):
    components = connected_components(vertices, faces)
    largest_component = max(components.values(), key=len)
    # Create a set for faster look-up
    largest_component_set = set(largest_component)

    # Filter out vertices not in the largest connected component
    filtered_vertices = vertices[largest_component]

    # Create a mapping from old vertex indices to new vertex indices
    vertex_index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component)}

    # Filter out faces not in the largest connected component and update their indices
    filtered_faces = []
    for face in faces:
        if all(v in largest_component_set for v in face):
            new_face = [vertex_index_mapping[v] for v in face]
            filtered_faces.append(new_face)

    filtered_faces = torch.tensor(np.array(filtered_faces))    
    return filtered_vertices, filtered_faces
    
def clean_trimesh_connected(vertices, triangles):    

    mesh = trimesh.Trimesh(vertices, triangles, process=False)
        
    connected_components = mesh.split(only_watertight=False)
        
    if len(connected_components) > 1:
        
        # 按面积排序连通域
        # area_sorted_components = sorted(connected_components, key=lambda x: x.area, reverse=True)
        # length = min(len(area_sorted_components), 5)
        # area_sorted_components = area_sorted_components[0: length]
        area_sorted_components = connected_components
        
        # 计算每个连通域的顶点数目，找到顶点数目大于500的连通域
        vertex_counts = []
        for component in area_sorted_components:
            vertex_count = len(component.vertices)
            vertex_counts.append((component, vertex_count))
        selected_components = [comp for comp, count in vertex_counts if count > 500]

        # Calculate AABB for each selected component
        aabbs = [comp.bounding_box.bounds for comp in selected_components]
        
        # Check if any AABB is completely contained within another AABB
        to_remove = set()
        for i, aabb1 in enumerate(aabbs):
            for j, aabb2 in enumerate(aabbs):
                if i != j and np.all(aabb1[0] >= aabb2[0]) and np.all(aabb1[1] <= aabb2[1]):
                    to_remove.add(i)
        
        # Remove the contained components
        cleaned_components = [comp for i, comp in enumerate(selected_components) if i not in to_remove]

        # 计算这些连通域的中心点到原点的距离
        centroids = [comp.centroid for comp in cleaned_components]
        distances = [np.linalg.norm(centroid) for centroid in centroids]

        # 找到距离原点最近的连通域
        min_distance_index = np.argmin(distances)
        closest_component = cleaned_components[min_distance_index]

        # 输出该连通域的顶点和面
        vertices = closest_component.vertices
        triangles = closest_component.faces

    return vertices, triangles    