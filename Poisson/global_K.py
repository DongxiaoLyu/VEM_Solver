import local_K
import numpy as np

def compute_global_K(nodes, elems):
    """
    compute the global stiffness matrix K for the entire mesh
    :param nodes: list of node coordinates
    :param elems: list of elements (each element is a list of node IDs)
    :return: global stiffness matrix K"""
    n = len(nodes)     
    K = np.zeros((n, n))

    for elem_ids in elems:
        coords = [nodes[nid] for nid in elem_ids]
        Ke = local_K.compute_local_K(coords)

        for i, ii in enumerate(elem_ids):
            for j, jj in enumerate(elem_ids):
                K[ii, jj] += Ke[i, j]

    return K