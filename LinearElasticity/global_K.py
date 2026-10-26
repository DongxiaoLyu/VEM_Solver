import local_K
import numpy as np

def compute_global_K(nodes, elems):
    """
    compute the global stiffness matrix K for the entire mesh
    :param nodes: list of node coordinates
    :param elems: list of elements (each element is a list of node IDs)
    :return: global stiffness matrix K"""
    n = len(nodes)     
    K = np.zeros((2*n, 2*n))

    for elem_ids in elems:
        coords = [nodes[nid] for nid in elem_ids]
        Ke = local_K.compute_local_K(coords)

        n_local = len(elem_ids)

        for i, ii in enumerate(elem_ids):
            for j, jj in enumerate(elem_ids):
                K[ii, jj] += Ke[i, j]
                K[ii, jj + n] += Ke[i, j + n_local]
                K[ii + n, jj] += Ke[i + n_local, j]
                K[ii + n, jj + n] += Ke[i + n_local, j + n_local]

    return K
