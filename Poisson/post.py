import numpy as np

def compute_gradient(u, nodes, elems):
    """
    Compute the gradient of u at each element using the VEM formula.
    :param u: solution vector at nodes
    :param nodes: list of node coordinates
    :param elems: list of elements (each element is a list of node IDs)
    :return: array of gradients at each element
    """
    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(u, dtype=float)

    grads = []
    return np.array(grads)