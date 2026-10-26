import numpy as np
"""
Construct load vector F
"""
def construct_F_vector(nodes, elems):
    """
    construct the load vector F for the entire mesh
    :param nodes: list of node coordinates
    :param elems: list of elements (each element is a list of node IDs)
    :return: load vector F
    """

    def compute_local_F(elem_nodes):
        """
        compute local load vector for a given element
        :param elem_nodes: list of node coordinates for the element
        :return: local load vector
        """
        elem_nodes = np.asarray(elem_nodes, dtype=np.float64)
        n_vertices = elem_nodes.shape[0]
        F_local = np.zeros(n_vertices)

        # Compute area
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross_product = x * y_next - x_next * y
        area_dirct = 0.5 * np.sum(cross_product)
        area = np.abs(area_dirct)

        # f at centroid
        cx = np.sum((x + x_next) * cross_product) / (6 * area_dirct)
        cy = np.sum((y + y_next) * cross_product) / (6 * area_dirct)
        centroid = np.array([cx, cy])
        f_centroid = f_exact(centroid[0], centroid[1])

        # Distribute the load values to the local load vector
        F_local = np.full(n_vertices, f_centroid * area / n_vertices)

        return F_local
    def f_exact(x, y):
        """
        exact load function f(x, y)
        :param x: x coordinate
        :param y: y coordinate
        :return: f value at (x, y)
        """
        return 1

    
    n = len(nodes)
    F = np.zeros(n)

    for elem_ids in elems:
        coords = [nodes[nid] for nid in elem_ids]
        Fe = compute_local_F(coords)

        for i, ii in enumerate(elem_ids):
            F[ii] += Fe[i]

    return F