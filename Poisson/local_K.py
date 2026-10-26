import numpy as np

def compute_local_K(elem_nodes):

    """
    compute local stiffness matrix for a given element
    :param elem_nodes: list of node coordinates for the element
    :return: local stiffness matrix
    """

    def geometry_quantities(elem_nodes):
        """
        compute geometry quantities for the element
        :param elem_nodes: list of node coordinates for the element
        :return: centroid, diameter, area, edge lengths, outer normal vectors
        """
        # compute centroid and area
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross_product = x * y_next - x_next * y
        area_dirct = 0.5 * np.sum(cross_product)
        area = np.abs(area_dirct)
        cx = np.sum((x + x_next) * cross_product) / (6 * area_dirct)
        cy = np.sum((y + y_next) * cross_product) / (6 * area_dirct)
        centroid = np.array([cx, cy])

        # compute diameter
        diff = elem_nodes[:, None, :] - elem_nodes[None, :, :]
        diameter = np.max(np.linalg.norm(diff, axis=-1))

        if area_dirct > 0: # CCW ordering
            CCW = 1
        else:
            CCW = -1

        # compute edge lengths and outer normal vectors
        edge_lengths = []
        outer_normals = []
        num_nodes = elem_nodes.shape[0]
        for i in range(num_nodes):
            p1 = elem_nodes[i]
            p2 = elem_nodes[(i + 1) % num_nodes]
            edge_length = np.linalg.norm(p2 - p1)
            edge_lengths.append(edge_length)

            # compute outer normal vector
            edge_vector = p2 - p1
            normal_vector = CCW * np.array([edge_vector[1], -edge_vector[0]])
            normal_vector /= np.linalg.norm(normal_vector)
            outer_normals.append(normal_vector)
        return centroid, diameter, area, edge_lengths, outer_normals

    def construct_D(elem_nodes, Ve, he):
        """
        construct the D matrix for the element
        :return: D matrix
        """
        n_vertices = elem_nodes.shape[0]
        D = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            D[i, 0] = 1
            D[i, 1] = (elem_nodes[i, 0] - Ve[0]) / he
            D[i, 2] = (elem_nodes[i, 1] - Ve[1]) / he
        return D
    def construct_G(Ae,he):
        """
        construct the G matrix for the element
        :return: G matrix
        """
        G = np.zeros((3, 3))
        G[1, 1] = Ae / (he ** 2)
        G[2, 2] = Ae / (he ** 2)
        return G
    def construct_B(elem_nodes,he,Le,Ne):
        """
        construct the B matrix for the element
        :return: B matrix
        """
        n_vertices = elem_nodes.shape[0]
        B = np.zeros((3, n_vertices))

        grad_m2 = np.array([1.0 / he, 0.0])
        grad_m3 = np.array([0.0, 1.0 / he])
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            ne = Ne[i]
            le = Le[i]

            d2 = np.dot(grad_m2, ne)         # ∇m2 · n_e
            d3 = np.dot(grad_m3, ne)         # ∇m3 · n_e

            B[1, i] += d2 * le / 2.0
            B[1, j] += d2 * le / 2.0
            B[2, i] += d3 * le / 2.0
            B[2, j] += d3 * le / 2.0
        return B
    def ritz_projection(D, G, B):
        """
        compute the Ritz projection matrix for the element
        :param D: D matrix
        :param G: G matrix
        :param B: B matrix
        :return: Ritz projection matrix
        """
        n_vertices = B.shape[1]

        # modified G and B
        G_mod = G.copy()
        B_mod = B.copy()
        G_mod[0, 0] = 1.0
        B_mod[0, :] = np.ones(n_vertices)/n_vertices

        Pi_poly = np.linalg.inv(G_mod) @ B_mod
        Pi_virtual = D @ Pi_poly

        return Pi_poly, Pi_virtual
    def K_consistency(Pi_poly, G):
        """
        compute the consistency part of the local stiffness matrix
        :param Pi_poly: Ritz projection matrix
        :param G: G matrix
        :return: consistency part of the local stiffness matrix
        """
        return Pi_poly.T @ G @ Pi_poly
    def K_stability(Pi_virtual):
        """
        compute the stability part of the local stiffness matrix
        :param Pi_virtual: Ritz projection matrix
        :return: stability part of the local stiffness matrix
        """
        n_vertices = Pi_virtual.shape[1]
        I = np.eye(n_vertices)
        return (I - Pi_virtual).T @ (I - Pi_virtual)
    
    Ke = []
    elem_nodes = np.asarray(elem_nodes, dtype=np.float64)
    Ve, he, Ae, Le, Ne = geometry_quantities(elem_nodes)
    D = construct_D(elem_nodes, Ve, he)
    G = construct_G(Ae, he)
    B = construct_B(elem_nodes, he, Le, Ne)
    Pi_poly, Pi_virtual = ritz_projection(D, G, B)
    Kc = K_consistency(Pi_poly, G)
    Ks = K_stability(Pi_virtual)
    Ke = Kc + Ks
    return Ke


if __name__ == "__main__":
    # Example usage
    elem_nodes = [[0, 0], [1, 0], [1, 1], [0, 1]]
    Ke = compute_local_K(elem_nodes)
    print("Local stiffness matrix Ke:")
    print(Ke)