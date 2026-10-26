import numpy as np

def compute_local_K(elem_nodes):

    """
    compute local stiffness matrix for a given element
    :param elem_nodes: list of node coordinates for the element
    :return: local stiffness matrix
    """

    def constitutive_matrix(plane_stress=True):
                """
                construct the constitutive matrix for the element
                :return: constitutive matrix
                """
                E = 1.0  # Young's modulus
                nu = 0.3  # Poisson's ratio
                C = np.zeros((3, 3))
                if plane_stress:
                    C[0, 0] = E / (1 - nu ** 2)
                    C[0, 1] = E * nu / (1 - nu ** 2)
                    C[1, 0] = E * nu / (1 - nu ** 2)
                    C[1, 1] = E / (1 - nu ** 2)
                    C[2, 2] = E / (2 * (1 + nu))
                else:
                    C[0, 0] = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
                    C[0, 1] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[0, 2] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[1, 0] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[1, 1] = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
                    C[1, 2] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[2, 0] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[2, 1] = E * nu / ((1 + nu) * (1 - 2 * nu))
                    C[2, 2] = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
                return C
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
        D = np.zeros((2 * n_vertices, 6))
        for i in range(n_vertices):
            D[i,0] = 1
            D[i,1] = (elem_nodes[i,0] - Ve[0])/he
            D[i,2] = (elem_nodes[i,1] - Ve[1])/he
            D[i+n_vertices,3] = 1
            D[i+n_vertices,4] = (elem_nodes[i,0] - Ve[0])/he
            D[i+n_vertices,5] = (elem_nodes[i,1] - Ve[1])/he
        return D
    def construct_G(Ae,he):
        """
        construct the G matrix for the element
        :return: G matrix
        """
        C = constitutive_matrix(plane_stress=True)
        strain_matrix = np.array([[0, 0, 0],[1, 0, 0],[0, 0, 1],[0, 0, 0],[0, 0, 1],[0, 1, 0]], dtype=np.float64).T / he
        G = Ae * strain_matrix.T @ C @ strain_matrix

        return G
    def construct_B(Ne, Le, he):
        """
        construct the B matrix for the element
        :return: B matrix
        """
        Ne = np.array(Ne)
        num_edges = len(Le)
        C = constitutive_matrix(plane_stress=True)
        strain_matrix = np.array([[0, 0, 0],[1, 0, 0],[0, 0, 1],[0, 0, 0],[0, 0, 1],[0, 1, 0]], dtype=np.float64).T / he
        B = np.zeros((6, 2 * num_edges))
        for e in range(len(Le)):
            length = Le[e]
            normal = np.array([[Ne[e,0],0],[0, Ne[e,1]],[Ne[e,1],Ne[e,0]]])

            i = e
            i_next = (e + 1) % num_edges

            for beta in range(6): # loop over the 6 shape functions

                sigma_beta = strain_matrix[:, beta].T @ C
                t_beta = sigma_beta @ normal

                B[beta, i] += t_beta[0] * length / 2.0
                B[beta, i_next] += t_beta[0] * length / 2.0

                B[beta, i + num_edges] += t_beta[1] * length / 2.0
                B[beta, i_next + num_edges] += t_beta[1] * length / 2.0

        return B
    import numpy as np

    def ritz_projection(D, G, B, elem_nodes, Ve, he):
        """
        compute the Ritz projection matrix with average-rotation constraint
        based on Eq. (3.75) - (3.79)
        """
        n = elem_nodes.shape[0]
        
        G_mod = G.copy()
        B_mod = B.copy()

        G_mod[0,:] = 0.0
        G_mod[2,:] = 0.0
        G_mod[3,:] = 0.0

        G_mod[0, 0] = 1.0
        G_mod[2, 2] = 1.0
        G_mod[3, 3] = 1.0

        xi = (elem_nodes[:, 0] - Ve[0]) / he
        eta = (elem_nodes[:, 1] - Ve[1]) / he

        B_mod[3:6, :] = B[3:6, :]

        for j in range(n):
            B_mod[0, j]         = 1.0 / n
            B_mod[0, j + n]     = 0.0
            
            B_mod[3, j]         = 0.0
            B_mod[3, j + n]     = 1.0 / n

            B_mod[2, j]         = -eta[j] / n
            B_mod[2, j + n]     =  xi[j] / n

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
    
    def K_stability(Pi_virtual, alpha=0.1):
        """
        compute the stability part of the local stiffness matrix
        :param Pi_virtual: Ritz projection matrix
        :return: stability part of the local stiffness matrix
        """
        n_vertices = Pi_virtual.shape[1]
        I = np.eye(n_vertices)
        return alpha * (I - Pi_virtual).T @ (I - Pi_virtual)
    
    Ke = []
    elem_nodes = np.asarray(elem_nodes, dtype=np.float64)
    Ve, he, Ae, Le, Ne = geometry_quantities(elem_nodes)
    D = construct_D(elem_nodes, Ve, he)
    G = construct_G(Ae, he)
    B = construct_B(Ne, Le, he)
    Pi_poly, Pi_virtual = ritz_projection(D, G, B, elem_nodes, Ve, he)
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