import numpy as np

def read_boundary_conditions(filename = "boundary_condition.txt"):
    """
    Read Dirichlet and Neumann boundary conditions from a text file.
    Dirichlet BCs #Node id #Dirichlet Value
    1   10
    Neumann BCs #Node id1 #Node id2 #Neumann Value
    1   2   10
    Returns
    -------
    diri_nodes : list of int
        Node IDs with Dirichlet BCs (1-based).
    diri_values : list of float
        Corresponding Dirichlet values.
    neu_edges : list of (int, int)
        Edges with Neumann BCs (1-based).
    neu_values : list of float
        Corresponding Neumann values.
    """
    diri_nodes = []
    diri_values = []
    neu_edges = []
    neu_values = []

    section = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith('dirichlet'):
                section = 'dirichlet'
                continue
            elif lower.startswith('neumann'):
                section = 'neumann'
                continue

            parts = line.split()

            if section == 'dirichlet':
                # Expected: node_id  value
                if len(parts) < 2:
                    raise ValueError(f"Bad Dirichlet line: '{line}'")
                nid = int(parts[0])
                val = float(parts[1])
                diri_nodes.append(nid)
                diri_values.append(val)

            elif section == 'neumann':
                # Expected: node_a  node_b  value
                if len(parts) < 3:
                    raise ValueError(f"Bad Neumann line: '{line}'")
                a = int(parts[0])
                b = int(parts[1])
                val = float(parts[2])
                neu_edges.append((a, b))
                neu_values.append(val)

    return diri_nodes, diri_values, neu_edges, neu_values

def assign_neumann_bcs(nodes,neu_edges,neu_values,F):
    """
    assign Neumann boundary condition to the element edges
    :param neu_edges: list of edges(node1,node2) with neumann boundary conditions
    :param neu_values: list of corresponding boudnary values
    :param F: right side force vector
    :return: updated F with boundary conditions
    """ 

    neu_edges = np.asarray(neu_edges, dtype=int) - 1
    neu_values = np.asarray(neu_values,dtype=float)
    F_bc = F.copy()

    for edge in range(len(neu_edges)):
        i = neu_edges[edge,0]
        j = neu_edges[edge,1]
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        gi = neu_values[edge]
        gj = neu_values[edge]
        L = np.hypot(xj - xi, yj - yi)
        F_bc[i] += 0.5 * L * gi
        F_bc[j] += 0.5 * L * gj

    return F_bc

    

def assign_dirichlet_bcs(diri_nodes, diri_values, Kg, F):
    """
    assign Dirichlet boundary conditions to the nodes
    :param diri_nodes: list of node IDs with Dirichlet boundary conditions
    :param diri_values: list of corresponding boundary values
    :param Kg: global stiffness matrix
    :param F: right side force vector
    :return: updated Kg, F with boundary conditions
    """

    diri_nodes = np.array(diri_nodes, dtype=int) - 1
    diri_values = np.array(diri_values, dtype=float)

    Kg_bc = Kg.copy()
    F_bc = F.copy()

    F_bc -= Kg_bc[:, diri_nodes] @ diri_values
    Kg_bc[diri_nodes, :] = 0.0
    Kg_bc[:, diri_nodes] = 0.0
    Kg_bc[diri_nodes, diri_nodes] = 1.0
    F_bc[diri_nodes] = diri_values

    
    return Kg_bc, F_bc