import numpy as np

"""
Construct the load vector F for 2-D linear elasticity (VEM).

DOF ordering -- identical to global_K.py:
    u_x of node id  ->  F[node_id - 1]
    u_y of node id  ->  F[node_id - 1 + n]
with n = number of nodes.  Node ids are the 1-based ids of voronoi_mesh.txt.

Two sources contribute to F:

  1. BODY FORCE, given as a lumped (nodal) force: the selected nodes each carry
     the force f = (f_x, f_y), so it is simply added to their two DOFs.
         F[node]     += f_x
         F[node + n] += f_y

  2. NEUMANN TRACTION t = (t_x, t_y) per unit length on a boundary edge.
     The virtual/test function is linear along an edge, so the edge integral of
     a constant traction is integrated exactly by the trapezoidal rule:
         F[a],     F[b]     += t_x * L / 2
         F[a + n], F[b + n] += t_y * L / 2
     with L the length of the edge (a, b).  Summing over all edges of a side
     therefore gives the exact resultant t * L_side at the right place.

The boundary conditions are read from the file written by
Tools/generate_bc_file.py, or passed directly as dicts.

Example
-------
    python construct_F.py                     # report the load of the BC file
"""


# ---------------------------------------------------------------------------
# I/O: boundary conditions written by Tools/generate_bc_file.py
# ---------------------------------------------------------------------------

def read_boundary_conditions(filename):
    """
    Read Dirichlet displacements, Neumann tractions and the nodal body force.

    File format
    -----------
    Dirichlet BCs #Node id #Component (1=ux, 2=uy) #Value
    Neumann BCs #Node id1 #Node id2 #t_x #t_y
    Body force #Node id #f_x #f_y

    Returns
    -------
    diri : {(node_id, comp): value}     comp = 1 (u_x) or 2 (u_y)
    neu  : {(node_a, node_b): (tx, ty)}
    body : {node_id: (fx, fy)}
    """
    diri, neu, body = {}, {}, {}
    section = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith('dirichlet'):
                section = 'dirichlet'
                continue
            if low.startswith('neumann'):
                section = 'neumann'
                continue
            if low.startswith('body'):
                section = 'body'
                continue

            parts = line.split()
            if section == 'dirichlet':
                diri[(int(parts[0]), int(parts[1]))] = float(parts[2])
            elif section == 'neumann':
                neu[(int(parts[0]), int(parts[1]))] = (float(parts[2]),
                                                       float(parts[3]))
            elif section == 'body':
                body[int(parts[0])] = (float(parts[1]), float(parts[2]))
            else:
                raise ValueError(f'undefined section in {filename}: {line!r}')

    return diri, neu, body


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def node_coord(nodes, node_id):
    """Coordinates of a 1-based node id, for a dict or a list of nodes."""
    if isinstance(nodes, dict):
        return np.asarray(nodes[node_id], dtype=np.float64)
    return np.asarray(nodes[node_id - 1], dtype=np.float64)


def edge_length(nodes, edge):
    """Length of the edge (node_a, node_b)."""
    a, b = edge
    return float(np.linalg.norm(node_coord(nodes, b) - node_coord(nodes, a)))


def boundary_edges_of(elems):
    """Edges belonging to exactly one element: the boundary of the mesh."""
    count = {}
    for elem in elems:
        elem = [int(n) + 1 for n in elem]        # mesh_input uses 0-based ids
        m = len(elem)
        for i in range(m):
            a, b = elem[i], elem[(i + 1) % m]
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    return {e for e, c in count.items() if c == 1}


# ---------------------------------------------------------------------------
# contributions to F
# ---------------------------------------------------------------------------

def add_body_force(F, nodes, body_force, verbose=True):
    """
    Add the lumped body force to F, in place.

    :param F: load vector of length 2n (modified in place)
    :param nodes: node coordinates (dict keyed by id, or list)
    :param body_force: {node_id: (fx, fy)} -- the force put on each node
    :return: F
    """
    n = F.size // 2
    for node_id, force in body_force.items():
        i = int(node_id) - 1
        if not 0 <= i < n:
            raise ValueError(f'body force on unknown node {node_id} '
                             f'(mesh has {n} nodes)')
        fx, fy = float(force[0]), float(force[1])
        F[i] += fx              # u_x
        F[i + n] += fy          # u_y

    if verbose and body_force:
        fx = sum(float(v[0]) for v in body_force.values())
        fy = sum(float(v[1]) for v in body_force.values())
        print(f"[construct_F] body force on {len(body_force)} nodes, "
              f"total ({fx:+.6e}, {fy:+.6e})")
    return F


def add_traction(F, nodes, neumann, elems=None, verbose=True):
    """
    Add the Neumann tractions to F, in place.

    The traction t = (t_x, t_y) acts per unit length on the edge (a, b); a
    constant traction over an edge gives t * L / 2 to each of its two nodes
    (exact, because the virtual functions are linear along an edge).

    :param F: load vector of length 2n (modified in place)
    :param nodes: node coordinates (dict keyed by id, or list)
    :param neumann: {(node_a, node_b): (tx, ty)}
    :param elems: optional, used only to warn about non-boundary edges
    :return: F
    """
    n = F.size // 2

    bnd = None
    if elems is not None:
        bnd = boundary_edges_of(elems)

    total = np.zeros(2)
    for (a, b), traction in neumann.items():
        a, b = int(a), int(b)
        if bnd is not None and (min(a, b), max(a, b)) not in bnd:
            print(f"[construct_F] warning: edge ({a}, {b}) is not a boundary "
                  f"edge, its traction is still applied")
        L = edge_length(nodes, (a, b))
        tx, ty = float(traction[0]), float(traction[1])
        for node_id in (a, b):
            i = node_id - 1
            F[i] += tx * L / 2.0
            F[i + n] += ty * L / 2.0
        total += np.array([tx, ty]) * L

    if verbose and neumann:
        print(f"[construct_F] traction on {len(neumann)} boundary edges, "
              f"total ({total[0]:+.6e}, {total[1]:+.6e})")
    return F


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def construct_F_vector(nodes, elems, body_force=None, neumann=None,
                       bc_file='boundary_condition.txt', verbose=True):
    """
    Construct the load vector F for the entire mesh.

    :param nodes: list (or dict) of node coordinates
    :param elems: list of elements (each element is a list of node indices)
    :param body_force: {node_id: (fx, fy)}, optional; read from bc_file if None
    :param neumann: {(node_a, node_b): (tx, ty)}, optional; same
    :param bc_file: boundary condition file used when body_force and neumann
                    are both None (written by Tools/generate_bc_file.py)
    :param verbose: print a short summary
    :return: load vector F of length 2 * len(nodes)
    """
    n = len(nodes)
    F = np.zeros(2 * n)

    if body_force is None and neumann is None:
        _, neumann, body_force = read_boundary_conditions(bc_file)
    body_force = {} if body_force is None else body_force
    neumann = {} if neumann is None else neumann

    add_body_force(F, nodes, body_force, verbose=verbose)
    add_traction(F, nodes, neumann, elems=elems, verbose=verbose)

    if verbose:
        xy = np.array([node_coord(nodes, i) for i in range(1, n + 1)])
        fx = F[:n].sum()
        fy = F[n:].sum()
        # net moment about the origin, m_z = sum(x * f_y - y * f_x)
        mz = float(np.dot(xy[:, 0], F[n:]) - np.dot(xy[:, 1], F[:n]))
        print(f"[construct_F] |F| = {np.abs(F).max():.6e}, net force "
              f"({fx:+.6e}, {fy:+.6e}), net moment about the origin "
              f"{mz:+.6e}")
    return F


if __name__ == "__main__":
    import mesh_input

    nodes, elems = mesh_input.read_mesh('voronoi_mesh.txt')
    print(f"Loaded {len(nodes)} nodes and {len(elems)} elements")

    diri, neu, body = read_boundary_conditions('boundary_condition.txt')
    print(f"BC file: {len(diri)} Dirichlet DOF, {len(neu)} traction edges, "
          f"{len(body)} nodes with a body force")

    F = construct_F_vector(nodes, elems, body_force=body, neumann=neu)
    loaded = np.nonzero(F)[0]
    print(f"F has {F.size} entries, {loaded.size} of them non-zero")
