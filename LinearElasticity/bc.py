import numpy as np

"""
Assign boundary conditions for 2-D linear elasticity (VEM).

DOF ordering -- identical to global_K.py and construct_F.py:
    u_x of node id  ->  index (node_id - 1)
    u_y of node id  ->  index (node_id - 1 + n)
with n = number of nodes and node ids the 1-based ids of voronoi_mesh.txt.

Because every node carries two DOFs, a Dirichlet condition is applied to a
single COMPONENT of a node, which makes rollers / symmetry lines possible:

    Dirichlet BCs #Node id #Component (1=ux, 2=uy) #Value
    Neumann BCs   #Node id1 #Node id2 #t_x #t_y
    Body force    #Node id #f_x #f_y

Dirichlet conditions are imposed by elimination: the known value is moved to
the right-hand side, then the row and the column of that DOF are cleared and a
unit diagonal entry is inserted.  The matrix therefore stays symmetric, which
matters for the solver and for the reaction forces.
"""


# ---------------------------------------------------------------------------
# DOF helpers
# ---------------------------------------------------------------------------

def dof_index(node_id, comp, n):
    """
    Global DOF index of a component of a node.

    :param node_id: 1-based node id
    :param comp: 1 for u_x, 2 for u_y
    :param n: number of nodes
    :return: index into the length-2n vector / (2n x 2n) matrix
    """
    i = int(node_id) - 1
    if not 0 <= i < n:
        raise ValueError(f'node id {node_id} out of range (mesh has {n} nodes)')
    if comp not in (1, 2):
        raise ValueError(f'component must be 1 (u_x) or 2 (u_y), got {comp!r}')
    return i if comp == 1 else i + n


def _normalize_diri(diri_nodes, diri_values, n):
    """
    Turn the (nodes, values) pair into [(dof, value), ...].

    `diri_nodes` entries may be
        (node_id, comp)   fix one component   <- what the BC file provides
        node_id           fix both components (scalar-Poisson style)
    """
    pairs = []
    seen = {}
    for node, val in zip(diri_nodes, diri_values):
        if isinstance(node, (tuple, list, np.ndarray)):
            node_id, comps = int(node[0]), [int(node[1])]
        else:
            node_id, comps = int(node), [1, 2]      # both directions
        for comp in comps:
            dof = dof_index(node_id, comp, n)
            if dof in seen and seen[dof] != float(val):
                print(f'[bc] warning: DOF {dof} (node {node_id}, component '
                      f'{comp}) is prescribed twice with different values '
                      f'{seen[dof]} and {val}; the last one wins')
            seen[dof] = float(val)
            pairs.append((dof, float(val)))
    return pairs


# ---------------------------------------------------------------------------
# Reading boundary_condition.txt
# ---------------------------------------------------------------------------

def read_bc_dicts(filename="boundary_condition.txt"):
    """
    Read the boundary condition file written by Tools/generate_bc_file.py.

    Returns
    -------
    diri : {(node_id, comp): value}      comp = 1 (u_x) or 2 (u_y)
    neu  : {(node_a, node_b): (tx, ty)}
    body : {node_id: (fx, fy)}
    """
    diri, neu, body = {}, {}, {}
    section = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()      # strip inline comments
            if not line:
                continue
            lower = line.lower()
            if lower.startswith('dirichlet'):
                section = 'dirichlet'
                continue
            if lower.startswith('neumann'):
                section = 'neumann'
                continue
            if lower.startswith('body'):
                section = 'body'
                continue

            parts = line.split()
            if section == 'dirichlet':
                if len(parts) != 3:
                    raise ValueError(
                        f"bad Dirichlet line {line!r}: expected "
                        f"'node_id component value' with component 1=ux, 2=uy")
                diri[(int(parts[0]), int(parts[1]))] = float(parts[2])
            elif section == 'neumann':
                if len(parts) != 4:
                    raise ValueError(
                        f"bad Neumann line {line!r}: expected "
                        f"'node_a node_b t_x t_y'")
                neu[(int(parts[0]), int(parts[1]))] = (float(parts[2]),
                                                       float(parts[3]))
            elif section == 'body':
                if len(parts) != 3:
                    raise ValueError(
                        f"bad body force line {line!r}: expected "
                        f"'node_id f_x f_y'")
                body[int(parts[0])] = (float(parts[1]), float(parts[2]))
            else:
                raise ValueError(f'no section header before line {line!r}')

    return diri, neu, body


def read_boundary_conditions(filename="boundary_condition.txt"):
    """
    Read Dirichlet and Neumann boundary conditions (main.py call style).

    Returns
    -------
    diri_nodes : list of (int, int)
        (node_id, component) pairs, component 1 = u_x, 2 = u_y.
    diri_values : list of float
        Corresponding prescribed displacements.
    neu_edges : list of (int, int)
        Edges with Neumann BCs (1-based node ids).
    neu_values : list of (float, float)
        Corresponding traction vectors (t_x, t_y).

    Notes
    -----
    The body force section of the file is ignored here; use read_bc_dicts() or
    construct_F.py to get it.
    """
    diri, neu, _ = read_bc_dicts(filename)

    diri_nodes = sorted(diri)
    diri_values = [diri[key] for key in diri_nodes]
    neu_edges = sorted(neu)
    neu_values = [neu[key] for key in neu_edges]
    return diri_nodes, diri_values, neu_edges, neu_values


# ---------------------------------------------------------------------------
# Dirichlet: elimination
# ---------------------------------------------------------------------------

def rigid_body_check(diri_nodes, diri_values, nodes, verbose=True):
    """
    Check that the Dirichlet conditions really remove the 3 rigid-body modes.

    A 2-D elasticity system still has 3 zero-energy modes (translation in x,
    translation in y and rotation), so the constrained DOFs must "see" all
    three of them.  Their values at the constrained DOFs are collected in a
    3 x m matrix; if its rank is below 3, some rigid motion is still free and
    the system stays singular.

    :param diri_nodes: [(node_id, comp), ...] or [node_id, ...]
    :param diri_values: corresponding values (unused, only for the DOF list)
    :param nodes: node coordinates (list or dict) -- needed for the rotation
    :return: True when the constraints are sufficient
    """
    n = len(nodes)
    dofs = constrained_dofs(diri_nodes, diri_values, n)
    if dofs.size == 0:
        if verbose:
            print('[bc] warning: no Dirichlet DOF at all -> singular system')
        return False

    def coord(node_id):
        if isinstance(nodes, dict):
            return np.asarray(nodes[node_id], dtype=np.float64)
        return np.asarray(nodes[int(node_id) - 1], dtype=np.float64)

    xy = np.array([coord(i) for i in range(1, n + 1)])
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()

    modes = np.zeros((3, 2 * n))
    modes[0, :n] = 1.0                 # translation in x
    modes[1, n:] = 1.0                 # translation in y
    modes[2, :n] = -(xy[:, 1] - cy)    # rotation about the centroid
    modes[2, n:] = xy[:, 0] - cx

    sampled = modes[:, dofs]                       # 3 x m
    rank = np.linalg.matrix_rank(
        sampled, tol=1e-10 * max(1.0, float(np.abs(sampled).max())))

    if verbose:
        if rank == 3:
            print(f'[bc] rigid-body modes removed '
                  f'(rank {rank}/3 on {dofs.size} constrained DOF)')
        else:
            names = ['translation in x', 'translation in y', 'rotation']
            free = [names[i] for i in range(3)
                    if np.linalg.norm(sampled[i]) < 1e-12
                    or np.linalg.matrix_rank(sampled[[i]], tol=1e-10) == 0]
            why = ', '.join(free) if free else \
                'a combination of them (add a constraint in another direction)'
            print(f'[bc] warning: only rank {rank}/3 of the rigid-body modes '
                  f'is constrained -> the system is still singular; '
                  f'not blocked: {why}')
    return bool(rank == 3)


def assign_dirichlet_bcs(diri_nodes, diri_values, Kg, F, nodes=None, verbose=True):
    """
    Impose the Dirichlet conditions on the global system by elimination.

    For every prescribed DOF d with value v the known part is first moved to
    the right-hand side, F -= K[:, d] * v (using the untouched matrix), and
    then the row and the column are cleared and K[d, d] = 1, F[d] = v.  The
    matrix stays symmetric and the prescribed values are reproduced exactly.

    :param diri_nodes: [(node_id, comp), ...] or [node_id, ...] (both comps)
    :param diri_values: corresponding prescribed values
    :param Kg: global stiffness matrix (2n x 2n)
    :param F: global load vector (2n)
    :param nodes: optional node coordinates; when given, the rigid-body modes
                  are checked before the system is handed to the solver
    :return: Kg_bc, F_bc
    """
    Kg_bc = np.array(Kg, dtype=np.float64, copy=True)
    F_bc = np.array(F, dtype=np.float64, copy=True)

    n = F_bc.size // 2
    if Kg_bc.shape != (2 * n, 2 * n):
        raise ValueError(f'Kg has shape {Kg_bc.shape}, expected '
                         f'{(2 * n, 2 * n)} for n = {n} nodes')

    if nodes is not None:
        rigid_body_check(diri_nodes, diri_values, nodes, verbose=verbose)

    pairs = _normalize_diri(diri_nodes, diri_values, n)
    if not pairs:
        if verbose:
            print('[bc] no Dirichlet condition: the system is singular '
                  '(rigid-body modes are not removed)')
        return Kg_bc, F_bc

    dofs = np.array([d for d, _ in pairs], dtype=int)
    vals = np.array([v for _, v in pairs], dtype=np.float64)

    # 1) move the known part to the right-hand side
    nonzero = vals != 0.0
    if np.any(nonzero):
        F_bc -= Kg_bc[:, dofs[nonzero]] @ vals[nonzero]

    # 2) clear row and column, keep the diagonal
    Kg_bc[dofs, :] = 0.0
    Kg_bc[:, dofs] = 0.0
    Kg_bc[dofs, dofs] = 1.0
    F_bc[dofs] = vals

    if verbose:
        # a DOF index d is node (d % n): u_x of node i -> i, u_y -> i + n
        n_nodes = len({int(d % n) for d in dofs})
        print(f'[bc] Dirichlet: {len(pairs)} DOF constrained on {n_nodes} '
              f'node(s); matrix stays symmetric: '
              f'{np.allclose(Kg_bc, Kg_bc.T)}')
    return Kg_bc, F_bc


def constrained_dofs(diri_nodes, diri_values, n):
    """Sorted array of the DOF indices fixed by the Dirichlet conditions."""
    return np.array(sorted({d for d, _ in
                            _normalize_diri(diri_nodes, diri_values, n)}),
                    dtype=int)


def free_dofs(diri_nodes, diri_values, n):
    """Sorted array of the DOF indices that stay free."""
    return np.setdiff1d(np.arange(2 * n), constrained_dofs(diri_nodes,
                                                           diri_values, n))


# ---------------------------------------------------------------------------
# Neumann
# ---------------------------------------------------------------------------

def assign_neumann_bcs(nodes, neu_edges, neu_values, F, verbose=True):
    """
    Add the Neumann tractions to the load vector F and return the result (F
    itself is left untouched).

    The traction t = (t_x, t_y) acts per unit length on the edge; a constant
    traction gives t * L / 2 to each of the two nodes of the edge.

    Careful: construct_F.construct_F_vector() already includes the tractions of
    the BC file, so only call this when F was built without them, otherwise the
    boundary load is counted twice.

    :param nodes: node coordinates (list indexed by node_id - 1, or a dict)
    :param neu_edges: [(node_a, node_b), ...]
    :param neu_values: [(tx, ty), ...]
    :param F: global load vector (2n)
    :return: F with the tractions added
    """
    F_bc = np.array(F, dtype=np.float64, copy=True)
    n = F_bc.size // 2

    def coord(node_id):
        if isinstance(nodes, dict):
            return np.asarray(nodes[node_id], dtype=np.float64)
        return np.asarray(nodes[int(node_id) - 1], dtype=np.float64)

    total = np.zeros(2)
    for edge, value in zip(neu_edges, neu_values):
        a, b = int(edge[0]), int(edge[1])
        t = np.asarray(value, dtype=np.float64)
        if t.size == 1:                       # legacy scalar: t_x only
            t = np.array([float(t), 0.0])
        L = float(np.linalg.norm(coord(b) - coord(a)))
        for node_id in (a, b):
            F_bc[dof_index(node_id, 1, n)] += t[0] * L / 2.0
            F_bc[dof_index(node_id, 2, n)] += t[1] * L / 2.0
        total += t * L

    if verbose and len(neu_edges):
        print(f'[bc] Neumann: {len(neu_edges)} edge(s), resultant '
              f'({total[0]:+.6e}, {total[1]:+.6e})')
    return F_bc


# ---------------------------------------------------------------------------
# Post-processing helper
# ---------------------------------------------------------------------------

def compute_reactions(Kg, F, u, constrained):
    """
    Reaction forces at the constrained DOFs: R = Kg @ u - F.

    :param Kg: stiffness matrix BEFORE the Dirichlet elimination
    :param F: load vector BEFORE the Dirichlet elimination
    :param u: displacement vector of the solved system
    :param constrained: DOF indices fixed by the Dirichlet conditions
    :return: reaction vector (zero outside the constrained DOFs)
    """
    R = np.zeros_like(u, dtype=np.float64)
    R[constrained] = (np.asarray(Kg) @ np.asarray(u) - np.asarray(F))[constrained]
    return R


if __name__ == "__main__":
    import mesh_input

    nodes, elems = mesh_input.read_mesh('voronoi_mesh.txt')
    n = len(nodes)
    print(f'{n} nodes -> {2 * n} DOF')

    diri, neu, body = read_bc_dicts('boundary_condition.txt')
    print(f'BC file: {len(diri)} Dirichlet DOF, {len(neu)} traction edges, '
          f'{len(body)} nodes with a body force')

    diri_nodes, diri_values, neu_edges, neu_values = read_boundary_conditions()
    print(f'read_boundary_conditions -> {len(diri_nodes)} (node, comp) pairs')
    print('  first pairs:', list(zip(diri_nodes, diri_values))[:4])
    print('  first tractions:', list(zip(neu_edges, neu_values))[:2])
    print('  constrained DOF:', constrained_dofs(diri_nodes, diri_values, n)[:8])
