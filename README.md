# VEM — Virtual Element Method solvers in Python

Two independent VEM solvers on unstructured polygonal (Voronoi) meshes: a scalar
Poisson solver and a 2-D linear elasticity solver, both written from scratch in
plain Python + NumPy.

The code implements the conforming $H^1$ virtual element space without ever
writing down the basis functions inside an element. Only the degrees of freedom
on the element boundary are used: a **Ritz projection** is built from the edge
data, and the element stiffness matrix is assembled as a *consistency* part plus
a *stability* part. The mesh is polygonal (Voronoi), so no triangle or quad
re-meshing is required.

- `Poisson/` — scalar Poisson equation, 4–8 vertices per element
- `LinearElasticity/` — 2-D linear elasticity, 2 DOF per node ($u_x$, $u_y$)

---

## Repository layout

```
VEM_Solver/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Poisson/                    # scalar Poisson solver
│   ├── main.py                 # pipeline: read mesh -> assemble -> apply BC -> solve -> plot
│   ├── mesh_input.py           # polygonal mesh reader
│   ├── local_K.py              # element stiffness matrix (Ritz projection + consistency + stability)
│   ├── global_K.py             # global stiffness assembly
│   ├── construct_F.py          # load vector (integral of f over each element)
│   ├── bc.py                   # Dirichlet / Neumann boundary conditions
│   ├── solve.py                # linear solver
│   ├── post.py                 # post-processing (gradient recovery, TODO)
│   ├── visual.py               # visualisation
│   ├── voronoi_mesh.txt        # input mesh
│   ├── boundary_condition.txt  # input boundary conditions
│   └── solution_plot.png       # result (reproducible by running main.py)
└── LinearElasticity/           # 2-D linear elasticity solver
    ├── main.py
    ├── mesh_input.py
    ├── local_K.py              # elasticity VEM element matrix (with mean-rotation constraint)
    ├── global_K.py             # global matrix, u_x and u_y blocks
    ├── construct_F.py          # nodal body force + Neumann tractions
    ├── bc.py                   # per-component Dirichlet, rigid-body check, reactions
    ├── solve.py
    ├── post.py                 # empty file (TODO)
    ├── visual.py               # solution field + deformed mesh + BC sketch
    ├── voronoi_mesh.txt
    ├── boundary_condition.txt
    ├── boundary_conditions.png
    ├── solution_plot.png
    └── deformation_plot.png
```

Both solvers use **flat modules**: `main.py` does `import mesh_input` from the
same directory, so the scripts must be run from inside their own folder.

---

## Requirements

- Python ≥ 3.9
- `numpy`, `matplotlib`

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts were verified on Python 3.11.17 with numpy 2.4.6 and
matplotlib 3.11.2, and the PNG files they produce are byte-identical to the
figures committed in this repository.

---

## Quick start

```bash
git clone https://github.com/DongxiaoLyu/VEM_Solver.git
cd VEM_Solver

# Poisson equation
cd Poisson
python main.py                     # writes solution_plot.png

# 2-D linear elasticity
cd LinearElasticity
python main.py                     # writes solution_plot.png, deformation_plot.png
```

Each module also has a small self-check entry point that prints intermediate
quantities (useful for debugging):

```bash
cd Poisson          && python mesh_input.py      # print the nodes and elements read
cd Poisson          && python local_K.py         # print Ke of a unit square
cd LinearElasticity && python bc.py              # print boundary condition statistics
cd LinearElasticity && python local_K.py         # print Ke
```

---

## Results

### Poisson equation

Unit source term $f=1$, $u=10$ on the left edge, normal flux $q=10$ on the right
edge: the solution rises monotonically from left to right, $u \in [10.0,\ 41.5]$.

![Poisson solution](Poisson/solution_plot.png)

### 2-D linear elasticity

Left edge clamped ($u_x=u_y=0$), downward uniform traction $t=(0,-10)$ on the
right edge. The material parameters ($E=1$, $\nu=0.3$, plane stress by default)
are set inside `local_K.py`, so the **absolute displacement is only meaningful in
relative terms**; what matters is the deformation pattern and the magnitude
ratios.

![Elasticity solution](LinearElasticity/solution_plot.png)

![Deformation](LinearElasticity/deformation_plot.png)

Boundary conditions (nodes, fixed components, traction direction):

![Boundary conditions](LinearElasticity/boundary_conditions.png)

---

## Numerical method

### Common part: the VEM element matrix

For a polygonal element $E$ with $n$ vertices, area $A_E$ and diameter $h_E$,
the lowest-order virtual element space is used:

1. **Degrees of freedom**: vertex values only.
2. **Polynomial space**: $P_1$ (3-D) for Poisson; $[P_1]^2$ (6-D) for elasticity.
3. **Ritz projection** $\Pi^\nabla$: built from the $G$ and $B$ matrices with
   $G_{ij}=(\nabla m_i,\nabla m_j)$ and $B_{ij}$ the boundary integral terms,
   solving $\Pi^{\nabla} = D\,G^{-1}B$ (the constant mode of $G$ is modified so
   that the projection is well defined on the constant polynomial).
4. **Element stiffness**:

   $$
   K_E = \underbrace{(\Pi^\nabla)^T G\, \Pi^\nabla}_{\text{consistency}} + \underbrace{\alpha\,(I-\Pi^\nabla)^T(I-\Pi^\nabla)}_{\text{stability}}
   $$

   Poisson takes the stability factor $\alpha = 1$; the elasticity problem takes
   $\alpha = 0.1$ (see `K_stability` in the respective `local_K.py`).

The implementation lives in `Poisson/local_K.py` and
`LinearElasticity/local_K.py`. The two share the same structure; the elasticity
version additionally has a plane-stress constitutive matrix, a $6\times6$ $G$ and
a mean-rotation constraint (the form of Eq. 3.75–3.79 in the reference).

> Note: when `area_dirct < 0` (vertices listed clockwise) the edge normals are
> flipped automatically so that they always point outward.

### Poisson

- Load: $f=1$, distributed equally over the vertices of each element
  (`construct_F.py`).
- Dirichlet: imposed by elimination, which keeps the matrix symmetric
  (`bc.assign_dirichlet_bcs`).
- Neumann: `bc.assign_neumann_bcs` distributes $tL/2$ to the two nodes of each
  boundary edge.

### 2-D linear elasticity

- DOF numbering: $u_x$ of node $i$ maps to global index $i-1$; $u_y$ maps to
  $i-1+n$.
- Load: a lumped nodal body force plus constant tractions on boundary edges
  (trapezoidal rule, exact for linear virtual functions).
- Dirichlet: applied **per component** (1 = $u_x$, 2 = $u_y$), which allows
  rollers and symmetry planes.
- `bc.rigid_body_check`: verifies that the constraints remove all three
  rigid-body modes ($x$-translation, $y$-translation, rotation); otherwise the
  stiffness matrix is singular.
- `bc.compute_reactions`: reaction forces from $R = K u - F$.

---

## Input file formats

### Mesh `voronoi_mesh.txt`

```
Nodes
1 3.7671210800 2.8056596700          # <node id> <x> <y>, node ids start at 1
...

Elements
1 1 2 3 4 5 6                        # <element id> <vertex 1> <vertex 2> ..., vertex ids start at 1
...
```

The current test case is the rectangle $[0,8]\times[0,4]$ with 140 nodes and
69 elements, each having 4–8 vertices.

### Boundary conditions

**Poisson** (`boundary_condition.txt`):

```
Dirichlet BCs #Node id #Dirichlet Value
129   10.0

Neumann BCs #Node id1 #Node id2 #Neumann Value
96   97   10.0
```

**Linear elasticity** (`boundary_condition.txt`):

```
Dirichlet BCs #Node id #Component (1=ux, 2=uy) #Value
129   1   0

Neumann BCs #Node id1 #Node id2 #t_x #t_y
96   97   0   -10

Body force #Node id #f_x #f_y
1    0.0   0.0
```

Everything after `#` is a comment, and the section headers (`Dirichlet`,
`Neumann`, `Body`) are matched case-insensitively by prefix.

---

## Known limitations / TODO

- `compute_gradient` in `Poisson/post.py` is a placeholder that returns an empty
  array: gradient recovery (the $L^2$ projection of $\nabla u$) is not
  implemented. `LinearElasticity/post.py` is an empty file, so stress recovery is
  still missing.
- Poisson's `main.py` only calls `assign_dirichlet_bcs`; it does **not** apply the
  Neumann conditions automatically. Call
  `bc.assign_neumann_bcs(nodes, neu_edges, neu_values, F)` manually when needed.
- The material parameters $E$ and $\nu$ are hard-coded in `constitutive_matrix`
  inside `local_K.py`; there is no CLI or config file for them yet.
- The comments in `bc.py` and `construct_F.py` mention `Tools/generate_bc_file.py`
  (the boundary condition generator); that script is **not part of this
  repository**.
- Assembly uses dense matrices and Python-level double loops, so it is intended
  for teaching and small-to-medium problems ($10^2$–$10^3$ unknowns) only: no
  sparse storage and no vectorisation.
- There are no automated tests yet, and no convergence study script (for example
  $L^2$ / $H^1$ error orders under mesh refinement).

---

## License

MIT License, see [LICENSE](LICENSE).
