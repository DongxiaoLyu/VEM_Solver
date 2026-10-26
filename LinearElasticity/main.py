import mesh_input
import global_K
import construct_F
import bc
import solve
import post
import visual

# read mesh
nodes, elems = mesh_input.read_mesh('voronoi_mesh.txt')
# caculate global stiffness matrix
global_K_matrix = global_K.compute_global_K(nodes, elems)
# caculate F
F = construct_F.construct_F_vector(nodes, elems)
# read boundary conditions
diri_nodes, diri_values, neu_edges, neu_values = bc.read_boundary_conditions()
# assign boundary conditions (nodes= lets bc.py check the rigid-body modes)
K_bc, F_bc = bc.assign_dirichlet_bcs(diri_nodes, diri_values, global_K_matrix, F,
                                     nodes=nodes)
# solve
u = solve.solve_linear_system(K_bc, F_bc).reshape(2, len(nodes))
# visulization
visual.plot_solution(nodes, elems, u, title='VEM Solution', save_path='solution_plot.png')
visual.plot_deformation(nodes, elems, u)