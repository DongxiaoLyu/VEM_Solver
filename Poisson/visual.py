import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

def plot_solution(nodes, elems, u, title='Solution', save_path='solution_plot.png'):
    """
    Plot the solution u over the mesh defined by nodes and elems
    :param nodes: list of node coordinates
    :param elems: list of elements (triangles) defined by node indices
    :param u: solution vector at the nodes
    """

    def plot_scalar_field(nodes, elems, u, cmap="viridis",
                          show_mesh=True, save = True):
            """
            Plot the scalar field u over the mesh
            :param nodes: list of node coordinates
            :param triangles: list of triangles defined by node indices
            :param u: solution vector at the nodes
            """
            u_elem = np.array([np.mean(u[elem_ids]) for elem_ids in elems])

            patches = []
            for elem_ids in elems:
                elem_ids = np.asarray(elem_ids, dtype=int)
                poly = nodes[elem_ids]
                patches.append(MplPolygon(poly, closed=True))

            fig, ax = plt.subplots(figsize=(8, 6))
            pc = PatchCollection(
                patches,
                cmap=cmap,
                edgecolor="k" if show_mesh else "none",
                linewidth=0.4,
            )
            pc.set_array(u_elem)
            ax.add_collection(pc)

            fig.colorbar(pc, ax=ax, label=title)

            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            if save:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()
            
    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        plot_scalar_field(nodes, elems, u)
    else:
        # vector field, plot each component separately
        pass
    