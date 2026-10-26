import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

def plot_solution(nodes, elems, u, title='Solution', save_path='solution_plot.png'):
    """
    Plot the solution u over the mesh defined by nodes and elems
    :param nodes: list of node coordinates
    :param elems: list of elements (polygons) defined by node indices
    :param u: (k, n) array -- k components (rows), n nodes (columns)
    """

    def plot_scalar_field(values, ax, label, cmap="viridis", show_mesh=True):
            """
            Draw the scalar field `values` over the mesh on the given axes
            :param values: scalar value at each node (length n)
            :param ax: axes to draw on
            :param label: colorbar label / panel title
            """
            u_elem = np.array([np.mean(values[elem_ids]) for elem_ids in elems])

            patches = []
            for elem_ids in elems:
                elem_ids = np.asarray(elem_ids, dtype=int)
                poly = nodes[elem_ids]
                patches.append(MplPolygon(poly, closed=True))

            pc = PatchCollection(
                patches,
                cmap=cmap,
                edgecolor="k" if show_mesh else "none",
                linewidth=0.4,
            )
            pc.set_array(u_elem)
            ax.add_collection(pc)

            ax.figure.colorbar(pc, ax=ax, label=label)

            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_title(label)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(u, dtype=float)

    k = u.shape[0]
    components = [u[i] for i in range(k)]
    labels = [f'component {i + 1}' for i in range(k)]

    if k == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_scalar_field(components[0], ax, labels[0])
        fig.suptitle(f'{title}')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig, ax

    ncols = min(3, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, label, values in zip(axes, labels, components):
        plot_scalar_field(values, ax, label)
    for ax in axes[k:]:                       # unused panels of the grid
        ax.axis('off')

    fig.suptitle(f'{title}')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, axes[:k]

def plot_deformation(nodes, elems, u, title='Deformation',
                     save_path='deformation_plot.png', scale=None):
    """
    Plot the deformed mesh for a displacement u
    :param nodes: list of node coordinates
    :param elems: list of elements (polygons) defined by node indices
    :param u: (2, n) displacement matrix -- row 0: u_x, row 1: u_y
    :param scale: magnification of the displacement; None -> the largest |u|
                  is drawn as 10 % of the model size
    """
    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(u, dtype=float)
    if u.shape[0] != 2:
        raise ValueError(f'u must be (2, n) = [u_x; u_y], got {u.shape}')

    deformed = u.T                          # (n, 2)
    mag = np.linalg.norm(deformed, axis=1)

    if scale is None:
        span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]))
        scale = 0.1 * span / mag.max() if mag.max() > 0 else 1.0

    coords = nodes + scale * deformed

    fig, ax = plt.subplots(figsize=(8, 6))

    # undeformed mesh, for reference
    for elem_ids in elems:
        elem_ids = np.asarray(elem_ids, dtype=int)
        ax.add_patch(MplPolygon(nodes[elem_ids], closed=True, fill=False,
                                edgecolor="0.75", linewidth=0.8))

    # deformed mesh, coloured by |u|
    patches = [MplPolygon(coords[np.asarray(elem_ids, dtype=int)], closed=True)
               for elem_ids in elems]
    pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.4)
    pc.set_array(np.array([np.mean(mag[elem_ids]) for elem_ids in elems]))
    ax.add_collection(pc)

    fig.colorbar(pc, ax=ax, label="|u|")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f'{title}  (displacement magnified x{scale:.3g})')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax, scale