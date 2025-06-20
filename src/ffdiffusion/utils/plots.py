from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0, title=True, xlabel=True, ylabel=True, xticks=True, yticks=True):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    https://stackoverflow.com/a/26432947/4417954
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = [ax]
    if title:
        items += [ax.title]
    if xlabel:
        items += [ax.xaxis.label]
    if ylabel:
        items += [ax.yaxis.label]
    if xticks:
        items += ax.get_xticklabels()
    if yticks:
        items += ax.get_yticklabels()
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)


def save_parts_of_figure(ax, filename, title=True, xlabel=True, ylabel=True, xticks=True, yticks=True, **kwargs):
    extent = full_extent(ax, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks).transformed(
        plt.gcf().dpi_scale_trans.inverted()
    )
    plt.savefig(filename, bbox_inches=extent, **kwargs)


def rasterize_contour(contour):
    """Rasterize a contour plot to avoid jagged edges."""
    for c in contour.collections:
        c.set_rasterized(True)


def phi_psi(phi, psi, title="Histogram", highlight=None, bins=60, vmin=None, vmax=None, **kwargs):
    plot_range = [-jnp.pi, jnp.pi]

    if title is not None:
        plt.title(title)
    plt.hist2d(
        phi,
        psi,
        bins=bins,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        range=[plot_range, plot_range],
        rasterized=True,
        **kwargs,
    )
    plt.xticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    plt.yticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\psi$")
    plt.gca().set_box_aspect(1)

    if highlight is not None:
        # plot a star for each highlight
        for h in highlight:
            plt.plot(phi[h], psi[h], "*", markersize=8, alpha=0.8)

    return phi, psi


def plot_fes(
    samples: jnp.ndarray,
    kBT: float,
    grid: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    bw_method: float = 0.05,
    linewidth: float = 3,
    *args,
    **kwargs,
):
    if grid is None:
        grid = jnp.linspace(samples.min(), samples.max(), 100)
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()

    plt.plot(grid, fes, linewidth=linewidth, *args, **kwargs)
    plt.xlim(grid.min(), grid.max())
    plt.ylabel(r"Energy / $k_BT$")

    return grid, fes


def plot_fes_angles(
    angles: jnp.ndarray, kBT: float, weights: Optional[jnp.ndarray] = None, bw_method: float = 0.05, *args, **kwargs
):
    grid = jnp.linspace(-jnp.pi, jnp.pi, 100)
    # we add the periodic boundary conditions so that the plot is continuous
    extended_angles = jnp.concatenate([angles, angles + 2 * jnp.pi, angles - 2 * jnp.pi])
    # extended_angles = angles
    grid, fes = plot_fes(extended_angles, kBT, grid, weights, bw_method, *args, **kwargs)
    plt.xticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    return grid, fes


def plot_potential_1d(potential, potential_net, x_min, x_max):
    xx = jnp.linspace(x_min, x_max, 1000)
    xx = xx.reshape(-1, 1)
    ground_truth_potential = potential(xx)
    plt.title("Potential")
    ax = plt.plot(xx, ground_truth_potential, label="Ground Truth")
    if potential_net is not None:
        model_potential = potential_net(xx)
        plt.plot(xx, model_potential - model_potential.min() + ground_truth_potential.min(), label="Model")
    plt.xlabel(r"$x$")
    plt.legend()

    return ax


def plot_potential_2d(potential, density, potential_net, xlim, ylim):
    n = 100j

    (x_min, x_max), (y_min, y_max) = xlim, ylim
    xx, yy = jnp.mgrid[x_min:x_max:n, y_min:y_max:n]
    positions = jnp.vstack([xx.ravel(), yy.ravel()]).T

    ground_truth_density = density(positions)
    ground_truth_potential = potential(positions)

    # Ground Truth
    ax0 = plt.subplot(2, 2, 1)
    plt.title(r"Density$\propto\exp (-U(x))$ (Truth)")
    contour = plt.contourf(xx, yy, ground_truth_density.reshape(xx.shape), 100)
    rasterize_contour(contour)
    plt.colorbar()

    ax1 = plt.subplot(2, 2, 3 if potential_net is not None else 2)
    plt.title(r"Potential $\propto U(x)$ (Truth)")
    contour = plt.contourf(xx, yy, ground_truth_potential.reshape(xx.shape), 100)
    rasterize_contour(contour)
    plt.colorbar()

    if potential_net is not None:
        model_potential = potential_net(positions)
        model_potential = model_potential - model_potential.min() + ground_truth_potential.min()
        model_density = jnp.exp(-model_potential)

        # Model
        ax3 = plt.subplot(2, 2, 2)
        plt.title(r"Density$\propto\exp (-U(x))$ (Model)")
        contour = plt.contourf(xx, yy, model_density.reshape(xx.shape), 100)
        rasterize_contour(contour)
        plt.colorbar()

        ax4 = plt.subplot(2, 2, 4)
        plt.title(r"Potential $\propto U(x)$ (Model)")
        contour = plt.contourf(xx, yy, model_potential.reshape(xx.shape), 100)
        rasterize_contour(contour)
        plt.colorbar()

    return ax0, ax1, ax3, ax4


def plot_force_2d(force_net, forces, xlim, ylim, equal_scaling=True, n=100j, every=5):
    """
    Args:
        equal_scaling (bool, optional): If we use equal scaling, the arrows will show comparable lengths. Defaults to True.

    Returns:
        _type_: _description_
    """
    (x_min, x_max), (y_min, y_max) = xlim, ylim

    def get_scale(xx, yy, uu, vv):
        """Get autoscale value by plotting to do-nothing backend."""
        backend = plt.matplotlib.get_backend()
        plt.matplotlib.use("template")
        Q = plt.quiver(xx, yy, uu, vv, scale=None)
        plt.matplotlib.use(backend)
        Q._init()
        return Q.scale

    xx, yy = jnp.mgrid[x_min:x_max:n, y_min:y_max:n]
    positions = jnp.vstack([xx.ravel(), yy.ravel()]).T
    u, v = forces(positions).T

    if equal_scaling:
        scale = get_scale(
            xx[::every, ::every],
            yy[::every, ::every],
            u.reshape(xx.shape)[::every, ::every],
            v.reshape(yy.shape)[::every, ::every],
        )
    else:
        scale = None

    plt.quiver(
        xx[::every, ::every],
        yy[::every, ::every],
        u.reshape(xx.shape)[::every, ::every],
        v.reshape(yy.shape)[::every, ::every],
        label="Ground Truth",
        color="blue",
        scale=scale,
    )

    u, v = force_net(positions).T
    plt.quiver(
        xx[::every, ::every],
        yy[::every, ::every],
        u.reshape(xx.shape)[::every, ::every],
        v.reshape(yy.shape)[::every, ::every],
        label="Model",
        color="orange",
        scale=scale,
    )
    plt.legend()
