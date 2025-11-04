import os.path
from dataclasses import dataclass, field
from typing import Tuple
from deeptime.util import energy2d
import jax.numpy as jnp
import jax
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from scoremd.data.dataset.base import Datapoints
from scoremd.utils.file import get_persistent_storage
import hashlib

from scoremd.utils.plots import rasterize_contour
from . import Dataset
from ...simulation import create_langevin_step_function, simulate

log = logging.getLogger(__name__)


def mueller_brown_potential(xs: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Compute the energy of the mueller brown potential.
    """
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)

    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * jnp.exp(-((x - 1) ** 2) - 10 * y**2)
    e2 = -100 * jnp.exp(-(x**2) - 10 * (y - 0.5) ** 2)
    e3 = -170 * jnp.exp(-6.5 * (0.5 + x) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
    e4 = 15.0 * jnp.exp(0.7 * (1 + x) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
    return beta * (e1 + e2 + e3 + e4)


@dataclass
class MuellerBrownSimulation(Dataset):
    n_samples: int = 10_000
    n_steps: int = 50
    mass: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 1.0]))
    gamma: float = 1.0
    dt: float = 1e-4
    beta: float = 1.0
    seed: int = 0

    def __init__(
        self,
        n_samples: int = 100_000,
        n_steps: int = 50,
        kbT: float = 23.0,
        mass: jnp.ndarray = jnp.array([1.0, 1.0]),
        gamma: float = 1.0,
        dt: float = 1e-4,
        beta: float = 1.0,
        seed: int = 0,
        name="mueller_brown",
    ):
        super().__init__(
            name=name,
            sample_shape=(2, 1),
            kbT=kbT,
        )
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.mass = jnp.array(mass)
        self.gamma = gamma
        self.dt = dt
        self.beta = beta
        self.seed = seed

    def range(self) -> jnp.ndarray:
        return jnp.array([[-1.8, 1.1], [-0.5, 2.0]])

    def potential(self, xs: jnp.ndarray) -> jnp.ndarray:
        return mueller_brown_potential(xs, self.beta)

    def likelihood(self, xs: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.potential(xs) / self.kbT)

    def force(self, xs: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda _x: -self.potential(_x).sum())(xs)

    def _get_data(self) -> Tuple[Datapoints, None, None]:
        key = jax.random.PRNGKey(self.seed)

        dir = os.path.join(get_persistent_storage(), "MuellerBrown")
        os.makedirs(dir, exist_ok=True)
        sha256 = hashlib.sha256()
        sha256.update(repr(self).encode("utf-8"))

        file_name = f"{sha256.hexdigest()}.npy"
        data = None
        if os.path.exists(os.path.join(dir, file_name)):
            # try loading file
            try:
                data = jnp.load(os.path.join(dir, file_name))
                log.info(f"Loaded data from {os.path.join(dir, file_name)}")
            except Exception as e:
                log.warning(f"Failed to load data: {e}")

        if data is None:
            log.info(f"Generating data for {self.name} dataset.")
            data = self._generate_data(key)
            jnp.save(os.path.join(dir, file_name), data)

        return Datapoints(data, None), None, None

    def _generate_data(self, key):
        key, velocity_key = jax.random.split(key)

        starting_point = jnp.array([-0.55828035, 1.44169])
        # Sample the starting velocity from the Boltzmann distribution
        starting_velocity = jnp.sqrt(self.kbT / self.mass) * jax.random.normal(velocity_key, (2,))

        step = jax.jit(
            create_langevin_step_function(self.force, self.mass, self.gamma, self.n_steps, self.dt, self.kbT)
        )
        trajectory, _ = simulate(starting_point, starting_velocity, step, self.n_samples, key)
        return trajectory

    def plot(self, samples: jnp.ndarray, cbar_range: Tuple[float, float] = None, cbar: bool = True):
        assert samples.ndim == 2 and samples.shape[1] == 2, "Data should be a 2D vector."

        bins = 150
        (x_min, x_max), (y_min, y_max) = self.range()
        x, y = jnp.linspace(x_min, x_max, bins), jnp.linspace(y_min, y_max, bins)
        x, y = jnp.meshgrid(x, y, indexing="ij")
        z = self.potential(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])

        plt.contour(x, y, z, levels=[-120, -90, -50, -20, 0, 20, 35, 70, 150, 250, 500, 1000], colors="black")

        energies = energy2d(*samples.T, bins=(100, 100), kbt=self.kbT, shift_energy=True)
        contourf_kws = {"cmap": "turbo"}
        if cbar_range is not None:
            contourf_kws["vmin"] = cbar_range[0]
            contourf_kws["vmax"] = cbar_range[1]
        ax, contour, cbar_obj = energies.plot(contourf_kws=contourf_kws, cbar=cbar_range is None and cbar)

        rasterize_contour(contour)

        if cbar:
            if cbar_range is not None:
                cbar_obj = ax.figure.colorbar(
                    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(cbar_range[0], cbar_range[1])), ax=ax
                )

            cbar_obj.set_label(r"Energy / $k_BT$")

        ax.set_xticks([])
        ax.set_yticks([])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        return energies
