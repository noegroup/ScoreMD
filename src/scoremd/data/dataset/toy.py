from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable
import jax.numpy as jnp
import jax
from scoremd.data.dataset import Dataset
import logging
import matplotlib.pyplot as plt

from scoremd.data.dataset.base import Datapoints
from scoremd.utils.plots import plot_potential_1d, plot_potential_2d, plot_force_2d

log = logging.getLogger(__name__)


class ToyDatasets(Enum):
    DoubleWell = "double_well"
    DoubleWell2D = "double_well_2d"
    CheckerBoard = "checker_board"

    def range(self):
        if self is ToyDatasets.DoubleWell:
            return jnp.array([-10, 10])
        elif self is ToyDatasets.DoubleWell2D:
            return jnp.array([[-7.5, 7.5], [-7.5, 7.5]])
        elif self is ToyDatasets.CheckerBoard:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown example: {self} of type {type(self)}")


@jax.jit
def _double_well_density(x):
    return 1 / 2 * jax.scipy.stats.norm.pdf(x, -5, 1) + 1 / 2 * jax.scipy.stats.norm.pdf(x, 5, 1)


@jax.jit
def _double_well_potential(x):
    return -jnp.log(_double_well_density(x))


@jax.jit
def _double_well_2d_density(x):
    return 1 / 5 * jax.scipy.stats.multivariate_normal.pdf(
        x, jnp.array([-5, -5]), 1
    ) + 4 / 5 * jax.scipy.stats.multivariate_normal.pdf(x, jnp.array([5, 5]), 1)


@jax.jit
def _double_well_2d_potential(x):
    return -jnp.log(_double_well_2d_density(x))


@dataclass
class ToyDataset(Dataset):
    example: ToyDatasets = ToyDatasets.DoubleWell
    n_samples: int = 10_000
    seed: int = 0
    potential: Optional[Callable] = None
    density: Optional[Callable] = None

    def __init__(
        self,
        example: ToyDatasets,
        n_samples: int,
        seed: int = 0,
        name: str = "toy",
    ):
        super().__init__(
            name=name,
            sample_shape=(1, 1) if example is ToyDatasets.DoubleWell else (2, 1),
            kbT=1.0,
        )
        self.example = example
        self.n_samples = n_samples
        self.seed = seed

        if self.example is ToyDatasets.DoubleWell:
            self.potential = _double_well_potential
            self.density = _double_well_density
        elif self.example is ToyDatasets.DoubleWell2D:
            self.potential = _double_well_2d_potential
            self.density = _double_well_2d_density

    def _get_data(self) -> Tuple[Datapoints, None, None]:
        log.info(f"Generating toy dataset: {self.example} with {self.n_samples} samples.")
        n_samples = int(self.n_samples * 1.2)  # we generate more samples and use the last part as validation set

        key = jax.random.PRNGKey(self.seed)
        if self.example is ToyDatasets.DoubleWell:
            data = jax.random.normal(key, (n_samples, 1))
            data = data.at[: n_samples // 2].add(-5)
            data = data.at[n_samples // 2 :].add(+5)
        elif self.example is ToyDatasets.DoubleWell2D:
            # 1/5 * N([-5, -5], 1) + 4/5 * N([5, 5], 1)
            data = jax.random.normal(key, (n_samples, 2))
            data = data.at[: n_samples // 5].add(jnp.array([-5, -5]))
            data = data.at[n_samples // 5 :].add(jnp.array([5, 5]))
        elif self.example is ToyDatasets.CheckerBoard:
            k1, k2, k3 = jax.random.split(key, 3)
            x1 = jax.random.uniform(k1, (n_samples,)) * 4 - 2
            x2 = jax.random.uniform(k2, (n_samples,)) - 2 * jax.random.randint(k3, (n_samples,), 0, 2)
            x2 += jnp.floor(x1) % 2
            data = jnp.concatenate([x1[:, None], x2[:, None]], 1) * 2
        else:
            raise ValueError(f"Unknown example: {self.example} of type {type(self.example)}")

        return Datapoints(data[: self.n_samples], None), Datapoints(data[self.n_samples :], None), None

    def plot_potential_evaluation(
        self, potential_net: Optional[Callable[[jnp.ndarray], jnp.ndarray]]
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes] | plt.Axes:
        if self.example is ToyDatasets.DoubleWell:
            x_min, x_max = self.example.range()
            return plot_potential_1d(self.potential, potential_net, x_min, x_max)
        elif self.example is ToyDatasets.DoubleWell2D:
            return plot_potential_2d(self.potential, self.density, potential_net, *self.example.range())
        else:
            raise NotImplementedError(f"Plotting for {self.example} is not implemented.")

    def plot_force_evaluation(self, force_net: Optional[Callable[[jnp.ndarray], jnp.ndarray]]):
        forces = jax.jit(jax.grad(lambda x: -self.potential(x).sum()))
        plt.title(r"Force $-\nabla_x U(x)$")
        if self.example is ToyDatasets.DoubleWell:
            x_min, x_max = self.example.range()
            xx = jnp.linspace(x_min, x_max, 1000)
            xx = xx.reshape(-1, 1)

            ground_truth_forces = forces(xx)

            plt.plot(xx, ground_truth_forces, label="Ground Truth")
            if force_net is not None:
                plt.plot(xx, force_net(xx), label="Model")

            plt.xlabel(r"$x$")
            plt.legend()
        elif self.example is ToyDatasets.DoubleWell2D:
            plot_force_2d(force_net, forces, *self.example.range())
        else:
            raise NotImplementedError(f"Plotting for {self.example} is not implemented.")
