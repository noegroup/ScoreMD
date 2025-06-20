from typing import Sequence, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ffdiffusion.models import RangedModel


def construct_global_constant_weighting_function(
    ranged_models: Sequence[RangedModel],
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Each model has equal range across all t's. Weights are constant over time. This basically discards the range."""

    def weight(_x, t):
        return jnp.ones((len(ranged_models), t.shape[0])) / len(ranged_models)

    return weight


def construct_ranged_constant_weighting_function(
    ranged_models: Sequence[RangedModel], normalize: bool = True
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Each model has the weight 1 in its range, 0 otherwise. In case two models overlap, they share the weight and sum up to 1."""

    def weight(_x, t):
        out = jnp.zeros((len(ranged_models), t.shape[0]))
        for i, model in enumerate(ranged_models):
            t1, t0 = model.range
            out = out.at[i].set(jnp.where((t >= t0) & (t <= t1), 1.0, 0.0).squeeze())
        if normalize:
            return out / jnp.sum(out, axis=0)
        else:
            return out

    return weight


def plot_weighting_function(
    weighting_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
):
    t = jnp.linspace(t0, t1, 10000).reshape(-1, 1)
    weights = weighting_function(x, t)

    for i in range(weights.shape[0]):
        mask = weights[i] > 0
        plt.plot(t[mask], weights[i][mask], label=f"Model {i}")

    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend()
