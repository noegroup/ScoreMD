import jax.numpy as jnp
from typing import Any, Callable, Optional
from tqdm import tqdm


def empty_if_none(x: Optional[Any], string: str) -> str:
    if x is None:
        return ""
    else:
        return string


def once_on_every_and_all(trajectories: jnp.ndarray, f: Callable[[jnp.ndarray, Optional[int]], None], desc=None):
    """This is a helper to apply a function to every trajectory and to the entire batch of trajectories.
    It is useful for evaluating the performance of a model on a single trajectory and on the entire batch of trajectories.
    For instance, if we simulate 100 trajectories in parallel, we want to plot the ramachandran plot of every trajectory and the entire batch.
    """
    num = trajectories.shape[0] + 1

    with tqdm(total=num, desc=desc) as pbar:
        for i, trajectory in enumerate(trajectories):
            f(trajectory, i)
            pbar.update(1)
        f(trajectories.reshape(-1, *trajectories.shape[2:]), None)
        pbar.update(1)
