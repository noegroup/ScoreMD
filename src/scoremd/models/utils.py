import jax
from typing import Callable, Tuple
import jax.numpy as jnp


def value_and_grad_sum(fn: Callable, x: jnp.ndarray, *args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the gradient of the function wrt. x and returns the gradient and the value (without the sum).
    The arguments and keyword arguments are passed to the function.
    """

    def sum_and_value(x):
        val = fn(x, *args, **kwargs)
        return val.sum(), val

    grad, value = jax.grad(sum_and_value, has_aux=True)(x)
    return value, grad
