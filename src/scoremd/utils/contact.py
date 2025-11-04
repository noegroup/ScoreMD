import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def compute_contact_map(samples, threshold=1):  # default threshold is 1 nm = 10 angstroms
    positions = samples.reshape(samples.shape[0], -1, 3)
    distance_matrix = jnp.linalg.norm(positions[:, None, :, :] - positions[:, :, None, :], axis=-1)
    return jnp.log(jnp.mean(distance_matrix < threshold, axis=0) + 0.0001)
