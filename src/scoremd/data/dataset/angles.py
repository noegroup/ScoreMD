import jax
import jax.numpy as jnp


@jax.jit
@jax.vmap
def dihedral(p: jnp.ndarray) -> jnp.ndarray:
    """The code is taken and adapted from: http://stackoverflow.com/q/20305272/1128289

    Args:
        p: A set of points in the form of a jax numpy array with shape (batch, 4, 3).
    Returns:
        The dihedral angles in radians.
    """
    b = p[:-1] - p[1:]
    b = b.at[0].set(-b[0])
    v = jnp.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= jnp.sqrt(jnp.einsum("...i,...i", v, v)).reshape(-1, 1)
    b1 = b[1] / jnp.linalg.norm(b[1])
    x = jnp.dot(v[0], v[1])
    m = jnp.cross(v[0], b1)
    y = jnp.dot(m, v[1])
    return jnp.arctan2(y, x)
