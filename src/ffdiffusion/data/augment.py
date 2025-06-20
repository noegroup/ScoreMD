from typing import Optional
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def batch_random_rotation_matrices(key: ArrayLike, batch_size: int) -> ArrayLike:
    """Generate a batch of uniform random 3D rotation matrices."""
    subkeys = jax.random.split(key, batch_size)  # Create a key for each molecule

    def random_rotation_matrix(subkey):
        """Generate a single uniform random 3D rotation matrix using quaternions."""
        q = jax.random.normal(subkey, (4,))
        q /= jnp.linalg.norm(q)  # Normalize to unit quaternion

        w, x, y, z = q

        # equivalent to scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()
        return jnp.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
            ]
        )

    return jax.vmap(random_rotation_matrix)(subkeys)  # (BS, 3, 3)


def apply_random_rotations(batch: ArrayLike, features: Optional[ArrayLike], key: ArrayLike) -> ArrayLike:
    """
    Apply different random rotations to each molecule in the batch.
    batch: (BS, num_atoms, 3) - A batch of molecules with num_atoms atoms each in 3D. Can also be shape (BS, num_atoms * 3).
    key: JAX PRNG key.
    """
    orig_shape = batch.shape
    batch = batch.reshape(batch.shape[0], -1, 3)  # (BS, num_atoms, 3)
    R_batch = batch_random_rotation_matrices(key, batch.shape[0])

    # we center the molecule to the origin before applying the rotation, and then move it back
    batch_offset = jnp.mean(batch, axis=1, keepdims=True)
    batch_centered = batch - batch_offset

    return (jnp.einsum("bij,bnj->bni", R_batch, batch_centered) + batch_offset).reshape(
        orig_shape
    )  # Efficient batch matrix multiplication
