from functools import partial
import jax.numpy as jnp
import jax

"""
From https://hunterheidenreich.com/posts/kabsch_algorithm/ and adapted
"""


@partial(jax.jit, static_argnums=(2,))
def kabsch_align(P, Q, return_rotation: bool = False):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points or a flattened array.
    :param Q: A Nx3 matrix of points or a flattened array.
    :param return_rotation: If True, also return the rotation matrix.
    :return: Return aligned P and Q. P will be aligned to Q, and Q will be shifted to the origin (hence we return both).
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    orig_dim = P.shape

    P = P.reshape(-1, 3)
    Q = Q.reshape(-1, 3)

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = jnp.dot(p.T, q)

    # SVD
    U, _, Vt = jnp.linalg.svd(H)

    # Validate right-handed coordinate system
    det = jnp.linalg.det(jnp.dot(Vt.T, U.T))
    Vt = jnp.where(det < 0.0, Vt.at[-1, :].set(Vt[-1, :] * -1.0), Vt)

    # Optimal rotation
    R = jnp.dot(Vt.T, U.T)

    p_aligned = jnp.dot(p, R.T).reshape(orig_dim)
    q_aligned = q.reshape(orig_dim)

    if return_rotation:
        return p_aligned, q_aligned, R

    return p_aligned, q_aligned


def kabsch_align_many(Ps, Q, return_rotation: bool = False):
    """
    Kabsch aligns a set of points to a reference set of points.
    Args:
        Ps: An array of shape (N, M * 3) or (N, M, 3) where N is the number of points and M is the number of sets of points.
        Q:  An array of shape (M * 3) or (M, 3) representing the reference set of points.
        return_rotation: If True, also return the rotation matrix.

    Returns:
        An array of shape (N, M*3) representing the aligned points and the aligned Q.
    """

    @jax.vmap
    def kabsch_aligned(p):
        return kabsch_align(p, Q, return_rotation=True)

    aligned_ps, aligned_qs, rotations = kabsch_aligned(Ps)
    if return_rotation:
        return aligned_ps, aligned_qs[0], rotations
    return aligned_ps, aligned_qs[0]


def kabsch_rmsd(P, Q):
    P_aligned, Q_aligned, _ = kabsch_align(P, Q)
    return jnp.sqrt(jnp.sum(jnp.square(P_aligned - Q_aligned)) / P.shape[0])
