from functools import partial
from typing import Callable, Optional, Tuple
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import scoremd.diffusion.classic.sde as sdes
import logging

logger = logging.getLogger(__name__)


def divergence(f: Callable, n: int, gaussian: bool, mean: bool):
    """
    Compute the divergence of a vector field using JAX.
    Implementation originally from https://github.com/jax-ml/jax/issues/3022#issuecomment-2100553108
    This has been adapted to support batches and work with score functions (x, t)

    Args:
    f : Callable
        The vector field function R^n -> R^n.
    n : int
        Mode of divergence computation. -1 for exact trace, 0 for efficient exact,
        and positive integers for stochastic estimation using Hutchinson's trace estimator.
    gaussian : bool
        Flag to use Gaussian (True) or Rademacher (False) vectors for stochastic estimation.
    mean: bool
        Flag to use the mean of the stochastic estimation. This only works for n > 0.

    Returns:
    Callable
        A function that computes the divergence at a point.
    """
    if n == -1:
        logger.info("Using exact trace calculation for divergence. This might use a lot of memory.")
    if n == 0:
        logger.info("Using exact calculation for divergence. This might be very slow to compile.")

    if mean and n <= 0:
        raise ValueError("Mean estimation only works for n > 0.")

    # Exact calculation using the trace of the Jacobian
    if n == -1:

        def div(x, features, t, _):
            def f_sum(x):
                f_val = f(x, features, t)
                return jnp.sum(f(x, features, t), axis=0), f_val

            jac, f_val = jax.jacobian(f_sum, has_aux=True)(x)
            jac = jnp.swapaxes(jac, 0, 1)
            return jnp.trace(jac, axis1=-1, axis2=-2), f_val

        return div

    # Efficient exact calculation using gradients
    if n == 0:

        @jax.vmap
        def div(x, features, t, _):
            def fi(i, *x):
                stacked_x = jnp.stack(x)
                return f(stacked_x, features, t)[i]

            def dfidxi(i, x):
                return jax.grad(fi, argnums=i + 1)(i, *x)

            # I tried to vmap this but somehow it doesn't work
            return sum(dfidxi(i, x) for i in range(x.shape[0])), f(x, features, t)

        return div

    # Hutchinson's trace estimator for stochastic estimation
    if n > 0:

        @jax.vmap
        def div(x, features, t, key):
            def f_and_val(x):
                fx = f(x, features, t)
                return fx, fx

            _, vjp, fx = jax.vjp(f_and_val, x, has_aux=True)

            def vJv(key):
                v = (
                    jax.random.normal(key, x.shape, dtype=x.dtype)
                    if gaussian
                    else jax.random.rademacher(key, x.shape, dtype=x.dtype)
                )
                return jnp.dot(vjp(v)[0], v)

            ret = jax.vmap(vJv)(jax.random.split(key, n))
            if mean:
                return ret.mean(), fx

            return ret, fx

        return div


def exact_partial_t(fn, x, features, t, output_dim):
    @jax.vmap
    def batched_grad_fn(x, features, t):
        def grad_fn(x, features, t, j):
            # We add this wrapper to ensure the shape because otherwise log q_t and s_t have different shapes
            def reshaped_fn(x, features, t):
                x = x.reshape(1, -1)
                features = features.reshape(1, *features.shape) if features is not None else None
                ret = fn(x, features, t.reshape(1, 1))  # we reshape to a batch containing a single element
                return ret[0] if ret.ndim == 2 else ret

            return jax.grad(lambda t: reshaped_fn(x, features, t)[j])(t)  # Gradient wrt t for the j-th output

        return jax.vmap(lambda j: grad_fn(x, features, t, j))(
            jnp.arange(output_dim)
        )  # Apply over all output dimensions

    # Map over the entire batch of x, t inputs
    return batched_grad_fn(x, features, t).reshape((x.shape[0], output_dim))


def exact_log_q_partial_t_and_score(fn, x, features, t, output_dim):
    @jax.vmap
    def batched_grad_fn(x, features, t):
        x = x.reshape(1, -1)
        features = features.reshape(1, *features.shape) if features is not None else None

        def grad_fn(x, features, t):
            # We add this wrapper to ensure the shape because otherwise log q_t and s_t have different shapes
            def reshaped_fn(t):
                ret, aux = fn(x, features, t.reshape(1, 1))  # we reshape to a batch containing a single element
                return jnp.sum(ret), aux[0]  # ret is just a (1, 1) but for the gradient we need a scalar

            return jax.grad(reshaped_fn, has_aux=True)(t)  # Gradient wrt t for the j-th output

        return grad_fn(x, features, t)

    # Map over the entire batch of x, t inputs
    derivative, aux = batched_grad_fn(x, features, t)
    return derivative.reshape((x.shape[0], output_dim)), aux


def finite_diff_t(fn, x, features, t, hs=0.001, hd=0.0005):
    fn_eval = fn(
        jnp.concatenate([x, x, x], axis=0),
        None if features is None else jnp.concatenate([features, features, features], axis=0),
        jnp.concatenate([t + hd, t, t - hs], axis=0),
    )
    single_element = fn_eval.shape[0] // 3
    fn_p, fn_c, fn_m = (
        fn_eval[:single_element],
        fn_eval[single_element : 2 * single_element],
        fn_eval[2 * single_element :],
    )

    up = hs**2 * fn_p + (hd**2 - hs**2) * fn_c - hd**2 * fn_m
    low = hs * hd * (hd + hs)
    return up / low


def finite_diff_t_and_score(fn, fn_and_aux, x, features, t, hs=0.001, hd=0.0005):
    fn_eval = fn(
        jnp.concatenate([x, x], axis=0),
        None if features is None else jnp.concatenate([features, features], axis=0),
        jnp.concatenate([t + hd, t - hs], axis=0),
    )
    single_element = fn_eval.shape[0] // 2
    fn_p, fn_m = (
        fn_eval[:single_element],
        fn_eval[single_element : 2 * single_element],
    )
    fn_c, aux = fn_and_aux(x, features, t)

    up = hs**2 * fn_p + (hd**2 - hs**2) * fn_c - hd**2 * fn_m
    low = hs * hd * (hd + hs)
    return up / low, aux


def residual_fp_vp_error(
    key: jax.random.PRNGKey,
    sde: sdes.VP,
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    vector: bool,
    scalar: bool,
    sigma: float,
    partial_t_approx: bool,
    single_gamma: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This FP loss minimizes the residual.
    The fastest way is to use partial_t_approx and single_gamma, although it might be less accurate.
    """
    _, diffusion = sde.sde(x, features, t)
    g_pow = diffusion**2

    if vector:
        raise NotImplementedError("Vector loss not implemented for fast_fp.")

    if partial_t_approx:
        partial_t = partial(finite_diff_t_and_score, log_q_fn)
    else:
        partial_t = partial(exact_log_q_partial_t_and_score, output_dim=1)

    v = sigma * jax.random.normal(key, shape=(2, *x.shape))

    x1_p, x1_m = x + v[0], x - v[0]
    x2_p, x2_m = x + v[1], x - v[1]

    if single_gamma:
        non_positive_scores = score_fn(
            jnp.concatenate([x, x1_m, x2_m], axis=0),
            None if features is None else jnp.concatenate([features, features, features], axis=0),
            jnp.concatenate([t, t, t], axis=0),
        )

        score, score1_m, score2_m = (
            non_positive_scores[: x.shape[0]],
            non_positive_scores[x.shape[0] : 2 * x.shape[0]],
            non_positive_scores[2 * x.shape[0] :],
        )

        # we compute all the scores and log_q_t at once
        all_dlogqdts, positive_scores = partial_t(
            log_q_and_score_fn,
            jnp.concatenate([x1_p, x2_p], axis=0),
            None if features is None else jnp.concatenate([features, features], axis=0),
            jnp.concatenate([t, t], axis=0),
        )

        dlogdt1_p, dlogdt2_p = (
            all_dlogqdts[: x.shape[0]],
            all_dlogqdts[x.shape[0] : 2 * x.shape[0]],
        )

        assert dlogdt1_p.shape == dlogdt2_p.shape

        score1_p, score2_p = (
            positive_scores[: x.shape[0]],
            positive_scores[x.shape[0] : 2 * x.shape[0]],
        )
    else:
        score = score_fn(x, features, t)  # for the score we don't need the partial t

        all_dlogqdts, all_scores = partial_t(
            log_q_and_score_fn,
            jnp.concatenate([x1_p, x1_m, x2_p, x2_m], axis=0),
            None if features is None else jnp.concatenate([features, features, features, features], axis=0),
            jnp.concatenate([t, t, t, t], axis=0),
        )

        dlogdt1_p, dlogdt1_m, dlogdt2_p, dlogdt2_m = (
            all_dlogqdts[: x.shape[0]],
            all_dlogqdts[x.shape[0] : 2 * x.shape[0]],
            all_dlogqdts[2 * x.shape[0] : 3 * x.shape[0]],
            all_dlogqdts[3 * x.shape[0] :],
        )

        assert dlogdt1_p.shape == dlogdt1_m.shape == dlogdt2_p.shape == dlogdt2_m.shape

        score1_p, score1_m, score2_p, score2_m = (
            all_scores[: x.shape[0]],
            all_scores[x.shape[0] : 2 * x.shape[0]],
            all_scores[2 * x.shape[0] : 3 * x.shape[0]],
            all_scores[3 * x.shape[0] :],
        )

    assert score1_p.shape == score1_m.shape == score2_p.shape == score2_m.shape

    # for VP f(x, t) = -0.5 beta(t) * x, and hence the divergence is -0.5 beta(t)
    div_f = -1 / 2 * g_pow

    def gamma_fn(x, score, dlogqdt):
        # an additional dimension is added (BS, 1), so we need to remove it
        return (
            1 / 2 * g_pow * jnp.linalg.norm(jnp.abs(score) + 1e-7, 2, axis=1)[:, None] ** 2
            + 1 / 2 * g_pow * jnp.sum(x * score, axis=1)[:, None]
            - div_f
            - dlogqdt
        )[:, 0]

    @jax.vmap
    def compute_estimate(x_p, x_m, v, score_p, score_m, dlogqdt_p, dlogqdt_m):
        R = 1 / 2 * g_pow * (v / sigma) * (score_p - score_m) / (2 * sigma)
        R = jnp.sum(R, axis=1)

        if single_gamma:
            gamma = gamma_fn(x_p, score_p, dlogqdt_p)
        else:
            gamma = (gamma_fn(x_p, score_p, dlogqdt_p) + gamma_fn(x_m, score_m, dlogqdt_m)) / 2
        return R + gamma

    if scalar:
        R = compute_estimate(
            jnp.vstack([x1_p[None, :], x2_p[None, :]]),
            None if single_gamma else jnp.vstack([x1_m[None, :], x2_m[None, :]]),
            v,
            jnp.vstack([score1_p[None, :], score2_p[None, :]]),
            jnp.vstack([score1_m[None, :], score2_m[None, :]]),
            jnp.vstack([dlogdt1_p[None, :], dlogdt2_p[None, :]]),
            None if single_gamma else jnp.vstack([dlogdt1_m[None, :], dlogdt2_m[None, :]]),
        )
        R1, R2 = R[0], R[1]
        R = R1 * R2
    else:
        R = 0.0

    return 0.0, R, score


def fp_vp_error(
    key: jax.random.PRNGKey,
    sde: sdes.VP,
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    vector: bool,
    scalar: bool,
    div_est: int = 0,
    gaussian_div_est: bool = False,
    partial_t_approx: bool = False,
    unbiased: bool = True,
    stop_gradient: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert type(sde) is sdes.VP, "VP Loss only works with VP."
    assert vector or scalar, "At least one of vector or scalar must be True."
    assert log_q_and_score_fn is not None if scalar else True, "log_q_and_score_fn must be provided if scalar is True."
    assert log_q_fn is not None if scalar else True, "log_q_fn must be provided if scalar is True."
    if unbiased and div_est != 2 and div_est > 0:
        logger.warning("Unbiased estimation requires div_est to be 2. Setting unbiased to False.")
    unbiased = unbiased and div_est == 2

    BS, _ = x.shape

    _, diffusion = sde.sde(x, features, t)
    g_pow = diffusion**2

    def compute_RHS(x, features, key):
        def div(f, x, features, t, key):
            return divergence(f, div_est, gaussian_div_est, mean=not unbiased and div_est > 0)(
                x, features, t, jax.random.split(key, x.shape[0])
            )

        div_s, score = div(score_fn, x, features, t, key)

        if stop_gradient:
            div_s = jax.lax.stop_gradient(div_s)

        s_l22 = jnp.linalg.norm(jnp.abs(score) + 1e-7, 2, axis=1) ** 2
        f_dot_s = jnp.sum(x * score, axis=1)

        # ok, so this is a bit of a hack. If we want the unbiased estimation, we don't want div to compute the sum but need the individual estimates.
        # so we need to reshape a bit to get this working.
        if unbiased:
            div_s = jnp.swapaxes(div_s, 0, 1)  # We want the first dimension to be the different estimates
            # add dimensions to match the shape
            s_l22 = s_l22[None, ...]
            f_dot_s = f_dot_s[None, ...]

        RHS = div_s + s_l22 + f_dot_s
        if unbiased:
            RHS = RHS.reshape(div_est, BS, 1)
        else:
            RHS = RHS.reshape(BS, 1)

        return (g_pow / 2) * RHS, score

    def compute_RHS_and_sum(x, features, key):
        rhs, score = compute_RHS(x, features, key)
        return jnp.sum(rhs), (rhs, score)

    if vector and scalar:
        # this is equivalent to computing vector_RHS and scalar_RHS separately, but it's more efficient
        (_, (scalar_RHS, score)), vector_RHS = jax.value_and_grad(compute_RHS_and_sum, has_aux=True)(x, features, key)

        if unbiased:
            scalar_RHS1 = scalar_RHS[0]
            scalar_RHS2 = scalar_RHS[1]

    elif vector:
        vector_RHS, (_, score) = jax.grad(compute_RHS_and_sum, has_aux=True)(x, features, key)
    elif scalar:
        scalar_RHS, score = compute_RHS(x, features, key)
        if unbiased:
            scalar_RHS1 = scalar_RHS[0]
            scalar_RHS2 = scalar_RHS[1]

    if vector:  # FP loss defined on the score
        partial_t = finite_diff_t if partial_t_approx else partial(exact_partial_t, output_dim=x.shape[-1])

        dsdt = partial_t(score_fn, x, features, t)
        vector_error = dsdt - vector_RHS
        vector_error = jnp.sum((dsdt - vector_RHS) ** 2, axis=1)
    else:
        vector_error = 0.0

    if scalar:  # FP loss defined on log_q
        partial_t = finite_diff_t if partial_t_approx else partial(exact_partial_t, output_dim=1)

        dlogqdt = partial_t(log_q_fn, x, features, t)

        if unbiased:
            scalar_error = dlogqdt**2 - dlogqdt * (scalar_RHS1 + scalar_RHS2) + scalar_RHS1 * scalar_RHS2
        else:
            scalar_error = (dlogqdt - scalar_RHS) ** 2
    else:
        scalar_error = 0.0

    return vector_error, scalar_error, score


def fp_vp_loss(
    key: jax.random.PRNGKey,
    sde: sdes.VP,
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    alpha: float,
    beta: float,
    time_weighting: Callable[[ArrayLike], ArrayLike],
    reduce: Callable[[ArrayLike], ArrayLike] = jnp.nanmean,
    div_est: int = 0,
    gaussian_div_est: bool = False,
    partial_t_approx: bool = False,
    unbiased: bool = True,
    min_value: float = 1e-8,
    residual_fp: bool = False,
    sigma: float = 1e-4,
    single_gamma: bool = False,
    stop_gradient: bool = False,
):
    """
    Compute the FP loss for a given score model and SDE.
    """

    _, D = x.shape

    if residual_fp:
        vector_error, scalar_error, score = residual_fp_vp_error(
            key,
            sde,
            score_fn,
            log_q_fn,
            log_q_and_score_fn,
            x,
            features,
            t,
            alpha > min_value,
            beta > min_value,
            sigma,
            partial_t_approx,
            single_gamma,
        )
    else:
        vector_error, scalar_error, score = fp_vp_error(
            key,
            sde,
            score_fn,
            log_q_fn,
            log_q_and_score_fn,
            x,
            features,
            t,
            alpha > min_value,
            beta > min_value,
            div_est,
            gaussian_div_est,
            partial_t_approx,
            unbiased,
            stop_gradient,
        )

    time_weight = time_weighting(t)

    if alpha > min_value:  # FP loss defined on the score
        vector_fp = vector_error
        vector_fp *= time_weight

        vector_fp = reduce(vector_fp)
        vector_fp = vector_fp / (D**2)  # normalize by the number of dimensions
    else:
        vector_fp = 0.0

    if beta > min_value:  # FP loss defined on log_q
        scalar_fp = scalar_error
        scalar_fp *= time_weight

        scalar_fp = reduce(scalar_fp)
        scalar_fp = scalar_fp / (D**2)  # normalize by the number of dimensions
    else:
        scalar_fp = 0.0

    return alpha * vector_fp, beta * scalar_fp, score
