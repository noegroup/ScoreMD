"""Utility functions, including all functions related to loss computation, optimization and sampling.
Taken from https://github.com/bb515/diffusionjax/blob/24fb2c8ee0ca85f618caf797c24614d2ee686be4/diffusionjax/utils.py
"""

from typing import Callable, Optional, Sequence, Any
import jax
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
import logging
from jax.typing import ArrayLike
from flax.core import FrozenDict
from scoremd.utils.diffusion import batch_mul, get_score
import scoremd.diffusion.classic.sde as sdes

log = logging.getLogger(__name__)


def get_exponential_sigma_function(sigma_min, sigma_max):
    log_sigma_min = jnp.log(sigma_min)
    log_sigma_max = jnp.log(sigma_max)

    def sigma(t):
        # return sigma_min * (sigma_max / sigma_min)**t  # Has large relative error close to zero compared to alternative, below
        return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))

    return sigma


def get_linear_beta_function(beta_min, beta_max):
    """Returns:
    Linear beta (cooling rate parameter) as a function of time,
    It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
    """

    def beta(t):
        return beta_min + t * (beta_max - beta_min)

    def log_mean_coeff(t):
        """..math:: -0.5 * \\int_{0}^{t} \beta(s) ds"""
        return -0.5 * t * beta_min - 0.25 * t**2 * (beta_max - beta_min)

    return beta, log_mean_coeff


def perturb(t, sde, rng, data):
    mean, std = sde.marginal_prob(data, t)

    rng, step_rng = jax.random.split(rng)
    noise = jax.random.normal(step_rng, data.shape)

    return mean + batch_mul(std, noise), noise, std  # noisy sample


def errors(noise, score, std, likelihood_weighting=False):
    """
    Args:
      likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    Returns:
      A Monte-Carlo approximation to the (likelihood weighted) score errors. And the perturbed data
    """
    if not likelihood_weighting:
        return noise + batch_mul(score, std)
    else:
        return batch_mul(noise, 1.0 / std) + score


def get_loss(
    model,
    time_weighting: Callable[[ArrayLike], ArrayLike],
    evaluated_models: Sequence[int],
    sliced=False,
    sliced_noise="rademacher",
    stop_gradient=False,
    likelihood_weighting=True,
    reduce_mean=True,
    alpha=0.0,
    beta=0.0,
    gamma=1.0,
    fp_dist="pert",
    **kwargs,
):
    """Create a loss function for score matching training.
    Args:
      model: A valid flax neural network `:class:flax.linen.Module` class.
      time_weighting: A function that scales the score. It takes the time t as input and returns a scaling factor.
      sliced: Whether to use sliced score matching or conditional score matching.
      sliced_noise: If sliced score matching is used, this field specifies the random vector.
      score_scaling: A function that scales the score. It takes the time t as input and returns a scaling factor.
      likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
      reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
      alpha: A float, the weight of the vector field FP loss.
      beta: A float, the weight of the scalar field FP loss.
      gamma: A float, the weight of the diffusion loss.
      fp_dist: A string, the distribution to use for the FP loss. Can be 'pert' for perturbed data or 'x' for original data.
        **kwargs: Additional keyword arguments that are passed to the FP loss.
    Returns:
      A loss function that can be used for score matching training and is an expectation of the regression loss over time.
    """
    log.info("Using VP-SDE")
    sde = sdes.VP()
    from scoremd.diffusion.fp import fp_vp_loss

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss(
        params: FrozenDict[str, Any],
        rng: ArrayLike,
        batch: ArrayLike,
        features: Optional[ArrayLike],
        ts: ArrayLike,
        is_special_epoch: bool,
        training: bool,
    ) -> Sequence[ArrayLike]:
        rng, error_rng, dropout_rng = jax.random.split(rng, 3)
        score_fn = get_score(model, params, training, evaluated_models, rngs={"dropout": dropout_rng})

        perturbed_data, perturbed_noise, std = perturb(ts, sde, error_rng, batch)

        score = None
        vector_fp, scalar_fp = 0, 0
        if is_special_epoch:
            min_alpha_beta = 1e-6
            if alpha > min_alpha_beta or beta > min_alpha_beta:
                rng, fp_rng = jax.random.split(rng)
                from .sde import VP

                assert type(sde) is VP, "VP Loss only works with VP."

                if fp_dist == "pert":
                    x = perturbed_data
                elif fp_dist == "x":
                    x = batch
                else:
                    raise ValueError(f"Unknown fp_dist: {fp_dist}")

                if beta > min_alpha_beta:
                    energy_models = model.energy_models()
                    if len(energy_models) == 0:
                        raise ValueError("No energy models found. Beta FP loss is not possible.")

                    log_q_evaluated_models = model.energy_models()
                    log_q_evaluated_models = [
                        model_id for model_id in log_q_evaluated_models if model_id in evaluated_models
                    ]
                    log.info(f"Using energy models: {log_q_evaluated_models} for beta FP loss")
                    assert len(log_q_evaluated_models) > 0
                else:
                    log_q_evaluated_models = evaluated_models

                def call_log_q_and_score(x, features, t):
                    return model.apply(
                        params,
                        x,
                        features,
                        t,
                        training,
                        log_q_evaluated_models,
                        rngs={"dropout": dropout_rng},
                        method=model.__class__.log_q_and_score,
                    )

                def call_log_q(x, features, t):
                    return model.apply(
                        params,
                        x,
                        features,
                        t,
                        training,
                        log_q_evaluated_models,
                        rngs={"dropout": dropout_rng},
                        method=model.__class__.log_q,
                    )

                log_q_and_score_fn = (
                    call_log_q_and_score
                    if hasattr(model.__class__, "log_q_and_score") and model.__class__.log_q_and_score is not None
                    else None
                )

                log_q_fn = (
                    call_log_q if hasattr(model.__class__, "log_q") and model.__class__.log_q is not None else None
                )

                actual_alpha, actual_beta = alpha, beta
                if alpha < min_alpha_beta:
                    actual_alpha = 0.0
                if beta < min_alpha_beta or log_q_and_score_fn is None or log_q_fn is None:
                    actual_beta = 0.0

                log.info(f"Using FP/VP loss with alpha={actual_alpha}, beta={actual_beta}, fp_dist={fp_dist}")

                vector_fp, scalar_fp, score = fp_vp_loss(
                    fp_rng,
                    sde,
                    get_score(model, params, training, log_q_evaluated_models, rngs={"dropout": dropout_rng}),
                    log_q_fn,
                    log_q_and_score_fn,
                    x,
                    features,
                    ts.reshape(-1, 1),
                    actual_alpha,
                    actual_beta,
                    time_weighting,
                    min_value=min_alpha_beta,
                    stop_gradient=stop_gradient,
                    **kwargs,
                )

                # In case we have a mixture, we might compute the score only partially.
                if len(log_q_evaluated_models) < len(evaluated_models):
                    score = None

        if score is None or fp_dist != "pert":
            if score is not None:
                log.info("Score already computed, but with a different fp_dist. fp_dist pert ist the faster option.")
            # Save a call to the score function
            score = score_fn(perturbed_data, features, ts)
        e = errors(perturbed_noise, score, std, likelihood_weighting)

        # sliced score matching
        if sliced:
            log.info("Using sliced score matching ...")
            vectors = jax.random.normal(error_rng, (batch.shape[0], batch.shape[1]))

            if sliced_noise == "rademacher":
                vectors = jnp.sign(vectors)
            elif sliced_noise == "sphere":
                vectors = (
                    vectors
                    / jnp.linalg.norm(vectors, axis=-1).reshape(vectors.shape[0], 1)
                    * jnp.sqrt(vectors.shape[-1])
                )
            elif sliced_noise != "gaussian":
                # for gaussian we don't need to change anything
                raise ValueError(f"Unknown sliced noise: {sliced_noise}")

            grad1 = score_fn(perturbed_data, features, ts)
            # loss1 = jnp.sum(grad1 * vectors, axis=-1) ** 2 * 0.5
            loss1 = jnp.linalg.norm(grad1, ord=2, axis=-1) ** 2 * 0.5  # results in a lower variance

            grad2 = jax.grad(lambda x: jnp.sum(score_fn(x, features, ts) * vectors))(perturbed_data)
            loss2 = jnp.sum(vectors * grad2, axis=-1)

            diffusion_loss = jnp.mean(loss1 + loss2)
        else:
            losses = (jnp.abs(e) + 1e-7) ** 2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(batch), ts)[1] ** 2
                losses = losses * g2

            losses *= time_weighting(ts)
            diffusion_loss = jnp.mean(losses)

        return gamma * diffusion_loss, vector_fp, scalar_fp

    return loss


def get_karras_sigma_function(sigma_min, sigma_max, rho=7):
    """
    A sigma function from Algorithm 2 from Karras et al. (2022) arxiv.org/abs/2206.00364

    Returns:
      A function that can be used like `sigmas = vmap(sigma)(ts)` where `ts.shape = (num_steps,)`, see `test_utils.py` for usage.

    Args:
      sigma_min: Minimum standard deviation of forawrd transition kernel.
      sigma_max: Maximum standard deviation of forward transition kernel.
      rho: Order of the polynomial in t (determines both smoothness and growth
        rate).
    """
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)

    def sigma(t):
        # NOTE: is defined in reverse time of the definition in arxiv.org/abs/2206.00364
        return (min_inv_rho + t * (max_inv_rho - min_inv_rho)) ** rho

    return sigma


def get_karras_gamma_function(num_steps, s_churn, s_min, s_max):
    """
    A gamma function from Algorithm 2 from Karras et al. (2022) arxiv.org/abs/2206.00364
    Returns:
      A function that can be used like `gammas = gamma(sigmas)` where `sigmas.shape = (num_steps,)`, see `test_utils.py` for usage.
    Args:
      num_steps:
      s_churn: "controls the overall amount of stochasticity" in Algorithm 2 from Karras et al. (2022)
      [s_min, s_max] : Range of noise levels that "stochasticity" is enabled.
    """

    def gamma(sigmas):
        gammas = jnp.where(sigmas <= s_max, min(s_churn / num_steps, jnp.sqrt(2) - 1), 0.0)
        gammas = jnp.where(s_min <= sigmas, gammas, 0.0)
        return gammas

    return gamma


def get_times(num_steps=1000, dt=None, t0=None):
    """
    Get linear, monotonically increasing time schedule.
    Args:
        num_steps: number of discretization time steps.
        dt: time step duration, float or `None`.
          Optional, if provided then final time, t1 = dt * num_steps.
        t0: A small float 0. < t0 << 1. The SDE or ODE are integrated to
            t0 to avoid numerical issues.
    Return:
        ts: JAX array of monotonically increasing values t \\in [t0, t1].
    """
    if dt is not None:
        if t0 is not None:
            t1 = dt * (num_steps - 1) + t0
            # Defined in forward time, t \in [t0, t1], 0 < t0 << t1
            ts, step = jnp.linspace(t0, t1, num_steps, retstep=True)
            ts = ts.reshape(-1, 1)
            assert jnp.isclose(step, (t1 - t0) / (num_steps - 1))
            assert jnp.isclose(step, dt)
            dt = step
            assert t0 == ts[0]
        else:
            t1 = dt * num_steps
            # Defined in forward time, t \in [dt , t1], 0 < \t0 << t1
            ts, step = jnp.linspace(0.0, t1, num_steps + 1, retstep=True)
            ts = ts[1:].reshape(-1, 1)
            assert jnp.isclose(step, dt)
            dt = step
            t0 = ts[0]
    else:
        t1 = 1.0
        if t0 is not None:
            ts, dt = jnp.linspace(t0, 1.0, num_steps, retstep=True)
            ts = ts.reshape(-1, 1)
            assert jnp.isclose(dt, (1.0 - t0) / (num_steps - 1))
            assert t0 == ts[0]
        else:
            # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
            ts, dt = jnp.linspace(0.0, 1.0, num_steps + 1, retstep=True)
            ts = ts[1:].reshape(-1, 1)
            assert jnp.isclose(dt, 1.0 / num_steps)
            t0 = ts[0]
    assert ts[0, 0] == t0
    assert ts[-1, 0] == t1
    dts = jnp.diff(ts)
    assert jnp.all(dts > 0.0)
    assert jnp.all(dts == dt)
    return ts, dt


def get_timestep(t, t0, t1, num_steps):
    return (jnp.rint((t - t0) * (num_steps - 1) / (t1 - t0))).astype(jnp.int32)


def continuous_to_discrete(betas, dt):
    discrete_betas = betas * dt
    return discrete_betas


def shared_update(rng, x, features, t, solver, probability_flow=None):
    """A wrapper that configures and returns the update function of the solvers.

    :probablity_flow: Placeholder for probability flow ODE.
    """
    return solver.update(rng, x, features, t)


def get_sampler(
    shape,
    outer_solver,
    inner_solver=None,
    denoise=True,
    stack_samples=False,
    inverse_scaler=None,
):
    """Get a sampler from (possibly interleaved) numerical solver(s).

    Args:
      shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
        of the object being sampled from, for example, an image may have
        x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
      outer_solver: A valid numerical solver class that will act on an outer loop.
      inner_solver: '' that will act on an inner loop.
      denoise: Bool, that if `True` applies one-step denoising to final samples.
      stack_samples: Bool, that if `True` return all of the sample path or
        just returns the last sample.
      inverse_scaler: The inverse data normalizer function.
    Returns:
      A sampler.
    """
    if inverse_scaler is None:
        inverse_scaler = lambda x: x  # noqa: E731

    def sampler(rng, features, x_0=None):
        """
        Args:
          rng: A JAX random state.
          x_0: Initial condition. If `None`, then samples an initial condition from the
              sde's initial condition prior. Note that this initial condition represents
              `x_T sim Normal(O, I)` in reverse-time diffusion.
        Returns:
            Samples and the number of score function (model) evaluations.
        """
        outer_update = partial(shared_update, solver=outer_solver)
        outer_ts = outer_solver.ts

        if inner_solver:
            inner_update = partial(shared_update, solver=inner_solver)
            inner_ts = inner_solver.ts
            num_function_evaluations = jnp.size(outer_ts) * (jnp.size(inner_ts) + 1)

            def inner_step(carry, t):
                rng, x, x_mean, vec_t = carry
                rng, step_rng = jax.random.split(rng)
                x, x_mean = inner_update(step_rng, x, features, vec_t)
                return (rng, x, x_mean, vec_t), ()

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full(shape[0], t)
                rng, step_rng = jax.random.split(rng)
                x, x_mean = outer_update(step_rng, x, features, vec_t)
                (rng, x, x_mean, vec_t), _ = scan(inner_step, (step_rng, x, x_mean, vec_t), inner_ts)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    if denoise:
                        return (rng, x, x_mean), x_mean
                    else:
                        return (rng, x, x_mean), x
        else:
            num_function_evaluations = jnp.size(outer_ts)

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full((shape[0],), t)
                rng, step_rng = jax.random.split(rng)
                x, x_mean = outer_update(step_rng, x, features, vec_t)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    return ((rng, x, x_mean), x_mean) if denoise else ((rng, x, x_mean), x)

        rng, step_rng = jax.random.split(rng)
        if x_0 is None:
            if inner_solver:
                x = inner_solver.prior(step_rng, shape)
            else:
                x = outer_solver.prior(step_rng, shape)
        else:
            assert x_0.shape == shape
            x = x_0
        if not stack_samples:
            (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(x_mean if denoise else x), num_function_evaluations
        else:
            (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(xs), num_function_evaluations

    # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
    return sampler
