from dataclasses import dataclass
from scoremd.data.dataset.base import Datapoints
from scoremd.data.dataset.minipeptide import CGMinipeptideDataset, PeptideDatapoints
from functools import partial
import logging
import math
import os
from typing import Any, Callable, Optional, Sequence
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm, trange
from scoremd.data.dataset.protein import SingleProteinDataset
import scoremd.diffusion.classic.sde as sdes
from scoremd.diffusion.classic.utils import perturb
from scoremd.diffusion.fp import fp_vp_error
import wandb as wandb_lib
from scoremd.data.dataset import ToyDataset, ToyDatasets, Dataset, ALDPDataset, MuellerBrownSimulation
from scoremd.models.mixture import MixtureOfModels
from scoremd.rmsd import kabsch_align_many
from scoremd.simulation import create_langevin_step_function, simulate
from scoremd.utils.evaluation import helper_metrics_2d, js_divergence
from scoremd.utils.plots import plot_force_2d, plot_potential_2d, save_parts_of_figure
from flax.core import FrozenDict
import numpy as onp
import scoremd.evaluate.molecules as molecules
import json
import gc

log = logging.getLogger(__name__)
DPI = 200


@dataclass
class EvaluationSettings:
    seed: Optional[int] = None  # The seed for the evaluation, if None, we use the seed from the training
    inference_bs: Optional[int] = None  # The batch size for inference
    fp_inference_bs: Optional[int] = None  # The batch size for FP inference (often we need a smaller number here.)
    num_iid_samples: Optional[int] = None  # The number of iid samples to generate
    num_fp_timepoints: int = 500  # The number of timepoints at which to evaluate the FP loss
    eval_t: float = 0.0  # The time at which the model is evaluated
    num_langevin_intermediate_steps: int = 10  # How many langevin steps are performed to get one sample
    num_langevin_samples: Optional[int] = None  # The number of langevin samples to generate
    num_parallel_langevin_samples: int = 5  # How many langevin simulations are run in parallel
    langevin_dt: Optional[float] = None  # The time step for the langevin simulation, in picoseconds
    max_num_openmm_evaluations: int = 1_000  # The maximum number of openmm calls we do for energy evaluations
    aldp_ensure_start_low_prob: bool = True  # Whether to ensure that one simulation starts in a low probability state
    aldp_evaluate_forces: bool = True  # Whether to evaluate the forces of the model
    limit_inference_peptides: Optional[Sequence[str]] = None  # If specified, only evaluate on these peptides
    only_store_results: bool = False  # If this is true, we only store the results with minimal evaluation.


def evaluate(
    model: MixtureOfModels,
    params: FrozenDict[str, Any],
    dataset: Dataset,
    evaluation: EvaluationSettings,
    orig_BS: int,
    norm_factor: jnp.ndarray,
    wandb: bool,
    out_dir: str,
):
    sde = sdes.VP()
    BS = orig_BS if evaluation.inference_bs is None else evaluation.inference_bs
    num_langevin_samples = (
        min(dataset.train.data.shape[0], 1_000_000)
        if evaluation.num_langevin_samples is None
        else evaluation.num_langevin_samples
    )

    def trained_unnormalized_score(x, features, t, *args, **kwargs):
        """The trained score function without any normalization. This is used for sampling."""
        return model.apply(params, x, features, t, training=False, *args, **kwargs)

    # define the force function (scaled score)
    def force(x: jnp.ndarray, features: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return (
            dataset.kbT
            * model.apply(
                params, x * norm_factor, features, evaluation.eval_t, training=False, method=model.__class__.force
            )
            * norm_factor
        )

    potential = None
    # check if the model has an energy function, and then define it
    if model.apply(
        params,
        dataset.train.data[0],
        evaluation.eval_t * jnp.ones((1,)),
        method=model.__class__.has_energy,
    ):
        log.info("The model(s) and their weights allow for the evaluation of the potential.")

        def potential(x, features):
            return dataset.kbT * model.apply(
                params, x * norm_factor, features, evaluation.eval_t, training=False, method=model.__class__.energy
            )

    metrics = {}

    # use different evaluation functions for different datasets
    if isinstance(dataset, ToyDataset):
        metrics |= evaluate_toy_dataset(dataset.train, dataset, force, potential, out_dir)
        metrics |= evaluate_toy_samples(
            dataset.train, dataset, trained_unnormalized_score, norm_factor, out_dir, seed=evaluation.seed
        )
        if dataset.train.data.shape[-1] == 2:
            metrics |= evaluate_toy_normalization_constant(model, params, out_dir)
    elif isinstance(dataset, MuellerBrownSimulation):
        metrics |= evaluate_mueller_brown(dataset, force, potential, out_dir)
        metrics |= evaluate_mueller_brown_samples(
            dataset.train, dataset, trained_unnormalized_score, norm_factor, out_dir, seed=evaluation.seed
        )
        metrics |= simulate_mueller_brown(dataset.train, dataset, force, out_dir, seed=evaluation.seed)
    elif isinstance(dataset, ALDPDataset):
        inference_bs = BS
        num_samples = (
            min(dataset.train.data.shape[0], 100_000)
            if evaluation.num_iid_samples is None
            else evaluation.num_iid_samples
        )
        if num_samples > 0:
            metrics |= evaluate_molecule_samples(
                dataset,
                dataset.train.data,
                dataset.train.data[0].reshape(dataset.sample_shape),
                None,
                trained_unnormalized_score,
                norm_factor,
                inference_bs,
                num_samples,
                dataset.write_animation,
                wandb,
                "aldp",
                out_dir,
                evaluation.seed,
                only_store_results=evaluation.only_store_results,
            )

        if evaluation.aldp_evaluate_forces:
            metrics |= evaluate_forces_aldp(dataset, force, inference_bs, wandb, out_dir)

        if num_langevin_samples > 0:
            metrics |= simulate_single_system(
                dataset.train,
                dataset,
                force,
                evaluation.num_parallel_langevin_samples,
                num_langevin_samples,
                evaluation.num_langevin_intermediate_steps,
                evaluation.langevin_dt,
                evaluation.max_num_openmm_evaluations,
                evaluation.aldp_ensure_start_low_prob,
                wandb,
                out_dir,
                "aldp",
                evaluation.seed,
                only_store_results=evaluation.only_store_results,
            )

    elif isinstance(dataset, CGMinipeptideDataset):
        inference_bs = BS

        if dataset.val is None:
            datapoints = dataset.train
            current_out_folder = os.path.join(out_dir, "train")
        else:
            datapoints = dataset.val
            current_out_folder = os.path.join(out_dir, "validation")

        metrics |= evaluate_peptide_samples(
            datapoints,
            evaluation.limit_inference_peptides,
            dataset,
            trained_unnormalized_score,
            norm_factor,
            evaluation.num_iid_samples,
            inference_bs,
            wandb,
            current_out_folder,
            evaluation.seed,
        )

        if num_langevin_samples > 0:
            metrics |= simulate_minipeptide(
                datapoints,
                evaluation.limit_inference_peptides,
                dataset,
                force,
                evaluation.num_parallel_langevin_samples,
                num_langevin_samples,
                evaluation.num_langevin_intermediate_steps,
                evaluation.langevin_dt,
                wandb,
                current_out_folder,
                evaluation.seed,
                only_store_results=evaluation.only_store_results,
            )
    elif isinstance(dataset, SingleProteinDataset):
        inference_bs = BS
        num_samples = (
            min(dataset.train.data.shape[0], 100_000)
            if evaluation.num_iid_samples is None
            else evaluation.num_iid_samples
        )
        if num_samples > 0:
            metrics |= evaluate_molecule_samples(
                dataset,
                dataset.train.data,
                dataset.train.data[0].reshape(dataset.sample_shape),
                None,
                trained_unnormalized_score,
                norm_factor,
                inference_bs,
                num_samples,
                None,
                wandb,
                dataset.name,
                out_dir,
                evaluation.seed,
                only_store_results=evaluation.only_store_results,
            )

        if num_langevin_samples > 0:
            metrics |= simulate_single_system(
                dataset.train,
                dataset,
                force,
                evaluation.num_parallel_langevin_samples,
                num_langevin_samples,
                evaluation.num_langevin_intermediate_steps,
                evaluation.langevin_dt,
                evaluation.max_num_openmm_evaluations,
                False,
                wandb,
                out_dir,
                dataset.name,
                evaluation.seed,
                only_store_results=evaluation.only_store_results,
            )

    FP_BS = evaluation.fp_inference_bs if evaluation.fp_inference_bs is not None else BS
    if evaluation.num_fp_timepoints > 0:
        metrics |= evaluate_fp_loss(
            dataset.train[jax.random.permutation(jax.random.PRNGKey(0), len(dataset.train))[:FP_BS]],
            evaluation.num_fp_timepoints,
            norm_factor,
            model,
            params,
            trained_unnormalized_score,
            sde,
            wandb,
            out_dir,
        )

    if wandb and len(metrics) > 0:
        wandb_lib.log(metrics)

    # write metrics to file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        # Skip values that are not JSON serializable
        serializable_metrics = {}
        for key, value in metrics.items():
            try:
                # Test if the value is JSON serializable
                json.dumps({key: value})
                serializable_metrics[key] = value
            except (TypeError, OverflowError):
                log.warning(f"Skipping non-serializable metric: {key}")

        json.dump(serializable_metrics, f)

    log.info(f"Metrics: {metrics}")
    log.info(f"Finished evaluation. You can find the results in {out_dir}")
    return metrics


def evaluate_fp_loss(
    data_points: Datapoints,
    num_fp_timepoints: int,
    norm_factor: jnp.ndarray,
    model: MixtureOfModels,
    params: FrozenDict[str, Any],
    unnormalized_score: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    sde: sdes.VP,
    wandb: bool,
    out_dir: str,
) -> dict:
    data = data_points.data
    data *= norm_factor  # convert the data to the correct scale

    vs, ss = [], []
    key = jax.random.PRNGKey(0)

    if hasattr(model, "log_q_and_score") and hasattr(model, "log_q"):

        def log_q_and_score_fn(x, features, t):
            return model.apply(
                params,
                x,
                features,
                t,
                training=False,
                evaluated_models=model.energy_models(),
                method=model.__class__.log_q_and_score,
            )

        def log_q_fn(x, features, t):
            return model.apply(
                params,
                x,
                features,
                t,
                training=False,
                evaluated_models=model.energy_models(),
                method=model.__class__.log_q,
            )
    else:
        log_q_and_score_fn = None
        log_q_fn = None

    D = data.shape[-1]
    log.warning("Hardcoded likelihood weighting to False")

    @partial(jax.jit, static_argnums=(1,))
    def fp_vp_error_at_t(
        t: jnp.ndarray, scalar: bool
    ):  # we intentially use the same key for all t to make the evaluation reproducible
        x, _, _ = perturb(t, sde, key, data)
        vector_error, scalar_error, _ = fp_vp_error(
            key,
            sde,
            unnormalized_score,
            log_q_fn,
            log_q_and_score_fn,
            x,
            data_points.features,
            t * jnp.ones((x.shape[0], 1)),
            True,
            scalar,
            div_est=2,
            unbiased=False,
        )
        return jnp.mean(jnp.sqrt(vector_error)) / D, jnp.mean(jnp.sqrt(scalar_error)) / D

    ts = jnp.linspace(0, 1, num_fp_timepoints)
    for t in tqdm(ts, desc="Evaluating FP Error"):
        scalar = (
            False
            if log_q_and_score_fn is None and log_q_fn is None
            else model.apply(params, data, t * jnp.ones((1,)), method=model.__class__.has_energy)
        )

        vector_error, scalar_error = fp_vp_error_at_t(t * jnp.ones((data.shape[0], 1)), scalar)
        vs.append(vector_error)
        ss.append(scalar_error)

    plt.figure(clear=True)
    plt.title("Vector FP Error (alpha)")
    plt.plot(ts[1:-1], vs[1:-1])
    plt.xlim(0, 1)
    plt.savefig(f"{out_dir}/vector_fp_error.png", bbox_inches="tight")
    plt.close()

    onp.save(f"{out_dir}/vector_fp_error.npy", vs)

    if jnp.nansum(jnp.abs(jnp.array(ss))) > 0:  # if there is any conservative model
        plt.figure(clear=True)
        plt.title("Scalar FP Error (beta)")
        plt.plot(ts[1:-1], ss[1:-1])
        plt.xlim(0, 1)
        plt.savefig(f"{out_dir}/scalar_fp_error.png", bbox_inches="tight")
        plt.close()

        onp.save(f"{out_dir}/scalar_fp_error.npy", ss)

    if wandb:
        wandb_lib.define_metric("eval/vector_fp_error", step_metric="eval/time")
        wandb_lib.define_metric("eval/scalar_fp_error", step_metric="eval/time")
        for i in range(ts.shape[0]):
            wandb_lib.log({"eval/vector_fp_error": vs[i], "eval/scalar_fp_error": ss[i], "eval/time": ts[i]})

    return {}


def evaluate_peptide_samples(
    datapoints: PeptideDatapoints,
    limit_inference_peptides: Optional[Sequence[str]],
    dataset: CGMinipeptideDataset,
    trained_unnormalized_score: Callable,
    norm_factor: jnp.ndarray,
    num_iid_samples: Optional[int],
    inference_bs: int,
    wandb: bool,
    out_dir: str,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {}
    current_idx = 0
    for peptide, peptide_length in zip(datapoints.peptides, datapoints.peptide_lengths):
        next_idx = current_idx + peptide_length
        num_samples = min(peptide_length, 100_000) if num_iid_samples is None else num_iid_samples
        if num_samples <= 0:
            continue

        if limit_inference_peptides is not None and peptide not in limit_inference_peptides:
            log.info(f"Skipping peptide {peptide}")
        else:
            log.info(f"Generating and evaluating {num_samples} samples for peptide {peptide}")
            metrics |= evaluate_molecule_samples(
                dataset,
                datapoints.data[current_idx:next_idx],
                datapoints.data[current_idx].reshape(dataset.sample_shape),
                datapoints.features[current_idx : current_idx + 1].repeat(num_samples, axis=0),
                trained_unnormalized_score,
                norm_factor,
                inference_bs,
                num_samples,
                lambda x, out_dir: dataset.write_animation(x, peptide, out_dir),
                wandb,
                peptide,
                out_dir,
                seed,
            )

            # free unused memory
            gc.collect()

        current_idx = next_idx

    return metrics


def get_samples(
    shape: jnp.ndarray,
    features: Optional[jnp.ndarray],
    score: Callable,
    norm_factor: jnp.ndarray,
    seed: int,
    BS: Optional[int] = None,
    t0: float = 0.0,
) -> jnp.ndarray:
    from scoremd.diffusion.classic.utils import get_times, get_sampler
    from scoremd.diffusion.classic.solvers import EulerMaruyama

    log.info(f"Generating {shape[0]} samples ...")

    updated_shape = shape if BS is None else (BS, *shape[1:])

    log.warning("Hardcoded SDE to VP, in case you want to change it, also change it here")
    rsde = sdes.VP().reverse(score)
    ts, _ = get_times(num_steps=1000, t0=t0)
    outer_solver = EulerMaruyama(rsde, ts)

    q_samples = []
    key = jax.random.PRNGKey(seed)

    sampler = get_sampler(updated_shape, outer_solver, denoise=False, inverse_scaler=lambda x: x)

    step_size = shape[0] if BS is None else BS
    num_steps = math.ceil(shape[0] / BS) if BS is not None else 1

    # we need to pad the features to the same length as num_steps * step_size
    if features is not None:
        log.info(f"Padding features to {num_steps * step_size} length")
        features = jnp.pad(
            features, ((0, num_steps * step_size - features.shape[0]), *([(0, 0)] * (features.ndim - 1)))
        )
        log.info(f"Features new shape: {features.shape}")

    def sample_batch(carry, features):
        key, _ = carry
        key, sample_key = jax.random.split(key)

        return (key, None), sampler(sample_key, features)[0]

    init_carry = (key, None)
    _, q_samples = jax.lax.scan(
        sample_batch,
        init_carry,
        features.reshape(num_steps, step_size, *features.shape[1:]) if features is not None else None,
        length=num_steps,
    )

    q_samples = q_samples.reshape(-1, q_samples.shape[-1])

    # filter out nan values
    nan_entries = jnp.isnan(q_samples).any(axis=1)
    if nan_entries.sum() > 0:
        log.warning(f"Found {nan_entries.sum()} NaN entries. Filtering them out...")

        q_samples = q_samples[~nan_entries]
    q_samples /= norm_factor

    if q_samples.shape[0] > shape[0]:
        q_samples = q_samples[: shape[0]]

    log.info(f"{q_samples.shape[0]} samples remaining")

    return q_samples


def evaluate_toy_dataset(
    datapoints: Datapoints,
    dataset: ToyDataset,
    force_with_features: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    potential_net_with_features: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]],
    out_dir: str,
):
    assert datapoints.features is None

    def force(x):
        return force_with_features(x, None)

    if potential_net_with_features is not None:

        def potential_net(x):
            return potential_net_with_features(x, None)
    else:
        potential_net = None

    metrics = {}
    if datapoints.data.shape[-1] == 1:
        plt.figure(figsize=(10, 4), clear=True)

        plt.subplot(1, 2, 1)
        dataset.plot_potential_evaluation(potential_net)

        plt.subplot(1, 2, 2)
        dataset.plot_force_evaluation(force)

        plt.savefig(f"{out_dir}/1d-results.png", bbox_inches="tight")
        plt.close()

    if datapoints.data.shape[-1] == 2:
        plt.figure(figsize=(12, 10), clear=True)
        ax0, _, ax2, _ = dataset.plot_potential_evaluation(potential_net)
        save_parts_of_figure(
            ax0,
            f"{out_dir}/2d-results-density-ground-truth.pdf",
            title=False,
            xticks=False,
            yticks=False,
            xlabel=False,
            ylabel=False,
            dpi=DPI,
        )
        save_parts_of_figure(
            ax2,
            f"{out_dir}/2d-results-density-learned.pdf",
            title=False,
            xticks=False,
            yticks=False,
            xlabel=False,
            ylabel=False,
            dpi=DPI,
        )
        plt.savefig(f"{out_dir}/2d-results-potential.png", bbox_inches="tight")
        plt.close()

        plt.figure(clear=True)
        dataset.plot_force_evaluation(force)
        plt.savefig(f"{out_dir}/2d-results-forces.png", bbox_inches="tight")
        plt.close()

        if potential_net and dataset.example is ToyDatasets.DoubleWell2D:
            from scoremd.data.dataset.toy import _double_well_2d_density

            (x_min, x_max), (y_min, y_max) = dataset.example.range()

            n = 100j
            xx, yy = jnp.mgrid[x_min:x_max:n, y_min:y_max:n]
            positions = jnp.vstack([xx.ravel(), yy.ravel()]).T

            def compute_mass(positions, values):
                upper_right = positions[:, 1] > -positions[:, 0]
                return (values[upper_right]).sum(), (values[~upper_right]).sum()

            ff = potential_net(positions)
            ff = jnp.exp(-ff + ff.min())
            mass_a, mass_b = compute_mass(positions, ff.reshape(-1, 1))
            mass_a_baseline, mass_b_baseline = compute_mass(positions, _double_well_2d_density(positions))

            metrics |= {
                "eval/relative_mass_distribution": (mass_a / (mass_a + mass_b)).item(),
            }

            log.info(
                f"{mass_a / (mass_a + mass_b) * 100:.2f}% of the learned potential mass lies above the diagonal threshold. This should be {mass_a_baseline / (mass_a_baseline + mass_b_baseline) * 100:.2f}%"
            )

    return metrics


def evaluate_toy_normalization_constant(
    model: MixtureOfModels,
    params: FrozenDict[str, Any],
    out_dir: str,
):
    # we now estimate the normalization constant Z
    # we draw a batch of samples from the learned potential
    test_ts = jnp.linspace(0.0, 1.0, 100)

    N = 100
    x0, x1 = -10, 10
    y0, y1 = -10, 10

    x = jnp.linspace(x0, x1, N)
    y = jnp.linspace(y0, y1, N)

    dx = (x1 - x0) / (N - 1)
    dy = (y1 - y0) / (N - 1)

    xx, yy = jnp.meshgrid(x, y)
    samples = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

    def log_q_fn(x, features, t):
        return model.apply(
            params,
            x,
            features,
            t,
            False,
            model.energy_models(),
            # rngs={"dropout": jax.random.PRNGKey(0)}, we don't need this here, because we don't train
            method=model.__class__.log_q,
        )

    Zs = []
    Z_grads = []

    def compute_Z(t):
        log_q = log_q_fn(samples, None, t)
        q = jnp.exp(log_q)

        # integrate over the grid
        Z = q.sum() * dx * dy
        return jnp.log(Z)

    Z_grad_t = jax.grad(compute_Z)

    for t in tqdm(test_ts):
        Z = compute_Z(t * jnp.ones((samples.shape[0], 1)))
        Z_grad = Z_grad_t(t * jnp.ones((samples.shape[0], 1))).mean()
        Zs.append(Z)
        Z_grads.append(Z_grad)

    plt.figure(clear=True)
    plt.plot(test_ts, Zs)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$Z_t$")
    plt.savefig(f"{out_dir}/Z-estimation.png", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    plt.plot(test_ts, Z_grads)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\partial_t Z_t$")
    plt.savefig(f"{out_dir}/Z-grad-estimation.png", bbox_inches="tight")
    plt.close()

    return {"eval/Z": jnp.log(jnp.array(Zs).mean())}


def evaluate_toy_samples(
    datapoints: Datapoints, dataset: ToyDataset, score: Callable, norm_factor: jnp.ndarray, out_dir: str, seed: int
):
    q_samples = get_samples(datapoints.data.shape, None, score, norm_factor=norm_factor, seed=seed)
    outliers = (jnp.abs(q_samples) > 20).sum(axis=1) > 0
    if outliers.sum() > 0:
        log.warning(f"There are {outliers.mean() * 100:.3f}% outliers. Filtering them out...")

    q_samples = q_samples[~outliers]

    if datapoints.data.shape[-1] == 1:
        plt.figure(figsize=(10, 4), clear=True)

        plt.subplot(1, 2, 1)
        plt.title("Histogram of new samples")
        plt.hist(q_samples, bins=100)

        plt.subplot(1, 2, 2)
        plt.title("KDE approximation of samples")
        kde_sampled = gaussian_kde(q_samples.T)

        x_min, x_max = dataset.example.range()
        xx = jnp.linspace(x_min, x_max, 1000)
        xx = xx.reshape(-1, 1)

        kde = gaussian_kde(datapoints.data.reshape(-1))
        vv_true = -jnp.log(kde(xx.reshape(-1)))

        vv_sampled = -jnp.log(kde_sampled(xx.reshape(-1)))
        plt.plot(xx, vv_sampled - vv_sampled.min(), label="KDE of samples")
        plt.plot(xx, vv_true - vv_true.min(), label="truth")
        plt.legend()
        plt.savefig(f"{out_dir}/1d-sampled.png", bbox_inches="tight")
        plt.close()
    elif datapoints.data.shape[-1] == 2:
        plt.figure(clear=True)
        plt.title("Sampled data")

        (x_min, x_max), (y_min, y_max) = dataset.example.range()

        # filter out all points that are not within x_min, x_max, y_min, y_max
        plot_q_samples = q_samples[
            (q_samples[:, 0] >= x_min)
            & (q_samples[:, 0] <= x_max)
            & (q_samples[:, 1] >= y_min)
            & (q_samples[:, 1] <= y_max)
        ]

        if dataset.example is ToyDatasets.DoubleWell2D:
            log.info(
                f"{(q_samples[:, 1] > -q_samples[:, 0]).mean() * 100:.2f}% of sampled mass lies above the diagonal threshold. This should be 80%"
            )

        plt.hist2d(plot_q_samples[:, 0], plot_q_samples[:, 1], bins=100)
        if dataset.example is ToyDatasets.DoubleWell2D:
            plt.scatter([-5, 5], [-5, 5], marker="*", color="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(f"{out_dir}/2d-sampled.png", bbox_inches="tight")
        plt.close()

    return {"eval/iid_js_divergence": js_divergence(datapoints.data, q_samples, bins=100)}


def evaluate_mueller_brown(
    dataset: MuellerBrownSimulation,
    force_with_features: Callable[[jnp.ndarray], jnp.ndarray],
    potential_net_with_features: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    out_dir: str,
):
    def force(x):
        return force_with_features(x, None)

    if potential_net_with_features is not None:

        def potential_net(x):
            return potential_net_with_features(x, None)
    else:
        potential_net = None

    plt.figure(figsize=(12, 10), clear=True)
    plot_potential_2d(dataset.potential, dataset.likelihood, potential_net, *dataset.range())
    plt.savefig(f"{out_dir}/mueller-brown-results-potential.png", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    ground_truth_force = jax.grad(lambda x: -dataset.potential(x).sum())
    plot_force_2d(force, ground_truth_force, *dataset.range())
    plt.savefig(f"{out_dir}/mueller-brown-results-forces.png", bbox_inches="tight")
    plt.close()
    plt.figure(clear=True)
    plot_force_2d(force, ground_truth_force, *dataset.range(), equal_scaling=False)
    plt.savefig(f"{out_dir}/mueller-brown-results-forces-autoscale.png", bbox_inches="tight")
    plt.close()

    return {}


def evaluate_mueller_brown_samples(
    datapoints: Datapoints,
    dataset: MuellerBrownSimulation,
    score: Callable,
    norm_factor: jnp.ndarray,
    out_dir: str,
    seed: int,
):
    q_samples = get_samples(datapoints.data.shape, None, score, norm_factor=norm_factor, seed=seed)
    outliers = (jnp.abs(q_samples) > 20).sum(axis=1) > 0
    if outliers.sum() > 0:
        log.warning(f"There are {outliers.mean() * 100:.3f}% outliers. Filtering them out...")

    q_samples = q_samples[~outliers]

    plt.figure(figsize=(12, 4), clear=True)

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Samples")
    dataset.plot(datapoints.data)

    plt.subplot(1, 2, 2)
    plt.title("New Samples")
    dataset.plot(q_samples)

    plt.savefig(f"{out_dir}/mueller-brown-samples.png", bbox_inches="tight")
    plt.close()

    # paper figures
    plt.figure(clear=True)
    dataset.plot(q_samples, cbar_range=(0, 152.5), cbar=False)
    plt.savefig(f"{out_dir}/mueller-brown-iid.pdf", bbox_inches="tight", dpi=DPI)
    plt.close()

    rms_fe_sq_error, rms_mjs_error = helper_metrics_2d(
        datapoints.data[:, 0],
        datapoints.data[:, 1],
        q_samples[:, 0],
        q_samples[:, 1],
        limits_x=dataset.range()[0],
        limits_y=dataset.range()[1],
    )

    return {
        "eval/iid_js_divergence": js_divergence(datapoints.data, q_samples, bins=100),
        "eval/iid_rms_fe_sq_error": rms_fe_sq_error,
        "eval/iid_rms_mjs_error": rms_mjs_error,
    }


def simulate_mueller_brown(
    datapoints: Datapoints,
    dataset: MuellerBrownSimulation,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    out_dir: str,
    seed: int,
):
    key = jax.random.PRNGKey(seed)
    key, velocity_key = jax.random.split(key)

    n_samples = 100_000  # number of samples to generate

    starting_point = jnp.array([-0.55828035, 1.44169])
    starting_velocity = jnp.sqrt(dataset.kbT / dataset.mass) * jax.random.normal(velocity_key, (2,))

    @jax.jit
    def adjusted_force(x):
        # we don't have any features for toy datasets
        return force(x.reshape(1, -1), None).reshape(-1)

    n_steps = 50
    step = jax.jit(
        create_langevin_step_function(adjusted_force, dataset.mass, dataset.gamma, n_steps, dataset.dt, dataset.kbT)
    )

    log.info(f"Simulating {n_samples} MD steps with {n_steps} intermediate steps")
    trajectory, velocities = simulate(starting_point, starting_velocity, step, n_samples, key)
    forces = jnp.array([adjusted_force(x) for x in trajectory])

    plt.figure(clear=True)
    plt.hist(jnp.linalg.norm(forces, axis=1), bins=100)
    plt.xlabel("Force Norm")
    plt.ylabel("Count")
    plt.savefig(f"{out_dir}/mueller-brown-forces-histogram.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 4), clear=True)

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Samples")
    dataset.plot(datapoints.data)

    plt.subplot(1, 2, 2)
    plt.title("New Samples")
    dataset.plot(trajectory)

    plt.savefig(f"{out_dir}/mueller-brown-langevin.png", bbox_inches="tight")
    plt.close()

    # paper figures
    plt.figure(clear=True)
    dataset.plot(datapoints.data, cbar_range=(0, 152.5), cbar=False)
    plt.savefig(f"{out_dir}/mueller-brown-langevin-ground-truth.pdf", bbox_inches="tight", dpi=200)
    plt.close()

    plt.figure(clear=True)
    dataset.plot(trajectory, cbar_range=(0, 152.5), cbar=False)
    plt.savefig(f"{out_dir}/mueller-brown-langevin.pdf", bbox_inches="tight", dpi=200)
    plt.close()

    rms_fe_sq_error, rms_mjs_error = helper_metrics_2d(
        datapoints.data[:, 0],
        datapoints.data[:, 1],
        trajectory[:, 0],
        trajectory[:, 1],
        limits_x=dataset.range()[0],
        limits_y=dataset.range()[1],
    )
    return {
        "eval/langevin_js_divergence": js_divergence(datapoints.data, trajectory, bins=100),
        "eval/langevin_rms_fe_sq_error": rms_fe_sq_error,
        "eval/langevin_rms_mjs_error": rms_mjs_error,
    }


def simulate_single_system(
    datapoints: Datapoints,
    dataset: ALDPDataset | SingleProteinDataset,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    n_points: int,
    n_samples: int,
    n_steps: int,
    langevin_dt: Optional[float],
    max_num_openmm_evaluations: int,
    aldp_ensure_start_low_prob: bool,
    wandb: bool,
    out_dir: str,
    prefix: str,
    seed: int,
    only_store_results: bool = False,
) -> dict:
    if datapoints.data.shape[0] < n_points:
        data = jnp.repeat(datapoints.data, n_points, axis=0)
    else:
        data = datapoints.data

    initial_positions = data[:n_points]

    if isinstance(dataset, ALDPDataset) and aldp_ensure_start_low_prob:
        ground_truth_phi, ground_truth_psi = dataset.get_2d_features(data)
        is_low_probability = (
            (ground_truth_phi > 0.0) & (ground_truth_phi < 2.0) & (ground_truth_psi > -2) & (ground_truth_psi < 2)
        )
        if is_low_probability.any():
            log.info("Found low probability state. Prepending it to the simulation...")
            first_low_probability_state = data[is_low_probability][0]
            initial_positions = jnp.concatenate([first_low_probability_state[None], initial_positions[:-1]], axis=0)
        else:
            log.warning("No low probability state found. Using random starting points...")

    trajectories, velocities = molecules.simulate_molecule(
        dataset,
        force,
        initial_positions,
        None,
        n_samples,
        n_steps,
        langevin_dt,
        seed,
    )

    return molecules.evaluate_langevin_samples(
        dataset,
        dataset.train.data,
        trajectories,
        velocities,
        max_num_openmm_evaluations,
        dataset.write_animation if hasattr(dataset, "write_animation") else None,
        wandb,
        prefix,
        out_dir,
        only_store_results=only_store_results,
    )


def simulate_minipeptide(
    datapoints: Datapoints,
    limit_inference_peptides: Optional[Sequence[str]],
    dataset: CGMinipeptideDataset,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    n_points: int,
    n_samples: int,
    n_steps: int,
    langevin_dt: Optional[float],
    wandb: bool,
    out_dir: str,
    seed: int,
    only_store_results: bool = False,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    initial_positions = []
    features = []

    current_idx = 0
    for peptide, peptide_length in zip(datapoints.peptides, datapoints.peptide_lengths):
        next_idx = current_idx + peptide_length

        if limit_inference_peptides is not None and peptide not in limit_inference_peptides:
            log.info(f"Skipping peptide {peptide}")
        else:
            if peptide_length < n_points:
                raise ValueError(f"Peptide {peptide} has only {peptide_length} points, but {n_points} are required")

            random_samples = jax.random.permutation(jax.random.PRNGKey(seed), peptide_length)[:n_points]
            initial_positions.append(datapoints.data[current_idx:next_idx][random_samples])
            features.append(datapoints.features[current_idx:next_idx][random_samples])

        current_idx = next_idx

    if len(initial_positions) == 0:
        log.warning("No peptides found to evaluate")
        return {}

    initial_positions = jnp.concatenate(initial_positions)
    features = jnp.concatenate(features)

    trajectories, velocities = molecules.simulate_molecule(
        dataset, force, initial_positions, features, n_samples, n_steps, langevin_dt, seed
    )

    metrics = {}
    current_idx = 0
    i = 0
    for peptide, peptide_length in zip(datapoints.peptides, datapoints.peptide_lengths):
        next_idx = current_idx + peptide_length
        if limit_inference_peptides is not None and peptide not in limit_inference_peptides:
            log.info(f"Skipping peptide {peptide}")
        else:
            current_trajectories = trajectories[i * n_points : (i + 1) * n_points]
            current_velocities = velocities[i * n_points : (i + 1) * n_points]

            baseline_samples = datapoints.data[current_idx:next_idx]

            metrics |= molecules.evaluate_langevin_samples(
                dataset,
                baseline_samples,
                current_trajectories,
                current_velocities,
                None,
                lambda x, out_dir: dataset.write_animation(x, peptide, out_dir),
                wandb,
                peptide,
                out_dir,
                only_store_results=only_store_results,
            )
            i += 1

        current_idx = next_idx
    return metrics


def evaluate_molecule_samples(
    dataset: ALDPDataset | CGMinipeptideDataset | SingleProteinDataset,
    ground_truth_samples: jnp.ndarray,
    reference_sample: jnp.ndarray,
    features: Optional[jnp.ndarray],
    unnormalized_score: Callable,
    norm_factor: jnp.ndarray,
    inference_bs: int,
    num_samples: int,
    write_animation: Optional[Callable[[jnp.ndarray, str], None]],
    wandb: bool,
    prefix: str,
    out_dir: str,
    seed: int,
    only_store_results: bool = False,
) -> dict:
    if not only_store_results:
        plt.figure(clear=True)
        dataset.plot_2d(ground_truth_samples, title=f"Histogram {prefix} (ground truth)")
        plt.savefig(f"{out_dir}/{prefix}_iid_phi_psi_target.png", bbox_inches="tight")
        plt.close()

    sample_shape = (num_samples, dataset.train.data.shape[1])
    q_samples = get_samples(
        sample_shape,
        features,
        unnormalized_score,
        norm_factor=norm_factor,
        BS=inference_bs,
        seed=seed,
    ).reshape(-1, *dataset.sample_shape)
    q_samples, _ = kabsch_align_many(q_samples, reference_sample)
    onp.save(f"{out_dir}/{prefix}_iid_samples.npy", q_samples)
    log.info(f"Saved iid samples to {out_dir}/{prefix}_iid_samples.npy")

    if only_store_results:
        return {}

    return molecules.evaluate_iid_samples(
        dataset, ground_truth_samples, q_samples, write_animation, wandb, prefix, out_dir
    )


def evaluate_forces_aldp(
    dataset: ALDPDataset,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    BS: int,
    wandb: bool,
    out_dir: str,
) -> dict:
    # This is a bit hacky
    xyz = jnp.array(dataset._dataset.xyz)[:, dataset._atoms_to_keep, ...]
    ground_truth_forces = jnp.array(dataset._dataset.forces)[:, dataset._atoms_to_keep, ...]
    ground_truth_forces = jnp.linalg.norm(ground_truth_forces, axis=(1, 2))
    assert dataset.train.features is None

    # iterata data with BS
    @jax.jit
    def adjusted_force(x):
        return force(x.reshape(x.shape[0], -1), None).reshape(x.shape[0], *dataset.sample_shape)

    forces = []
    for i in trange(0, xyz.shape[0], BS, desc="Computing forces"):
        trajectory = xyz[i : min(i + BS, xyz.shape[0])]
        forces.append(adjusted_force(trajectory))

    forces = jnp.concatenate(forces)  # shape: (n_samples, *sample_shape)
    forces = jnp.linalg.norm(forces, axis=(1, 2))

    plt.figure(clear=True)
    plt.title("Predicted forces")
    plt.hist(ground_truth_forces, bins=100, alpha=0.7, label="Ground Truth", density=True)
    plt.hist(forces, bins=100, alpha=0.7, label="Model", density=True)
    plt.xlabel("Force Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{out_dir}/forces_histogram.png", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    plt.title("Relative force error")
    plt.hist(ground_truth_forces / forces, bins=100, density=True)
    plt.axvline(jnp.mean(ground_truth_forces / forces), color="k", linestyle="dashed", linewidth=1)
    plt.xlabel("Relative Force Error")
    plt.ylabel("Density")
    plt.savefig(f"{out_dir}/forces_error_histogram.png", bbox_inches="tight")
    plt.close()

    if wandb:
        wandb_lib.log({"eval/forces_histogram": wandb_lib.Image(f"{out_dir}/forces_histogram.png")})
        wandb_lib.log({"eval/forces_error_histogram:": wandb_lib.Image(f"{out_dir}/forces_error_histogram.png")})
        wandb_lib.log(
            {
                "eval/mean_abs_force_error": jnp.mean(jnp.abs(ground_truth_forces - forces)),
                "eval/mean_relative_force_error": jnp.mean(ground_truth_forces / forces),
            }
        )

    return {}
