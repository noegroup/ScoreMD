from evaluate import EvaluationSettings, evaluate
from scoremd.utils import slurm
from scoremd.utils.file import (
    get_persistent_storage,
)  # this should be the very first line so that the .env file is loaded
import logging
import os
from typing import Callable, List, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp
from hydra.conf import HydraConf, RunDir
from hydra.core.hydra_config import HydraConfig
from hydra_zen import builds, zen, store
from hydra_zen.typing import Builds
from scoremd.data.preprocess import CenterMolecule
import scoremd.models as diffusion_models
import wandb as wandb_lib
from scoremd.data.dataset import ToyDataset, Dataset, MuellerBrownSimulation
from scoremd.models.mixture import MixtureOfModels
from scoremd.training import load_and_train, TrainingSchedule
from scoremd.training.train_state import EmaTrainState
from scoremd.training.weighting import (
    plot_weighting_function,
)
from scoremd.utils.hydra import SlurmInitializer, WandbInitializer
from scoremd.utils.ranges import assert_range_continuity
from stores import create_dataset_store, create_trainig_schedule_store, create_weighting_store, create_optimizer_store

log = logging.getLogger(__name__)


def training_routine(
    dataset: Builds[Dataset],
    optimizer: Callable[[int], optax.GradientTransformation],
    ranged_models: List[Builds[diffusion_models.RangedModel]],
    training_schedule: Builds[TrainingSchedule],
    num_devices: Optional[int],
    weighting_function: Callable[
        [List[diffusion_models.RangedModel]], Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ],
    checkpoint_options: ocp.CheckpointManagerOptions,
    load_dir: Optional[str],
    continue_from: Optional[str],
    load_epoch: Optional[int],
    evaluation: EvaluationSettings,
    no_evaluation: bool,
    seed: int,
    wandb: dict,
):
    assert len(ranged_models) > 0, "At least one model is required"
    assert continue_from is None or load_dir is None, "Please specify either load_dir or continue_from, not both."

    base_dir = HydraConfig.get().runtime.output_dir

    if continue_from is None:
        log.info(f"The unique output directory for this folder is {base_dir}.")
        model_load_dir = os.path.join(load_dir, "model") if load_dir is not None else None
    else:
        continue_from = os.path.abspath(os.path.expanduser(continue_from))
        if not os.path.exists(continue_from):
            log.warning(
                f"Directory {continue_from} does not exist, we will start a new run, but use the specified output directory {continue_from}."
            )
        log.info(
            f"The unique run directory for this folder is {base_dir} (i.e., hydra config will be stored there), however we will continue from {continue_from} and add our results to it."
        )
        base_dir = continue_from
        model_load_dir = os.path.join(continue_from, "model")

    if model_load_dir is not None:
        if not os.path.exists(model_load_dir):
            log.warning(
                f"Model directory {model_load_dir} does not exist, we will start a new run, not loading any models."
            )
            model_load_dir = None

    out_dir = os.path.join(base_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    log.info(
        f"Working directory: {os.getcwd()}, output directory: {out_dir}, tmpdir: {os.getenv('TMPDIR')}, persistent storage: {get_persistent_storage()}"
    )
    log.info(f"Loading dataset '{dataset.name}'")

    train_data, val_data = dataset.train, dataset.val
    log.info(
        f"Loaded dataset '{dataset.name}' with data shape: {train_data.data.shape} and features shape: {train_data.features.shape if train_data.features is not None else None}"
    )
    if val_data is not None:
        log.info(
            f"Validation data shape: {val_data.data.shape} and features shape: {val_data.features.shape if val_data.features is not None else None}"
        )

    BS = training_schedule.BS
    if BS > train_data.data.shape[0]:
        # replicate the training data to match the batch size
        log.warning(
            f"Batch size is larger than the dataset size. "
            f"To prevent errors, we increase the training data from {train_data.data.shape[0]} to {BS}."
        )

        new_train_data = jnp.concatenate([train_data.data] * BS, axis=0)[:BS]
        new_train_features = (
            None
            if train_data.features is None
            else jnp.concatenate([train_data.features] * BS, axis=0, dtype=train_data.features.dtype)[:BS]
        )
        train_data = train_data.replace(data=new_train_data, features=new_train_features)

    ranged_models = sorted(ranged_models, key=lambda x: x.range[0], reverse=True)
    log.info(f"Specified {len(ranged_models)} model(s) with range(s) {' -> '.join([str(m) for m in ranged_models])}")
    assert_range_continuity([m.range for m in ranged_models])

    # create the weighting function with the given models
    weighting_function = weighting_function(ranged_models)

    if len(ranged_models) > 1:
        plt.figure(clear=True)
        plot_weighting_function(weighting_function, x=None)
        plt.savefig(f"{out_dir}/model_weights.png", bbox_inches="tight")
        plt.close()

        if wandb["enabled"]:
            wandb_lib.save(f"{out_dir}/model_weights.png", base_path=out_dir)

    if dataset.sample_shape[-1] == 3:
        norm_factor = 1.0 / dataset.std
    else:
        # We don't normalize toy data because the distribution is not really hard
        # also, not everything is perfectly centered
        norm_factor = 1.0
    log.info(f"Normalization factor: {norm_factor}")

    unified_model = MixtureOfModels(
        [m.build(dataset, norm_factor) for m in ranged_models], weighting_function, CenterMolecule(dataset)
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    if num_devices is None:
        if isinstance(dataset, ToyDataset) or isinstance(dataset, MuellerBrownSimulation):
            num_devices = 1
            log.info("num_devices not specified, using 1 device since we are using a simple dataset")
        else:
            num_devices = jax.device_count()
            log.info(f"num_devices not specified, using all {num_devices} devices")
    else:
        if num_devices > jax.device_count():
            raise ValueError(
                f"Number of devices ({num_devices}) must be less than or equal to the number of available devices ({jax.device_count()})"
            )

    if training_schedule.BS % num_devices != 0:
        raise ValueError(f"Batch size ({training_schedule.BS}) must be divisible by number of devices ({num_devices})")

    log.info(
        f"Using {num_devices} devices with batch size {training_schedule.BS} ({training_schedule.BS // num_devices} per device)"
    )

    init_params = unified_model.init(
        init_key,
        jnp.ones([BS // num_devices, train_data.data.shape[1]]),
        jnp.ones([BS // num_devices, *train_data.features.shape[1:]], dtype=train_data.features.dtype)
        if train_data.features is not None
        else None,
        jnp.ones([BS // num_devices, 1]),
        training=False,
    )
    # for ema weight see https://arxiv.org/pdf/2011.13456 pages 26, we use same as in two for one 0.995
    state = EmaTrainState.create(apply_fn=unified_model.apply, params=init_params, tx=optimizer(1), ema_weight=0.995)

    log.debug("Initialized model with params: %s", state.params)
    log.info(f"Initialized model {unified_model} with {sum(x.size for x in jax.tree.leaves(state.params))} parameters.")

    if wandb["enabled"]:
        wandb_lib.define_metric("train/*", step_metric="train/epoch")
        wandb_lib.define_metric("val/*", step_metric="val/epoch")

    log.info(
        f"Training for {training_schedule.total()} epochs, which is about {train_data.data.shape[0] // BS * training_schedule.total()} steps"
    )

    state, train_losses, val_losses = load_and_train(
        training_schedule,
        unified_model,
        state,
        optimizer,
        train_data,
        val_data,
        num_devices,
        norm_factor,
        key,
        checkpoint_options,
        os.path.join(base_dir, "model"),
        model_load_dir,
        continue_from is not None and model_load_dir is not None,
        load_epoch,
        wandb["enabled"],
    )
    log.info("Finished training. Evaluating model...")

    for losses, prefix, filename in zip([train_losses, val_losses], ["Training", "Validation"], ["loss", "val_loss"]):
        if losses is None or len(losses) == 0:
            continue
        plt.figure(clear=True)
        plt.title(f"{prefix} - Loss")
        plt.plot(losses.sum(axis=-1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{out_dir}/{filename}.png", bbox_inches="tight")
        plt.close()

        for i, (loss, title) in enumerate(zip(losses.T, ["Diffusion Loss", "Vector FP Loss", "Scalar FP Loss"])):
            if jnp.any(jnp.abs(loss) > 1e-6):
                plt.figure(clear=True)
                plt.title(f"{prefix} - {title}")
                plt.plot(loss)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(f"{out_dir}/{filename}{i}.png", bbox_inches="tight")
                plt.close()

    if no_evaluation:
        log.info("Skipping evaluation")
        return

    if evaluation.seed is None:
        evaluation.seed = seed
    evaluate(
        unified_model,
        state.ema_params,
        dataset,
        evaluation,
        BS // num_devices,  # This uses only a single GPU for evaluation
        norm_factor,
        wandb["enabled"],
        out_dir,
    )


if __name__ == "__main__":
    create_dataset_store(store)
    create_weighting_store(store)
    create_optimizer_store(store)
    create_trainig_schedule_store(store)

    TrainConfig = builds(
        training_routine,
        checkpoint_options=builds(
            ocp.CheckpointManagerOptions,
            save_interval_steps=100,
            max_to_keep=3,
            create=True,
            cleanup_tmp_directories=True,
            enable_background_delete=True,
        ),
        load_dir=None,
        continue_from=None,
        load_epoch=None,
        num_devices=None,
        no_evaluation=False,
        evaluation=builds(EvaluationSettings, populate_full_signature=True),
        seed=1,
        wandb={"enabled": False, "project": "scoremd"},
        zen_meta={"model": None},
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"dataset": "???"},
            {"training_schedule": "vp_standard"},
            {"optimizer": "cosine"},
            {"weighting_function": "ranged"},
            {"override /hydra/job_logging": "colorlog"},
            {"override /hydra/hydra_logging": "colorlog"},
        ],
    )

    run_dir_path = "outputs/${dataset.name}/${now:%Y-%m-%d/%H-%M-%S}"

    slurm_job_id = slurm.get_slurm_job_id()
    if slurm_job_id is not None:
        run_dir_path += f"-{slurm_job_id}"

    store(TrainConfig, name="train")
    store(
        HydraConf(
            callbacks={
                "wandb": builds(WandbInitializer),
                "slurm": builds(SlurmInitializer),
            },
            run=RunDir(run_dir_path),
        )
    )

    store.add_to_hydra_store(overwrite_ok=True)

    zen(training_routine).hydra_main(
        config_name="train",
        version_base="1.3",
        config_path="config",
    )
