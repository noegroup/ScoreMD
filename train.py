from evaluate import EvaluationSettings, evaluate
from ffdiffusion.utils.file import (
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
from ffdiffusion.data.preprocess import CenterMolecule
import ffdiffusion.models as diffusion_models
import wandb as wandb_lib
from ffdiffusion.data.dataset import ToyDataset, Dataset, MuellerBrownSimulation
from ffdiffusion.models.mixture import MixtureOfModels
from ffdiffusion.training import load_and_train, TrainingSchedule
from ffdiffusion.training.train_state import EmaTrainState
from ffdiffusion.training.weighting import (
    plot_weighting_function,
)
from ffdiffusion.utils.hydra import SlurmInitializer, WandbInitializer
from ffdiffusion.utils.ranges import assert_range_continuity
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
    load_epoch: Optional[int],
    evaluation: EvaluationSettings,
    no_evaluation: bool,
    seed: int,
    wandb: dict,
):
    assert len(ranged_models) > 0, "At least one model is required"
    out_dir = f"{HydraConfig.get().runtime.output_dir}/out"
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
        os.path.join(HydraConfig.get().runtime.output_dir, "model"),
        os.path.join(load_dir, "model") if load_dir is not None else None,
        load_epoch,
        wandb["enabled"],
    )
    log.info("Finished training. Evaluating model...")

    for losses, prefix, filename in zip([train_losses, val_losses], ["Training", "Validation"], ["loss", "val_loss"]):
        if losses is None:
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
        BS // num_devices,  # TODO: we could also shard the data in evaluation
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
        load_epoch=None,
        num_devices=None,
        no_evaluation=False,
        evaluation=builds(EvaluationSettings, populate_full_signature=True),
        seed=1,
        wandb={"enabled": False, "project": "ffdiffusion"},
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

    store(TrainConfig, name="train")
    store(
        HydraConf(
            callbacks={
                "wandb": builds(WandbInitializer),
                "slurm": builds(SlurmInitializer),
            },
            run=RunDir("outputs/${dataset.name}/${now:%Y-%m-%d/%H-%M-%S}"),
        )
    )

    store.add_to_hydra_store(overwrite_ok=True)

    zen(training_routine).hydra_main(
        config_name="train",
        version_base="1.3",
        config_path="config",
    )
