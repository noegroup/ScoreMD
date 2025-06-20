from functools import partial
import optax
import orbax.checkpoint as ocp
from jax.typing import ArrayLike
from typing import Callable, Tuple, Any, Optional
import jax.numpy as jnp
import jax
from tqdm import tqdm
import flax.linen as nn
import logging
from flax.core import FrozenDict
import wandb
import os
from orbax.checkpoint import args as ocp_args
from ffdiffusion.data.dataset.base import Datapoints
from ffdiffusion.training.schedule import TrainingSchedule
from ffdiffusion.training.train_state import EmaTrainState

log = logging.getLogger(__name__)

# We disable orbax logging
logging.getLogger("absl").setLevel(logging.WARNING)


def equal_weight(t, midpoint, t0=0.0, t1=1.0):
    """
    A time-weighting that gives the same weight to all time points left of midpoint and right of midpoint.
    """
    assert t0 <= midpoint <= t1
    t = t.reshape(-1)
    return jnp.where(t <= midpoint, (t1 - t0) / (midpoint - t0), (t1 - t0) / (t1 - midpoint)) / 2


def exponential_decay(t, n, t0=0.0, t1=1.0):
    """
    A time-weighting that decays exponentially from t0 to t1 with a rate of n. The higher n, the faster the decay.
    """
    t01 = (t - t0) / (t1 - t0)
    return (n * jnp.exp(-n * t01)) / (1 - jnp.exp(-n))  # the division is to normalize the integral to 1


def constant(_t):
    return 1.0


@partial(jax.jit, static_argnums=2)
def eval_model(
    data: ArrayLike,
    state: EmaTrainState,
    loss_fn: Callable[[FrozenDict[str, Any], ArrayLike, ArrayLike], ArrayLike],
    key: ArrayLike,
) -> ArrayLike:
    return jnp.array(loss_fn(state.ema_params, key, data))


def train(
    schedule: TrainingSchedule,
    model: nn.Module,
    state: EmaTrainState,
    optimizer: Callable[[int], optax.GradientTransformation],
    loaded_opt_state: Optional[dict],
    train_data: Datapoints,
    val_data: Optional[Datapoints],
    num_devices: int,
    norm_factor: jnp.ndarray,
    start_epoch: int,
    key: ArrayLike,
    checkpoint_manager: Optional[ocp.CheckpointManager] = None,
    enable_wandb: bool = False,
) -> Tuple[EmaTrainState, ArrayLike, Optional[ArrayLike]]:
    key, train_key = jax.random.split(key)
    losses = []
    with tqdm(
        schedule.train(
            state,
            model,
            optimizer,
            loaded_opt_state,
            start_epoch,
            train_data,
            val_data,
            num_devices,
            norm_factor,
            train_key,
        ),
        total=schedule.total(),
        initial=start_epoch,
    ) as pbar:
        val_losses = []
        for epoch, (state, loss, val_loss) in enumerate(pbar):
            info = schedule.parse_loss(loss, epoch, prefix="train/")
            info |= {"train/epoch": epoch}

            # losses.shape = (num_steps_in_epoch, num_loss_terms)
            losses.append(loss.mean(axis=0))  # we just want to store the mean loss

            if val_loss is not None:
                info |= schedule.parse_loss(val_loss, epoch, prefix="val/")
                info |= {"val/epoch": epoch}

                val_losses.append(val_loss.mean(axis=0))

            if enable_wandb:
                wandb.log(info)

            pbar_fields = ["loss", "val_loss"]
            postfix = {i: f"{info[i]:.3f}" for i in pbar_fields if i in info}

            pbar.set_postfix(postfix)

            if checkpoint_manager is not None and checkpoint_manager.should_save(epoch + 1):
                log.info(f"Saving checkpoint at epoch {epoch + 1}")
                checkpoint_manager.save(
                    epoch + 1,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardSave(state.params),
                        ema_params=ocp.args.StandardSave(state.ema_params),
                        opt_state=ocp.args.PyTreeSave(state.opt_state),
                    ),
                )

    return state, jnp.array(losses), jnp.array(val_losses) if len(val_losses) > 0 else None


def load_and_train(
    schedule: TrainingSchedule,
    model: nn.Module,
    state: EmaTrainState,
    optimizer: Callable[[int], optax.GradientTransformation],
    train_data: Datapoints,
    val_data: Optional[Datapoints],
    num_devices: int,
    norm_factor: jnp.ndarray,
    key: ArrayLike,
    checkpoint_options: ocp.CheckpointManagerOptions,
    checkpoint_directory: str,
    load_directory: Optional[str] = None,
    load_epoch: Optional[int] = None,
    wandb: bool = False,
) -> Tuple[EmaTrainState, ArrayLike, Optional[ArrayLike]]:
    if load_epoch is not None and load_directory is None:
        raise ValueError("load_epoch has been specified, but load_directory is None")

    loaded_opt_state = None

    # We always want to store the last model
    cumulative_epochs = 0
    for epochs in schedule.important_epochs():
        cumulative_epochs += epochs
        if cumulative_epochs not in checkpoint_options.save_on_steps:
            checkpoint_options.save_on_steps = list(checkpoint_options.save_on_steps) + [cumulative_epochs]

    start_epoch = 0
    if load_directory is not None:
        logging.info(f"Trying to load model from {load_directory}")
        with ocp.CheckpointManager(os.path.abspath(os.path.expanduser(load_directory))) as checkpoint_manager:
            if checkpoint_manager.latest_step() is None:
                logging.error(f"No checkpoint found for model in directory {load_directory}.")
                raise ValueError(f"No checkpoint found for model in directory {load_directory}.")

            start_epoch = load_epoch if load_epoch is not None else checkpoint_manager.latest_step()
            log.info(f"Restoring model from {load_directory} at step {start_epoch}")

            # this is hacky, but we don't know the opt state ...
            try:
                restored = checkpoint_manager.restore(start_epoch)
                loaded_opt_state = restored.opt_state
            except Exception as e:
                log.error(f"Error restoring model: {e}")
                log.info("Trying to load model without opt state")

                from jax.sharding import NamedSharding, PartitionSpec

                mesh = jax.make_mesh((num_devices,), ("data",))
                named_sharding = NamedSharding(mesh, PartitionSpec())

                # Build abstract trees
                abstract_params = jax.device_put(state.params, named_sharding)
                abstract_ema = jax.device_put(state.ema_params, named_sharding)

                restored = checkpoint_manager.restore(
                    start_epoch,
                    args=ocp_args.Composite(
                        params=ocp_args.StandardRestore(abstract_params),
                        ema_params=ocp_args.StandardRestore(abstract_ema),
                    ),
                )

            state = EmaTrainState.create(
                apply_fn=state.apply_fn,
                params=restored.params,
                ema_params=restored.ema_params,
                tx=state.tx,
                ema_weight=state.ema_weight,
            )

            logging.info(f"Restored model from {load_directory} at step {start_epoch}")

    log.info("Training models...")
    with ocp.CheckpointManager(
        os.path.abspath(os.path.expanduser(checkpoint_directory)), options=checkpoint_options
    ) as checkpoint_manager:
        state, losses, val_losses = train(
            schedule,
            model,
            state,
            optimizer,
            loaded_opt_state,
            train_data,
            val_data,
            num_devices,
            norm_factor,
            start_epoch,
            key,
            checkpoint_manager,
            wandb,
        )

    log.info("Finished training models")
    # If there is no training involved, we return the losses as is
    average_losses = losses.sum(axis=1) if len(losses) > 0 else losses
    average_val_losses = val_losses.sum(axis=1) if val_losses is not None and len(val_losses) > 0 else val_losses

    return state, average_losses, average_val_losses
