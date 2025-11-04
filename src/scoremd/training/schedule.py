from dataclasses import dataclass
from functools import partial
from typing import Mapping, Optional, Sequence, Tuple, Callable, Any, Literal, Generator, List
import jax
import jax.numpy as jnp
import abc
from flax import traverse_util
from jax.typing import ArrayLike
from flax.core import FrozenDict
import logging
from itertools import accumulate
import optax
from scoremd.data.dataset.base import Datapoints
from scoremd.loss import RangedLoss
from scoremd.models.mixture import MixtureOfModels
from scoremd.training.train_state import EmaTrainState
from scoremd.utils.parallel import generate_sharding
from scoremd.utils.ranges import assert_range_continuity

log = logging.getLogger(__name__)


def IDENTITY(x: ArrayLike, _: Optional[ArrayLike], key: ArrayLike) -> ArrayLike:
    return x


def special_epoch_every_n_epochs(n: int) -> Callable[[int], bool]:
    """Returns a function that returns True every n epochs."""
    return lambda epoch: epoch % n == 0


def always_special(_: int) -> bool:
    return True


@dataclass
class TrainingSchedule(abc.ABC):
    """
    This schedule provides an iterator to train a model.
    Since we want to support training a mixture of models, this class has quite a few abstract method and
    sometimes functions are passed.
    The idea is that the class itself is frozen, but the iterable object can change the training function over time.
    i.e., start with training 200 epochs the first model, then 200 epochs the second model, then go back to the first.
    """

    losses: List[RangedLoss]
    BS: int
    BS_factor: int
    augment: Callable[[ArrayLike, Optional[ArrayLike], ArrayLike], ArrayLike]
    # A function that takes the epoch number and returns whether it is a special epoch.
    # For our purposes, a special epoch is one where we do something different, like FP-loss.
    # This parameter will be passed to the loss functions.
    is_special_epoch: Callable[[int], bool]
    validation_every: int

    def __init__(
        self,
        losses: Sequence[RangedLoss],
        BS: int,
        BS_factor: int,
        augment: Callable[[ArrayLike, ArrayLike], ArrayLike],
        is_special_epoch: Callable[[int], bool],
        validation_every: int,
    ):
        losses = sorted(losses, key=lambda x: x.range[0], reverse=True)

        log.info(f"Specified {len(losses)} loss(es) with range(s) {' -> '.join([str(rl) for rl in losses])}")
        assert_range_continuity([rl.range for rl in losses])

        self.losses = losses
        self.BS = BS
        self.BS_factor = BS_factor
        self.augment = augment
        self.is_special_epoch = is_special_epoch
        self.validation_every = validation_every

    @staticmethod
    def _shuffle(datapoints: Datapoints, BS: int, key: jnp.ndarray) -> jnp.ndarray:
        """Shuffles the data and reshapes it into a 2D array of shape (steps_per_epoch, BS)."""
        steps_per_epoch = min(datapoints.data.shape[0] // BS, datapoints.data.shape[0])

        perms = jax.random.permutation(key, datapoints.data.shape[0])
        perms = perms[: steps_per_epoch * BS]  # Skip incomplete batches
        return perms.reshape((steps_per_epoch, BS))

    @staticmethod
    def _summed(loss_fn):
        """Wraps the specified function to compute the sum of all outputs while returning all outputs as a second value."""

        def wrapper(*args, **kwargs):
            ret = jnp.array(loss_fn(*args, **kwargs))
            return jnp.sum(ret), jnp.array(ret)

        return wrapper

    def _mixed_t_train_step(
        self,
        loss_fns: Sequence[Callable[[FrozenDict[str, Any], Tuple[ArrayLike, ArrayLike]], ArrayLike]],
        state: EmaTrainState,
        batch: ArrayLike,
        features: Optional[ArrayLike],
        ts: ArrayLike,
        is_special_epoch: bool,
        validation: bool,
        key: ArrayLike,
    ) -> Tuple[EmaTrainState, ArrayLike]:
        """Use this train step when you have no clue about the ordering of t and it could be that each loss function needs to be called on it."""

        def merged_loss(
            params: FrozenDict[str, Any],
            key: ArrayLike,
            batch: ArrayLike,
            features: Optional[ArrayLike],
            ts: ArrayLike,
            is_special_epoch: bool,
        ) -> Tuple[ArrayLike, ArrayLike]:
            # return loss_fns[0](params, key, batch, ts)
            sum_loss, sum_aux = 0.0, jnp.array([0.0, 0.0, 0.0])
            for (t1, t0), loss_fn in zip(self.training_ranges(), loss_fns):
                loss_match = jnp.any((ts < t1) & (ts > t0))
                cur_loss, cur_aux = jax.lax.cond(
                    loss_match,
                    lambda: loss_fn(params, key, batch, features, ts, is_special_epoch),
                    lambda: (jnp.sum(jnp.array([0.0])), jnp.array([0.0, 0.0, 0.0])),
                )

                sum_loss += loss_match * cur_loss
                sum_aux += loss_match * cur_aux
            return sum_loss, sum_aux

        return TrainingSchedule._train_step_single_loss(
            merged_loss, state, batch, features, ts, is_special_epoch, validation, key
        )

    @staticmethod
    def _train_step_single_loss(
        loss_fn: Callable[[FrozenDict[str, Any], ArrayLike, ArrayLike, ArrayLike, bool], Tuple[ArrayLike, ArrayLike]],
        state: EmaTrainState,
        batch: ArrayLike,
        features: Optional[ArrayLike],
        ts: ArrayLike,
        is_special_epoch: bool,
        validation: bool,
        key: ArrayLike,
    ) -> Tuple[EmaTrainState, ArrayLike]:
        if validation:
            _, loss = loss_fn(state.params, key, batch, features, ts, is_special_epoch, False)
            return state, loss
        (_, loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, key, batch, features, ts, is_special_epoch, True
        )

        state = state.apply_gradients(grads=grads)
        return state, loss

    def _train_epoch(
        self,
        sample_ts: Callable[[jnp.ndarray], Sequence[Tuple[jnp.ndarray, int]]],
        reweight_losses: Callable[[Sequence[jnp.ndarray]], jnp.ndarray],
        loss_fns: Sequence[Callable[[FrozenDict[str, Any], Tuple[ArrayLike, ArrayLike]], ArrayLike]],
        state: EmaTrainState,
        datapoints: Datapoints,
        norm_factor: jnp.ndarray,
        is_special_epoch: bool,
        validation: bool,
        key: jnp.ndarray,
        data_sharding: Optional[jax.sharding.Sharding],
        replicated_sharding: Optional[jax.sharding.Sharding],
    ) -> Tuple[EmaTrainState, jnp.ndarray]:
        """Trains the model for one epoch."""
        key, shuffle_key = jax.random.split(key)
        perms = TrainingSchedule._shuffle(datapoints, self.BS, shuffle_key)

        summed_loss_fns = [TrainingSchedule._summed(loss) for loss in loss_fns]

        if replicated_sharding is not None:
            state = jax.lax.with_sharding_constraint(state, replicated_sharding)

        def train_step_scan(carry, perm):
            key, state = carry
            key, iter_key, augment_key = jax.random.split(key, 3)

            current_data = datapoints.data[perm, ...]
            current_features = datapoints.features[perm, ...] if datapoints.features is not None else None
            if data_sharding is not None:
                current_data = jax.lax.with_sharding_constraint(current_data, data_sharding)
                if current_features is not None:
                    current_features = jax.lax.with_sharding_constraint(current_features, data_sharding)

            # augment the data (e.g., random rotations)
            current_data = self.augment(current_data, current_features, augment_key)
            current_data *= norm_factor

            cur_losses = []
            for ts, possible_loss in sample_ts(iter_key):
                if data_sharding is not None:
                    ts = jax.lax.with_sharding_constraint(ts, data_sharding)
                key, step_key = jax.random.split(key)
                if possible_loss >= 0:  # we can be certain that a single loss function is responsible
                    state, cur_loss = TrainingSchedule._train_step_single_loss(
                        summed_loss_fns[possible_loss],
                        state,
                        current_data,
                        current_features,
                        ts,
                        is_special_epoch,
                        validation,
                        step_key,
                    )
                else:
                    state, cur_loss = self._mixed_t_train_step(
                        summed_loss_fns,
                        state,
                        current_data,
                        current_features,
                        ts,
                        is_special_epoch,
                        validation,
                        step_key,
                    )
                cur_losses.append(cur_loss)

            return (key, state), reweight_losses(jnp.array(cur_losses))

        if validation:
            log.info(f"Validating {len(perms)} batches in each validation epoch.")
        else:
            log.info(f"Training {len(perms)} batches in each epoch.")
        (_, state), losses = jax.lax.scan(train_step_scan, (key, state), perms)
        losses = jnp.array(losses)

        # apply SEMA https://arxiv.org/abs/2402.09240, similar to https://arxiv.org/abs/2312.07551
        state = state.replace(params=state.ema_params)

        return state, losses

    def _train_n_epochs(
        self,
        train_single_epoch: Callable[
            [EmaTrainState, Datapoints, bool, bool, jnp.ndarray], Tuple[EmaTrainState, jnp.ndarray]
        ],
        data: Datapoints,
        val_data: Optional[Datapoints],
        state: EmaTrainState,
        start_epoch: int,
        epochs: int,
        key,
    ) -> Generator[Tuple[EmaTrainState, jnp.ndarray, Optional[jnp.ndarray]], None, None]:
        """This is a helper function that trains a certain number of epochs and takes care of occuring nans."""
        MAX_RETRIES = 20
        stuck_at = (start_epoch, 0)  # epoch, retry_count
        retry_key = None

        # ensure that the keys is the same for all epochs (even after loading)
        for _ in range(start_epoch):
            key, _, _ = jax.random.split(key, 3)

        epoch = start_epoch
        while epoch < epochs:
            stuck_epoch, stuck_count = stuck_at

            if stuck_epoch == epoch and stuck_count > 0:  # we are stuck, so we split the retry key
                retry_key, iter_key = jax.random.split(retry_key)
            else:
                key, retry_key, iter_key = jax.random.split(key, 3)

            special_epoch = self.is_special_epoch(epoch)

            # we sometimes face the issue that the loss is nan
            # the reason is that the gradient computation can fail (despite no nan values involved)
            # we haven't really figured out the cause, but usually just retrying the epoch fixes the problem
            # however, sometimes this doesn't help, so we can also try to disable FP loss (and any other behavior of special_epoch)
            # the ideal solution would be to retry the epoch with a float64 precision, but there is no trivial way to change jax's precision for a single run
            # we would need to set JAX_ENABLE_X64, but then all arrays will be float64, so we would need to specify the dtype everywhere.
            # this would make it really annoying to use external libraries and would also make the code less clean.
            # so we just try to mark the epoch as non-special (i.e., turn off FP-Loss) and hope for the best.
            if stuck_epoch == epoch and stuck_count > MAX_RETRIES // 2:
                if special_epoch:
                    log.warning(
                        "Taking more drastic measures: marking this epoch as non-special to disable FP-Loss calculations."
                    )
                    special_epoch = False

            new_state, loss = train_single_epoch(state, data, False, special_epoch, iter_key)

            # ensure that we don't get stuck in a nan loop
            if jnp.isnan(loss).any():
                if stuck_epoch == epoch:
                    stuck_count += 1
                else:  # previous epoch was fine
                    stuck_count = 1
                stuck_at = (epoch, stuck_count)

                if stuck_count < MAX_RETRIES:
                    log.warning(f"Loss contains nan. Retrying epoch {epoch} ({stuck_count}/{MAX_RETRIES}) ...")
                else:
                    log.error(f"Loss is still nan after {stuck_count} retries. Stopping training ...")
                    raise ValueError(f"Loss is still nan after {stuck_count} retries. Stopping training ...")

            else:  # continue to next epoch
                epoch += 1
                state = new_state  # only update the parameters if there was no nan
                val_loss = None
                if val_data is not None and epoch % self.validation_every == 0:
                    _, val_loss = train_single_epoch(state, val_data, True, special_epoch, iter_key)
                yield state, loss, val_loss

    @staticmethod
    def restore_optimizer_state(opt_state: optax.OptState, restored: Mapping[str, ...]) -> optax.OptState:
        """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(opt_state), jax.tree_util.tree_leaves(restored)
        )

    @abc.abstractmethod
    def train(
        self,
        state: EmaTrainState,
        model: MixtureOfModels,
        optimizer: Callable[[int], optax.GradientTransformation],
        loaded_opt_state: Optional[dict],
        start_epoch: int,
        train_data: Datapoints,
        val_data: Optional[Datapoints],
        num_devices: int,
        norm_factor: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Generator[Tuple[EmaTrainState, jnp.ndarray], None, None]:
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_loss(self, loss: jnp.ndarray, epoch: int, prefix: str = "") -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def total(self) -> int:
        """Returns the total number of epochs."""
        raise NotImplementedError()

    @abc.abstractmethod
    def important_epochs(self) -> Sequence[int]:
        raise NotImplementedError()

    def training_ranges(self):
        return [rl.range for rl in self.losses]


@dataclass
class AllAtOnce(TrainingSchedule):
    """Trains all models at the same time, and uses the same time range for all models."""

    epochs: int
    # If True, a single batch contains a mixture of randomly sampled time points from all training ranges.
    # Otherwise, each batch contains time points from a single training range. This also changes the training distribution if the training ranges are not equally sized.
    mixed: bool

    def __init__(
        self,
        losses: Sequence[RangedLoss],
        BS: int,
        BS_factor: int,
        epochs: int,
        mixed: bool,
        augment: Callable[[ArrayLike, ArrayLike], ArrayLike] = IDENTITY,
        is_special_epoch: Callable[[int], bool] = always_special,
        validation_every: int = 1,
    ):
        super().__init__(losses, BS, BS_factor, augment, is_special_epoch, validation_every)
        self.epochs = epochs
        self.mixed = mixed

    def _sample_ts(self, key: jnp.ndarray) -> Generator[Tuple[jnp.ndarray, int], None, None]:
        if self.mixed:
            t0 = min([r[1] for r in self.training_ranges()])
            t1 = max([r[0] for r in self.training_ranges()])
            log.info(f"Mixed training between range {t0} and {t1}")
            yield jax.random.uniform(key, (self.BS,), minval=t0, maxval=t1), -1
        else:
            log.info(f"Training each range separately: {self.training_ranges()}")
            for i, (t1, t0) in enumerate(self.training_ranges()):
                key, t_key = jax.random.split(key)
                yield jax.random.uniform(t_key, (self.BS,), minval=t0, maxval=t1), i

    def _reweight_losses(self, loss: jnp.ndarray) -> jnp.ndarray:
        if self.mixed:
            # The samples are already normalized
            return jnp.mean(loss, axis=0).reshape(1, -1)

        weights = jnp.array([t1 - t0 for t1, t0 in self.training_ranges()])
        return loss * weights.reshape(-1, 1) / jnp.sum(weights)

    def parse_loss(self, loss: jnp.ndarray, epoch: int, prefix: str = "") -> dict:
        info = {
            f"{prefix}loss": jnp.mean(loss, axis=0).sum(),
            f"{prefix}diffusion_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[0],
            f"{prefix}vector_fp_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[1],
            f"{prefix}scalar_fp_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[2],
        }

        info |= {f"{prefix}loss_{i}": jnp.mean(loss, axis=0)[i].sum() for i in range(loss.shape[1])}
        info |= {f"{prefix}diffusion_loss_{i}": jnp.mean(loss, axis=0)[i][0] for i in range(loss.shape[1])}
        info |= {f"{prefix}vector_fp_loss_{i}": jnp.mean(loss, axis=0)[i][1] for i in range(loss.shape[1])}
        info |= {f"{prefix}scalar_fp_loss_{i}": jnp.mean(loss, axis=0)[i][2] for i in range(loss.shape[1])}

        return info

    def train(
        self,
        state: EmaTrainState,
        model: MixtureOfModels,
        optimizer: Callable[[int], optax.GradientTransformation],
        loaded_opt_state: Optional[dict],
        start_epoch: int,
        train_data: Datapoints,
        val_data: Optional[Datapoints],
        num_devices: int,
        norm_factor: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Generator[Tuple[EmaTrainState, jnp.ndarray], None, None]:
        loss_fns = [rl.loss(model, rl.time_weighting) for rl in self.losses]

        if self.BS_factor > 1:

            def wrapped_optimizer(num_steps):
                return optax.MultiSteps(optimizer(num_steps), every_k_schedule=self.BS_factor)
        else:
            wrapped_optimizer = optimizer

        data_sharding, replicated_sharding = generate_sharding(num_devices)

        @partial(jax.jit, static_argnums=(2, 3))
        def step(
            state: EmaTrainState, data: Datapoints, validation: bool, is_special_epoch: bool, key: jax.random.PRNGKey
        ):
            return self._train_epoch(
                self._sample_ts,
                self._reweight_losses,
                loss_fns,
                state,
                data,
                norm_factor,
                is_special_epoch,
                validation,
                key,
                data_sharding,
                replicated_sharding,
            )

        tx = wrapped_optimizer(self.epochs * train_data.data.shape[0] // (self.BS * self.BS_factor))
        dummy_opt_state = tx.init(state.params)

        state = EmaTrainState.create(
            apply_fn=state.apply_fn,
            params=state.params,
            ema_params=state.ema_params,
            ema_weight=state.ema_weight,
            tx=tx,
            opt_state=AllAtOnce.restore_optimizer_state(dummy_opt_state, loaded_opt_state) if start_epoch > 0 else None,
        )
        state = state.replace(step=start_epoch * train_data.data.shape[0] // (self.BS * self.BS_factor))

        for state, loss, val_loss in self._train_n_epochs(
            step, train_data, val_data, state, start_epoch, self.epochs, key
        ):
            yield state, loss, val_loss

    def important_epochs(self) -> Sequence[int]:
        return [self.epochs]

    def total(self) -> int:
        return self.epochs


@dataclass
class OneAfterAnother(TrainingSchedule):
    epochs: Sequence[int]
    order: Literal["HTL", "LTH"] = "HTL"  # Either train high to low (HTL) or low to high (LTH)

    def __init__(
        self,
        losses: Sequence[RangedLoss],
        BS: int,
        BS_factor: int,
        epochs: int | Sequence[int],
        order: Literal["HTL", "LTH"] = "HTL",
        augment: Callable[[ArrayLike, ArrayLike], ArrayLike] = IDENTITY,
        is_special_epoch: Callable[[int], bool] = always_special,
        validation_every: int = 1,
    ):
        super().__init__(losses, BS, BS_factor, augment, is_special_epoch, validation_every)
        self.epochs = [epochs] if isinstance(epochs, int) else epochs
        self.order = order

    def __post_init__(self):
        assert len(self.epochs) == 1 or len(self.epochs) == len(self.losses)

    def _expand_epochs(self):
        return list(self.epochs) * len(self.losses) if len(self.epochs) == 1 else self.epochs

    @staticmethod
    def _create_mask(params, trainable):
        """Label params for current model as 'trainable', others as 'frozen'"""

        def traverse(path, _):
            if len(trainable) == 0:
                return "trainable"
            for t in trainable:  # all possible states that are trainable
                if t in path:
                    return "trainable"
            return "frozen"

        return traverse_util.path_aware_map(traverse, params)

    @staticmethod
    def _has_trainable_params(params, trainable):
        """Check if a model has trainable parameters"""
        if len(trainable) == 0:
            return True

        for t in trainable:
            if any(t in path for path in params.keys()):
                return True
        return False

    @staticmethod
    def _idx_to_identifiers(trainable):
        return [f"models_{mdl}" for mdl in trainable]

    def train(
        self,
        state: EmaTrainState,
        model: MixtureOfModels,
        optimizer: Callable[[int], optax.GradientTransformation],
        loaded_opt_state: Optional[dict],
        start_epoch: int,
        train_data: Datapoints,
        val_data: Optional[Datapoints],
        num_devices: int,
        norm_factor: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Generator[Tuple[EmaTrainState, jnp.ndarray], None, None]:
        trainable = [rl.trainable for rl in self.losses]

        if self.BS_factor > 1:

            def wrapped_optimizer(num_steps):
                return optax.MultiSteps(optimizer(num_steps), every_k_schedule=self.BS_factor)
        else:
            wrapped_optimizer = optimizer

        # check that we have trainable params
        for rl in self.losses:
            assert OneAfterAnother._has_trainable_params(
                state.params["params"], OneAfterAnother._idx_to_identifiers(rl.trainable)
            ), f"Loss {rl} does not lead to any trainable parameters. Is there no model with id {rl.trainable}?"

            for m in rl.evaluated_models:
                if m < 0 or m >= len(model.models):
                    raise ValueError(f"Evaluated model {m} is out of range. Must be between 0 and {len(model.models)}")

        data_sharding, replicated_sharding = generate_sharding(num_devices)

        epochs = self._expand_epochs()
        loss_fns = [
            rl.loss(
                model,
                rl.time_weighting,
                rl.evaluated_models if len(rl.evaluated_models) > 0 else range(len(model.models)),
            )
            for rl in self.losses
        ]

        if self.order == "HTL":
            iterate = zip(epochs, loss_fns, self.training_ranges(), trainable, range(len(self.losses)))
        else:
            iterate = zip(
                reversed(epochs),
                reversed(loss_fns),
                reversed(self.training_ranges()),
                reversed(trainable),
                reversed(range(len(self.losses))),
            )

        for num_epochs, loss_fn, (t1, t0), trainable, cur_idx in iterate:
            if start_epoch >= num_epochs:  # we have already performed this loss
                start_epoch -= num_epochs
                continue

            original_optimizer = wrapped_optimizer(num_epochs * train_data.data.shape[0] // (self.BS * self.BS_factor))

            @partial(jax.jit, static_argnums=(2, 3))
            def step(
                state: EmaTrainState,
                data: Datapoints,
                validation: bool,
                is_special_epoch: bool,
                key: jax.random.PRNGKey,
            ):
                def sample_ts(key):
                    yield jax.random.uniform(key, (self.BS,), minval=t0, maxval=t1), 0

                def reweight_losses(loss):
                    return jnp.mean(loss, axis=0).reshape(1, -1)

                return self._train_epoch(
                    sample_ts,
                    reweight_losses,
                    [loss_fn],
                    state,
                    data,
                    norm_factor,
                    is_special_epoch,
                    validation,
                    key,
                    data_sharding,
                    replicated_sharding,
                )

            partition_optimizers = {"trainable": original_optimizer, "frozen": optax.set_to_zero()}
            param_partitions = OneAfterAnother._create_mask(
                state.params, OneAfterAnother._idx_to_identifiers(trainable)
            )
            tx = optax.multi_transform(partition_optimizers, param_partitions)
            dummy_opt_state = tx.init(state.params)

            # Freeze all other models in state
            # New optimizer parameters are initialized here
            state = EmaTrainState.create(
                apply_fn=state.apply_fn,
                params=state.params,
                ema_weight=state.ema_weight,
                ema_params=state.ema_params,
                tx=tx,
                opt_state=OneAfterAnother.restore_optimizer_state(dummy_opt_state, loaded_opt_state)
                if start_epoch > 0
                else None,
            )
            state = state.replace(step=start_epoch * train_data.data.shape[0] // (self.BS * self.BS_factor))

            for state, loss, val_loss in self._train_n_epochs(
                step, train_data, val_data, state, start_epoch, num_epochs, key
            ):
                yield state, loss, val_loss

            start_epoch = 0

    def parse_loss(self, loss: jnp.ndarray, epoch: int, prefix: str = "") -> dict:
        epochs = self._expand_epochs()
        epoch_ranges = list(accumulate(epochs))
        # find the idx of the current epoch
        idx = next(i for i, v in enumerate(epoch_ranges) if v > epoch)

        info = {
            f"{prefix}loss": jnp.mean(loss, axis=0).sum(),
            f"{prefix}diffusion_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[0],
            f"{prefix}vector_fp_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[1],
            f"{prefix}scalar_fp_loss": jnp.sum(jnp.mean(loss, axis=0), axis=0)[2],
        }

        info |= {
            f"{prefix}loss_{idx}": jnp.mean(loss, axis=0).sum(),
            f"{prefix}diffusion_loss_{idx}": jnp.sum(jnp.mean(loss, axis=0), axis=0)[0],
            f"{prefix}vector_fp_loss_{idx}": jnp.sum(jnp.mean(loss, axis=0), axis=0)[1],
            f"{prefix}scalar_fp_loss_{idx}": jnp.sum(jnp.mean(loss, axis=0), axis=0)[2],
        }

        return info

    def total(self) -> int:
        return sum(self._expand_epochs())

    def important_epochs(self) -> Sequence[int]:
        return self._expand_epochs()
