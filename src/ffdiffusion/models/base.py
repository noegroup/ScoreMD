import jax.numpy as jnp
from dataclasses import dataclass
import abc
from typing import Optional, Tuple
import flax.linen as nn

from ..data.dataset import Dataset


@dataclass
class BaseDiffusionModel(abc.ABC):
    """A base class for all diffusion models defined on (t0, t1).
    It takes care of the diffusion time and also uses some sinusoidal features.
    """

    t0: float
    t1: float
    rescale_time: bool
    clip_time: bool

    def _prepare_input(self, x, features, t, concat=True):
        if self.clip_time:
            t = jnp.clip(t, self.t0, self.t1)

        t_features = [
            t - (self.t1 - self.t0) / 2,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t),
        ]

        if self.rescale_time:
            if self.t0 - 0.0 >= 1e-6 or self.t1 - 1.0 >= 1e-6:
                # only add these features, if either t0 or t1 is different from 0 or 1.
                # in this case, we allow the model to learn either from the absolute time or from the relative time.
                t01 = (t - self.t0) / (self.t1 - self.t0)
                t_features += [
                    jnp.cos(2 * jnp.pi * t01),
                    jnp.sin(2 * jnp.pi * t01),
                    -jnp.cos(4 * jnp.pi * t01),
                ]

        t = jnp.concatenate(t_features, axis=-1)
        if concat:
            if features is not None:
                return jnp.concatenate([x, features.reshape(features.shape[0], -1), t], axis=-1)
            return jnp.concatenate([x, t], axis=-1)
        return x, features, t

    @staticmethod
    def _reshape_input(x, features, t):
        assert x.ndim == 1 or x.ndim == 2
        original_shape = x.shape

        if x.ndim != 2:
            x = x.reshape(1, -1)
        if features is not None:
            if features.ndim != 3:
                assert features.ndim == 2, "Features must have either 2 or 3 dimensions"
                features = features.reshape(1, features.shape[0], features.shape[1])
        t = t.reshape(x.shape[0], 1)

        return x, features, t, original_shape

    def __call__(self, x, features, t, training):
        x, features, t, original_shape = BaseDiffusionModel._reshape_input(x, features, t)

        out = self._forward(x, features, t, training)
        return out.reshape(original_shape)  # ensure that we return the same shape as the original input

    @abc.abstractmethod
    def _forward(self, x, features, t, training):
        raise NotImplementedError

    def force(self, x, features, t, training):
        return self(x, features, t * jnp.ones((x.shape[0], 1)), training)


class EnergyModel(abc.ABC):
    """A model that can compute the energy of a given configuration.
    It is assumed that the energy is given by the negative log-probability of the model.
    If you are not sure if your model satisfies this, you have to override supports_energy().
    """

    def energy(self, x: jnp.ndarray, features: Optional[jnp.ndarray], t: jnp.ndarray, training: bool) -> jnp.ndarray:
        return -self.log_q(x, features, t * jnp.ones((x.shape[0], 1)), training)

    @abc.abstractmethod
    def log_q(self, x: jnp.ndarray, features: Optional[jnp.ndarray], t: jnp.ndarray, training: bool) -> jnp.ndarray:
        raise NotImplementedError

    def log_q_and_score(
        self, x: jnp.ndarray, features: Optional[jnp.ndarray], t: jnp.ndarray, training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def supports_energy(self):
        return True


@dataclass
class ModelInfo(abc.ABC):
    """This is a wrapper because directly instantiating a list of models does not work nicely with hydra"""

    @abc.abstractmethod
    def build(
        self, dataset: Dataset, t0: float, t1: float, rescale_time: bool, clip_time: bool, norm_factor: jnp.ndarray
    ) -> nn.Module:
        raise NotImplementedError


@dataclass
class RangedModel:
    """
    A model that is defined for a specific range of diffusion times.
    This is a wrapper that can build a model for a given range of times.
    """

    model: ModelInfo
    range: Tuple[float, float] = (1.0, 0.0)  # t1, t0, t1 > t0
    rescale_time: bool = True  # Whether to rescale the time to [0, 1]
    clip_time: bool = True  # Whether to clip the time within the range when doing a forward pass

    def build(self, dataset: Dataset, norm_factor: jnp.ndarray) -> nn.Module:
        t1, t0 = self.range
        return self.model.build(dataset, t0, t1, self.rescale_time, self.clip_time, norm_factor)
