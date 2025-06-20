from dataclasses import dataclass
from typing import List, Optional, Sequence, Callable, Tuple
import flax.linen as nn
import jax.numpy as jnp
from ffdiffusion.models.utils import value_and_grad_sum
from . import BaseDiffusionModel, EnergyModel, ModelInfo
from jax.typing import ArrayLike
import jax
from ..data.dataset import Dataset


@dataclass
class MLPModelInfo(ModelInfo):
    """This is a wrapper because directly instantiating a list of models does not work nicely with hydra"""

    hidden_dims: List[int]
    activation: str
    potential: bool

    def build(
        self, dataset: Dataset, t0: float, t1: float, rescale_time: bool, clip_time: bool, norm_factor: jnp.ndarray
    ) -> nn.Module:
        activation = getattr(nn, self.activation)
        if self.potential:
            return PotentialScoreModel(t0, t1, rescale_time, clip_time, self.hidden_dims, activation)
        return ScoreModel(t0, t1, rescale_time, clip_time, self.hidden_dims, activation)


@dataclass
class ScoreModel(BaseDiffusionModel, nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[ArrayLike], ArrayLike]

    @nn.compact
    def _forward(self, x, features, t, training):
        h = self._prepare_input(x, features, t)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = self.activation(h)

        return nn.Dense(x.shape[1])(h)


@dataclass
class PotentialScoreModel(BaseDiffusionModel, EnergyModel, nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[ArrayLike], ArrayLike]

    def _forward(self, x, features, t, training):
        return -jax.grad(lambda x: self._energy_forward(x, features, t, training).sum())(x)

    @nn.compact
    def _energy_forward(self, x, features, t, training):
        h = self._prepare_input(x, features, t)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = self.activation(h)

        return nn.Dense(1)(h)

    def log_q(self, x: jnp.ndarray, features: Optional[jnp.ndarray], t: jnp.ndarray, training: bool) -> jnp.ndarray:
        x, features, t, _ = BaseDiffusionModel._reshape_input(x, features, t)
        return -self._energy_forward(x, features, t, training)

    def log_q_and_score(
        self, x: jnp.ndarray, features: Optional[jnp.ndarray], t: jnp.ndarray, training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, features, t, _ = BaseDiffusionModel._reshape_input(x, features, t)

        val, grad = value_and_grad_sum(self._energy_forward, x, features, t, training)
        return -val, -grad
