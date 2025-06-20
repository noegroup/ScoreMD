from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, Sequence
import flax.linen as nn
from jax.typing import ArrayLike
from flax.core import FrozenDict


@dataclass
class RangedLoss:
    loss: Callable[
        [nn.Module, Callable[[ArrayLike], ArrayLike], Sequence[int]],
        Callable[[FrozenDict[str, Any], ArrayLike, Optional[ArrayLike], ArrayLike], ArrayLike],
    ]  # This is a function that takes a flax nn.Module and returns a loss function
    time_weighting: Callable[[ArrayLike], ArrayLike]
    range: Tuple[float, float] = (1.0, 0.0)
    trainable: Sequence[int] = ()  # Indices of trainable models. Empty list means that all models are trained
    evaluated_models: Sequence[
        int
    ] = ()  # Indices of models that are evaluated. Empty list means that all models are evaluated.
