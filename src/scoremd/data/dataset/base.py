import abc
from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp
import logging
from flax import struct

log = logging.getLogger(__name__)


class Datapoints(struct.PyTreeNode):
    data: jnp.ndarray
    features: Optional[jnp.ndarray]

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {self.data.ndim}D.")
        if self.features is not None:
            if self.features.ndim != 3:
                raise ValueError(f"Features must be 3D, got {self.features.ndim}D. (BS, num_atoms, num_features)")
            if self.data.shape[0] != self.features.shape[0]:
                raise ValueError(
                    f"First dimension of data ({self.data.shape[0]}) and features ({self.features.shape[0]}) must match."
                )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return Datapoints(self.data[idx], self.features[idx] if self.features is not None else None)


@dataclass
class Dataset:
    name: str
    sample_shape: Tuple[int, ...]
    kbT: float

    def __init__(self, name: str, sample_shape: Tuple[int, ...], kbT: float):
        self.name = name
        self.sample_shape = sample_shape
        self.kbT = kbT
        self._train: Optional[Datapoints] = None
        self._val: Optional[Datapoints] = None
        self._test: Optional[Datapoints] = None

    @abc.abstractmethod
    def _get_data(self) -> Tuple[Datapoints, Optional[Datapoints], Optional[Datapoints]]:
        raise NotImplementedError()

    def load_data(self):
        if self._train is None:
            self._train, self._val, self._test = self._get_data()

    def _std(self, data: jnp.ndarray) -> jnp.ndarray:
        return jnp.std(
            data
        )  # we don't want to compute std for each coordinate, because we assume that the data can be arbitrarily rotated

    @property
    def train(self) -> Datapoints:
        self.load_data()
        return self._train

    @property
    def val(self) -> Optional[Datapoints]:
        self.load_data()
        return self._val

    @property
    def test(self) -> Optional[Datapoints]:
        self.load_data()
        return self._test

    @property
    def std(self) -> jnp.ndarray:
        return self._std(self.train.data.reshape(self.train.data.shape[0], -1, 3))
