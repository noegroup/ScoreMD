from functools import partial
from typing import Callable, Tuple
from typing_extensions import Literal
from ffdiffusion.data.dataset import Dataset
import jax.numpy as jnp
import logging

log = logging.getLogger(__name__)


class Preprocessor:
    """
    A data preprocessor that can handle both preprocessing and postprocessing.
    These method prepare the input for the score network, and then postprocess the output.
    For instance, we could mean-zero the input and then scale it to unit variance.

    However, since we are dealing with scores, the inverse would just undo the scaling but not the mean-zeroing.
    """

    def preprocess(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        return x, {}

    def postprocess(
        self,
        network_output: jnp.ndarray,
        evaluation_mode: Literal["log_q", "log_q_and_score", "score"],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return network_output

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray, bool], jnp.ndarray]]:
        processed, kwargs = self.preprocess(x)
        return processed, partial(self.postprocess, **kwargs)


class CenterMolecule(Preprocessor):
    def __init__(self, dataset: Dataset):
        self.sample_shape = dataset.sample_shape
        assert len(self.sample_shape) == 2, "Expected 2D data shape."
        self.data_shape = self.sample_shape

        if self.data_shape[1] == 3:
            assert jnp.allclose(
                dataset.train.data.reshape(-1, *self.sample_shape).mean(axis=1), 0.0, rtol=1e-6, atol=1e-6
            ), "Data is not centered but trying to center molecules. This might make diffusion training challenging."

    def preprocess(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        # we only want to center if we have higher-dimensional data (i.e., molecules with atoms in 3d space)
        if self.sample_shape[1] != 3:
            log.debug("Not centering data, as it is not in 3D space.")
            return x, {}

        orig_dim = x.shape
        x = x.reshape(x.shape[0], *self.sample_shape)
        assert x.ndim == 3, "Expected (BS, atoms, 3) data."

        x -= x.mean(axis=1, keepdims=True)
        return x.reshape(orig_dim), {}


IDENTITY = Preprocessor()
