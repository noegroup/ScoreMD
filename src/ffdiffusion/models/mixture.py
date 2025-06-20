import logging
from typing import Callable, Literal, Optional, Sequence, Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
from ffdiffusion.data.preprocess import IDENTITY, Preprocessor
from ffdiffusion.models import BaseDiffusionModel, EnergyModel


class MixtureOfModels(EnergyModel, nn.Module):
    models: Sequence[nn.Module]
    weight: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # (x, t) -> (num_models, BS)
    processor: Preprocessor = IDENTITY  # This usually centers the molecules
    threshold: float = 1e-2  # Skip if weight is below this value

    def __call__(
        self,
        x: jnp.ndarray,
        features: Optional[jnp.ndarray],
        t: jnp.ndarray,
        training,
        evaluated_models: Sequence[int] = [],
    ):
        x, features, t, original_shape = BaseDiffusionModel._reshape_input(x, features, t)

        out = self._forward(x, features, t, training, evaluated_models, evaluation_mode="score")
        return out.reshape(original_shape)  # ensure that we return the same shape as the original input

    @nn.compact
    def _forward(
        self,
        x: jnp.ndarray,
        features: Optional[jnp.ndarray],
        t: jnp.ndarray,
        training: bool,
        evaluated_models: Sequence[int],
        evaluation_mode: Literal["log_q", "log_q_and_score", "score"],
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        weights = self.weight(x, jnp.clip(t, 0.0, 1.0))  # (num_models, BS)
        weights = jnp.where(weights <= self.threshold, jnp.zeros_like(weights), weights)

        # set all weights to zero that are not in evaluated_models
        if len(evaluated_models) > 0 and not len(set(evaluated_models)) == len(evaluated_models):
            # set weights of non-evaluated models to zero
            multiplication_matrix = jnp.zeros_like(weights)
            multiplication_matrix = multiplication_matrix.at[jnp.array(evaluated_models)].set(1)
            weights *= multiplication_matrix
            weights = weights / jnp.sum(weights, axis=0)  # normalize the weights because we might have set some to zero

        assert weights.shape[0] == len(self.models), "Number of models and weights do not match"

        def zeros():
            return jnp.zeros_like(t), jnp.zeros_like(x)

        out_log_q, out_score = zeros()
        x, inverse = self.processor(x)

        def predict(m):
            ret_log_q, ret_score = zeros()
            if evaluation_mode == "log_q":
                ret_log_q = m.log_q(x, features, t, training)
            elif evaluation_mode == "log_q_and_score":
                ret_log_q, ret_score = m.log_q_and_score(x, features, t, training)
            elif evaluation_mode == "score":
                ret_score = m(x, features, t, training)

            r1, r2 = inverse((ret_log_q, ret_score), evaluation_mode)  # undo the preprocessing
            return r1.reshape(t.shape), r2.reshape(x.shape)

        if self.is_initializing():  # Force all models to be evaluated
            for i, m in enumerate(self.models):
                w = weights[i].reshape(-1, 1)
                p_log_q, p_score = predict(m)

                out_log_q += w * p_log_q
                out_score += w * p_score

        else:
            if len(evaluated_models) == 0:  # evaluate all models
                models = enumerate(self.models)
            else:
                models = [(i, m) for i, m in enumerate(self.models) if i in evaluated_models]
            for i, m in models:

                def f():
                    if evaluation_mode == "log_q" and (not hasattr(m, "log_q") or not m.supports_energy()):
                        logging.warning(f"Model {i} does not support energy evaluation. Ensure that it has no weight.")
                        return zeros()
                    elif evaluation_mode == "log_q_and_score" and (
                        not hasattr(m, "log_q_and_score") or not m.supports_energy()
                    ):
                        logging.warning(f"Model {i} does not support energy evaluation. Ensure that it has no weight.")
                        return zeros()

                    p_log_q, p_score = predict(m)
                    w = weights[i].reshape(-1, 1)

                    return w * p_log_q, w * p_score

                ret_log_q, ret_score = jax.lax.cond(jnp.any(weights[i] > self.threshold), f, zeros)
                out_log_q += ret_log_q
                out_score += ret_score

        if evaluation_mode == "log_q":
            return out_log_q
        elif evaluation_mode == "log_q_and_score":
            return out_log_q, out_score
        elif evaluation_mode == "score":
            return out_score

    def has_energy(self, x: jnp.ndarray, t: jnp.ndarray) -> bool:
        weights = self.weight(x, t).flatten()
        for i, w in enumerate(weights):
            if w > self.threshold:
                if not isinstance(self.models[i], EnergyModel):
                    return False
                # now we know it is an energy model so we can call supports_energy
                if not self.models[i].supports_energy():
                    return False

        return True

    def energy_models(self):
        return [i for i, m in enumerate(self.models) if isinstance(m, EnergyModel) and m.supports_energy()]

    def force(self, x, features: Optional[jnp.ndarray], t, training, evaluated_models: Sequence[int] = []):
        return self(x, features, t * jnp.ones((x.shape[0], 1)), training, evaluated_models)

    def log_q(
        self,
        x: jnp.ndarray,
        features: Optional[jnp.ndarray],
        t: jnp.ndarray,
        training,
        evaluated_models: Sequence[int] = [],
    ) -> jnp.ndarray:
        return self._forward(x, features, t, training, evaluated_models, evaluation_mode="log_q")

    def log_q_and_score(
        self,
        x: jnp.ndarray,
        features: Optional[jnp.ndarray],
        t: jnp.ndarray,
        training,
        evaluated_models: Sequence[int] = [],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._forward(x, features, t, training, evaluated_models, evaluation_mode="log_q_and_score")
