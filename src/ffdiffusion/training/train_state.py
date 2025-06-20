# This code is a variation of the original code from the Flax library.

# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable
import optax

import jax
import jax.numpy as jnp
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT


class EmaTrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Example usage::

      >>> import flax.linen as nn
      >>> from ffdiffusion.training.train_state import EmaTrainState
      >>> import jax, jax.numpy as jnp
      >>> import optax

      >>> x = jnp.ones((1, 2))
      >>> y = jnp.ones((1, 2))
      >>> model = nn.Dense(2)
      >>> variables = model.init(jax.random.key(0), x)
      >>> tx = optax.adam(1e-3)

      >>> state = EmaTrainState.create(
      ...     apply_fn=model.apply,
      ...     params=variables['params'],
      ...     tx=tx)

      >>> def loss_fn(params, x, y):
      ...   predictions = state.apply_fn({'params': params}, x)
      ...   loss = optax.l2_loss(predictions=predictions, targets=y).mean()
      ...   return loss
      >>> loss_fn(state.params, x, y)
      Array(3.3514676, dtype=float32)

      >>> grads = jax.grad(loss_fn)(state.params, x, y)
      >>> state = state.apply_gradients(grads=grads)
      >>> loss_fn(state.params, x, y)
      Array(3.343844, dtype=float32)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
      step: Counter starts at 0 and is incremented by every call to
        ``.apply_gradients()``.
      apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for
        convenience to have a shorter params list for the ``train_step()`` function
        in your training loop.
      params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
      tx: An Optax gradient transformation.
      opt_state: The state for ``tx``.
    """

    step: int | jax.Array
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    ema_weight: float = struct.field(pytree_node=False)  # Exponential moving average weight. Usually close to 1.0
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

        Note that internally this function calls ``.tx.update()`` followed by a call
        to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

        Args:
          grads: Gradients that have the same pytree structure as ``.params``.
          **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

        Returns:
          An updated instance of ``self`` with ``step`` incremented by one, ``params``
          and ``opt_state`` updated by applying ``grads``, and additional attributes
          replaced as specified by ``kwargs``.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads["params"]
            params_with_opt = self.params["params"]
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(grads_with_opt, self.opt_state, params_with_opt)
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                "params": new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt

        def conditional_ema_update(ema_param, new_param, old_param):
            # Only update if any of the paramters changed.
            # Since this function is called for each 'module' in the params tree, it handles frozen parameters.
            return jnp.where(
                (jnp.abs(new_param - old_param) > 1e-8).any(),
                self.ema_weight * new_param + (1 - self.ema_weight) * ema_param,
                ema_param,
            )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=jax.tree_util.tree_map(
                lambda ema_param, new_param, old_param: conditional_ema_update(ema_param, new_param, old_param),
                self.ema_params,
                new_params,
                self.params,
            ),
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema_weight, ema_params=None, opt_state=None, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        assert ema_weight >= 0.0 and ema_weight <= 1.0, "ema_weight must be in [0, 1]"

        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = params["params"] if OVERWRITE_WITH_GRADIENT in params else params
        if opt_state is None:
            opt_state = tx.init(params_with_opt)

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_weight=ema_weight,
            ema_params=params if ema_params is None else ema_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
