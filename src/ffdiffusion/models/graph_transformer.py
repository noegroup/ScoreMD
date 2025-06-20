"""
This code is based on:
https://github.com/microsoft/two-for-one-diffusion/blob/main/models/graph_transformer.py
Which in itself is basically an adaptation of the following code:
(lucidrains - graph-transformer-pytorch)
https://github.com/lucidrains/graph-transformer-pytorch
with (reduced) features
"""

from dataclasses import dataclass
from typing import Callable, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from ffdiffusion.data.dataset import Dataset
from ffdiffusion.models import BaseDiffusionModel, EnergyModel, ModelInfo
from ffdiffusion.models.utils import value_and_grad_sum
import logging


log = logging.getLogger(__name__)


@dataclass
class GraphTransformerModelInfo(ModelInfo):
    hidden_nf: int = 96
    feature_embedding_dim: int = 16
    n_layers: int = 2
    potential: bool = True
    dropout: float = 0.0

    def build(
        self, dataset: Dataset, t0: float, t1: float, rescale_time: bool, clip_time: bool, norm_factor: jnp.ndarray
    ) -> nn.Module:
        if hasattr(dataset, "max_z"):
            max_z = dataset.max_z
        else:
            max_z = []

        return GraphTransformer(
            t0,
            t1,
            rescale_time,
            clip_time,
            hidden_nf=self.hidden_nf,
            feature_embedding_dim=self.feature_embedding_dim,
            max_z=max_z,
            n_layers=self.n_layers,
            potential=self.potential,
            use_intrinsic_coords=True,
            use_abs_coords=False,
            use_distances=False,
            dropout=self.dropout,
        )


@dataclass
class GraphTransformer(BaseDiffusionModel, EnergyModel, nn.Module):
    hidden_nf: int
    feature_embedding_dim: int
    max_z: Sequence[int]
    n_layers: int
    potential: bool
    use_intrinsic_coords: bool = True
    use_abs_coords: bool = False
    use_distances: bool = False
    dropout: float = 0.0

    def _forward(self, x, features, t, training):
        if self.potential:
            # we predict an energy for every node, so we need to sum over all nodes
            # we also sum over the batch dimension to get a scalar
            return -jax.grad(lambda x: self._differentiable_forward(x, features, t, training).sum())(x).reshape(x.shape)
        return -self._differentiable_forward(x, features, t, training).reshape(x.shape)

    @nn.compact
    def _differentiable_forward(self, x, features, t, training):
        x, features, t = self._prepare_input(x, features, t, concat=False)
        x = x.reshape(x.shape[0], -1, 3)  # This architecture expects the input to be of shape [bs, n_nodes, 3]
        bs, n_nodes, _ = x.shape

        t = jnp.tile(t[:, None, ...], (1, n_nodes, 1))  # shape = (bs, n_nodes, time_features)

        if features is None:
            h = jnp.eye(n_nodes)
            h = jnp.tile(h[None, ...], (bs, 1, 1))  # shape = (bs, n_nodes, n_node_features)
        else:
            assert len(self.max_z) == features.shape[-1], f"len(max_z) = {len(self.max_z)} != {features.shape[-1]}"
            new_features = []
            for i, max_z in enumerate(self.max_z):
                new_features.append(
                    nn.Embed(
                        num_embeddings=max_z,
                        features=self.feature_embedding_dim,
                        name=f"scalar_embedding_{i}",
                    )(features[:, :, i])
                )

            h = jnp.concatenate(new_features, axis=-1)

        edge_attr = self.get_edge_attr(x)
        edge_attr = nn.Dense(self.hidden_nf, name="edge_embedding")(edge_attr)

        if self.use_abs_coords:
            nodes = jnp.concatenate([h, x, t], axis=-1)
        else:
            nodes = jnp.concatenate([h, t], axis=-1)

        nodes = nn.Dense(self.hidden_nf, name="node_embedding")(nodes)
        mask = jnp.ones((bs, n_nodes), dtype=bool)

        nodes, _ = GraphTransformerLucid(depth=self.n_layers, with_feedforwards=True, dropout=self.dropout)(
            nodes, edge_attr, training, mask=mask
        )

        if self.potential:
            return nn.Dense(1, name="node_decoder")(nodes)

        return nn.Dense(x.shape[1] * x.shape[2], name="node_decoder")(nodes.reshape(bs, -1)).reshape(x.shape)

    def get_edge_attr(self, x):
        # x shape: [bs, n_nodes, 3]
        if self.use_distances and not self.use_intrinsic_coords:
            # compute squared distances between nodes
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            dist = jnp.sum(diff**2, axis=-1, keepdims=True)
            return dist
        elif self.use_intrinsic_coords and not self.use_distances:
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            return diff
        elif self.use_intrinsic_coords and self.use_distances:
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            dist = jnp.sum(diff**2, axis=-1, keepdims=True)
            return jnp.concatenate([diff, dist], axis=-1)
        else:
            raise ValueError("Invalid configuration. I don't think we want to use this")
            bs, n_nodes, _ = x.shape
            return jnp.zeros((bs, n_nodes, n_nodes, 1))

    def log_q(self, x: jnp.ndarray, features: jnp.ndarray, t: jnp.ndarray, training: bool) -> jnp.ndarray:
        if not self.potential:
            log.error(f"Tried to call log_q on a non-potential model with x.shape = {x.shape} and t.shape = {t.shape}")
            log.error(f"x = {x}")
            log.error(f"t = {t}")
            raise ValueError("This model does not have a potential, only a force")

        assert x.ndim == t.ndim, f"{x.ndim} != {t.ndim}"

        return -self._differentiable_forward(x, features, t, training).sum(axis=(1, 2))[:, None]

    def log_q_and_score(self, x: jnp.ndarray, features: jnp.ndarray, t: jnp.ndarray, training: bool) -> jnp.ndarray:
        if not self.potential:
            log.error(
                f"Tried to call log_q_and_score on a non-potential model with x.shape = {x.shape} and t.shape = {t.shape}"
            )
            log.error(f"x = {x}")
            log.error(f"t = {t}")
            raise ValueError("This model does not have a potential, only a force")

        assert x.ndim == t.ndim, f"{x.ndim} != {t.ndim}"

        val, grad = value_and_grad_sum(self._differentiable_forward, x, features, t, training)
        return -val.sum(axis=(1, 2))[:, None], (-grad).reshape(x.shape)

    def supports_energy(self):
        return self.potential


class PreNorm(nn.Module):
    """Apply layer norm to the first argument of the function."""

    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.LayerNorm()(x)
        return self.fn(x, *args, **kwargs)


class GatedResidual(nn.Module):
    @nn.compact
    def __call__(self, x, res):
        gate_input = jnp.concatenate([x, res, x - res], axis=-1)
        gate = nn.Dense(1, use_bias=False)(gate_input)
        gate = nn.sigmoid(gate)
        return x * gate + res * (1 - gate)


class Attention(nn.Module):
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, nodes, edges, mask=None):
        h = self.heads
        inner_dim = self.dim_head * h

        scale = self.dim_head**-0.5

        q = nn.Dense(inner_dim)(nodes)
        k = nn.Dense(inner_dim)(nodes)
        v = nn.Dense(inner_dim)(nodes)
        e_kv = nn.Dense(inner_dim)(edges)

        q = rearrange(q, "b ... (h d) -> (b h) ... d", h=h)
        k = rearrange(k, "b ... (h d) -> (b h) ... d", h=h)
        v = rearrange(v, "b ... (h d) -> (b h) ... d", h=h)
        e_kv = rearrange(e_kv, "b ... (h d) -> (b h) ... d", h=h)

        ek, ev = e_kv, e_kv

        k = rearrange(k, "b j d -> b () j d")
        v = rearrange(v, "b j d -> b () j d")

        k += ek
        v += ev

        sim = jnp.einsum("b i d, b i j d -> b i j", q, k) * scale

        if mask is not None:
            mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim = jnp.where(mask, sim, -jnp.finfo(sim.dtype).max)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return nn.Dense(nodes.shape[-1])(out)


@dataclass
class GraphTransformerLucid(nn.Module):
    depth: int
    dim_head: int = 64
    heads: int = 8
    with_feedforwards: bool = True
    norm_edges: bool = False
    dropout: float = 0.0
    ff_mult: int = 4  # This is the multiplier for how many neurons are in the feedforward layer

    @nn.compact
    def __call__(self, nodes, edges, training, mask=None):
        if self.norm_edges:
            edges = nn.LayerNorm()(edges)

        for _ in range(self.depth):
            attn_out = PreNorm(Attention(heads=self.heads, dim_head=self.dim_head))(nodes, edges, mask=mask)
            nodes = GatedResidual()(attn_out, nodes)

            if self.with_feedforwards:
                feed_forward_out = PreNorm(
                    nn.Sequential(
                        [
                            nn.Dense(nodes.shape[-1] * self.ff_mult),
                            nn.gelu,
                            nn.Dense(nodes.shape[-1]),
                            nn.Dropout(self.dropout, deterministic=not training),
                        ]
                    )
                )(nodes)
                nodes = GatedResidual()(feed_forward_out, nodes)

        return nodes, edges


if __name__ == "__main__":
    import jax
    import time
    from tqdm import trange

    # Init variables
    n_nodes = 10
    hidden_nf = 256
    bs = 128

    # Init model
    model = GraphTransformer(
        0.0, 1.0, False, hidden_nf=hidden_nf, feature_embedding_dim=32, max_z=[], n_layers=5, potential=True
    )

    # Init parameters
    x = jnp.ones((bs, n_nodes, 3))
    x = x.reshape(x.shape[0], -1)
    # h = jnp.ones((n_nodes, n_nodes))
    t = jnp.ones((bs, 1))

    key = jax.random.PRNGKey(0)
    model_params = model.init(key, x, None, t)

    @jax.jit
    def forward(x, t):
        return model.apply(model_params, x, None, t)

    # Run model
    t1 = time.time()
    n_iterations = 100
    for i in trange(n_iterations):
        forces = forward(x, t).reshape(bs, n_nodes, 3)
    t2 = time.time()

    # Print output shape
    print(forces.shape)
    print(f"Average time per model pass {(t2 - t1) / n_iterations}")
