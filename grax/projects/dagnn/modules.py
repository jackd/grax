import typing as tp
from functools import partial

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from huf.types import Activation
from jax.experimental.sparse.ops import JAXSparse

from grax.hk_utils import mlp
from grax.projects.dagnn.ops import krylov

configurable = partial(gin.configurable, module="dagnn")


class GatedSum(hk.Module):
    def __init__(
        self,
        *args,
        gate_activation: tp.Optional[Activation] = jax.nn.sigmoid,
        name=None,
        **kwargs
    ):
        super().__init__(name=name)
        self._args = args
        self._kwargs = kwargs
        self.gate_activation = gate_activation

    def __call__(self, unscaled_features, gate_features=None):
        if gate_features is None:
            gate_features = unscaled_features
        scale = hk.Linear(1, *self._args, **self._kwargs)(gate_features)
        scale = jnp.squeeze(scale, axis=-1)
        if self.gate_activation is not None:
            scale = self.gate_activation(scale)
        return jnp.einsum("nkl,nk->nl", unscaled_features, scale)


@configurable
class DAGNN(hk.Module):
    def __init__(
        self,
        node_transform: tp.Callable[[jnp.ndarray, bool], jnp.ndarray] = mlp,
        num_propagations: int = 20,
        gate_activation: tp.Callable = jax.nn.sigmoid,
        adaptive: bool = True,
        scale_factor: float = 1.0,
        scale_method: str = "all",
        name=None,
    ):
        super().__init__(name=name)
        self.node_transform = node_transform
        self.num_propagations = num_propagations
        self.gate_activation = gate_activation
        self.adaptive = adaptive
        self.scale_factor = scale_factor
        self.scale_method = scale_method

    def __call__(
        self,
        graph: tp.Union[JAXSparse, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        x = self.node_transform(node_features, is_training=is_training)
        x = krylov(graph, x, self.num_propagations)
        if self.adaptive:
            assert self.scale_factor == 1, self.scale_factor
            logits = GatedSum(gate_activation=self.gate_activation)(x)
        else:
            if self.scale_method == "all":
                logits = self.scale_factor * jnp.sum(x, axis=1)
            else:
                assert self.scale_method == "most"
                x, x_leading = jnp.split(x, (x.shape[1] - 1,), axis=1)
                logits = self.scale_factor * jnp.sum(x, axis=1) + jnp.squeeze(
                    x_leading, axis=1
                )
        return logits
