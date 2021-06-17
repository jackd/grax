import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp
from jax.experimental.sparse_ops import JAXSparse

import haiku as hk
from grax.hk_utils import mlp
from grax.projects.dagnn.ops import krylov
from huf.types import Activation

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
        name=None,
    ):
        super().__init__(name=name)
        self.node_transform = node_transform
        self.num_propagations = num_propagations
        self.gate_activation = gate_activation

    def __call__(
        self,
        graph: tp.Union[JAXSparse, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        x = self.node_transform(node_features, is_training=is_training)
        x = krylov(graph, x, self.num_propagations)
        logits = GatedSum(gate_activation=self.gate_activation)(x)
        return logits
