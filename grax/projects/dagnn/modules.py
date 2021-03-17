import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
import spax
from grax.projects.dagnn.ops import krylov
from huf.module_ops import dropout
from huf.modules.vmap_linear import VmapLinear
from huf.types import Activation

configurable = partial(gin.configurable, module="dagnn")


class GatedSum(hk.Module):
    def __init__(
        self, *args, gate_activation: Activation = jax.nn.sigmoid, name=None, **kwargs
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
        scale = self.gate_activation(scale)
        return jnp.einsum("nkl,nk->nl", unscaled_features, scale)


class VmapGatedSum(hk.Module):
    def __init__(
        self, *args, gate_activation: Activation = jax.nn.sigmoid, name=None, **kwargs
    ):
        super().__init__(name=name)
        self._args = args
        self._kwargs = kwargs
        self.gate_activation = gate_activation

    def __call__(self, unscaled_features, gate_features=None):
        if gate_features is None:
            gate_features = unscaled_features
        scale = VmapLinear(1, *self._args, **self._kwargs)(gate_features)
        scale = self.gate_activation(scale)
        scale = jnp.squeeze(scale, axis=-1)
        return jnp.einsum("nkhl,nkh->nhl", unscaled_features, scale)


@configurable
class DAGNN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: int = 64,
        dropout_rate: float = 0.8,
        num_propagations: int = 20,
        gate_activation: Activation = jax.nn.sigmoid,
        name=None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout_rate
        self.num_propagations = num_propagations
        self.gate_activation = gate_activation

    def __call__(
        self,
        graph: tp.Union[spax.SparseArray, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        node_features = dropout(node_features, self.dropout_rate, is_training)
        node_features = hk.Linear(self.hidden_filters)(node_features)
        node_features = jax.nn.relu(node_features)
        node_features = dropout(node_features, self.dropout_rate, is_training)
        logits = hk.Linear(self.num_classes)(node_features)
        logits = krylov(graph, logits, self.num_propagations)
        logits = GatedSum(gate_activation=self.gate_activation)(logits)
        return logits
