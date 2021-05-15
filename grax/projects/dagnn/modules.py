import typing as tp
from functools import partial

import gin

import haiku as hk
import jax
import jax.numpy as jnp
from grax.projects.dagnn.ops import krylov
from huf.module_ops import Linear, dropout
from huf.modules.vmap_linear import VmapLinear
from huf.types import Activation
from jax.experimental.sparse_ops import JAXSparse

# from jax.experimental.host_callback import id_print


configurable = partial(gin.configurable, module="dagnn")


class GatedSum(hk.Module):
    def __init__(
        self,
        *args,
        gate_activation: Activation = tp.Optional[jax.nn.sigmoid],
        add_weighted_range: bool = False,
        name=None,
        **kwargs
    ):
        super().__init__(name=name)
        self._args = args
        self._kwargs = kwargs
        self.gate_activation = gate_activation
        self.add_weighted_range = add_weighted_range

    def __call__(self, unscaled_features, gate_features=None):
        if gate_features is None:
            gate_features = unscaled_features
        scale = hk.Linear(1, *self._args, **self._kwargs)(gate_features)
        scale = jnp.squeeze(scale, axis=-1)
        if self.add_weighted_range:
            range_weight = hk.get_parameter(
                "range_weight", shape=(), init=jnp.zeros, dtype=scale.dtype
            )
            num_scales = scale.shape[-1]
            scale = scale + range_weight * jnp.linspace(0, 1, num_scales)
            # id_print(range_weight)
        if self.gate_activation is not None:
            scale = self.gate_activation(scale)
        # id_print(unscaled_features[0, -1, :])
        # id_print(scale[0])
        # id_print(unscaled_features[:, -1, 0])
        # gf = gate_features[:, :, 0]
        # diff = jnp.abs(gf[:, :-1] - gf[:, 1:])
        # eps = 1e-4
        # id_print((diff / (jnp.abs(gf[:, :-1]) + eps)).mean(axis=0))
        # id_print(
        #     jnp.linalg.norm(unscaled_features[:, -1, 0])
        #     / jnp.linalg.norm(unscaled_features[:, -1, 1])
        # )
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


def _propagate(graph, logits, num_propagations: int, gate_activation: Activation):
    logits = krylov(graph, logits, num_propagations)
    logits = GatedSum(gate_activation=gate_activation)(logits)
    return logits


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
        graph: tp.Union[JAXSparse, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        node_features = dropout(node_features, self.dropout_rate, is_training)
        node_features = Linear(self.hidden_filters)(node_features)
        node_features = jax.nn.relu(node_features)
        node_features = dropout(node_features, self.dropout_rate, is_training)
        logits = hk.Linear(self.num_classes)(node_features)
        return _propagate(graph, logits, self.num_propagations, self.gate_activation)


@configurable
class DAGNNArxiv(hk.Module):
    """Used for ogbn-arxiv."""

    def __init__(
        self,
        num_classes: int,
        hidden_filters: int = 256,
        dropout_rate: float = 0.2,
        num_propagations: int = 16,
        gate_activation: Activation = jax.nn.sigmoid,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout_rate
        self.num_propagations = num_propagations
        self.gate_activation = gate_activation

    def __call__(
        self,
        graph: tp.Union[JAXSparse, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        node_features = Linear(self.hidden_filters)(node_features)
        node_features = hk.LayerNorm(0, True, True)(node_features)
        node_features = jax.nn.relu(node_features)
        node_features = dropout(
            node_features, self.dropout_rate, is_training=is_training
        )
        logits = hk.Linear(self.num_classes)(node_features)
        return _propagate(graph, logits, self.num_propagations, self.gate_activation)
