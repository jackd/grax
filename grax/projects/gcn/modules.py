import typing as tp
from functools import partial

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from huf import initializers
from huf.module_ops import Linear, dropout
from jax.experimental.sparse.ops import JAXSparse

from grax.projects.gcn.ops import graph_convolution
from grax.types import Activation

configurable = partial(gin.configurable, module="gcn")


@configurable
class GraphConvolution(hk.Module):
    def __init__(
        self,
        filters: int,
        use_bias: bool = True,
        kernel_initializer=initializers.glorot_uniform,
        bias_initializer=jnp.zeros,
        name=None,
    ):
        self.filters = filters
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super().__init__(name=name)

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, JAXSparse],
        features: tp.Union[jnp.ndarray, JAXSparse],
    ) -> jnp.ndarray:
        assert len(features.shape) == 2, features.shape
        assert graph.shape[1] == features.shape[0]
        filters_in = features.shape[1]
        filters_out = int(self.filters)
        kernel = hk.get_parameter(
            "kernel",
            shape=(filters_in, filters_out),
            dtype=features.dtype,
            init=self.kernel_initializer,
        )
        out = graph_convolution(graph, features, kernel)
        if self.use_bias:
            b = hk.get_parameter(
                "bias",
                shape=(filters_out,),
                dtype=features.dtype,
                init=self.bias_initializer,
            )
            out = out + b
        return out


@configurable
class GCN2(hk.Module):
    """
    Based on ogbn-arxiv leaderboard GCN.

    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
    """

    def __init__(
        self,
        num_classes: int,
        hidden_filters: tp.Iterable[int] = (16,),
        dropout_rate: float = 0.5,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, JAXSparse],
        node_features: jnp.ndarray,
        is_training: tp.Optional[bool] = None,
    ):
        x = node_features
        for f in self.hidden_filters:
            x = GCN(f)(graph, x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training=is_training)
            x = jax.nn.relu(x)
            x = dropout(x, self.dropout_rate, is_training=is_training)
        logits = GCN(self.num_classes)(graph, x)
        return logits


@configurable
class GCN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: tp.Iterable[int] = (16,),
        dropout_rate: float = 0.5,
        activation: Activation = jax.nn.relu,
        final_activation: Activation = lambda x: x,
        residual_connections: bool = False,
        name: tp.Optional[str] = None,
    ):
        if not hasattr(hidden_filters, "__iter__"):
            hidden_filters = (hidden_filters,)
        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.hidden_filters = tuple(hidden_filters)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation
        self.residual_connections = residual_connections
        if residual_connections:
            assert (
                self.hidden_filters[:-1] == self.hidden_filters[1:]
            ), self.hidden_filters

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, JAXSparse],
        node_features: jnp.ndarray,
        is_training: tp.Optional[bool] = None,
    ):
        if self.residual_connections:
            node_features = Linear(self.hidden_filters[0])(node_features)
            node_features = self.activation(node_features)
        for f in self.hidden_filters:
            n0 = node_features
            node_features = dropout(node_features, self.dropout_rate, is_training)
            node_features = GraphConvolution(f)(graph, node_features)
            node_features = self.activation(node_features)
            if self.residual_connections:
                node_features = node_features + n0
        node_features = dropout(node_features, self.dropout_rate, is_training)
        preds = GraphConvolution(self.num_classes)(graph, node_features)
        preds = self.final_activation(preds)
        return preds
