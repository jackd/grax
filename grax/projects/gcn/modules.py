import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
from grax.projects.gcn.ops import graph_convolution
from grax.types import Activation
from huf import initializers
from huf.module_ops import dropout
from spax import SparseArray

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
        self, graph: tp.Union[jnp.ndarray, SparseArray], features: jnp.ndarray
    ):
        assert graph.ndim == 2, graph.shape
        assert features.ndim == 2, features.shape
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
class GCN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: tp.Iterable[int] = (16,),
        dropout_rate: float = 0.5,
        activation: Activation = jax.nn.relu,
        final_activation: Activation = lambda x: x,
        **kwargs,
    ):
        if not hasattr(hidden_filters, "__iter__"):
            hidden_filters = (hidden_filters,)
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.hidden_filters = tuple(hidden_filters)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation
        self.hidden_layers = tuple(
            GraphConvolution(f, name=f"graph_conv_{i}")
            for i, f in enumerate(hidden_filters)
        )
        self.final_layer = GraphConvolution(num_classes, name="graph_conv_final")

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, SparseArray],
        node_features: jnp.ndarray,
        is_training: tp.Optional[bool] = None,
    ):
        for layer in self.hidden_layers:
            node_features = dropout(node_features, self.dropout_rate, is_training)
            node_features = layer(graph, node_features)
            node_features = self.activation(node_features)
        node_features = dropout(node_features, self.dropout_rate, is_training)
        preds = self.final_layer(graph, node_features)
        preds = self.final_activation(preds)
        return preds
