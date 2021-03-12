from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
import spax
from grax.projects.gat import ops as gat_ops
from huf import initializers
from huf.module_ops import dropout

configurable = partial(gin.configurable, module="gat")


@configurable
class GATConv(hk.Module):
    def __init__(
        self,
        filters: int,
        dropout_rate: float = 0.0,
        with_bias: bool = True,
        b_init=jnp.zeros,
        name=None,
    ):
        super().__init__(name=name)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.with_bias = with_bias
        self.b_init = b_init

    def __call__(
        self, graph: spax.SparseArray, node_features: jnp.ndarray, is_training: bool
    ):
        x = node_features
        del node_features
        x = dropout(x, self.dropout_rate, is_training)
        x = hk.Linear(
            self.filters,
            name="values",
            with_bias=False,
            w_init=initializers.glorot_uniform,
        )(x)
        # values = hk.Linear(self.filters + 2, w_init=initializers.lecun_uniform)(x)
        # key, query, values = jnp.split(values, (1, 2), axis=1)
        query = hk.Linear(
            1, name="query", with_bias=False, w_init=initializers.glorot_uniform
        )(x)
        key = hk.Linear(
            1, name="key", with_bias=False, w_init=initializers.glorot_uniform
        )(x)

        query = jnp.squeeze(query, axis=1)
        key = jnp.squeeze(key, axis=1)

        coords = spax.ops.get_coords(graph)
        row, col = coords

        query = query[row]
        key = key[col]

        attn = jax.nn.leaky_relu(key + query, negative_slope=0.2)
        attn = spax.utils.segment_softmax(attn, row, num_segments=graph.shape[0])
        attn = dropout(attn, self.dropout_rate, is_training)

        x = dropout(x, self.dropout_rate, is_training)
        # x = spax.ops.matmul(spax.ops.with_data(graph, attn), x)
        x = gat_ops.graph_conv(graph, attn, x)
        if self.with_bias:
            x = x + hk.get_parameter(
                "b", shape=(self.filters,), dtype=x.dtype, init=self.b_init
            )
        return x


# class GATConvBlock(hk.Module):
#     def __init__(self, filters: int, dropout_rate: float = 0.0, name=None):
#         super().__init__(name=name)
#         self.filters = filters
#         self.dropout_rate = dropout_rate

#     def __call__(
#         self, graph: spax.SparseArray, node_features: jnp.ndarray, is_training: bool
#     ):
#         x = dropout(node_features, self.dropout_rate, is_training)
#         x = hk.Linear(self.filters)(x)
#         return GATConv(self.filters, self.dropout_rate)(graph, x, is_training)


# @configurable
# class MultiHeadGATConv(hk.Module):
#     def __init__(
#         self, filters: int, num_heads: int, dropout_rate: float = 0.0, name=None,
#     ):
#         super().__init__(name=name)
#         self.filters = filters
#         self.num_heads = num_heads
#         self.dropout_rate = dropout_rate

#     def __call__(
#         self,
#         graph: spax.SparseArray,
#         node_features: jnp.ndarray,
#         is_training: tp.Optional[bool] = None,
#     ):
#         # no initial dropout / dense
#         # coords = spax.ops.to_coo(graph).coords
#         coords = spax.ops.get_coords(graph)
#         row, col = coords

#         values = hk.Linear(
#             (self.filters + 2) * self.num_heads, w_init=initializers.lecun_uniform,
#         )(node_features)
#         values = values.reshape(values.shape[0], self.num_heads, self.filters + 2)
#         key, query, values = jnp.split(values, (1, 2), axis=2)
#         key = jnp.squeeze(key, axis=2)
#         query = jnp.squeeze(query, axis=2)

#         key = key[row]
#         query = key[col]
#         attn = jax.nn.leaky_relu(key + query, negative_slope=0.2)
#         attn = spax.utils.segment_softmax(attn, row, num_segments=graph.shape[0])

#         attn = dropout(attn, self.dropout_rate, is_training)
#         values = dropout(values, self.dropout_rate, is_training)

#         out = gat_ops.multi_head_graph_conv(graph, attn, values)
#         assert out.shape == (graph.shape[0], self.num_heads, self.filters)
#         return out


class MultiHeadGATConv(hk.Module):
    def __init__(self, *args, num_heads=1, name=None, **kwargs):
        super().__init__(name=name)
        self.num_heads = num_heads
        self._head_fun = partial(GATConv, *args, **kwargs)

    def __call__(
        self, graph: spax.SparseArray, node_features: jnp.ndarray, is_training: bool
    ):
        heads = [
            self._head_fun(name=f"head{i}")(graph, node_features, is_training)
            for i in range(self.num_heads)
        ]
        return jnp.stack(heads, axis=1)


@configurable
class GAT(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: int = 8,
        hidden_heads: int = 8,
        final_heads: int = 1,
        dropout_rate: float = 0.6,
        name=None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.hidden_heads = hidden_heads
        self.final_heads = final_heads
        self.dropout_rate = dropout_rate

    def __call__(
        self, graph: spax.SparseArray, node_features: jnp.ndarray, is_training: bool,
    ):
        # x = dropout(node_features, self.dropout_rate, training)
        # x = hk.Linear(self.hidden_filters)(x)
        # x = MultiHeadGATConv(
        #     self.hidden_filters, self.hidden_heads, self.dropout_rate
        # )(graph, x, training)
        # x = x.reshape(graph.shape[0], self.hidden_filters * self.hidden_heads)
        # x = jax.nn.elu(x)
        # x = dropout(x, self.dropout_rate, training)
        # x = MultiHeadGATConv(self.num_classes, self.final_heads, self.dropout_rate)(
        #     graph, x, training
        # )  # [N, heads, cls]
        # x = x.sum(axis=1)  # [N, cls]
        # return x
        x = node_features
        x = MultiHeadGATConv(
            num_heads=self.hidden_heads,
            filters=self.hidden_filters,
            dropout_rate=self.dropout_rate,
        )(graph, x, is_training)
        assert x.shape == (
            node_features.shape[0],
            self.hidden_heads,
            self.hidden_filters,
        )
        x = x.reshape(x.shape[0], self.hidden_filters * self.hidden_heads)
        x = jax.nn.elu(x)
        x = MultiHeadGATConv(
            num_heads=self.final_heads,
            filters=self.num_classes,
            dropout_rate=self.dropout_rate,
        )(graph, x, is_training)
        x = x.mean(axis=1)
        return x
