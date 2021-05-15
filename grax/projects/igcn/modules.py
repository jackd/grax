import operator
import typing as tp
from functools import partial

import gin

import haiku as hk
import jax
import jax.numpy as jnp
from huf.module_ops import Linear, dropout
from huf.types import Activation
from jax.experimental.sparse_ops import JAXSparse

configurable = partial(gin.configurable, module="igcn")


def _propagate(graph, logits, tol: float):
    logits, _ = jax.scipy.sparse.linalg.cg(
        jax.tree_util.Partial(operator.matmul, graph), logits, tol=tol
    )
    return logits


@configurable
def mlp(
    x,
    is_training: bool,
    num_classes: int = gin.REQUIRED,
    hidden_filters: tp.Union[int, tp.Iterable[int]] = 64,
    dropout_rate: float = 0.8,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    activation: Activation = jax.nn.relu,
    final_activation: Activation = lambda x: x,
):
    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)
    for filters in hidden_filters:
        x = dropout(x, dropout_rate, is_training=is_training)
        x = Linear(filters)(x)
        x = activation(x)
        if use_batch_norm:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        if use_layer_norm:
            x = hk.LayerNorm(0, True, True)(x)
    x = dropout(x, dropout_rate, is_training=is_training)
    x = hk.Linear(num_classes)(x)
    return final_activation(x)


@configurable
class IGCN(hk.Module):
    def __init__(
        self,
        node_transform: tp.Callable[[jnp.ndarray, bool], jnp.ndarray] = mlp,
        name=None,
    ):
        super().__init__(name=name)
        self.node_transform = node_transform

    def __call__(
        self,
        smoother: tp.Any,  # anything with a __matmul__ operator
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        return smoother @ self.node_transform(node_features, is_training=is_training)


# @configurable
# class IGCNArxiv(hk.Module):
#     """Used for ogbn-arxiv."""

#     def __init__(
#         self,
#         num_classes: int,
#         hidden_filters: int = 256,
#         dropout_rate: float = 0.2,
#         tol: float = 1e-5,
#         name: tp.Optional[str] = None,
#     ):
#         super().__init__(name=name)
#         self.num_classes = num_classes
#         self.hidden_filters = hidden_filters
#         self.dropout_rate = dropout_rate
#         self.tol = tol

#     def __call__(
#         self,
#         graph: tp.Union[JAXSparse, jnp.ndarray],
#         node_features: jnp.ndarray,
#         is_training: bool,
#     ):
#         node_features = Linear(self.hidden_filters)(node_features)
#         node_features = hk.LayerNorm(0, True, True)(node_features)
#         node_features = jax.nn.relu(node_features)
#         node_features = dropout(
#             node_features, self.dropout_rate, is_training=is_training
#         )
#         logits = hk.Linear(self.num_classes)(node_features)
#         return _propagate(graph, logits, self.tol)


@configurable
class IGCNArxiv(hk.Module):
    """Used for ogbn-arxiv."""

    def __init__(
        self,
        num_classes: int,
        num_layers: int = 3,
        hidden_filters: int = 256,
        dropout_rate: float = 0.5,
        tol: float = 1e-5,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.tol = tol

    def __call__(
        self,
        graph: tp.Union[JAXSparse, jnp.ndarray],
        node_features: jnp.ndarray,
        is_training: bool,
    ):
        for _ in range(self.num_layers - 1):
            node_features = Linear(self.hidden_filters)(node_features)
            node_features = hk.LayerNorm(0, True, True)(node_features)
            node_features = jax.nn.relu(node_features)
            node_features = dropout(
                node_features, self.dropout_rate, is_training=is_training
            )
        logits = hk.Linear(self.num_classes)(node_features)
        return _propagate(graph, logits, self.tol)
