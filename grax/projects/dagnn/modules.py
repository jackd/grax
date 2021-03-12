import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
import spax
from grax.projects.dagnn.ops import krylov
from huf.module_ops import dropout

configurable = partial(gin.configurable, module="dagnn")


class GatedSum(hk.Module):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args):
        if len(args) == 1:
            gate_features = unscaled_features = args[0]
        else:
            unscaled_features, gate_features = args
        scale = hk.Linear(1, *self._args, **self._kwargs)(gate_features)
        scale = jax.nn.sigmoid(scale)
        scale = jnp.squeeze(scale, axis=-1)
        return jnp.einsum("nkl,nk->nl", unscaled_features, scale)


@configurable
class DAGNN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: int = 64,
        dropout_rate: float = 0.8,
        num_propagations: int = 20,
        name=None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout_rate
        self.num_propagations = num_propagations

    def __call__(
        self,
        graph: tp.Union[spax.SparseArray, jnp.ndarray],
        node_features: jnp.ndarray,
        training: bool,
    ):
        def Linear(*args, **kwargs):
            # fan_in = node_features.shape[-1]
            return hk.Linear(
                *args,
                # w_init=initializers.torch_linear_kernel_initializer,
                # b_init=initializers.torch_linear_bias_initializer(fan_in),
                **kwargs
            )

        node_features = dropout(node_features, self.dropout_rate, training)
        node_features = Linear(self.hidden_filters)(node_features)
        node_features = jax.nn.relu(node_features)
        node_features = dropout(node_features, self.dropout_rate, training)
        logits = Linear(self.num_classes)(node_features)
        logits = krylov(graph, logits, self.num_propagations)
        logits = GatedSum(
            # w_init=initializers.torch_linear_kernel_initializer,
            # b_init=initializers.torch_linear_bias_initializer(logits.shape[-1]),
        )(logits)
        return logits
