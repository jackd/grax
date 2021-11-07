import functools
import typing as tp

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.sparse.ops import JAXSparse

from grax import hk_utils
from grax.optax_utils import partition

configurable = functools.partial(gin.configurable, module="gcn2")


@configurable
def default_w_init(shape, dtype):
    features_out, _ = shape
    std = 1 / np.sqrt(float(features_out))
    return hk.initializers.RandomUniform(-std, std)(shape, dtype)


@configurable
def partitioned_additive_weight_decay(
    conv_weight_decay: float,  # wd1 from original repo
    linear_weight_decay: float,  # wd2 from original repo
) -> optax.GradientTransformation:
    def predicate(layer_name, param_name, value):
        del param_name, value
        return layer_name.split("/")[-1].startswith("linear")

    return partition(
        predicate,
        optax.additive_weight_decay(linear_weight_decay),
        optax.additive_weight_decay(conv_weight_decay),
    )


@configurable
class GraphConvolution(hk.Module):
    def __init__(
        self,
        filters: int,
        beta: float,
        alpha: float,
        with_bias: bool = True,
        w_init=default_w_init,
        b_init=jnp.zeros,
        variant: bool = False,
        name=None,
    ):
        super().__init__(name=name)
        self.w_init = w_init
        self.b_init = b_init
        self.filters = filters
        self.with_bias = with_bias
        self.beta = beta
        self.alpha = alpha
        self.variant = variant

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, JAXSparse],
        features: jnp.ndarray,
        features0: jnp.ndarray,
    ):
        hi = graph @ features
        if self.variant:
            support = jnp.concatenate([hi, features0], axis=1)
            r = (1 - self.alpha) * hi + self.alpha * features
        else:
            support = (1 - self.alpha) * hi + self.alpha * features0
            r = support
        w = hk.get_parameter(
            "w",
            shape=(self.filters, support.shape[1]),
            dtype=support.dtype,
            init=self.w_init,
        )
        output = self.beta * support @ w + (1 - self.beta) * r
        if self.with_bias:
            b = hk.get_parameter(
                "bias", shape=(self.filters,), dtype=output.dtype, init=self.b_init
            )
            output = output + b
        return output


@configurable
class GCN2(hk.Module):
    def __init__(
        self,
        num_classes: int,
        filters: int = 64,
        num_hidden_layers: int = 64,
        dropout_rate: float = 0.6,
        lam: float = 0.5,
        alpha: float = 0.1,
        variant: bool = False,
        activation=jax.nn.relu,
        name=None,
    ):
        super().__init__(name=name)
        self.filters = filters
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.alpha = alpha
        self.variant = variant
        self.activation = activation

    def __call__(self, graph, features, is_training: bool = False) -> jnp.ndarray:
        dropout = functools.partial(
            hk_utils.dropout, rate=self.dropout_rate, is_training=is_training
        )
        x = dropout(features)
        x = hk_utils.Linear(self.filters, name="linear_0")(x)
        x = self.activation(x)
        x0 = x
        for i in range(self.num_hidden_layers):
            x = dropout(x)
            x = GraphConvolution(
                self.filters,
                variant=self.variant,
                beta=np.log(self.lam / (i + 1) + 1),
                alpha=self.alpha,
                with_bias=False,
                name=f"gcn2_{i}",
            )(graph, x, x0)
            x = self.activation(x)

        x = dropout(x)
        x = hk.Linear(self.num_classes, name="linear_1")(x)
        return x
