import functools
import typing as tp

import gin
import haiku as hk
import jax.numpy as jnp
from jax.experimental.sparse import COO
from spax.ops import map_data

from grax import hk_utils

configurable = functools.partial(gin.configurable, module="appnp")


@configurable
class APPNP(hk.Module):
    def __init__(
        self,
        head_transform: tp.Callable = hk_utils.mlp,
        num_propagations: int = 10,
        alpha: float = 0.1,
        name: tp.Optional[str] = None,
        edge_dropout_rate: float = 0,
    ):
        super().__init__(name=name)
        self.head_transform = head_transform
        self.num_propagations = num_propagations
        self.alpha = alpha
        self.edge_dropout_rate = edge_dropout_rate

    def __call__(
        self, A: COO, features: jnp.ndarray, is_training: bool = False,
    ) -> jnp.ndarray:
        x = self.head_transform(features, is_training)
        A = map_data(
            A,
            lambda v: hk_utils.dropout(
                v, is_training=is_training, rate=self.edge_dropout_rate
            ),
        )
        x0 = x
        for _ in range(self.num_propagations):
            x = (1 - self.alpha) * (A @ x) + self.alpha * x0
        return x


@configurable
class PPNP(hk.Module):
    def __init__(
        self,
        head_transform: tp.Callable = hk_utils.mlp,
        propagator_dropout_rate: float = 0.0,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.head_transform = head_transform
        self.propagator_dropout_rate = propagator_dropout_rate

    def __call__(
        self, A: jnp.ndarray, features: jnp.ndarray, is_training: bool = False
    ):
        X = self.head_transform(features, is_training=is_training)
        A = hk_utils.dropout(
            A, rate=self.propagator_dropout_rate, is_training=is_training
        )
        return A @ X
