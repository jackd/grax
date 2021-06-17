import typing as tp
from functools import partial

import gin
import haiku as hk
import jax.numpy as jnp

from grax.hk_utils import mlp

configurable = partial(gin.configurable, module="igcn")


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
        ids=None,
        is_training: bool = False,
    ):
        x = self.node_transform(node_features, is_training=is_training, ids=ids)
        logits = smoother @ x
        return logits
