import functools
import typing as tp

import gin
import haiku as hk
import jax.numpy as jnp

configurable = functools.partial(gin.configurable, module="sgc")


@configurable
class SGC(hk.Module):
    """Just a linear layer with an interface similar to other projects."""

    def __init__(self, num_classes: int, name: tp.Optional[str] = None):
        super().__init__(name=name)
        self.num_classes = num_classes

    def __call__(self, features: jnp.ndarray, is_training=False) -> jnp.ndarray:
        del is_training
        return hk.Linear(self.num_classes)(features)
