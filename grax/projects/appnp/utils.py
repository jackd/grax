import functools

import gin
import optax

from grax.optax_utils import partition

configurable = functools.partial(gin.configurable, module="appnp.utils")


@configurable
def named_predicate(layer_name, variable_name):
    def predicate(a, b, c):
        del c
        p = (a.split("/")[-1], b) == (layer_name, variable_name)
        return p

    return predicate


@configurable
def selective_additive_weight_decay(predicate, weight_decay: float):
    return partition(predicate, optax.additive_weight_decay(weight_decay),)
