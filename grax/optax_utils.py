import typing as tp
from functools import partial

import gin
import haiku as hk
import jax.numpy as jnp
import optax
from optax import GradientTransformation

configurable = partial(gin.configurable, module="grax.optax_utils")


@configurable
def partition(
    predicate: tp.Callable[[str, str, jnp.ndarray], bool],
    if_true: GradientTransformation = optax.identity(),
    if_false: GradientTransformation = optax.identity(),
) -> optax.GradientTransformation:
    def init(params):
        true_params, false_params = hk.data_structures.partition(predicate, params)
        true_init = if_true.init(true_params)
        false_init = if_false.init(false_params)
        return hk.data_structures.merge(true_init, false_init)

    def update(updates, opt_state, params=None):
        if params is None:
            true_params = false_params = None
        else:
            true_params, false_params = hk.data_structures.partition(predicate, params)
        true_updates, false_updates = hk.data_structures.partition(predicate, updates)
        true_opt_state, false_opt_state = hk.data_structures.partition(
            predicate, opt_state
        )
        true_updates, true_opt_state = if_true.update(
            true_updates, true_opt_state, true_params
        )
        false_updates, false_opt_state = if_false.update(
            false_updates, false_opt_state, false_params
        )

        updates = hk.data_structures.merge(true_updates, false_updates)
        opt_state = hk.data_structures.merge(true_updates, false_updates)
        return updates, opt_state

    return optax.GradientTransformation(init, update)
