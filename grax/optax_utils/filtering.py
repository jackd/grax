import typing as tp
from functools import partial

import gin
import jax.numpy as jnp
import optax

import haiku as hk

configurable = partial(gin.configurable, module="grax.optax_utils")

skip = optax.GradientTransformation(
    lambda params: (), lambda updates, state, params=None: (updates, state)
)


@configurable
def additive_weight_decay_w(weight_decay: float):
    return partition(
        lambda _, n, __: n == "w", optax.additive_weight_decay(weight_decay)
    )


@configurable
def partition(
    predicate: tp.Callable[[str, str, jnp.ndarray], bool],
    true_optimizer: optax.GradientTransformation,
    false_optimizer: optax.GradientTransformation = skip,
) -> optax.GradientTransformation:
    partition = partial(hk.data_structures.partition, predicate)

    def init(params):
        true_params, false_params = partition(params)
        return (true_optimizer.init(true_params), false_optimizer.init(false_params))

    def update(updates, states, params=None):
        true_states, false_states = states
        if params is None:
            true_params, false_params = None, None
        else:
            true_params, false_params = partition(params)
        true_updates, false_updates = partition(updates)
        true_updates, true_states = true_optimizer.update(
            true_updates, true_states, true_params
        )
        false_updates, false_states = false_optimizer.update(
            false_updates, false_states, false_params
        )
        updates = hk.data_structures.merge(true_updates, false_updates)
        state = (true_states, false_states)
        return updates, state

    return optax.GradientTransformation(init, update)


@configurable
def partition_n(
    fn: tp.Callable[[str, str, jnp.ndarray], int], *optimizers
) -> optax.GradientTransformation:
    n = len(optimizers)
    partition = partial(hk.data_structures.partition_n, fn, n=n)

    def init(params):
        return tuple(opt.init(p) for opt, p in zip(optimizers, partition(params)))

    def update(updates, states, params=None):
        if params is None:
            params = [None] * n
        else:
            params = partition(params)
        updates = partition(updates)
        updates, states = zip(
            *(
                opt.update(u, s, p)
                for (opt, u, s, p) in zip(optimizers, updates, states, params)
            )
        )
        updates = hk.data_structures.merge(*updates)
        return updates, states

    return optax.GradientTransformation(init, update)
