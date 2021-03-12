import typing as tp
from collections import defaultdict

import optax
from jax import tree_util

from grax.optax_utils.utils import nested_items


def partition(
    partition_fun: tp.Callable[[str, str], int],
    *optimizers: optax.GradientTransformation
) -> optax.GradientTransformation:
    num_partitions = len(optimizers)

    def split(tree):
        partitioned = [defaultdict(dict) for _ in optimizers]
        for (k0, k1), v1 in nested_items(tree):
            p = partition_fun(k0, k1)
            assert isinstance(p, int) and 0 <= p < num_partitions, (k0, k1, p)
            partitioned[p][k0][k1] = v1
        return partitioned

    def merge(trees):
        out = defaultdict(dict)
        for tree in trees:
            for (k0, k1), v in nested_items(tree):
                layer_dict = out[k0]
                assert k1 not in layer_dict
                layer_dict[k1] = v
        return out

    def init(params):
        partitioned = split(params)
        return tuple(opt.init(p) for opt, p in zip(optimizers, partitioned))

    def update(updates, states, params=None):
        # HACK DEBUG
        import jax.numpy as jnp

        for (k0, k1), v in nested_items(updates):
            print((k0, k1), jnp.max(jnp.abs(v)))
        _, treedef = tree_util.tree_flatten(updates)
        split_updates = split(updates)
        params = [None] * len(optimizers) if params is None else split(params)
        split_updates, states = zip(
            *(
                opt.update(up, st, p)
                for opt, up, st, p in zip(optimizers, split_updates, states, params)
            )
        )
        merged_updates = merge(split_updates)
        states = tuple(states)
        # repack as same dict type as input updates
        flat_updates = [
            merged_updates[k0][k1] for ((k0, k1), _) in nested_items(updates)
        ]
        updates = tree_util.tree_unflatten(treedef, flat_updates)
        # print(treedef)
        # print(treedef2)
        # updates = tree_util.tree_unflatten(treedef, updates)
        return updates, states

    return optax.GradientTransformation(init, update)
