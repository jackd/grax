import typing as tp
from functools import partial

import gin
import jax.numpy as jnp

from spax import SparseArray, ops, utils

T = tp.TypeVar("T", SparseArray, jnp.ndarray)

configurable = partial(gin.configurable, module="grax.graph_utils.transforms")


@configurable
def row_normalize(x: T, ord=1) -> T:  # pylint: disable=redefined-builtin
    assert x.ndim == 2
    return ops.scale_rows(x, 1.0 / ops.norm(x, axis=1, ord=ord))


@configurable
def symmetric_normalize(x: T) -> T:
    factor = ops.norm(x, axis=1, ord=1) ** -0.5
    return ops.scale_columns(ops.scale_rows(x, factor), factor)


@configurable
def add_identity(x: T, scale: float = 1.0) -> T:
    assert x.ndim == 2
    if utils.is_sparse(x):
        return ops.add(x, ops.mul(utils.eye(x.shape[0], dtype=x.dtype), scale))
    return x + scale * jnp.eye(x.shape[0], dtype=x.dtype)


@configurable
def to_format(arr: SparseArray, fmt: str):
    if fmt == "coo":
        return arr.tocoo()
    if fmt == "csr":
        return arr.tocsr()
    if fmt == "ell":
        return arr.toell()
    if fmt == "dense":
        return arr.todense()
    raise ValueError(f"`fmt` must be in ('coo', 'csr', 'ell', 'dense'), got {fmt}")


@configurable
def chain(*transforms):
    """Get a transform that chains the single-arg transforms together."""

    def f(x):
        for t in transforms:
            x = t(x)
        return x

    return f
