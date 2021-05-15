import typing as tp
from functools import partial

import gin

import jax.numpy as jnp
from jax.experimental.sparse_ops import JAXSparse
from spax import ops, utils

T = tp.TypeVar("T", JAXSparse, jnp.ndarray)

configurable = partial(gin.configurable, module="grax.graph_utils.transforms")


@configurable
def row_normalize(x: T, ord=1) -> T:  # pylint: disable=redefined-builtin
    if isinstance(x, jnp.ndarray):
        norm = jnp.linalg.norm(x, axis=1, ord=ord, keepdims=True)
        return jnp.where(norm == 0, jnp.zeros_like(x), x / norm)
    return ops.scale_rows(x, 1.0 / ops.norm(x, axis=1, ord=ord))


@configurable
def symmetric_normalize(x: T) -> T:
    factor = ops.norm(x, axis=1, ord=1) ** -0.5
    return ops.scale_columns(ops.scale_rows(x, factor), factor)


@configurable
def add_identity(x: T, scale: float = 1.0) -> T:
    if isinstance(x, JAXSparse):
        return ops.add(x, ops.mul(utils.eye(x.shape[0], dtype=x.dtype), scale))
    return x + scale * jnp.eye(x.shape[0], dtype=x.dtype)


@configurable
def linear_transform(x: T, shift: float = 0.0, scale: float = 1.0):
    if shift:
        x = add_identity(x, -shift)
    if scale != 1:
        x = ops.scale(x, 1.0 / scale)
    return x


@configurable
def to_format(arr: JAXSparse, fmt: str):
    if fmt == "coo":
        return ops.to_coo(arr)
    if fmt == "csr":
        return ops.to_csr(arr)
    if fmt == "dense":
        return ops.to_dense(arr)
    raise ValueError(f"`fmt` must be in ('coo', 'csr', 'dense'), got {fmt}")


@configurable
def chain(*transforms):
    """Get a transform that chains the single-arg transforms together."""

    def f(x):
        for t in transforms:
            x = t(x)
        return x

    return f
