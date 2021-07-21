import typing as tp
from functools import partial

import gin
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental.sparse.ops import COO, CSR, JAXSparse
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
def symmetric_renormalize(x: T) -> T:
    x = add_identity(x)
    return symmetric_normalize(x)


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
def to_format(arr: tp.Union[JAXSparse, jnp.ndarray, np.ndarray], fmt: str):
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


def to_scipy(mat: JAXSparse):
    assert isinstance(mat, JAXSparse)
    if isinstance(mat, COO):
        return sp.coo_matrix(
            (mat.data.to_py(), (mat.row.to_py(), mat.col.to_py())), shape=mat.shape
        )
    if isinstance(mat, CSR):
        return sp.csr_matrix(
            (mat.data.to_py(), mat.indices.to_py(), mat.indptr.to_py()),
            shape=mat.shape,
        )
    raise NotImplementedError(f"Only COO and CSR supported, got {type(mat)}")


def from_scipy(mat_sp) -> JAXSparse:
    assert sp.isspmatrix(mat_sp)
    if sp.isspmatrix_coo(mat_sp):
        return COO((mat_sp.data, mat_sp.row, mat_sp.col), shape=mat_sp.shape)
    if sp.isspmatrix_csr(mat_sp):
        return CSR((mat_sp.data, mat_sp.indices, mat_sp.indptr), shape=mat_sp.shape)
    raise NotImplementedError(
        f"Only coo_matrix and csr_matrix supported, got {type(mat_sp)}"
    )
