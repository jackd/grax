import typing as tp
from functools import partial

import gin
import jax.numpy as jnp

from spax import ops
from spax.sparse import SparseArray
from spax.utils import diag, eye

configurable = partial(gin.configurable, module="grax.graph_utils.laplacians")


@configurable
def laplacian(
    adj: SparseArray, shift: float = 0.0
) -> tp.Tuple[SparseArray, jnp.ndarray]:
    """
    Get a possibly shifted Laplacian matrix.

    L = D - shift * I - adj

    where `D = diag(sum(adj, axis=1))`

    Args:
        adj: [n, n] adjacency matrix.
        shift: scalar diagonal shift.

    Returns:
        L: [n, n] Sparse Laplacian as above
        row_sum: [n] sum over rows of `adj`.
    """
    row_sum = ops.sum(adj, axis=1)
    return ops.subtract(diag(row_sum - shift), adj), row_sum


@configurable
def normalized_laplacian(
    adj: tp.Union[SparseArray, jnp.ndarray], shift: float = 0.0
) -> tp.Tuple[SparseArray, jnp.ndarray]:
    """
    Get a possibly-shifted normalized laplacian matrix.

    L - shift*I = (1 - shift) * I - D**-0.5 @ adj @ D**-0.5

    where D = diag(sum(adj, axis=1))

    Args:
        adj: [n, n] possibly-weighted adjacency matrix.
        shift: optional shift to apply to diagonal.

    Returns:
        laplacian: [n, n] normalized laplacian matrix.
        row_sum: [n] sum of weights for each row.
    """
    row_sum = ops.sum(adj, axis=1)
    D = row_sum ** -0.5
    dtype = adj.dtype
    L = ops.subtract(
        ops.mul(eye(adj.shape[0], dtype=dtype), jnp.asarray(1 - shift, dtype)),
        ops.scale_columns(ops.scale_rows(adj, D), D),
    )
    return L, row_sum


def normalized_laplacian_zero_eigenvector(row_sum: jnp.ndarray):
    """Get the normalized eigenvector for eigenvalue zero of a normalized Laplacian."""
    v0 = row_sum ** 0.5
    return v0 / jnp.linalg.norm(v0)
