import typing as tp
from functools import partial

import gin
import jax.numpy as jnp
from jax.experimental.sparse.ops import JAXSparse
from spax import ops
from spax.utils import diag, eye

register = partial(gin.register, module="grax.graph_utils.laplacians")


@register
def laplacian(
    adj: JAXSparse, shift: float = 0.0, return_row_sum: bool = True,
) -> tp.Union[JAXSparse, tp.Tuple[JAXSparse, jnp.ndarray]]:
    """
    Get a possibly shifted Laplacian matrix.

    L = D - shift * I - adj

    where `D = diag(sum(adj, axis=1))`

    Args:
        adj: [n, n] adjacency matrix.
        shift: scalar diagonal shift.
        return_row_sum: if True, returns the row sum as well.

    Returns:
        L: [n, n] Sparse Laplacian as above
        row_sum: [n] sum over rows of `adj` if `return_row_sum`.
    """
    row_sum = ops.sum(adj, axis=1)
    L = ops.subtract(diag(row_sum - shift), adj)
    if return_row_sum:
        return L, row_sum
    return L


@register
def normalized_laplacian(
    adj: tp.Union[JAXSparse, jnp.ndarray], shift: float = 0.0, return_row_sum=True
) -> tp.Union[JAXSparse, tp.Tuple[JAXSparse, jnp.ndarray]]:
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
    if return_row_sum:
        return L, row_sum
    return L


def normalized_laplacian_zero_eigenvector(row_sum: jnp.ndarray):
    """Get the normalized eigenvector for eigenvalue zero of a normalized Laplacian."""
    v0 = jnp.sqrt(row_sum)
    return v0 / jnp.linalg.norm(v0)
