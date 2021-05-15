from functools import partial

import gin

import jax
import spax
from jax.experimental.sparse_ops import JAXSparse

configurable = partial(gin.configurable, module="igcn.ops")


@configurable
def shifted_laplacian(
    adjacency: JAXSparse, eps: float, rescale: bool = False
) -> JAXSparse:
    with jax.experimental.enable_x64():
        rs = spax.ops.sum(adjacency, axis=0)
        d = jax.lax.rsqrt(rs)
        A = spax.ops.scale_rows(spax.ops.scale_columns(adjacency, d), d)
        n = A.shape[0]
        lap = spax.ops.add(spax.eye(n), spax.ops.scale(A, -(1 - eps)))
    if rescale:
        lap = spax.ops.scale(lap, 1 / eps)
    return lap
