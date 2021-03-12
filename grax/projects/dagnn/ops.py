import typing as tp

import jax.numpy as jnp

import spax
from spax.linalg.utils import as_array_fun


def krylov(a: tp.Union[jnp.ndarray, spax.SparseArray], b, dim: int, axis: int = 1):
    out = [b]
    a = as_array_fun(a)
    for _ in range(dim):
        b = a(b)
        out.append(b)
    return jnp.stack(out, axis=axis)
