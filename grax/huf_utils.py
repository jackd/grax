import jax.numpy as jnp

import spax
from huf.avals import register_zeros


@register_zeros(spax.AbstractCOO)
def zeros_like_coo(aval: spax.AbstractCOO):
    return spax.COO(
        jnp.zeros((aval.ndim, aval.nnz), dtype=jnp.int32),
        jnp.zeros((aval.nnz,), dtype=aval.dtype),
        shape=aval.shape,
    )


@register_zeros(spax.AbstractCSR)
def zeros_like_csr(aval: spax.AbstractCSR):
    return spax.CSR(
        jnp.zeros((aval.nnz,), dtype=jnp.int32),
        jnp.concatenate(([0], jnp.full((aval.shape[0],), aval.nnz, dtype=jnp.int32)),),
        jnp.zeros((aval.nnz,), dtype=aval.dtype),
        aval.shape,
    )
