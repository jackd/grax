"""Gradients derived in [Seeger et al](https://arxiv.org/pdf/1710.08717.pdf)."""

import jax
import jax.numpy as jnp


def _lq(x):
    q, r = jnp.linalg.qr(x.T)
    return r.T, q.T


def _lq_fwd(x):
    l, q = _lq(x)
    return (l, q), (l, q)


def _copyltu(x):
    """Copy lower triangular matrix to upper."""
    return jnp.tril(x) + jnp.tril(x, -1).T


def _lq_bwd(res, g):
    l, q = res
    grad_l, grad_q = g

    m = l.T @ grad_l - grad_q @ q.T
    grad_x = jax.scipy.linalg.solve_triangular(l.T, grad_q + _copyltu(m) @ q)
    return (grad_x,)


lq = jax.custom_vjp(_lq)
lq.defvjp(_lq_fwd, _lq_bwd)


def qr(x):
    """With backward gradient."""
    l, q = lq(x.T)
    return q.T, l.T
