import jax
import jax.numpy as jnp
from spax.linalg.polynomials import iterate_chebyshev1
from spax.linalg.utils import ArrayOrFun, as_array_fun


def _normalize(b, eps=1e-4):
    return b / jnp.maximum(jnp.linalg.norm(b, axis=0), eps)


def krylov(a: ArrayOrFun, b, dim: int, normalize: bool = False):
    a = as_array_fun(a)

    if normalize:
        b = _normalize(b)

    out = [b]
    for _ in range(dim):
        b = a(b)
        if normalize:
            b = _normalize(b)
        out.append(b)
    out = jnp.stack(out, axis=1)
    return out


def lanczos(a: ArrayOrFun, b, dim: int, eps: float = 1e-4):
    a = as_array_fun(a)
    assert b.ndim == 1
    q = [jnp.zeros_like(b), b / jnp.maximum(jnp.linalg.norm(b, ord=2), eps)]
    alpha = []
    beta = [jnp.zeros((), dtype=b.dtype)]
    for _ in range(dim):
        v = a(q[-1])
        alpha.append(jnp.dot(q[-1], v))
        v = v - beta[-1] * q[-2] - alpha[-1] * q[-1]
        beta.append(jnp.maximum(jnp.linalg.norm(v), eps))
        q.append(v / beta[-1])

    q = q[1:]
    beta = beta[1:]
    return jnp.stack(q, axis=1), jnp.stack(alpha), jnp.stack(beta)


def chebyshev1(a: ArrayOrFun, b, dim: int):
    assert dim >= 2
    a = as_array_fun(a)
    p0 = b
    p1 = a(b)
    out = [p0, p1]
    for _ in range(dim - 2):
        p0, p1 = iterate_chebyshev1(a, p0, p1)
        out.append(p1)
    return jnp.stack(out, axis=1)


def gated_sum(gate_features, unscaled_features, w, b=None):
    if unscaled_features is None:
        unscaled_features = gate_features
    scale = jnp.dot(gate_features, w)
    if b is not None:
        scale = scale + b
    scale = jnp.squeeze(scale, axis=-1)
    scale = jax.nn.sigmoid(scale)
    return jnp.einsum("nkl,nk->nl", unscaled_features, scale)
