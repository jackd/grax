from functools import partial
from typing import Callable

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
from absl import flags
from jax.config import config
from spax.linalg import eigh_jvp
from spax.linalg import subspace_iteration as si
from spax.types import ArrayOrFun

from grax.graph_utils import laplacians as lap
from grax.problems.single.data import dgl_data, get_largest_component

flags.DEFINE_bool("deflate", default=False, help="deflate known laplacians eigenvetor")
flags.DEFINE_bool("rev", default=False, help="benchmark forward + gradient computation")
flags.DEFINE_bool(
    "subsequent",
    default=False,
    help="benchmark subsequent guess after small weights shift",
)
flags.DEFINE_integer("max_iters", default=10000, help="maximum number of iterations")
flags.DEFINE_integer("k", default=4, help="number of extreme eigenpairs to find")
flags.DEFINE_integer("seed", default=0, help="seed used for initial solution")
flags.DEFINE_float("tol", default=None, help="tolerance")
flags.DEFINE_string("dtype", default="float32", help="data type to use")
flags.DEFINE_string("data", default="pubmed", help="data name")

FLAGS = flags.FLAGS

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)

backend = "gpu"

_cache = {}


def get_citations_inputs(name):

    if name in _cache:
        return _cache[name]

    # pubmed: 19717 nodes, 108365 edges
    data = dgl_data(name=name)
    data = get_largest_component(data)
    _cache[name] = data
    return data


def _benchmark_subspace_iteration_method(
    state, method: Callable, a: ArrayOrFun, v0: jnp.ndarray, outer_impl: Callable
):
    kwargs = dict(max_iters=FLAGS.max_iters, tol=FLAGS.tol)
    if FLAGS.rev:
        keys = jax.random.split(jax.random.PRNGKey(123), 3)

        @jax.jit
        def method_with_grad(a, v0, **kwargs):
            w, v, info = method(a, v0, **kwargs)
            # grad_w = jax.random.normal(keys[0], shape=w.shape, dtype=w.dtype)
            # grad_v = jax.random.normal(keys[1], shape=v.shape, dtype=v.dtype)
            grad_w = jnp.ones_like(w)
            grad_v = jnp.ones_like(v)
            x0 = jax.random.normal(keys[2], shape=v.shape, dtype=v.dtype)
            grad_data, _ = eigh_jvp.eigh_partial_rev(
                grad_w,
                grad_v,
                w,
                v,
                x0,
                a,
                outer_impl=outer_impl,
                tol=si.default_tol(v.dtype),
            )
            return w, v, info, grad_data

        fun = partial(method_with_grad, a, v0, **kwargs)
    else:
        fun = partial(method, a, v0, **kwargs)
    out = fun()
    if FLAGS.rev:
        grad_data = out[3]
        assert jnp.all(jnp.isfinite(grad_data))
    # info = out[2]
    # print(info)
    while state:
        fun()


def _get_a(row, col, size, weights):
    data, row, col, row_sum = lap.normalized_laplacian_coo(
        row, col, size, dtype=FLAGS.dtype, shift=2.0, weights=weights
    )
    a = coo.matmul_fun(data, row, col, jnp.zeros((size,), dtype=bool))
    if FLAGS.deflate:
        a = utils.deflate_eigenvector(
            a, lap.normalized_laplacian_zero_eigenvector(row_sum)
        )
    return a, row, col, row_sum


def _benchmark_citations(state, method: Callable):
    k = FLAGS.k
    dtype = FLAGS.dtype
    row0, col0, m = get_citations_inputs(name=FLAGS.data)
    a, row, col, row_sum = _get_a(row0, col0, m, weights=None)
    outer_impl = coo.masked_outer_fun(row, col)
    v0_key, weights_key = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    if FLAGS.deflate:
        v0 = jax.random.normal(v0_key, shape=(m, k - 1), dtype=dtype)
    else:
        v0 = jax.random.normal(v0_key, shape=(m, k), dtype=dtype)
        v0 = jax.ops.index_update(
            v0, jax.ops.index[:, 0], lap.normalized_laplacian_zero_eigenvector(row_sum)
        )

    if FLAGS.subsequent:
        _, v0, _ = method(a, v0, max_iters=FLAGS.max_iters, tol=FLAGS.tol)
        weights = 1 + 0.01 * jax.random.normal(
            weights_key, shape=row0.shape, dtype=dtype
        )
        weights = coo.symmetrize(weights, row0, col0, m)
        a = _get_a(row0, col0, m, weights)[0]

    _benchmark_subspace_iteration_method(state, method, a, v0, outer_impl)


# @benchmark.register
# def benchmark_scipy_citations(state):
#     import numpy as np
#     import scipy.sparse as sp

#     dtype = FLAGS.dtype
#     row, col, m = get_citations_inputs(name=FLAGS.data)
#     data, row, col, row_sum = lap.normalized_laplacian_coo(
#         row, col, m, dtype=dtype, shift=2.0
#     )
#     v0 = np.asarray(lap.normalized_laplacian_zero_eigenvector(row_sum))
#     coo = sp.coo_matrix(
#         (np.asarray(data), (np.asarray(row), np.asarray(col))),
#         shape=(m, m),
#         dtype=dtype,
#         copy=True,
#     )
#     fun = partial(
#         sp.linalg.eigsh,
#         A=coo,
#         k=FLAGS.k,
#         v0=v0,
#         maxiter=FLAGS.max_iters,
#         tol=FLAGS.tol or si.default_tol(dtype),
#     )

#     w, v = fun()
#     err = np.linalg.norm(coo @ v - v * w, axis=0)
#     print(f"coo err: {err}")
#     while state:
#         fun()


@benchmark.register
def benchmark_basic_citations(state):
    _benchmark_citations(state, method=jax.jit(si.basic_subspace_iteration))


@benchmark.register
def benchmark_basic_chebyshev_citations(state):
    @jax.jit
    def basic_chebyshev_subspace_iteration(a, v0, **kwargs):
        # accelerator = si.chebyshev_accelerator_fun(a, 4 * v0.shape[1])
        accelerator = si.chebyshev_accelerator_fun(4 * v0.shape[1], 2, a)
        return si.basic_subspace_iteration(a, v0, accelerator=accelerator, **kwargs)

    _benchmark_citations(state, method=basic_chebyshev_subspace_iteration)


@benchmark.register
def benchmark_projected_citations(state):
    _benchmark_citations(state, method=jax.jit(si.projected_subspace_iteration))


@benchmark.register
def benchmark_projected_chebyshev_citations(state):
    @jax.jit
    def projected_chebyshev_subspace_iteration(a, v0, **kwargs):
        # accelerator = si.chebyshev_accelerator_fun(a, 4 * v0.shape[1])
        accelerator = si.chebyshev_accelerator_fun(4 * v0.shape[1], 2.0, a)
        return si.projected_subspace_iteration(a, v0, accelerator=accelerator, **kwargs)

    _benchmark_citations(state, method=projected_chebyshev_subspace_iteration)


# @benchmark.register
# def benchmark_locking_projected_citations(state):
#     _benchmark_citations(state, method=si.locking_projected_subspace_iteration)


# @benchmark.register
# def benchmark_subspace_iteration_coo_vmapped(state):
#     k = 2
#     num_heads = 8
#     tol = 1e-30
#     row, col, m = get_citations_inputs()
#     sized = jnp.zeros((m,), dtype=bool)
#     weights, row, col = normalized_shifted_laplacian(row, col, m, num_heads=num_heads)
#     v0 = jnp.asarray(
#         np.random.default_rng(1).normal(size=(m, k, num_heads)), dtype=dtype
#     )

#     def base_fun(data, v0, row, col, sized):
#         A = coo.matmul_fun(data, row, col, sized)
#         w, v, info = subspace_iteration(A, v0, tol=tol)
#         return w, v, info

#     vmapped_fun = jax.vmap(
#         base_fun, in_axes=(1, 2, None, None, None), out_axes=(1, 2, 0)
#     )

#     fun = partial(jax.jit(vmapped_fun, backend=backend), weights, v0, row, col, sized)
#     w, v, info = fun()
#     del v
#     print("vmapped")
#     print(w)
#     print(info.err.T)
#     print(f"iters = {info.iterations}")

#     while state:
#         fun()

# @benchmark.register
# def benchmark_subspace_iteration_0coo_again(state):
#     row, col, v0 = get_citations_inputs(dtype=dtype)
#     m = v0.shape[0]

#     weights = get_shifted_normalized_laplacian_weights(m, row, col)
#     weights = jnp.asarray(weights, dtype=dtype)
#     row = jnp.asarray(row, dtype=np.int32)
#     col = jnp.asarray(col, dtype=np.int32)
#     sized = jnp.zeros((m,), dtype=bool)

#     A = coo.matmul_fun(weights, row, col, sized)
#     fun = partial(subspace_iteration, A, v0, tol=1e-3)

#     w, v, state = fun()
#     print(w)
#     print(f"err = {state.err}")
#     print(f"iters = {state.iterations}")

#     weights = np.random.default_rng(132).normal(size=(row.size,), loc=1, scale=0.006)
#     weights = get_shifted_normalized_laplacian_weights(m, row, col, weights)
#     weights = jnp.asarray(weights, dtype=dtype)

#     A = coo.matmul_fun(weights, row, col, sized)
#     fun = partial(subspace_iteration, A, v, tol=1e-3)
#     print("Again:")
#     w, v, state = fun()
#     print(w)
#     print(f"err = {state.err}")
#     print(f"iters = {state.iterations}")

#     from time import time

#     while state:
#         t0 = time()
#         fun()
#         print(time() - t0)


if __name__ == "__main__":
    benchmark.main()
