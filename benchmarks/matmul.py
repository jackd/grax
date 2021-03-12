import typing as tp
from functools import partial

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
from jax.config import config

import spax
from grax.graph_utils import laplacians
from grax.problems.single import data

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


data_cache = {}


def load_data(name="pub_med"):
    if name in data_cache:
        return data_cache[name]
    data = data.citations_data(name)
    data_cache[name] = data
    return data


def matmul_benchmark(
    state,
    fmt,
    data_name: str,
    num_vecs: int = 8,
    backend: tp.Optional[str] = "cpu",
    dtype=jnp.float32,
    seed: int = 0,
):
    s = load_data(data_name)
    adj = s.graph
    b = jax.random.normal(
        jax.random.PRNGKey(seed), (s.num_nodes, num_vecs), dtype=dtype
    )
    a, _ = laplacians.normalized_laplacian(adj, shift=2.0)
    if dtype != a.dtype:
        a = spax.ops.with_data(a, a.data.astype(dtype))
    if fmt == "coo":
        a = a.tocoo()
    elif fmt == "csr":
        a = a.tocsr()
    else:
        raise NotImplementedError(f"fmt '{fmt}' not supported")

    def fun(a, b):
        return spax.ops.matmul(a, b)

    if backend:
        fun = jax.jit(fun, backend=backend)

    fun(a, b).block_until_ready()  # ensure jit has finished
    while state:
        fun(a, b).block_until_ready()


for data_name in ("pub_med", "cite_seer", "cora"):
    for dtype, dtype_str in ((jnp.float32, "f32"), (jnp.float64, "f64")):
        for backend in (None, "cpu", "gpu"):
            backend_str = "nojit" if backend is None else backend
            for fmt in "csr", "coo":
                benchmark.register(
                    partial(
                        matmul_benchmark,
                        fmt=fmt,
                        dtype=dtype,
                        backend=backend,
                        data_name=data_name,
                    ),
                    name="-".join((data_name, dtype_str, backend_str, fmt)),
                )


if __name__ == "__main__":
    benchmark.main()
