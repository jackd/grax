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


def load_data(name="pubmed"):
    if name in data_cache:
        return data_cache[name]
    res = data.dgl_data(name)
    data_cache[name] = res
    return res


def matmul_benchmark(
    state,
    fmt,
    data_name: str,
    num_vecs: int = 8,
    backend: str = "cpu",
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
        a = spax.ops.to_coo(a)
    elif fmt == "csr":
        a = spax.ops.to_csr(a)
    else:
        raise NotImplementedError(f"fmt '{fmt}' not supported")

    @partial(jax.jit, backend=backend)
    def fun(a, b):
        return a @ b

    fun(a, b).block_until_ready()  # ensure jit has finished
    while state:
        fun(a, b).block_until_ready()


datasets = ("pubmed", "citeseer", "cora")
# preload datasets to avoid spam later
for data_name in datasets:
    load_data(data_name)
for data_name in datasets:
    for dtype, dtype_str in ((jnp.float32, "f32"), (jnp.float64, "f64")):
        for backend in ("cpu", "gpu"):
            for fmt in "csr", "coo":
                benchmark.register(
                    partial(
                        matmul_benchmark,
                        fmt=fmt,
                        dtype=dtype,
                        backend=backend,
                        data_name=data_name,
                    ),
                    name="-".join((data_name, dtype_str, backend, fmt)),
                )


if __name__ == "__main__":
    benchmark.main()
