import operator
from functools import partial

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
import spax
from jax.config import config

from grax.problems.single import data

config.parse_flags_with_absl()


data_cache = {}


def load_data(name="pubmed"):
    if name in data_cache:
        return data_cache[name]
    if name == "arxiv":
        res = data.ogbn_data(name)
    else:
        res = data.dgl_data(name)
    data_cache[name] = res
    return res


def matmul_benchmark(
    state,
    fmt,
    data_name: str,
    epsilon: float = 0.1,
    num_vecs: int = 4,
    backend: str = "cpu",
    seed: int = 0,
):
    s = load_data(data_name)
    adj = spax.ops.to_coo(s.graph)
    d = jax.lax.rsqrt(spax.ops.sum(adj, axis=1))
    adj = spax.ops.with_data(adj, adj.data * d[adj.row] * d[adj.col])
    n = adj.shape[0]
    with jax.experimental.enable_x64():
        Le = spax.ops.add(spax.eye(n), spax.ops.scale(adj, epsilon - 1))
    b = jax.random.normal(jax.random.PRNGKey(seed), (n, num_vecs), dtype=jnp.float32)
    if fmt == "coo":
        Le = spax.ops.to_coo(Le)
    elif fmt == "csr":
        Le = spax.ops.to_csr(Le)
    else:
        raise NotImplementedError(f"fmt '{fmt}' not supported")

    @partial(jax.jit, backend=backend)
    def fun(a, b):
        return jax.scipy.sparse.linalg.cg(partial(operator.matmul, a), b)[0]

    fun(Le, b).block_until_ready()  # ensure jit has finished
    while state:
        fun(Le, b).block_until_ready()


# datasets = ("pubmed", "citeseer", "cora")
datasets = ("arxiv",)
# preload datasets to avoid spam later
for data_name in datasets:
    load_data(data_name)
for data_name in datasets:
    for backend in ("cpu", "gpu"):
        for fmt in "csr", "coo":
            for num_vecs in (4, 8, 16):
                benchmark.register(
                    partial(
                        matmul_benchmark,
                        fmt=fmt,
                        backend=backend,
                        data_name=data_name,
                        num_vecs=num_vecs,
                    ),
                    name="-".join((data_name, backend, fmt, str(num_vecs))),
                )


if __name__ == "__main__":
    benchmark.main()
