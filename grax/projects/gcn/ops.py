import typing as tp

import jax.numpy as jnp

from spax import SparseArray, ops


def graph_convolution(
    graph: tp.Union[jnp.ndarray, SparseArray],
    features: jnp.ndarray,
    kernel: jnp.ndarray,
    transform_first: tp.Optional[bool] = None,
) -> jnp.ndarray:
    assert graph.ndim == 2, graph.shape
    assert features.ndim == 2, features.shape
    assert kernel.ndim == 2, kernel.shape
    assert graph.shape[1] == features.shape[0], (graph.shape, features.shape)
    assert features.shape[1] == kernel.shape[0], (features.shape, kernel.shape)

    if transform_first is None:
        transform_first = kernel.shape[0] >= kernel.shape[1]

    if transform_first:
        return ops.matmul(graph, features @ kernel)
    return ops.matmul(graph, features) @ kernel
