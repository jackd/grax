import typing as tp

import jax.numpy as jnp
from jax.experimental.sparse.ops import JAXSparse


def graph_convolution(
    graph: tp.Union[jnp.ndarray, JAXSparse],
    features: tp.Union[jnp.ndarray, JAXSparse],
    kernel: jnp.ndarray,
    transform_first: tp.Optional[bool] = None,
) -> jnp.ndarray:
    assert len(graph.shape) == 2, graph.shape
    assert len(features.shape) == 2, features.shape
    assert kernel.ndim == 2, kernel.shape
    assert graph.shape[1] == features.shape[0], (graph.shape, features.shape)
    assert features.shape[1] == kernel.shape[0], (features.shape, kernel.shape)

    if transform_first is None:
        transform_first = kernel.shape[0] >= kernel.shape[1]

    if transform_first:
        return graph @ (features @ kernel)
    return (graph @ features) @ kernel
