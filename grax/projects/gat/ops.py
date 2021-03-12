import jax
import jax.numpy as jnp

import spax


def graph_conv(graph: spax.SparseArray, graph_data: jnp.ndarray, values: jnp.ndarray):
    return spax.ops.matmul(spax.ops.with_data(graph, graph_data), values)


def multi_head_graph_conv(
    graph: spax.SparseArray, graph_data: jnp.ndarray, values: jnp.ndarray
):
    """
    Args:
        graph: [N, N] sparsity. `data` is not used.
        graph_data: [E, heads]
        values: N, heads, filters]

    Out:
        [N, heads, filters]
    """
    assert graph_data.ndim == 2, graph_data.shape
    assert values.ndim == 3, values.shape
    assert graph_data.shape[1] == values.shape[1], (graph_data.shape, values.shape)
    assert graph.shape[0] == graph.shape[1] == values.shape[0], (
        graph.shape,
        values.shape,
    )

    # out = []
    # for i in range(graph_data.shape[1]):
    #     out.append(graph_conv(graph, graph_data[:, i], values[:, i]))
    # return jnp.stack(out, axis=1)

    return jax.vmap(graph_conv, in_axes=(None, 1, 1), out_axes=1)(
        graph, graph_data, values
    )
