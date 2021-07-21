import jax
import jax.numpy as jnp
from jax.experimental.sparse.ops import COO


def random_adjacency(
    key: jnp.ndarray, num_nodes: int, num_edges: int, dtype=jnp.float32
) -> COO:
    """
    Get the adjacency matrix of a random fully connected undirected graph.

    Note that `num_edges` is only approximate. The process of creating edges it:
    - sample `num_edges` random edges
    - remove self-edges
    - add ring edges
    - add reverse edges
    - filter duplicates

    Args:
        key: `jax.random.PRNGKey`.
        num_nodes: number of nodes in returned graph.
        num_edges: number of random internal edges initially added.
        dtype: dtype of returned JAXSparse.

    Returns:
        COO, shape (num_nodes, num_nodes), weights all ones.
    """
    shape = num_nodes, num_nodes

    internal_indices = jax.random.uniform(
        key, shape=(num_edges,), dtype=jnp.float32, maxval=num_nodes ** 2,
    ).astype(jnp.int32)
    # remove randomly sampled self-edges.
    self_edges = (internal_indices // num_nodes) == (internal_indices % num_nodes)
    internal_indices = internal_indices[jnp.logical_not(self_edges)]

    # add a ring so we know the graph is connected
    r = jnp.arange(num_nodes, dtype=jnp.int32)
    ring_indices = r * num_nodes + (r + 1) % num_nodes
    indices = jnp.concatenate((internal_indices, ring_indices))

    # add reverse indices
    coords = jnp.unravel_index(indices, shape)
    coords_rev = coords[-1::-1]
    indices_rev = jnp.ravel_multi_index(coords_rev, shape)
    indices = jnp.concatenate((indices, indices_rev))

    # filter out duplicates
    indices = jnp.unique(indices)
    row, col = jnp.unravel_index(indices, shape)
    return COO((jnp.ones((row.size,), dtype=dtype), row, col), shape=shape)


def star_adjacency(num_nodes: int, dtype=jnp.float32) -> COO:
    """Get the adjacency matrix of an undirected star graph."""
    row = jnp.zeros((num_nodes - 1), dtype=jnp.int32)
    col = jnp.arange(1, num_nodes, dtype=jnp.int32)
    row, col = jnp.concatenate((row, col)), jnp.concatenate((col, row))
    return COO(
        (jnp.ones((2 * (num_nodes - 1),), dtype=dtype), row, col),
        shape=(num_nodes, num_nodes),
    )
