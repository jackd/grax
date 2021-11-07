"""
Approximate personalized propagation of neural predictions.

https://github.com/benedekrozemberczki/APPNP
"""

import functools

import gin
import jax.numpy as jnp
from spax.linalg import linear_operators as lin_ops
from spax.ops import to_dense

from grax.huf_utils import SplitData
from grax.problems.single.data import SemiSupervisedSingle

configurable = functools.partial(gin.configurable, module="appnp.data")


def dense_propagator(A: lin_ops.LinearOperator, *, alpha: float = 0.1) -> jnp.ndarray:
    return alpha * jnp.linalg.inv(jnp.eye(A.shape[0]) - (1 - alpha) * to_dense(A))


@configurable
def get_exact_split_data(data: SemiSupervisedSingle, alpha=0.1) -> SplitData:
    """Get SplitData with the precomputed ("exact") inverse propagator."""
    prop = dense_propagator(data.graph, alpha=alpha)

    def ids_to_data_split(ids):
        inputs = (prop, data.node_features)
        labels = data.labels
        weights = jnp.zeros((data.num_nodes,), jnp.float32).at[ids].set(1)
        return ((inputs, labels, weights),)

    return SplitData(
        *(
            ids_to_data_split(i)
            for i in (data.train_ids, data.validation_ids, data.test_ids)
        )
    )
