import functools

import gin
import jax.numpy as jnp

from grax.huf_utils import SplitData
from grax.problems.single.data import SemiSupervisedSingle, ids_to_mask

configurable = functools.partial(gin.configurable, module="igat.data")


@configurable
def get_split_data(data: SemiSupervisedSingle) -> SplitData:
    row = data.graph.row
    col = data.graph.col
    f = data.node_features
    size = data.num_nodes
    train_ex, validation_ex, test_ex = (
        ((row, col, f), data.labels, ids_to_mask(ids, size, dtype=jnp.float32))
        for ids in (data.train_ids, data.validation_ids, data.test_ids)
    )
    return SplitData((train_ex,), (validation_ex,), (test_ex,))
