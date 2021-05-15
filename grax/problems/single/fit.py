from functools import partial

import gin

import huf
import jax
import jax.numpy as jnp
import spax
from grax.huf_utils import FitData
from grax.problems.single import data
from huf.types import Splits
from jax.experimental.sparse_ops import COO, CSR, JAXSparse

configurable = partial(gin.configurable, module="grax.problems.single")


@configurable
def semi_supervised_fit_data(
    model: huf.models.Model, data: data.SemiSupervisedSingle, dtype=jnp.float32
) -> FitData:
    train_example, validation_example, test_example = as_examples(data, dtype=dtype)
    return FitData(model, (train_example,), (validation_example,), (test_example,))


def get_mask(data: data.SemiSupervisedSingle, split: str):
    if split == Splits.TRAIN:
        return data.train_mask
    if split == Splits.VALIDATION:
        return data.validation_mask
    if split == Splits.TEST:
        return data.test_mask
    raise ValueError(f"Invalid split {split}, must be in {Splits.all()}")


@configurable
def as_examples(
    data: data.SemiSupervisedSingle,
    splits=(Splits.TRAIN, Splits.VALIDATION, Splits.TEST),
    dtype=jnp.float32,
):
    def cast(x):
        if isinstance(x, jnp.ndarray):
            return x.astype(dtype)
        if isinstance(x, COO):
            return COO((cast(x.data), x.row, x.col), shape=x.shape)
        if isinstance(x, CSR):
            return CSR((cast(x.data), x.indices, x.indptr), shape=x.shape)
        raise TypeError(f"Unrecognized type for x `{type(x)}`")

    # def map_node_features(x):
    #     if isinstance(x, jnp.ndarray):
    #         return cast(x)
    #     assert isinstance(x, COO)
    #     return (cast(x.data), x.row, x.col)

    # not sure why we need to unpack/repack graph, but avoids errors in GAT
    graph = cast(spax.ops.to_coo(data.graph))
    node_features = jax.tree_map(
        cast,
        data.node_features,
        is_leaf=lambda x: isinstance(x, (jnp.ndarray, JAXSparse)),
    )

    # inputs = (
    #     (cast(graph.data), graph.row, graph.col),
    #     tuple(map_node_features(x) for x in data.node_features),
    # )
    return tuple(
        ((graph, node_features), data.labels, cast(get_mask(data, split)))
        for split in splits
    )
