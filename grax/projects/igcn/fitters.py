import operator
from functools import partial

import gin

import jax
import jax.numpy as jnp
from grax.huf_utils import FitData
from grax.problems.single.data import SemiSupervisedSingle, ids_to_mask
from grax.projects.igcn.ops import shifted_laplacian
from grax.pytrees import MatrixInverse
from huf.models import Model

configurable = partial(gin.configurable, module="igcn")


@configurable
def batched_fit_data(
    model: Model,
    data: SemiSupervisedSingle,
    labels_per_batch: int,
    nodes_per_batch: int,
    eps: float = 0.1,
) -> FitData:
    raise NotImplementedError("TODO")


@configurable
def simple_fit_data(
    model: Model, data: SemiSupervisedSingle, eps: float = 0.1, tol: float = 1e-5,
):
    inv = MatrixInverse(shifted_laplacian(data.graph, eps), tol=tol)
    node_features = data.node_features
    train_ex, validation_ex, test_ex = (
        ((inv, node_features), data.labels, ids_to_mask(ids, node_features.size[0]))
        for ids in (data.train_ids, data.validation_ids, data.test_ids)
    )
    return FitData(model, (train_ex,), (validation_ex,), (test_ex,))


@configurable
def preprocessed_fit_data(
    model: Model,
    data: SemiSupervisedSingle,
    eps: float = 0.1,
    tol: float = 1e-5,
    preprocess_validation: bool = True,
    preprocess_test: bool = False,
) -> FitData:
    graph = data.graph
    Le = shifted_laplacian(graph, eps)
    node_features = data.node_features

    def get_m(ids):
        n = data.node_features.shape[0]
        m = ids.size
        return jnp.zeros((n, m)).at[ids, jnp.arange(m)].set(jnp.ones((m,)))

    def get_inv(ids):
        m = get_m(ids)
        return jax.scipy.sparse.linalg.cg(
            jax.tree_util.Partial(operator.matmul, Le), m, tol=tol,
        )[0].T

    def get_data(ids, preprocess):
        if preprocess:
            example = (
                (get_inv(ids), node_features),
                data.labels[ids],
            )
        else:
            example = (
                (MatrixInverse(Le), node_features),
                data.labels,
                ids_to_mask(ids, node_features.shape[0]),
            )
        return (example,)

    train_data, validation_data, test_data = (
        get_data(i, p)
        for i, p in (
            (data.train_ids, True),
            (data.validation_ids, preprocess_validation),
            (data.test_ids, preprocess_test),
        )
    )

    return FitData(model, train_data, validation_data, test_data)
