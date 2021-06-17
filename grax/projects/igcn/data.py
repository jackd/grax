import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp
import spax
from jax.experimental.sparse_ops import COO
from spax.linalg import linear_operators as lin_ops

from grax.graph_utils.transforms import symmetric_normalize
from grax.huf_utils import SplitData
from grax.problems.single.data import SemiSupervisedSingle, ids_to_mask

configurable = partial(gin.configurable, module="igcn.data")


def ids_to_mask_matrix(ids: jnp.ndarray, size: int, dtype=jnp.float32):
    nl = ids.size
    return (
        jnp.zeros((size, nl), dtype=dtype)
        .at[ids, jnp.arange(nl, dtype=ids.dtype)]
        .set(jnp.ones((nl,), dtype=dtype))
    )


def _get_propagator(
    adj: COO, epsilon: float, tol: float = 1e-5, rescale: bool = False
) -> lin_ops.LinearOperator:
    x = adj
    x = spax.ops.scale(symmetric_normalize(x), -(1 - epsilon))
    # x = lin_ops.identity_plus(lin_ops.MatrixWrapper(x, is_self_adjoint=True))
    with jax.experimental.enable_x64():
        x = spax.ops.add(spax.eye(x.shape[0], x.dtype, x.row.dtype), x)
    x = lin_ops.SelfAdjointInverse(
        lin_ops.MatrixWrapper(x, is_self_adjoint=True), tol=tol
    )
    if rescale:
        x = lin_ops.scale(x, epsilon / 2)
    return x


@configurable
def preprocessed_logit_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
):
    prop = (propagator @ ids_to_mask_matrix(ids, propagator.shape[1])).T
    return (((prop, features), labels[ids]),)


@configurable
def lazy_logit_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
):
    # return (((lin_ops.take(propagator, ids), features), labels[ids]),)
    return (
        (
            (propagator, features, ids),
            labels,
            ids_to_mask(ids, propagator.shape[0], dtype=jnp.float32),
        ),
    )


@configurable
def input_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
    concat_original: bool = True,
):
    dense_features = spax.ops.to_dense(features)
    prop_features = (propagator @ dense_features)[ids]
    dense_features = dense_features[ids]
    if concat_original:
        if isinstance(features, COO):
            features = lin_ops.HStacked(spax.ops.to_coo(dense_features), prop_features)
        else:
            features = jnp.concatenate((dense_features, prop_features), axis=-1)
    else:
        features = prop_features
    return ((features, labels[ids]),)


@configurable
def get_split_data(
    data: SemiSupervisedSingle,
    epsilon: float,
    tol: float = 1e-5,
    rescale: bool = False,
    train_fun: tp.Callable = preprocessed_logit_propagated_data,
    validation_fun: tp.Callable = preprocessed_logit_propagated_data,
    test_fun: tp.Callable = lazy_logit_propagated_data,
) -> SplitData:
    """
    Get train/validation/test data for logit-propagated model.

    Args:
        data: `SemiSupervisedSingle` instance
        epsilon: value used in Le = I - (1 - epsilon)*normalized_adjacency.
        tol: value used in conjugate gradient to solve linear system.
        rescale: if True, rescale propagator by epsilon
        train_fun, validation_fun, test_fun: callables that map
            (propagator, node_features, labels, ids) to datasets / iterables.
    Returns:
        `SplitData` where each dataset is a single-example tuple of the form
            `((propagator, node_features), labels[ids])`
    """
    propagator = _get_propagator(data.graph, epsilon, tol, rescale)

    # jit so multiple input_propagated_data funs use the same propagator @ features
    @jax.jit
    def get_data():
        train_data, validation_data, test_data = (
            fun(propagator, data.node_features, data.labels, ids)
            for (fun, ids) in (
                (train_fun, data.train_ids),
                (validation_fun, data.validation_ids),
                (test_fun, data.test_ids),
            )
        )
        return SplitData(train_data, validation_data, test_data)

    return get_data()
