import typing as tp
from functools import partial

import gin
import jax.numpy as jnp

import haiku as hk
import huf
import spax
from grax.problems.single import data
from huf.metrics import get_combined_metrics
from huf.types import FitState, Metrics, Splits

configurable = partial(gin.configurable, module="grax.problems.single")


@configurable
def semi_supervised_net_fun(
    inputs, is_training: bool, *, module_fun: tp.Callable[[], hk.Module], **kwargs
):
    # graph, node_features = inputs
    coords, data, node_features = inputs
    size = node_features.shape[0]
    graph = spax.COO(coords, data, (size, size))
    module = module_fun(**kwargs)
    return module(graph, node_features, is_training)


@configurable
def as_example(
    data: data.SemiSupervisedSingle, split: str = Splits.TRAIN, dtype=jnp.float32
):
    # not sure why we need to unpack/repack graph, but avoids errors in GAT
    if split == Splits.TRAIN:
        mask = data.train_mask
    elif split == Splits.VALIDATION:
        mask = data.validation_mask
    elif split == Splits.TEST:
        mask = data.test_mask
    else:
        raise ValueError(f"Invalid split {split}.")
    graph = data.graph
    inputs = graph.coords, graph.data.astype(dtype), data.node_features.astype(dtype)
    return (inputs, data.labels, mask.astype(dtype))


@configurable
def fit_semi_supervised(
    model: huf.models.Model,
    initial_state: tp.Union[int, huf.types.PRNGKey, FitState],
    data: data.SemiSupervisedSingle,
    steps: int,
    callbacks: tp.Iterable[huf.callbacks.Callback] = (),
    verbose: bool = True,
    dtype=jnp.float32,
) -> Metrics:
    train_example = as_example(data, Splits.TRAIN, dtype=dtype)
    validation_example = as_example(data, Splits.VALIDATION, dtype=dtype)
    test_example = as_example(data, Splits.TEST, dtype=dtype)

    result = model.fit(
        initial_state,
        [train_example],
        epochs=steps,
        validation_data=[validation_example],
        callbacks=callbacks,
        verbose=verbose,
    )
    model_state = result.state.model_state
    test_metrics = model.evaluate(model_state, [test_example])
    metrics = get_combined_metrics(
        result.train_metrics, result.validation_metrics, test_metrics
    )
    return metrics
